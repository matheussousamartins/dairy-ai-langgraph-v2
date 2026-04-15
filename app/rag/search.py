"""
rag/search.py — Busca no Knowledge Base (vetorial, textual, híbrida)

Este módulo é o CORAÇÃO do RAG. Quando um agente recebe uma pergunta,
ele chama a função daqui para buscar os chunks mais relevantes na base
de conhecimento. É uma adaptação do app/rag/tools.py do projeto original.

Fluxo de uma busca:
  1. Recebe a query do usuário ("como fabricar mussarela?")
  2. (Opcional) HyDE: gera um documento hipotético para melhorar a busca
  3. Gera o embedding da query via OpenAI
  4. Chama uma função SQL no Supabase:
     - kb_vector_search: busca por similaridade de cosseno
     - kb_text_search: busca por palavras-chave (FTS)
     - kb_hybrid_search: combina ambas com RRF (Reciprocal Rank Fusion)
  5. (Opcional) Reranking: reordena os resultados com Cohere
  6. Retorna os top K chunks mais relevantes

Funções SQL (definidas em sql/03_kb_functions.sql):
  Essas funções rodam DENTRO do Postgres, não no Python. O Python
  apenas chama a função passando os parâmetros. A vantagem é performance:
  a busca vetorial e a busca textual acontecem no banco, sem transferir
  todos os chunks para o Python.

  kb_vector_search(embedding, k, threshold, agent_table):
    Calcula score = 1 - distância_cosseno entre o embedding da query
    e cada chunk da tabela. Retorna os top K com score > threshold.

  kb_text_search(query_text, k, agent_table):
    Usa Full-Text Search (FTS) do Postgres. Converte a query em
    ts_query ("fabricar" & "mussarela") e busca nos chunks que
    contêm essas palavras. Ranqueia por ts_rank_cd (frequência +
    proximidade dos termos).

  kb_hybrid_search(query_text, embedding, k, threshold, agent_table):
    Executa AMBAS as buscas (vector + text) e combina os resultados
    com Reciprocal Rank Fusion (RRF). RRF é uma fórmula que pondera
    a posição de cada resultado nas duas listas:
      score_rrf = 1.5/(60 + rank_vector) + 1.0/(60 + rank_text)
    O peso 1.5 para vector e 1.0 para text foi definido empiricamente
    no projeto original (funciona bem para a maioria dos casos).

Adaptações em relação ao original:
  - Original filtra por empresa/client_id (CRM multi-tenant)
  - Adaptado filtra por table_name (cada agente = uma tabela)
  - Original tem busca global; adaptado é sempre por agente
  - Adicionado: a tool LangChain (kb_search_tool) que os agentes usam
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool

from app.config import (
    EMBEDDING_MODEL,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_K,
    MATCH_THRESHOLD,
    USE_HYDE,
    HYDE_LLM_MODEL,
    RERANKER,
    RERANK_CANDIDATES,
)
from app.db.connection import get_supabase_conn

_embeddings_model = None
_hyde_model = None
try:
    _hybrid_workers = int(os.getenv("RAG_HYBRID_WORKERS", "8"))
except ValueError:
    _hybrid_workers = 8
_hybrid_executor = ThreadPoolExecutor(max_workers=max(2, _hybrid_workers))


# ============================================================
# Utilitário: converter vetor Python em literal SQL
# ============================================================

def vec_to_literal(v: List[float]) -> str:
    """Converte uma lista de floats em formato aceito pelo pgvector.
    
    O pgvector espera vetores como texto no formato "[0.1, 0.2, ...]".
    Esta função converte a lista Python para essa string.
    
    Exemplo:
        vec_to_literal([0.1, 0.2, 0.3]) → "[0.100000,0.200000,0.300000]"
    
    O formato com 6 casas decimais é suficiente para preservar a precisão
    dos embeddings da OpenAI (que retornam ~7 casas significativas).
    
    Função idêntica ao original (tools.py linha 17).
    """
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ============================================================
# Geração de embeddings
# ============================================================

def embed_query(text: str) -> List[float]:
    """Gera o embedding (vetor) de um texto.
    
    Usado para transformar a pergunta do usuário em vetor
    antes de buscar por similaridade no banco.
    
    O modelo text-embedding-3-small gera vetores de 1536 dimensões.
    Cada dimensão é um float entre -1 e 1 que captura um aspecto
    do significado semântico do texto.
    
    Textos com significado parecido geram vetores próximos no espaço
    vetorial. "Como fabricar mussarela?" e "Processo de produção de
    queijo mussarela" geram vetores quase idênticos, mesmo com
    palavras diferentes.
    
    Custo: ~$0.00002 por chamada (extremamente barato).
    Latência: ~100-200ms.
    """
    return _get_embeddings_model().embed_query(text)


# ============================================================
# HyDE — Hypothetical Document Embeddings
# ============================================================

def generate_hyde_document(query: str) -> str:
    """Gera um documento hipotético relevante para a query.
    
    HyDE (Hypothetical Document Embeddings) é uma técnica que melhora
    o retrieval para perguntas vagas ou curtas.
    
    Problema: a pergunta "queijo" gera um embedding genérico que pode
    retornar chunks sobre qualquer aspecto de queijo.
    
    Solução: antes de buscar, pedimos ao LLM para imaginar como seria
    um documento que respondesse essa pergunta. O LLM gera algo como:
    "O queijo é um produto lácteo obtido pela coagulação do leite.
    Existem diversos tipos como mussarela, prato, minas..."
    
    Esse documento hipotético gera um embedding muito mais rico e
    específico do que a pergunta original. Usamos esse embedding
    para a busca vetorial.
    
    Trade-off:
      + Melhora retrieval para perguntas curtas/vagas
      - Adiciona ~500ms + custo de 1 chamada LLM por busca
      - Pode enviesar a busca se o LLM "inventar" algo errado
    
    Função baseada no original (tools.py linhas 147-157).
    Diferença: usa HYDE_LLM_MODEL do config em vez de hardcoded.
    """
    llm = _get_hyde_model()
    
    prompt = (
        "Você é um especialista em tecnologia de laticínios. "
        "Escreva um parágrafo técnico conciso que seria altamente "
        "relevante para a seguinte pergunta. Não invente dados "
        "numéricos específicos.\n\n"
        f"Pergunta: {query}"
    )
    
    response = llm.invoke(prompt)
    return (response.content or "").strip()


def _get_embeddings_model() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings_model


def _get_hyde_model() -> ChatOpenAI:
    global _hyde_model
    if _hyde_model is None:
        _hyde_model = ChatOpenAI(
            model=HYDE_LLM_MODEL,
            temperature=0.3,  # Baixa criatividade (queremos algo factual)
        )
    return _hyde_model


# ============================================================
# Busca no banco (chamadas às funções SQL)
# ============================================================

def search_vector(
    query_embedding: List[float],
    table_name: str,
    k: int,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Busca vetorial pura (similaridade de cosseno).
    
    Chama a função SQL kb_vector_search no Supabase.
    
    Como funciona internamente (no SQL):
      1. Converte o embedding da query para tipo vector(1536)
      2. Calcula a distância de cosseno entre a query e CADA chunk
      3. Converte distância em score: score = 1 - distância
         (score 1.0 = idêntico, 0.0 = completamente diferente)
      4. Filtra chunks com score >= threshold (se definido)
      5. Ordena por score decrescente
      6. Retorna os top K
    
    A busca usa o índice HNSW (Hierarchical Navigable Small World),
    que é uma estrutura de dados otimizada para busca aproximada
    de vizinhos mais próximos. Em vez de comparar com TODOS os chunks
    (O(n)), o HNSW navega por uma estrutura em camadas e encontra
    os vizinhos mais próximos em O(log n). Para 10.000 chunks,
    a diferença é: brute force = 10.000 comparações, HNSW ≈ 13.
    
    Parâmetros:
        query_embedding: Vetor de 1536 dimensões da pergunta.
        table_name: Nome da tabela (ex: "embeddings_agente_1_queijos").
        k: Quantos resultados retornar.
        threshold: Score mínimo (None = sem filtro).
    
    Retorna:
        Lista de dicts com: content, score, metadata.
    """
    vec_lit = vec_to_literal(query_embedding)
    
    with get_supabase_conn() as conn:
        with conn.cursor() as cur:
            # Chama a busca vetorial na tabela específica do agente.
            # NOTA sobre segurança: table_name é formatado com .format()
            # porque não pode ser parâmetro SQL (%s). Isso é seguro aqui
            # pois table_name vem de configuração interna (não input do usuário).
            if threshold is None:
                cur.execute(
                    """
                    SELECT content,
                           1 - (embedding <=> %s::vector) as score,
                           metadata
                    FROM {}
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                    """.format(table_name),
                    (vec_lit, k),
                )
            else:
                cur.execute(
                    """
                    SELECT content,
                           1 - (embedding <=> %s::vector) as score,
                           metadata
                    FROM {}
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                    """.format(table_name),
                    (vec_lit, vec_lit, threshold, vec_lit, k),
                )

            rows = cur.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "content": row[0],
            "score": float(row[1]) if row[1] is not None else 0.0,
            "metadata": row[2] or {},
        })
    
    return results


def search_text(
    query: str,
    table_name: str,
    k: int,
) -> List[Dict[str, Any]]:
    """Busca textual pura (Full-Text Search do Postgres).
    
    Usa o operador @@ do Postgres para buscar palavras-chave.
    
    Como funciona:
      1. plainto_tsquery('portuguese', query) converte a pergunta em
         tokens de busca. Ex: "fabricar mussarela" → "fabricar & mussarela"
      2. O operador @@ verifica se o campo FTS do chunk contém esses tokens
      3. ts_rank_cd calcula a relevância (baseado em frequência e proximidade)
      4. Ordena por relevância e retorna top K
    
    A busca FTS usa o índice GIN (Generalized Inverted Index),
    que é um índice invertido (como o Google): mapeia cada palavra
    para a lista de documentos que a contêm. A busca é O(1) por
    palavra — extremamente rápida.
    
    Quando usar FTS em vez de vector:
      - Termos exatos: "IN 76", "RDC 331", "Art. 15"
      - Nomes próprios: "Clostridium tyrobutyricum"
      - Códigos: "CCS > 400.000"
    A busca vetorial NÃO é boa para termos exatos porque embeddings
    capturam significado, não palavras específicas.
    
    Requer: coluna FTS na tabela (gerada automaticamente pelo SQL schema).
    """
    with get_supabase_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content,
                       ts_rank_cd(fts, plainto_tsquery('portuguese', %s)) as score,
                       metadata
                FROM {}
                WHERE fts @@ plainto_tsquery('portuguese', %s)
                ORDER BY score DESC
                LIMIT %s
                """.format(table_name),
                (query, query, k),
            )
            
            rows = cur.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "content": row[0],
            "score": float(row[1]) if row[1] is not None else 0.0,
            "metadata": row[2] or {},
        })
    
    return results


def search_hybrid_rrf(
    query: str,
    query_embedding: List[float],
    table_name: str,
    k: int,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Busca híbrida com Reciprocal Rank Fusion (RRF).
    
    Combina busca vetorial + busca textual e funde os resultados
    usando RRF. Esta é a busca mais poderosa do sistema.
    
    Como funciona o RRF:
      1. Executa busca vetorial → lista ordenada por score_vector
      2. Executa busca textual → lista ordenada por score_text
      3. Para cada chunk que aparece em pelo menos uma lista:
         score_rrf = peso_vector/(K_rrf + rank_vector) + peso_text/(K_rrf + rank_text)
         onde K_rrf=60 (constante de suavização), peso_vector=1.5, peso_text=1.0
      4. Ordena por score_rrf e retorna top K
    
    Por que RRF e não média simples?
    A média simples (score_vector + score_text)/2 penaliza chunks
    que aparecem em apenas uma lista (score 0 na outra).
    O RRF usa RANK (posição na lista), não score absoluto.
    Um chunk em 1º lugar na busca vetorial e ausente na textual
    recebe: 1.5/(60+1) + 0 = 0.025. Se estiver em 3º na textual
    também: 1.5/(60+1) + 1.0/(60+3) = 0.025 + 0.016 = 0.041.
    A presença em ambas as listas SOMA, não penaliza.
    
    Os pesos (1.5 vector, 1.0 text) dão mais importância à busca
    semântica. Foram definidos empiricamente no projeto original
    e funcionam bem para a maioria dos casos.
    
    Lógica baseada em sql/kb/03_functions.sql → kb_hybrid_search.
    Aqui executamos em Python em vez de SQL para maior flexibilidade,
    mas a lógica é idêntica.
    """
    # Busca vetorial e textual em paralelo.
    # Como cada busca abre sua própria conexão no pool, conseguimos reduzir
    # a latência total da estratégia híbrida para próximo do ramo mais lento.
    fut_vector = _hybrid_executor.submit(
        search_vector, query_embedding, table_name, k * 3, threshold
    )
    fut_text = _hybrid_executor.submit(search_text, query, table_name, k * 3)
    vector_results = fut_vector.result()
    text_results = fut_text.result()
    
    # RRF: funde as duas listas
    # Mapa: content → { vector_rank, text_rank, content, metadata }
    rrf_map: Dict[str, Dict[str, Any]] = {}
    
    K_RRF = 60          # Constante de suavização (padrão da literatura)
    WEIGHT_VECTOR = 1.5  # Peso da busca vetorial
    WEIGHT_TEXT = 1.0    # Peso da busca textual
    
    # Adiciona resultados da busca vetorial
    for rank, item in enumerate(vector_results, start=1):
        key = item["content"][:200]  # Usa os primeiros 200 chars como chave
        if key not in rrf_map:
            rrf_map[key] = {
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_rank": rank,
                "text_rank": None,
            }
        else:
            rrf_map[key]["vector_rank"] = rank
    
    # Adiciona resultados da busca textual
    for rank, item in enumerate(text_results, start=1):
        key = item["content"][:200]
        if key not in rrf_map:
            rrf_map[key] = {
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_rank": None,
                "text_rank": rank,
            }
        else:
            rrf_map[key]["text_rank"] = rank
    
    # Calcula score RRF para cada chunk
    scored = []
    for item in rrf_map.values():
        score = 0.0
        if item["vector_rank"] is not None:
            score += WEIGHT_VECTOR / (K_RRF + item["vector_rank"])
        if item["text_rank"] is not None:
            score += WEIGHT_TEXT / (K_RRF + item["text_rank"])
        scored.append({
            "content": item["content"],
            "score": score,
            "metadata": item["metadata"],
        })
    
    # Ordena por score RRF (maior = mais relevante) e retorna top K
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ============================================================
# Função principal de busca (ponto de entrada)
# ============================================================

def search_knowledge_base(
    query: str,
    table_name: str,
    search_type: Optional[str] = None,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    use_hyde: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Busca unificada no knowledge base.
    
    Esta é a função principal que os agentes chamam (via tool).
    Ela resolve os parâmetros (defaults do config ou customizados),
    aplica HyDE se ativado, escolhe o tipo de busca, e retorna
    os chunks mais relevantes.
    
    Parâmetros:
        query: Pergunta do usuário (texto).
        table_name: Tabela de embeddings do agente.
        search_type: Tipo de busca (override do config). 
                     None = usa DEFAULT_SEARCH_TYPE.
        k: Quantidade de resultados (override). None = usa DEFAULT_K.
        threshold: Score mínimo (override). None = usa MATCH_THRESHOLD.
        use_hyde: Ativar HyDE (override). None = usa USE_HYDE.
    
    Retorna:
        Lista de dicts: [{"content": "...", "score": 0.85, "metadata": {...}}, ...]
    
    Exemplo de uso direto:
        results = search_knowledge_base(
            query="como fabricar mussarela?",
            table_name="embeddings_agente_1_queijos",
        )
    
    Exemplo de uso com overrides:
        results = search_knowledge_base(
            query="IN 76",
            table_name="embeddings_agente_3_regulatorios",
            search_type="hybrid_rrf",  # Override para este agente
            k=8,                        # Mais resultados
        )
    """
    # Resolve parâmetros: usa o override se fornecido, senão o default
    _search_type = search_type or DEFAULT_SEARCH_TYPE
    _k = k or DEFAULT_K
    _threshold = threshold if threshold is not None else MATCH_THRESHOLD
    _use_hyde = use_hyde if use_hyde is not None else USE_HYDE
    
    # Texto a ser embedado: query original ou documento hipotético (HyDE)
    text_to_embed = query
    if _use_hyde:
        text_to_embed = generate_hyde_document(query)
    
    # Gera embedding apenas quando necessário
    # (busca "text" pura não precisa de embedding)
    query_embedding = None
    if _search_type != "text":
        query_embedding = embed_query(text_to_embed)
    
    # Executa a busca conforme o tipo
    if _search_type == "vector":
        results = search_vector(query_embedding, table_name, _k, _threshold)
    
    elif _search_type == "text":
        results = search_text(query, table_name, _k)
    
    elif _search_type in ("hybrid", "hybrid_rrf"):
        results = search_hybrid_rrf(
            query, query_embedding, table_name, _k, _threshold
        )
    
    else:
        raise ValueError(
            f"search_type inválido: '{_search_type}'. "
            f"Use 'vector', 'text' ou 'hybrid_rrf'."
        )
    
    return results


# ============================================================
# Tool LangChain (usada pelos agentes no grafo ReAct)
# ============================================================

def create_kb_search_tool(
    table_name: str,
    agent_name: str,
    search_config: Optional[Dict[str, Any]] = None,
):
    """Cria uma tool de busca no KB configurada para um agente específico.
    
    Esta função retorna uma tool LangChain que o agente usa dentro
    do loop ReAct. Quando o LLM decide que precisa buscar informação,
    ele chama essa tool passando a query.
    
    Por que criar a tool dinamicamente (factory)?
    Porque cada agente busca em uma tabela diferente. O Agente 1
    busca em embeddings_agente_1_queijos, o Agente 3 busca em
    embeddings_agente_3_regulatorios. Em vez de criar 6 funções
    diferentes, criamos uma factory que recebe o table_name e
    retorna uma tool configurada.
    
    No projeto original (tools.py), existe apenas uma tool
    kb_search_client que filtra por empresa. Aqui, cada agente
    tem sua própria tool que aponta para sua tabela.
    
    Parâmetros:
        table_name: Tabela de embeddings do agente.
        agent_name: Nome do agente (para a descrição da tool).
        search_config: Parâmetros customizados (do agent_config.py).
    
    Retorna:
        Uma tool LangChain decorada com @tool.
    
    Uso (em base_agent.py):
        search_tool = create_kb_search_tool(
            table_name="embeddings_agente_1_queijos",
            agent_name="Tecnologia de Queijos",
            search_config={"search_type": "vector"},
        )
        # search_tool agora é uma tool que o ReAct executor pode chamar
    """
    # Extrai configs customizadas (ou usa defaults)
    _config = search_config or {}
    _search_type = _config.get("search_type")
    _k = _config.get("k")
    _use_hyde = _config.get("use_hyde")
    
    @tool
    def kb_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Busca informações na base de conhecimento.
        
        Use esta ferramenta para encontrar informações técnicas
        relevantes para responder a pergunta do usuário.
        
        Parâmetros:
            query: Texto da busca (reformule se necessário para melhorar resultados).
            k: Quantidade de resultados (padrão 5, aumente para perguntas amplas).
        
        Retorna:
            Lista de trechos relevantes da base de conhecimento,
            cada um com o texto (content) e score de relevância.
        """
        return search_knowledge_base(
            query=query,
            table_name=table_name,
            search_type=_search_type,
            k=_k or k,
            use_hyde=_use_hyde,
        )
    
    # Personaliza o nome e descrição da tool para o agente
    kb_search.name = f"buscar_base_{table_name.replace('embeddings_agente_', '').replace('_', '_')}"
    kb_search.description = (
        f"Busca informações na base de conhecimento de {agent_name}. "
        f"Use SEMPRE para responder perguntas do usuário sobre {agent_name.lower()}."
    )
    
    return kb_search
