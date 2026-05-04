"""
rag/search.py â€” Busca no Knowledge Base (vetorial, textual, hÃ­brida)

Este mÃ³dulo Ã© o CORAÃ‡ÃƒO do RAG. Quando um agente recebe uma pergunta,
ele chama a funÃ§Ã£o daqui para buscar os chunks mais relevantes na base
de conhecimento. Ã‰ uma adaptaÃ§Ã£o do app/rag/tools.py do projeto original.

Fluxo de uma busca:
  1. Recebe a query do usuÃ¡rio ("como fabricar mussarela?")
  2. (Opcional) HyDE: gera um documento hipotÃ©tico para melhorar a busca
  3. Gera o embedding da query via OpenAI
  4. Chama uma funÃ§Ã£o SQL no Supabase:
     - kb_vector_search: busca por similaridade de cosseno
     - kb_text_search: busca por palavras-chave (FTS)
     - kb_hybrid_search: combina ambas com RRF (Reciprocal Rank Fusion)
  5. (Opcional) Reranking: reordena os resultados com Cohere
  6. Retorna os top K chunks mais relevantes

FunÃ§Ãµes SQL (definidas em sql/03_kb_functions.sql):
  Essas funÃ§Ãµes rodam DENTRO do Postgres, nÃ£o no Python. O Python
  apenas chama a funÃ§Ã£o passando os parÃ¢metros. A vantagem Ã© performance:
  a busca vetorial e a busca textual acontecem no banco, sem transferir
  todos os chunks para o Python.

  kb_vector_search(embedding, k, threshold, agent_table):
    Calcula score = 1 - distÃ¢ncia_cosseno entre o embedding da query
    e cada chunk da tabela. Retorna os top K com score > threshold.

  kb_text_search(query_text, k, agent_table):
    Usa Full-Text Search (FTS) do Postgres. Converte a query em
    ts_query ("fabricar" & "mussarela") e busca nos chunks que
    contÃªm essas palavras. Ranqueia por ts_rank_cd (frequÃªncia +
    proximidade dos termos).

  kb_hybrid_search(query_text, embedding, k, threshold, agent_table):
    Executa AMBAS as buscas (vector + text) e combina os resultados
    com Reciprocal Rank Fusion (RRF). RRF Ã© uma fÃ³rmula que pondera
    a posiÃ§Ã£o de cada resultado nas duas listas:
      score_rrf = 1.5/(60 + rank_vector) + 1.0/(60 + rank_text)
    O peso 1.5 para vector e 1.0 para text foi definido empiricamente
    no projeto original (funciona bem para a maioria dos casos).

AdaptaÃ§Ãµes em relaÃ§Ã£o ao original:
  - Original filtra por empresa/client_id (CRM multi-tenant)
  - Adaptado filtra por table_name (cada agente = uma tabela)
  - Original tem busca global; adaptado Ã© sempre por agente
  - Adicionado: a tool LangChain (kb_search_tool) que os agentes usam
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import logging
import os
import hashlib
import json
import re

_log = logging.getLogger(__name__)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool

from app.config import (
    EMBEDDING_MODEL,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_K,
    MATCH_THRESHOLD,
    USE_HYDE,
    HYDE_LLM_MODEL,
    USE_QUERY_REWRITE,
    QUERY_REWRITE_MODEL,
    QUERY_REWRITE_VARIANTS,
    RAG_MIN_CHUNK_CHARS,
    RAG_MIN_ALNUM_RATIO,
    RAG_SECOND_PASS_ENABLED,
    RAG_SECOND_PASS_MIN_RESULTS,
    RAG_SECOND_PASS_MIN_KEYWORD_HITS,
    RAG_SECOND_PASS_EXPAND_FACTOR,
    RAG_SECOND_PASS_MAX_K,
    RAG_SECOND_PASS_FORCE_HYBRID,
    RAG_SECOND_PASS_DISABLE_THRESHOLD,
    RAG_SECOND_PASS_USE_QUERY_REWRITE,
    RERANKER,
    RERANK_CANDIDATES,
)
from app.db.connection import get_supabase_conn

_embeddings_model = None
_hyde_model = None
_query_rewrite_model = None
try:
    _hybrid_workers = int(os.getenv("RAG_HYBRID_WORKERS", "8"))
except ValueError:
    _hybrid_workers = 8
_hybrid_executor = ThreadPoolExecutor(max_workers=max(2, _hybrid_workers))
# Pool separado para a busca em múltiplas tabelas do fallback geral.
# Isolado de _hybrid_executor para evitar deadlock: cada tarefa deste pool
# pode chamar search_hybrid_rrf que submete ao _hybrid_executor — pools
# independentes garantem que não há espera circular.
_general_fallback_executor = ThreadPoolExecutor(max_workers=5)


def _rrf_key(item: Dict[str, Any]) -> str:
    """Gera chave estÃ¡vel para deduplicaÃ§Ã£o no RRF.

    Evita colisÃµes ao usar apenas prefixo de conteÃºdo (que pode repetir
    em documentos parecidos), situaÃ§Ã£o comum em bases normativas/tÃ©cnicas.
    """
    metadata = item.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in ("id", "chunk_id", "source", "doc_id", "path"):
            value = metadata.get(key)
            if value:
                return f"{key}:{value}:{hashlib.md5(item.get('content', '').encode('utf-8')).hexdigest()}"
    return hashlib.md5(item.get("content", "").encode("utf-8")).hexdigest()


# ============================================================
# UtilitÃ¡rio: converter vetor Python em literal SQL
# ============================================================

def vec_to_literal(v: List[float]) -> str:
    """Converte uma lista de floats em formato aceito pelo pgvector.
    
    O pgvector espera vetores como texto no formato "[0.1, 0.2, ...]".
    Esta funÃ§Ã£o converte a lista Python para essa string.
    
    Exemplo:
        vec_to_literal([0.1, 0.2, 0.3]) â†’ "[0.100000,0.200000,0.300000]"
    
    O formato com 6 casas decimais Ã© suficiente para preservar a precisÃ£o
    dos embeddings da OpenAI (que retornam ~7 casas significativas).
    
    FunÃ§Ã£o idÃªntica ao original (tools.py linha 17).
    """
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ============================================================
# GeraÃ§Ã£o de embeddings
# ============================================================

def embed_query(text: str) -> List[float]:
    """Gera o embedding (vetor) de um texto.
    
    Usado para transformar a pergunta do usuÃ¡rio em vetor
    antes de buscar por similaridade no banco.
    
    O modelo text-embedding-3-small gera vetores de 1536 dimensÃµes.
    Cada dimensÃ£o Ã© um float entre -1 e 1 que captura um aspecto
    do significado semÃ¢ntico do texto.
    
    Textos com significado parecido geram vetores prÃ³ximos no espaÃ§o
    vetorial. "Como fabricar mussarela?" e "Processo de produÃ§Ã£o de
    queijo mussarela" geram vetores quase idÃªnticos, mesmo com
    palavras diferentes.
    
    Custo: ~$0.00002 por chamada (extremamente barato).
    LatÃªncia: ~100-200ms.
    """
    return _get_embeddings_model().embed_query(text)


# ============================================================
# HyDE â€” Hypothetical Document Embeddings
# ============================================================

def generate_hyde_document(query: str) -> str:
    """Gera um documento hipotÃ©tico relevante para a query.
    
    HyDE (Hypothetical Document Embeddings) Ã© uma tÃ©cnica que melhora
    o retrieval para perguntas vagas ou curtas.
    
    Problema: a pergunta "queijo" gera um embedding genÃ©rico que pode
    retornar chunks sobre qualquer aspecto de queijo.
    
    SoluÃ§Ã£o: antes de buscar, pedimos ao LLM para imaginar como seria
    um documento que respondesse essa pergunta. O LLM gera algo como:
    "O queijo Ã© um produto lÃ¡cteo obtido pela coagulaÃ§Ã£o do leite.
    Existem diversos tipos como mussarela, prato, minas..."
    
    Esse documento hipotÃ©tico gera um embedding muito mais rico e
    especÃ­fico do que a pergunta original. Usamos esse embedding
    para a busca vetorial.
    
    Trade-off:
      + Melhora retrieval para perguntas curtas/vagas
      - Adiciona ~500ms + custo de 1 chamada LLM por busca
      - Pode enviesar a busca se o LLM "inventar" algo errado
    
    FunÃ§Ã£o baseada no original (tools.py linhas 147-157).
    DiferenÃ§a: usa HYDE_LLM_MODEL do config em vez de hardcoded.
    """
    llm = _get_hyde_model()
    
    prompt = (
        "VocÃª Ã© um especialista em tecnologia de laticÃ­nios. "
        "Escreva um parÃ¡grafo tÃ©cnico conciso que seria altamente "
        "relevante para a seguinte pergunta. NÃ£o invente dados "
        "numÃ©ricos especÃ­ficos.\n\n"
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


def _get_query_rewrite_model() -> ChatOpenAI:
    global _query_rewrite_model
    if _query_rewrite_model is None:
        _query_rewrite_model = ChatOpenAI(
            model=QUERY_REWRITE_MODEL,
            temperature=0,
        )
    return _query_rewrite_model


def generate_query_rewrites(query: str, variants: int) -> List[str]:
    """Gera variacoes tecnicas da query mantendo intencao e dominio.

    Retorna lista unica contendo a query original + N variacoes.
    """
    max_variants = max(0, int(variants))
    base = (query or "").strip()
    if not base or max_variants == 0:
        return [base] if base else []

    llm = _get_query_rewrite_model()
    prompt = (
        "VocÃª reescreve perguntas para retrieval tÃ©cnico em laticÃ­nios.\n"
        "Tarefa: gerar variaÃ§Ãµes curtas e tÃ©cnicas da pergunta abaixo, sem mudar o sentido.\n"
        "Regras:\n"
        "- NÃ£o inventar fatos ou nÃºmeros.\n"
        "- NÃ£o responder a pergunta.\n"
        "- Retornar APENAS JSON vÃ¡lido: array de strings.\n"
        f"- Gere exatamente {max_variants} variaÃ§Ãµes.\n\n"
        f"Pergunta original: {base}"
    )

    raw = ""
    try:
        resp = llm.invoke(prompt)
        raw = (resp.content or "").strip()
    except Exception:
        return [base]

    candidates: List[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            candidates = [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        lines = re.split(r"[\r\n]+", raw)
        for line in lines:
            cleaned = re.sub(r"^\s*[-*\d\.\)]\s*", "", line).strip()
            if cleaned:
                candidates.append(cleaned)

    out: List[str] = []
    seen = set()
    for item in [base] + candidates:
        norm = re.sub(r"\s+", " ", item).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item.strip())
        if len(out) >= (max_variants + 1):
            break

    return out or [base]


def _fuse_results_rrf(result_lists: List[List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
    """Funde multiplas listas de resultados usando Reciprocal Rank Fusion."""
    k_rrf = 60
    weight = 1.0
    rrf_map: Dict[str, Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            key = _rrf_key(item)
            entry = rrf_map.get(key)
            score_inc = weight / (k_rrf + rank)
            if entry is None:
                rrf_map[key] = {
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "score": score_inc,
                }
            else:
                entry["score"] += score_inc

    merged = list(rrf_map.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:k]


_PT_STOPWORDS = {
    "como", "qual", "quais", "para", "com", "sem", "uma", "uns", "umas", "dos",
    "das", "de", "do", "da", "e", "ou", "o", "a", "os", "as", "em", "no", "na",
    "nos", "nas", "por", "que", "se", "ao", "aos", "sao", "sÃ£o", "antes", "apos",
    "apÃ³s", "deve", "devem", "ser", "fazer", "tomar",
}


def _normalize_text_for_match(text: str) -> str:
    normalized = text or ""
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_query_keywords(query: str) -> List[str]:
    # Unicode-safe tokenization (letters+digits), avoiding brittle char ranges.
    tokens = re.findall(r"[^\W_]+", _normalize_text_for_match(query), flags=re.UNICODE)
    keywords = []
    for token in tokens:
        if len(token) < 4:
            continue
        if token in _PT_STOPWORDS:
            continue
        keywords.append(token)
    # Ordem estavel sem duplicar
    dedup = []
    seen = set()
    for k in keywords:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup


def _is_definition_query(query: str) -> bool:
    q = _normalize_text_for_match(query)
    markers = (
        "o que significa",
        "que significa",
        "significa",
        "o que quer dizer",
        "termo",
    )
    return any(m in q for m in markers)


def _is_high_quality_chunk(content: str) -> bool:
    text = (content or "").strip()
    if len(text) < RAG_MIN_CHUNK_CHARS:
        return False
    if text in {".", "-", "|", "||"}:
        return False
    alnum = sum(1 for ch in text if ch.isalnum())
    ratio = alnum / max(1, len(text))
    if ratio < RAG_MIN_ALNUM_RATIO:
        return False
    return True


def _rerank_results(query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Reranks candidates using Cohere rerank-multilingual-v3.0.

    Falls back to original truncated list on any API error so retrieval
    never fails silently — callers always get k results.
    """
    if not results:
        return results
    try:
        import cohere
        client = cohere.Client(api_key=COHERE_API_KEY)
        documents = [r.get("content", "") for r in results]
        response = client.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=documents,
            top_n=min(k, len(documents)),
        )
        reranked: List[Dict[str, Any]] = []
        for hit in response.results:
            item = dict(results[hit.index])
            item["score"] = float(hit.relevance_score)
            reranked.append(item)
        return reranked
    except Exception as exc:
        _log.warning("Cohere rerank failed, falling back to original order: %s", exc)
        return results[:k]


def _maybe_rerank(query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Applies Cohere reranking when configured, otherwise truncates to k."""
    if RERANKER == "cohere" and COHERE_API_KEY and results:
        return _rerank_results(query, results, k)
    return results[:k]


def _postprocess_results(query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Aplica qualidade + priorizacao lexical para reduzir ruÃ­do no retrieval."""
    if not results:
        return []

    filtered = [r for r in results if _is_high_quality_chunk(str(r.get("content", "")))]
    if not filtered:
        return []

    keywords = _extract_query_keywords(query)
    is_definition = _is_definition_query(query)
    if not keywords:
        return filtered[:k]

    scored: List[Dict[str, Any]] = []
    for item in filtered:
        base_score = float(item.get("score", 0.0) or 0.0)
        content = _normalize_text_for_match(str(item.get("content", "")))
        metadata = item.get("metadata") or {}
        source = _normalize_text_for_match(str(metadata.get("source", ""))) if isinstance(metadata, dict) else ""
        haystack = f"{content} {source}".strip()

        matches = sum(1 for kw in keywords if kw in haystack)
        # Bonus pequeno para preservar ordenacao semÃ¢ntica original.
        boosted = base_score + (0.01 * matches)

        # Perguntas de definicao: prioriza fortemente entradas canÃ´nicas de glossario.
        if is_definition:
            if "termo:" in content and "substituicao:" in content:
                boosted += 0.20
            # Se o termo da pergunta aparece como chave canÃ´nica do glossÃ¡rio, dÃ¡ prioridade.
            exact_term_hits = sum(1 for kw in keywords if f"termo: {kw}" in content)
            boosted += 0.15 * exact_term_hits
            # Penaliza chunks de metadados/keywords para evitar desvio de resposta.
            if "palavras_chave" in content:
                boosted -= 0.05

        scored.append({
            "content": item.get("content", ""),
            "metadata": metadata,
            "score": boosted,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def _keyword_hits(query: str, item: Dict[str, Any]) -> int:
    """Conta quantas palavras-chave da query aparecem no chunk + source."""
    keywords = _extract_query_keywords(query)
    if not keywords:
        return 0
    content = _normalize_text_for_match(str(item.get("content", "")))
    metadata = item.get("metadata") or {}
    source = _normalize_text_for_match(str(metadata.get("source", ""))) if isinstance(metadata, dict) else ""
    haystack = f"{content} {source}".strip()
    return sum(1 for kw in keywords if kw in haystack)


def _needs_second_pass(query: str, results: List[Dict[str, Any]], requested_k: int) -> bool:
    """Decide se vale executar uma segunda passada de retrieval."""
    if not RAG_SECOND_PASS_ENABLED:
        return False

    if not results:
        return True

    min_results = max(1, min(requested_k, RAG_SECOND_PASS_MIN_RESULTS))
    if len(results) < min_results:
        return True

    keywords = _extract_query_keywords(query)
    if not keywords:
        return False

    top = results[: min(3, len(results))]
    best_hits = max((_keyword_hits(query, item) for item in top), default=0)
    min_hits = max(1, min(len(keywords), RAG_SECOND_PASS_MIN_KEYWORD_HITS))
    return best_hits < min_hits


# ============================================================
# Busca no banco (chamadas Ã s funÃ§Ãµes SQL)
# ============================================================

def search_vector(
    query_embedding: List[float],
    table_name: str,
    k: int,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Busca vetorial pura (similaridade de cosseno).
    
    Chama a funÃ§Ã£o SQL kb_vector_search no Supabase.
    
    Como funciona internamente (no SQL):
      1. Converte o embedding da query para tipo vector(1536)
      2. Calcula a distÃ¢ncia de cosseno entre a query e CADA chunk
      3. Converte distÃ¢ncia em score: score = 1 - distÃ¢ncia
         (score 1.0 = idÃªntico, 0.0 = completamente diferente)
      4. Filtra chunks com score >= threshold (se definido)
      5. Ordena por score decrescente
      6. Retorna os top K
    
    A busca usa o Ã­ndice HNSW (Hierarchical Navigable Small World),
    que Ã© uma estrutura de dados otimizada para busca aproximada
    de vizinhos mais prÃ³ximos. Em vez de comparar com TODOS os chunks
    (O(n)), o HNSW navega por uma estrutura em camadas e encontra
    os vizinhos mais prÃ³ximos em O(log n). Para 10.000 chunks,
    a diferenÃ§a Ã©: brute force = 10.000 comparaÃ§Ãµes, HNSW â‰ˆ 13.
    
    ParÃ¢metros:
        query_embedding: Vetor de 1536 dimensÃµes da pergunta.
        table_name: Nome da tabela (ex: "embeddings_agente_1_queijos").
        k: Quantos resultados retornar.
        threshold: Score mÃ­nimo (None = sem filtro).
    
    Retorna:
        Lista de dicts com: content, score, metadata.
    """
    vec_lit = vec_to_literal(query_embedding)
    
    with get_supabase_conn() as conn:
        with conn.cursor() as cur:
            # Chama a busca vetorial na tabela especÃ­fica do agente.
            # NOTA sobre seguranÃ§a: table_name Ã© formatado com .format()
            # porque nÃ£o pode ser parÃ¢metro SQL (%s). Isso Ã© seguro aqui
            # pois table_name vem de configuraÃ§Ã£o interna (nÃ£o input do usuÃ¡rio).
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
         tokens de busca. Ex: "fabricar mussarela" â†’ "fabricar & mussarela"
      2. O operador @@ verifica se o campo FTS do chunk contÃ©m esses tokens
      3. ts_rank_cd calcula a relevÃ¢ncia (baseado em frequÃªncia e proximidade)
      4. Ordena por relevÃ¢ncia e retorna top K
    
    A busca FTS usa o Ã­ndice GIN (Generalized Inverted Index),
    que Ã© um Ã­ndice invertido (como o Google): mapeia cada palavra
    para a lista de documentos que a contÃªm. A busca Ã© O(1) por
    palavra â€” extremamente rÃ¡pida.
    
    Quando usar FTS em vez de vector:
      - Termos exatos: "IN 76", "RDC 331", "Art. 15"
      - Nomes prÃ³prios: "Clostridium tyrobutyricum"
      - CÃ³digos: "CCS > 400.000"
    A busca vetorial NÃƒO Ã© boa para termos exatos porque embeddings
    capturam significado, nÃ£o palavras especÃ­ficas.
    
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
    """Busca hÃ­brida com Reciprocal Rank Fusion (RRF).
    
    Combina busca vetorial + busca textual e funde os resultados
    usando RRF. Esta Ã© a busca mais poderosa do sistema.
    
    Como funciona o RRF:
      1. Executa busca vetorial â†’ lista ordenada por score_vector
      2. Executa busca textual â†’ lista ordenada por score_text
      3. Para cada chunk que aparece em pelo menos uma lista:
         score_rrf = peso_vector/(K_rrf + rank_vector) + peso_text/(K_rrf + rank_text)
         onde K_rrf=60 (constante de suavizaÃ§Ã£o), peso_vector=1.5, peso_text=1.0
      4. Ordena por score_rrf e retorna top K
    
    Por que RRF e nÃ£o mÃ©dia simples?
    A mÃ©dia simples (score_vector + score_text)/2 penaliza chunks
    que aparecem em apenas uma lista (score 0 na outra).
    O RRF usa RANK (posiÃ§Ã£o na lista), nÃ£o score absoluto.
    Um chunk em 1Âº lugar na busca vetorial e ausente na textual
    recebe: 1.5/(60+1) + 0 = 0.025. Se estiver em 3Âº na textual
    tambÃ©m: 1.5/(60+1) + 1.0/(60+3) = 0.025 + 0.016 = 0.041.
    A presenÃ§a em ambas as listas SOMA, nÃ£o penaliza.
    
    Os pesos (1.5 vector, 1.0 text) dÃ£o mais importÃ¢ncia Ã  busca
    semÃ¢ntica. Foram definidos empiricamente no projeto original
    e funcionam bem para a maioria dos casos.
    
    LÃ³gica baseada em sql/kb/03_functions.sql â†’ kb_hybrid_search.
    Aqui executamos em Python em vez de SQL para maior flexibilidade,
    mas a lÃ³gica Ã© idÃªntica.
    """
    # Busca vetorial e textual em paralelo.
    # Como cada busca abre sua prÃ³pria conexÃ£o no pool, conseguimos reduzir
    # a latÃªncia total da estratÃ©gia hÃ­brida para prÃ³ximo do ramo mais lento.
    fut_vector = _hybrid_executor.submit(
        search_vector, query_embedding, table_name, k * 3, threshold
    )
    fut_text = _hybrid_executor.submit(search_text, query, table_name, k * 3)
    vector_results = fut_vector.result()
    text_results = fut_text.result()
    
    # RRF: funde as duas listas
    # Mapa: content â†’ { vector_rank, text_rank, content, metadata }
    rrf_map: Dict[str, Dict[str, Any]] = {}
    
    K_RRF = 60          # Constante de suavizaÃ§Ã£o (padrÃ£o da literatura)
    WEIGHT_VECTOR = 1.5  # Peso da busca vetorial
    WEIGHT_TEXT = 1.0    # Peso da busca textual
    
    # Adiciona resultados da busca vetorial
    for rank, item in enumerate(vector_results, start=1):
        key = _rrf_key(item)
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
        key = _rrf_key(item)
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
# FunÃ§Ã£o principal de busca (ponto de entrada)
# ============================================================

def search_knowledge_base(
    query: str,
    table_name: str,
    search_type: Optional[str] = None,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    use_hyde: Optional[bool] = None,
    use_query_rewrite: Optional[bool] = None,
    precomputed_embedding: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Busca unificada no knowledge base.
    
    Esta Ã© a funÃ§Ã£o principal que os agentes chamam (via tool).
    Ela resolve os parÃ¢metros (defaults do config ou customizados),
    aplica HyDE se ativado, escolhe o tipo de busca, e retorna
    os chunks mais relevantes.
    
    ParÃ¢metros:
        query: Pergunta do usuÃ¡rio (texto).
        table_name: Tabela de embeddings do agente.
        search_type: Tipo de busca (override do config). 
                     None = usa DEFAULT_SEARCH_TYPE.
        k: Quantidade de resultados (override). None = usa DEFAULT_K.
        threshold: Score mÃ­nimo (override). None = usa MATCH_THRESHOLD.
        use_hyde: Ativar HyDE (override). None = usa USE_HYDE.
        use_query_rewrite: Ativar query rewriting (override).
    
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
    # Resolve parÃ¢metros: usa o override se fornecido, senÃ£o o default
    _search_type = search_type or DEFAULT_SEARCH_TYPE
    _k = k or DEFAULT_K
    _threshold = threshold if threshold is not None else MATCH_THRESHOLD
    _use_hyde = use_hyde if use_hyde is not None else USE_HYDE
    _use_query_rewrite = (
        use_query_rewrite if use_query_rewrite is not None else USE_QUERY_REWRITE
    )

    def _run_single_search(
        single_query: str,
        limit: int,
        search_type_override: Optional[str] = None,
        threshold_override: Optional[float] = None,
        use_hyde_override: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        # Texto a ser embedado: query original ou documento hipotetico (HyDE)
        run_search_type = search_type_override or _search_type
        run_threshold = _threshold if threshold_override is None else threshold_override
        run_use_hyde = _use_hyde if use_hyde_override is None else use_hyde_override

        text_to_embed = single_query
        if run_use_hyde:
            text_to_embed = generate_hyde_document(single_query)

        query_embedding = None
        if run_search_type != "text":
            # Reutiliza embedding pré-computado quando disponível e HyDE não está ativo.
            # HyDE gera um documento diferente da query original, então não pode reaproveitar.
            if precomputed_embedding is not None and not run_use_hyde:
                query_embedding = precomputed_embedding
            else:
                query_embedding = embed_query(text_to_embed)

        if run_search_type == "vector":
            return search_vector(query_embedding, table_name, limit, run_threshold)
        if run_search_type == "text":
            return search_text(single_query, table_name, limit)
        if run_search_type in ("hybrid", "hybrid_rrf"):
            return search_hybrid_rrf(
                single_query, query_embedding, table_name, limit, run_threshold
            )
        raise ValueError(
            f"search_type invalido: '{run_search_type}'. "
            f"Use 'vector', 'text' ou 'hybrid_rrf'."
        )

    def _run_search_with_optional_rewrite(
        base_query: str,
        limit: int,
        enable_query_rewrite: bool,
        search_type_override: Optional[str] = None,
        threshold_override: Optional[float] = None,
        use_hyde_override: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        if not enable_query_rewrite:
            return _run_single_search(
                base_query,
                limit,
                search_type_override=search_type_override,
                threshold_override=threshold_override,
                use_hyde_override=use_hyde_override,
            )

        queries = generate_query_rewrites(base_query, QUERY_REWRITE_VARIANTS)
        if len(queries) <= 1:
            return _run_single_search(
                base_query,
                limit,
                search_type_override=search_type_override,
                threshold_override=threshold_override,
                use_hyde_override=use_hyde_override,
            )

        expanded_limit = max(limit * 2, limit + 2)
        result_lists: List[List[Dict[str, Any]]] = []
        for q in queries:
            result_lists.append(
                _run_single_search(
                    q,
                    expanded_limit,
                    search_type_override=search_type_override,
                    threshold_override=threshold_override,
                    use_hyde_override=use_hyde_override,
                )
            )
        return _fuse_results_rrf(result_lists, max(limit * 2, limit + 2))

    # Quando reranker habilitado, busca mais candidatos para dar ao Cohere
    # mais material para trabalhar antes de reordenar para os top _k finais.
    _eff_k = max(_k, RERANK_CANDIDATES) if RERANKER == "cohere" and COHERE_API_KEY else _k

    # Primeira passada: caminho quente e mais frequente.
    # O query rewrite fica reservado para a segunda passada (fraca),
    # reduzindo custo fixo sem perder a ferramenta quando o retrieval vier ruim.
    primary_raw = _run_search_with_optional_rewrite(
        query,
        _eff_k,
        enable_query_rewrite=False,
    )
    primary = _postprocess_results(query, primary_raw, _eff_k)

    if not _needs_second_pass(query, primary, _k):
        return _maybe_rerank(query, primary, _k)

    second_k = int(round(_k * max(1.0, RAG_SECOND_PASS_EXPAND_FACTOR)))
    second_k = max(_k + 2, second_k)
    second_k = min(second_k, max(_k + 2, RAG_SECOND_PASS_MAX_K))

    second_search_type = "hybrid_rrf" if RAG_SECOND_PASS_FORCE_HYBRID else _search_type
    second_threshold = None if RAG_SECOND_PASS_DISABLE_THRESHOLD else _threshold
    second_use_qr = _use_query_rewrite and RAG_SECOND_PASS_USE_QUERY_REWRITE

    second_raw = _run_search_with_optional_rewrite(
        query,
        second_k,
        enable_query_rewrite=second_use_qr,
        search_type_override=second_search_type,
        threshold_override=second_threshold,
    )
    second = _postprocess_results(query, second_raw, second_k)
    if not second:
        return _maybe_rerank(query, primary, _k)

    merged = _fuse_results_rrf([primary, second], max(_eff_k * 2, _eff_k + 2))
    return _maybe_rerank(query, _postprocess_results(query, merged, _eff_k), _k)


def search_general_knowledge_base(
    query: str,
    table_names: List[str],
    search_type: str = "hybrid_rrf",
    per_table_k: int = 3,
    final_k: int = 6,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Busca unificada em múltiplas tabelas de embeddings.

    Uso recomendado: fallback final do orquestrador quando a rota especialista
    não trouxe evidência suficiente.

    Guardrails desta função:
    - Compartilha o mesmo embedding para todas as tabelas (eficiente).
    - Ignora erros por tabela e continua com as demais (robusto).
    - Mescla resultados por RRF entre tabelas e aplica pós-processamento.
    """
    base_query = (query or "").strip()
    if not base_query:
        return []

    clean_tables: List[str] = []
    seen = set()
    for raw_name in table_names or []:
        name = str(raw_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        clean_tables.append(name)
    if not clean_tables:
        return []

    mode = (search_type or "hybrid_rrf").strip().lower()
    mode = "hybrid_rrf" if mode in ("hybrid", "hybrid_rrf") else mode
    if mode not in ("vector", "text", "hybrid_rrf"):
        mode = "hybrid_rrf"

    limit_per_table = max(1, int(per_table_k or 1))
    limit_final = max(1, int(final_k or 1))
    run_threshold = threshold if threshold is not None else MATCH_THRESHOLD

    query_embedding = None
    if mode != "text":
        query_embedding = embed_query(base_query)

    def _search_one_table(table_name: str) -> List[Dict[str, Any]]:
        if mode == "vector":
            rows = search_vector(query_embedding, table_name, limit_per_table, run_threshold)
        elif mode == "text":
            rows = search_text(base_query, table_name, limit_per_table)
        else:
            rows = search_hybrid_rrf(
                base_query,
                query_embedding,
                table_name,
                limit_per_table,
                run_threshold,
            )
        enriched: List[Dict[str, Any]] = []
        for item in rows:
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("source_table", table_name)
            enriched.append(
                {
                    "content": item.get("content", ""),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "metadata": metadata,
                }
            )
        return enriched

    result_lists: List[List[Dict[str, Any]]] = []
    futures = {
        _general_fallback_executor.submit(_search_one_table, tbl): tbl
        for tbl in clean_tables
    }
    for fut in as_completed(futures):
        try:
            enriched = fut.result()
        except Exception:
            # Falha de tabela isolada não deve derrubar fallback geral.
            continue
        if enriched:
            result_lists.append(enriched)

    if not result_lists:
        return []

    fused = _fuse_results_rrf(result_lists, max(limit_final * 2, limit_final + 2))
    return _postprocess_results(base_query, fused, limit_final)


# ============================================================
# Tool LangChain (usada pelos agentes no grafo ReAct)
# ============================================================

def create_kb_search_tool(
    table_name: str,
    agent_name: str,
    search_config: Optional[Dict[str, Any]] = None,
):
    """Cria uma tool de busca no KB configurada para um agente especÃ­fico.
    
    Esta funÃ§Ã£o retorna uma tool LangChain que o agente usa dentro
    do loop ReAct. Quando o LLM decide que precisa buscar informaÃ§Ã£o,
    ele chama essa tool passando a query.
    
    Por que criar a tool dinamicamente (factory)?
    Porque cada agente busca em uma tabela diferente. O Agente 1
    busca em embeddings_agente_1_queijos, o Agente 3 busca em
    embeddings_agente_3_regulatorios. Em vez de criar 6 funÃ§Ãµes
    diferentes, criamos uma factory que recebe o table_name e
    retorna uma tool configurada.
    
    No projeto original (tools.py), existe apenas uma tool
    kb_search_client que filtra por empresa. Aqui, cada agente
    tem sua prÃ³pria tool que aponta para sua tabela.
    
    ParÃ¢metros:
        table_name: Tabela de embeddings do agente.
        agent_name: Nome do agente (para a descriÃ§Ã£o da tool).
        search_config: ParÃ¢metros customizados (do agent_config.py).
    
    Retorna:
        Uma tool LangChain decorada com @tool.
    
    Uso (em base_agent.py):
        search_tool = create_kb_search_tool(
            table_name="embeddings_agente_1_queijos",
            agent_name="Tecnologia de Queijos",
            search_config={"search_type": "vector"},
        )
        # search_tool agora Ã© uma tool que o ReAct executor pode chamar
    """
    # Extrai configs customizadas (ou usa defaults)
    _config = search_config or {}
    _search_type = _config.get("search_type")
    _k = _config.get("k")
    _use_hyde = _config.get("use_hyde")
    _use_query_rewrite = _config.get("use_query_rewrite")
    
    @tool
    def kb_search(query: str, k: int = 5, embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Busca informaÃ§Ãµes na base de conhecimento.

        Use esta ferramenta para encontrar informaÃ§Ãµes tÃ©cnicas
        relevantes para responder a pergunta do usuÃ¡rio.

        ParÃ¢metros:
            query: Texto da busca (reformule se necessÃ¡rio para melhorar resultados).
            k: Quantidade de resultados (padrÃ£o 5, aumente para perguntas amplas).
            embedding: Embedding pré-computado (opcional, uso interno do orquestrador).

        Retorna:
            Lista de trechos relevantes da base de conhecimento,
            cada um com o texto (content) e score de relevÃ¢ncia.
        """
        return search_knowledge_base(
            query=query,
            table_name=table_name,
            search_type=_search_type,
            k=_k or k,
            use_hyde=_use_hyde,
            use_query_rewrite=_use_query_rewrite,
            precomputed_embedding=embedding,
        )
    
    # Personaliza o nome e descriÃ§Ã£o da tool para o agente
    kb_search.name = f"buscar_base_{table_name.replace('embeddings_agente_', '').replace('_', '_')}"
    kb_search.description = (
        f"Busca informaÃ§Ãµes na base de conhecimento de {agent_name}. "
        f"Use SEMPRE para responder perguntas do usuÃ¡rio sobre {agent_name.lower()}."
    )
    
    return kb_search
