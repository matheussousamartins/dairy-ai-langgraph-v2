"""
rag/rerank.py — Reranking de resultados de busca

Este módulo reordena os chunks retornados pela busca (search.py) usando
um modelo de reranking mais inteligente. É uma adaptação direta da função
apply_rerank do app/rag/tools.py do projeto original (linhas 85-144).

Por que reranking?
A busca vetorial retorna chunks por SIMILARIDADE — o quão parecido o
embedding do chunk é com o embedding da pergunta. Mas similaridade nem
sempre é relevância.

Exemplo:
  Pergunta: "temperatura de filagem da mussarela"
  
  Busca vetorial retorna (por score de cosseno):
    1. "A mussarela é um queijo de massa filada..." (score 0.89)
    2. "A temperatura de pasteurização do leite é 72°C..." (score 0.87)
    3. "A filagem deve ser feita entre 78-82°C..." (score 0.86)
    4. "A temperatura do tanque de salga é 10°C..." (score 0.85)
    5. "O pH ideal para filagem é 5.2-5.4..." (score 0.84)

  O chunk #3 é o mais relevante, mas está em 3º lugar. Os chunks #2
  e #4 falam de temperatura mas não de filagem.

  Depois do reranking:
    1. "A filagem deve ser feita entre 78-82°C..." (rerank score 0.95)
    2. "O pH ideal para filagem é 5.2-5.4..." (rerank score 0.82)
    3. "A mussarela é um queijo de massa filada..." (rerank score 0.71)
    4. "A temperatura de pasteurização é 72°C..." (rerank score 0.35)
    5. "A temperatura do tanque de salga é 10°C..." (rerank score 0.22)

  O reranker entende que a pergunta é sobre "temperatura DE FILAGEM"
  (não qualquer temperatura), e reordena corretamente.

Como funciona tecnicamente:
  O reranker é um modelo de cross-encoder. Diferente do embedding
  (que codifica query e documento SEPARADAMENTE e depois compara),
  o cross-encoder recebe query + documento JUNTOS e produz um score
  de relevância diretamente. É mais preciso mas mais lento.

  Fluxo: busca retorna 24 candidatos → reranker analisa cada um
  em relação à query → reordena → retorna top 5.

  Por que não usar o reranker direto (sem busca)?
  Porque o reranker é lento (~200ms por documento). Analisar 10.000
  chunks com reranker levaria 33 minutos. A busca vetorial é rápida
  (~50ms para qualquer quantidade) e filtra para ~24 candidatos.
  O reranker então refina esses 24 — rápido e preciso.

Rerankers disponíveis:
  - "none": sem reranking (usa a ordem da busca direto)
  - "cohere": usa Cohere Rerank v3 (precisa de COHERE_API_KEY)
  
  Futuros rerankers podem ser adicionados (BAAI/bge-reranker-v2,
  Jina Reranker, etc.) seguindo o mesmo padrão.
"""

from typing import Any, Dict, List, Optional
import os
from collections import defaultdict, deque

from app.config import RERANKER, COHERE_API_KEY, RERANK_CANDIDATES, DEFAULT_K


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    reranker: Optional[str] = None,
    k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Reordena os resultados de busca por relevância.
    
    Ponto de entrada principal do módulo. Chamada pelo search.py
    após a busca, quando o reranking está ativado.
    
    Parâmetros:
        query: Pergunta original do usuário.
        results: Lista de chunks retornados pela busca
                 (cada item tem "content", "score", "metadata").
        reranker: Qual reranker usar. None = usa RERANKER do config.
        k: Quantos resultados retornar após reranking.
           None = usa DEFAULT_K do config.
    
    Retorna:
        Lista reordenada, limitada a k itens.
        Se reranker="none" ou lista vazia, retorna os primeiros k
        sem alteração.
    
    Exemplo:
        # Busca retorna 24 candidatos
        candidates = search_knowledge_base(query, table, k=24)
        # Reranker reordena e retorna top 5
        final = rerank_results(query, candidates, k=5)
    """
    _reranker = reranker or RERANKER
    _k = k or DEFAULT_K
    
    # Sem reranking: retorna os primeiros k diretamente
    if _reranker == "none" or not results:
        return results[:_k]
    
    # Cohere Rerank
    if _reranker == "cohere":
        return _rerank_cohere(query, results, _k)
    
    # Reranker desconhecido: avisa e retorna sem alterar
    # (melhor retornar algo do que falhar)
    print(f"[rerank] Reranker desconhecido: '{_reranker}'. Retornando sem reranking.")
    return results[:_k]


def _rerank_cohere(
    query: str,
    results: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Reranking usando Cohere Rerank v3.
    
    Cohere é um dos melhores rerankers disponíveis. O modelo
    rerank-english-v3.0 funciona bem para português também
    (é multilingual apesar do nome).
    
    Custo: ~$0.001 por 1000 documentos rerankeados.
    Para 24 candidatos por busca, ~$0.000024 por busca.
    Extremamente barato.
    
    Latência: ~200-400ms para 24 documentos.
    
    Esta função é uma adaptação direta da apply_rerank do projeto
    original (tools.py linhas 85-144). A lógica de reconstrução
    da ordem por conteúdo é idêntica.
    """
    # Valida que o pacote está instalado
    try:
        from langchain_cohere import CohereRerank
    except ImportError as e:
        raise RuntimeError(
            "Reranker 'cohere' solicitado, mas pacote langchain-cohere "
            "não está instalado. Instale com: pip install langchain-cohere"
        ) from e
    
    # Valida que a API key está configurada
    if not COHERE_API_KEY:
        raise RuntimeError(
            "Reranker 'cohere' solicitado, mas COHERE_API_KEY não está "
            "definida no .env. Obtenha uma chave em https://cohere.com"
        )
    
    # Inicializa o modelo de reranking
    # rerank-english-v3.0 é o modelo mais recente e performante
    reranker_model = CohereRerank(model="rerank-english-v3.0")
    
    # Extrai apenas os textos dos chunks (o reranker precisa de strings)
    docs = [item["content"] for item in results]
    
    # Chama o reranker
    # A API do Cohere aceita uma query e uma lista de documentos,
    # e retorna os documentos reordenados por relevância.
    #
    # O LangChain pode expor essa funcionalidade de formas diferentes
    # dependendo da versão, por isso tentamos múltiplos métodos.
    # Isso é idêntico ao original (tools.py linhas 100-108).
    reranked = None
    
    if hasattr(reranker_model, "rank"):
        # Versão mais recente do langchain-cohere
        reranked = reranker_model.rank(query=query, documents=docs)
    elif hasattr(reranker_model, "rerank"):
        # Versão intermediária
        reranked = reranker_model.rerank(query=query, documents=docs)
    else:
        # Fallback: tenta via invoke (interface genérica)
        try:
            reranked = reranker_model.invoke(
                {"query": query, "documents": docs}
            )
        except Exception:
            reranked = None
    
    # Se o reranking retornou resultados, reconstrói a ordem original
    # mapeando pelo conteúdo do documento
    #
    # Por que esse mapeamento complicado?
    # O Cohere retorna objetos com .document e .relevance_score,
    # mas NÃO retorna o índice original. Precisamos mapear de volta
    # para os nossos dicts originais (que têm metadata, score original, etc.)
    #
    # Lógica idêntica ao original (tools.py linhas 111-139).
    if isinstance(reranked, list) and reranked:
        # Extrai conteúdo e score de cada resultado rerankeado
        reranked_pairs = []
        for obj in reranked:
            # O Cohere retorna objetos com diferentes atributos
            # dependendo da versão. Tentamos todos os formatos.
            content = _extract_content(obj)
            score = _extract_score(obj)
            reranked_pairs.append((content, score))
        
        # Mapa: conteúdo → fila de índices no array original
        # (deque para suportar chunks com conteúdo duplicado)
        idx_map = defaultdict(deque)
        for i, doc in enumerate(docs):
            idx_map[doc].append(i)
        
        # Reconstrói a lista na nova ordem
        ordered = []
        for content, _score in reranked_pairs:
            if content in idx_map and idx_map[content]:
                original_idx = idx_map[content].popleft()
                # Pega o dict original (com metadata) e atualiza o score
                item = dict(results[original_idx])
                item["rerank_score"] = _score
                ordered.append(item)
        
        if ordered:
            return ordered[:k]
    
    # Fallback: se o reranking falhou, retorna na ordem original
    # Melhor retornar algo do que nada
    print("[rerank] Cohere reranking falhou. Retornando ordem original.")
    return results[:k]


def _extract_content(obj) -> str:
    """Extrai o texto de um resultado do Cohere.
    
    O Cohere retorna objetos em formatos diferentes dependendo
    da versão do SDK. Esta função tenta todos os formatos conhecidos.
    
    Possíveis formatos:
      - obj.document.page_content (LangChain Document)
      - obj.document.text (Cohere nativo)
      - obj.page_content (Document direto)
      - obj["document"]["text"] (dict)
    
    Função baseada na lógica do original (tools.py linhas 113-125).
    """
    # Tenta atributo .document primeiro
    doc = getattr(obj, "document", None)
    if doc is not None:
        txt = getattr(doc, "page_content", None) or getattr(doc, "text", None)
        if txt:
            return str(txt)
        return str(doc)
    
    # Tenta atributos diretos
    txt = getattr(obj, "page_content", None) or getattr(obj, "text", None)
    if txt:
        return str(txt)
    
    # Tenta formato dict
    if isinstance(obj, dict):
        d = obj.get("document", obj)
        return str(d.get("text") or d.get("page_content") or d)
    
    return str(obj)


def _extract_score(obj) -> float:
    """Extrai o score de relevância de um resultado do Cohere.
    
    O Cohere retorna um relevance_score entre 0 e 1 para cada
    documento rerankeado. 1.0 = altamente relevante, 0.0 = irrelevante.
    
    Função baseada na lógica do original (tools.py linhas 125-126).
    """
    score = getattr(obj, "relevance_score", None)
    if score is None and isinstance(obj, dict):
        score = obj.get("relevance_score")
    return float(score or 0.0)
