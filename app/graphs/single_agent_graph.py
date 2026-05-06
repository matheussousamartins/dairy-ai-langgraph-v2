# -*- coding: utf-8 -*-
"""
graphs/single_agent_graph.py - Grafo LangGraph V2 (Single-Agent)

Arquitetura simplificada sem orquestrador multiagente:

  START
    analyze_query       - classifica intencao, resolve anafora (sem LLM quando possivel)
    retrieve_context    - RAG multi-tabela com filtros de metadata por dominio
    generate_answer     - 1 chamada LLM com todos os chunks consolidados
    validate_response   - strip de frases proibidas + controle de qualidade minima
  END

Comparado ao orquestrador (V1):
  V1: classify -> route -> execute (N agentes em paralelo) -> consolidate -> END
  V2: analyze_query -> retrieve_context -> generate_answer -> validate_response -> END

Ganhos esperados:
  - Latencia: 1-2 chamadas LLM vs. 2-4 no orquestrador
  - Custo: proporcional a reducao de chamadas
  - Depurabilidade: fluxo linear, sem cascata de fallbacks

O que e reutilizado do V1:
  - search_knowledge_base / embed_query (rag/search.py) - inalterado
  - contextualize_query_for_rag (rag/search.py) - inalterado
  - metadata_filters.classify_query_intent (rag/metadata_filters.py) - novo, reusa orch_signals
  - strip_prohibited_phrases / detect_question_type (agents/orch_quality.py) - inalterado
  - observability (app/observability.py) - inalterado
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.config import (
    LLM_MODEL,
    AGENT_TEMPERATURE,
    DEFAULT_SEARCH_TYPE,
    DEFAULT_K,
    MATCH_THRESHOLD,
    SINGLE_AGENT_MAX_TABLES,
    SINGLE_AGENT_K_PER_TABLE,
    SINGLE_AGENT_SEARCH_TYPE,
    SINGLE_AGENT_ANSWER_TIMEOUT_SEC,
    SINGLE_AGENT_REGULATORY_K,
    SINGLE_AGENT_REGULATORY_MIN_SCORE,
)
from app.rag.search import (
    search_knowledge_base,
    embed_query,
    contextualize_query_for_rag,
)
from app.rag.metadata_filters import classify_query_intent_async, QueryIntent, _TABLE_REGULATORIOS
from app.agents.single_agent_prompts import get_single_agent_prompt
from app.agents.synthesis_rules import build_synthesis_prompt
from app.agents.evidence_reducer import reduce_evidence_for_question
from app.agents.orch_quality import (
    detect_question_type,
    strip_prohibited_phrases,
    classify_response_quality,
)
from app.agents.orch_text import (
    _extract_current_user_segment,
    _strip_profile_suffix,
    _postprocess_consolidated_answer,
)
from app.agents.orchestrator import (
    _ZERO_EVIDENCE_MSG,
    _fetch_web_fallback_evidence,
    _render_web_sources_block,
)
from app.observability import log_event, NodeTimer, LLMSlot

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SingleAgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    llm_model: str
    user_profile: Optional[Dict[str, Any]]
    query_intent: Optional[QueryIntent]
    retrieved_chunks: List[Dict[str, Any]]
    specialist_chunks: List[Dict[str, Any]]
    regulatory_chunks: List[Dict[str, Any]]
    context_text: str
    reduced_specialist_text: str
    reduced_regulatory_text: str
    evidence_reduction_stats: Dict[str, Any]
    final_response: str
    response_quality: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_user_query(state: SingleAgentState) -> str:
    """Extrai a query do usuario do ultimo HumanMessage do estado."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            raw = msg.content or ""
            stripped = _strip_profile_suffix(raw)
            return _extract_current_user_segment(stripped).strip() or stripped
    return ""


def _extract_context_lines(state: SingleAgentState) -> List[str]:
    """Constroi linhas de contexto recente para resolucao de anafora."""
    messages = state.get("messages") or []
    lines: List[str] = []
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
            lines.append(f"Usuario: {(msg.content or '').strip()[:300]}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Dairy AI: {(msg.content or '').strip()[:300]}")
    return lines[-8:]


def _format_chunks_as_context(chunks: List[Dict[str, Any]]) -> str:
    """Formata chunks recuperados em bloco de contexto para o LLM."""
    if not chunks:
        return ""
    parts: List[str] = []
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _reduce_chunks_for_prompt(
    *,
    question: str,
    chunks: List[Dict[str, Any]],
    agent_id: int,
    max_sentences: Optional[int] = None,
) -> tuple[str, Dict[str, Any]]:
    """Compress retrieved chunks into answer-focused evidence.

    The reducer is intentionally best-effort: when it cannot select useful
    snippets, the caller can fall back to the original chunk text.
    """
    original_text = _format_chunks_as_context(chunks)
    if not original_text.strip():
        return "", {
            "agent_id": agent_id,
            "input_chunks": len(chunks),
            "used_reducer": False,
            "selected_snippets": 0,
            "direct_answer": False,
            "input_chars": 0,
            "output_chars": 0,
        }

    top_score = max((float(chunk.get("score") or 0.0) for chunk in chunks), default=0.0)
    reduced = reduce_evidence_for_question(
        question,
        original_text,
        agent_id=agent_id,
        top_score=top_score,
        max_sentences=max_sentences,
    )
    reduced_text = (reduced.text or "").strip()

    if reduced.direct_answer:
        if reduced_text and reduced.direct_answer not in reduced_text:
            reduced_text = f"{reduced.direct_answer} {reduced_text}".strip()
        elif not reduced_text:
            reduced_text = reduced.direct_answer.strip()

    if not reduced_text:
        return original_text, {
            "agent_id": agent_id,
            "input_chunks": len(chunks),
            "used_reducer": False,
            "selected_snippets": 0,
            "direct_answer": False,
            "input_chars": len(original_text),
            "output_chars": len(original_text),
        }

    return reduced_text, {
        "agent_id": agent_id,
        "input_chunks": len(chunks),
        "used_reducer": True,
        "selected_snippets": len(reduced.snippets),
        "direct_answer": bool(reduced.direct_answer),
        "input_chars": len(original_text),
        "output_chars": len(reduced_text),
    }


def _get_llm(state: SingleAgentState) -> ChatOpenAI:
    model = state.get("llm_model") or LLM_MODEL
    return ChatOpenAI(
        model=model,
        temperature=AGENT_TEMPERATURE,
        max_tokens=1200,
    )


# ---------------------------------------------------------------------------
# No 1: analyze_query
# ---------------------------------------------------------------------------

async def analyze_query(state: SingleAgentState) -> dict:
    """Classifica a intencao da query e resolve anafora. Sem chamada LLM na maioria dos casos."""
    async with NodeTimer("analyze_query"):
        raw_query = _extract_user_query(state)
        context_lines = _extract_context_lines(state)

        resolved_query = contextualize_query_for_rag(raw_query, context_lines)
        intent = await classify_query_intent_async(resolved_query)

        log_event("analyze_query_complete",
                  domain=intent.domain,
                  tables=str(intent.search_tables),
                  needs_regulatory=intent.needs_regulatory,
                  is_greeting=intent.is_greeting,
                  query_len=len(resolved_query))

        if intent.is_greeting:
            greeting_response = (
                "Oi! Eu sou o Dairy AI, seu assistente tecnico especializado em "
                "laticinios. Como posso ajudar com suas duvidas sobre queijos, "
                "fermentados, qualidade do leite ou legislacao?"
            )
            return {
                "query_intent": intent,
                "retrieved_chunks": [],
                "context_text": "",
                "final_response": greeting_response,
                "messages": [AIMessage(content=greeting_response)],
            }

        return {"query_intent": intent}


# ---------------------------------------------------------------------------
# No 2: retrieve_context
# ---------------------------------------------------------------------------

async def retrieve_context(state: SingleAgentState) -> dict:
    """Busca chunks nas tabelas do intent + busca regulatoria complementar em paralelo.

    A busca regulatoria roda sempre (k=SINGLE_AGENT_REGULATORY_K), mas seus chunks
    so entram no contexto se score >= SINGLE_AGENT_REGULATORY_MIN_SCORE.
    Embedding pre-computado uma unica vez e reutilizado em todas as buscas.
    """
    intent: Optional[QueryIntent] = state.get("query_intent")

    if not intent or intent.is_greeting:
        return {"retrieved_chunks": [], "context_text": ""}

    tables = intent.search_tables[:SINGLE_AGENT_MAX_TABLES]
    if not tables:
        return {"retrieved_chunks": [], "context_text": ""}

    raw_query = _extract_user_query(state)
    context_lines = _extract_context_lines(state)
    resolved_query = contextualize_query_for_rag(raw_query, context_lines)

    search_type = SINGLE_AGENT_SEARCH_TYPE or DEFAULT_SEARCH_TYPE
    k = SINGLE_AGENT_K_PER_TABLE or DEFAULT_K
    # hybrid_rrf usa scores RRF (max ~0.04) — threshold cosine nao se aplica
    effective_threshold = None if search_type == "hybrid_rrf" else MATCH_THRESHOLD

    async with NodeTimer("retrieve_context"):
        # Embedding pre-computado uma vez, reutilizado em todas as tabelas
        precomputed: Optional[List[float]] = None
        if search_type != "text":
            try:
                precomputed = embed_query(resolved_query)
            except Exception as exc:
                _log.warning("retrieve_context: falha ao pre-computar embedding: %s", exc)

        loop = asyncio.get_event_loop()

        def _search(table: str, k_override: int) -> List[Dict[str, Any]]:
            try:
                return search_knowledge_base(
                    query=resolved_query,
                    table_name=table,
                    search_type=search_type,
                    k=k_override,
                    threshold=effective_threshold,
                    precomputed_embedding=precomputed,
                )
            except Exception as exc:
                _log.warning("retrieve_context: falha na tabela %s: %s", table, exc)
                return []

        # Tabelas principais + regulatoria complementar (se ainda nao esta nas principais)
        regulatory_is_primary = _TABLE_REGULATORIOS in tables
        all_tables_to_search = list(tables)
        if not regulatory_is_primary:
            all_tables_to_search.append(_TABLE_REGULATORIOS)

        # Executa todas as buscas em paralelo via executor
        futures = []
        for table in all_tables_to_search:
            is_regulatory_complement = (table == _TABLE_REGULATORIOS and not regulatory_is_primary)
            k_for_table = SINGLE_AGENT_REGULATORY_K if is_regulatory_complement else k
            futures.append(loop.run_in_executor(None, _search, table, k_for_table))

        results_per_table = await asyncio.gather(*futures)

        # Separa chunks por origem: especialista vs regulatorio complementar
        specialist_raw: List[Dict[str, Any]] = []
        regulatory_raw: List[Dict[str, Any]] = []
        reg_included = 0
        reg_skipped = 0

        for table, chunks in zip(all_tables_to_search, results_per_table):
            is_regulatory_complement = (table == _TABLE_REGULATORIOS and not regulatory_is_primary)
            if is_regulatory_complement:
                for chunk in chunks:
                    score = float(chunk.get("score") or 0.0)
                    if score >= SINGLE_AGENT_REGULATORY_MIN_SCORE:
                        regulatory_raw.append(chunk)
                        reg_included += 1
                    else:
                        reg_skipped += 1
            elif table == _TABLE_REGULATORIOS:
                # Tabela regulatoria como busca primaria (quando a query e regulatoria)
                regulatory_raw.extend(chunks)
            else:
                specialist_raw.extend(chunks)

        def _dedup(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen: set = set()
            out: List[Dict[str, Any]] = []
            for chunk in chunks:
                content = (chunk.get("content") or "").strip()
                key = hash(content[:200])
                if key not in seen and content:
                    seen.add(key)
                    out.append(chunk)
            return out

        specialist_chunks = _dedup(specialist_raw)
        regulatory_chunks = _dedup(regulatory_raw)

        specialist_chunks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        regulatory_chunks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

        specialist_chunks = specialist_chunks[:k]
        regulatory_chunks = regulatory_chunks[:SINGLE_AGENT_REGULATORY_K]

        # context_text combinado (para backwards compat e logs)
        all_final = specialist_chunks + [
            c for c in regulatory_chunks if c not in specialist_chunks
        ]
        context_text = _format_chunks_as_context(all_final)

        log_event("retrieve_context_complete",
                  tables=str(all_tables_to_search),
                  specialist_chunks=len(specialist_chunks),
                  regulatory_chunks=len(regulatory_chunks),
                  reg_complement_included=reg_included,
                  reg_complement_skipped=reg_skipped,
                  context_chars=len(context_text))

        return {
            "retrieved_chunks": all_final,
            "specialist_chunks": specialist_chunks,
            "regulatory_chunks": regulatory_chunks,
            "context_text": context_text,
        }


# ---------------------------------------------------------------------------
# No 3: generate_answer
# ---------------------------------------------------------------------------

async def generate_answer(state: SingleAgentState) -> dict:
    """Gera a resposta final com base nos chunks recuperados. Uma unica chamada LLM.

    Quando ha chunks regulatorios separados, monta prompt hierarquico:
    tecnico lidera, regulatorio complementa — igual ao consolidador V1.

    Sem evidencia na KB: tenta web fallback (se habilitado) antes de
    retornar _ZERO_EVIDENCE_MSG — nunca delega o texto de ausencia ao LLM.
    """
    intent: Optional[QueryIntent] = state.get("query_intent")

    if not intent or intent.is_greeting or state.get("final_response"):
        return {}

    raw_query = _extract_user_query(state)
    specialist_chunks = state.get("specialist_chunks") or []
    regulatory_chunks = state.get("regulatory_chunks") or []

    # Fallback para context_text quando os campos separados nao estao presentes
    # (compatibilidade com invocacoes que nao passam pelo retrieve_context normal)
    if not specialist_chunks and not regulatory_chunks:
        context_text = state.get("context_text") or ""
        if context_text:
            specialist_chunks = [{"content": context_text, "score": 1.0}]

    # Sem nenhuma evidencia na KB → web fallback direto (sem gate de sinal de laticinios)
    # Quando a KB esta vazia, ja e last-resort — qualquer fonte web e valida.
    if not specialist_chunks and not regulatory_chunks:
        log_event("generate_answer_no_kb_evidence", query_len=len(raw_query))
        web_text, web_sources, _ = await _fetch_web_fallback_evidence(raw_query)
        if web_text:
            answer = web_text
            sources_block = _render_web_sources_block(web_sources)
            if sources_block:
                answer = f"{answer.rstrip()}\n\n{sources_block}"
            return {"final_response": answer}
        # Web também falhou (sem conexão, timeout) — único caso onde retorna mensagem fixa
        return {"final_response": _ZERO_EVIDENCE_MSG}

    question_type = detect_question_type(raw_query)

    specialist_text, specialist_reduction = _reduce_chunks_for_prompt(
        question=raw_query,
        chunks=specialist_chunks,
        agent_id=1,
    )
    regulatory_text, regulatory_reduction = _reduce_chunks_for_prompt(
        question=raw_query,
        chunks=regulatory_chunks,
        agent_id=3,
        max_sentences=3,
    )

    log_event("evidence_reduction_complete",
              specialist_used=specialist_reduction.get("used_reducer", False),
              specialist_input_chars=specialist_reduction.get("input_chars", 0),
              specialist_output_chars=specialist_reduction.get("output_chars", 0),
              regulatory_used=regulatory_reduction.get("used_reducer", False),
              regulatory_input_chars=regulatory_reduction.get("input_chars", 0),
              regulatory_output_chars=regulatory_reduction.get("output_chars", 0))

    human_content = build_synthesis_prompt(
        question=raw_query,
        question_type=question_type,
        specialist_text=specialist_text,
        regulatory_text=regulatory_text,
    )

    messages_for_llm = [
        SystemMessage(content=get_single_agent_prompt()),
        HumanMessage(content=human_content),
    ]

    async with NodeTimer("generate_answer"):
        try:
            async with LLMSlot():
                llm = _get_llm(state)
                response = await asyncio.wait_for(
                    llm.ainvoke(messages_for_llm),
                    timeout=SINGLE_AGENT_ANSWER_TIMEOUT_SEC,
                )
                answer = (response.content or "").strip()
        except asyncio.TimeoutError:
            _log.warning("generate_answer: timeout apos %ss", SINGLE_AGENT_ANSWER_TIMEOUT_SEC)
            answer = "Nao foi possivel gerar uma resposta no tempo esperado. Tente novamente."
        except Exception as exc:
            _log.error("generate_answer: erro na chamada LLM: %s", exc)
            answer = "Ocorreu um erro ao processar a resposta. Tente novamente."

    log_event("generate_answer_complete",
              answer_chars=len(answer),
              question_type=question_type,
              has_specialist=bool(specialist_chunks),
              has_regulatory=bool(regulatory_chunks))

    return {
        "final_response": answer,
        "reduced_specialist_text": specialist_text,
        "reduced_regulatory_text": regulatory_text,
        "evidence_reduction_stats": {
            "specialist": specialist_reduction,
            "regulatory": regulatory_reduction,
        },
    }


# ---------------------------------------------------------------------------
# No 4: validate_response
# ---------------------------------------------------------------------------

async def validate_response(state: SingleAgentState) -> dict:
    """Aplica pos-processamento e controle de qualidade minima na resposta."""
    final_response = state.get("final_response") or ""

    if not final_response:
        final_response = "Nao foi possivel gerar uma resposta. Tente novamente."

    raw_query = _extract_user_query(state)

    async with NodeTimer("validate_response"):
        cleaned = strip_prohibited_phrases(final_response)
        cleaned = _postprocess_consolidated_answer(raw_query, cleaned)
        quality = classify_response_quality(cleaned)

        log_event("validate_response_complete",
                  quality=quality,
                  response_chars=len(cleaned))

    return {
        "final_response": cleaned,
        "response_quality": quality,
        "messages": [AIMessage(content=cleaned)],
    }


# ---------------------------------------------------------------------------
# Roteamento condicional
# ---------------------------------------------------------------------------

def _should_skip_to_validate(state: SingleAgentState) -> str:
    """Depois de analyze_query, pula para validate se for saudacao."""
    intent: Optional[QueryIntent] = state.get("query_intent")
    if intent and intent.is_greeting:
        return "validate_response"
    return "retrieve_context"


# ---------------------------------------------------------------------------
# Construcao do grafo
# ---------------------------------------------------------------------------

_graph_cache: Optional[Any] = None  # invalidado ao reiniciar o processo


def get_single_agent_graph():
    """Retorna o grafo compilado (lazy, singleton)."""
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache

    builder = StateGraph(SingleAgentState)

    builder.add_node("analyze_query", analyze_query)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("validate_response", validate_response)

    builder.set_entry_point("analyze_query")

    builder.add_conditional_edges(
        "analyze_query",
        _should_skip_to_validate,
        {
            "retrieve_context": "retrieve_context",
            "validate_response": "validate_response",
        },
    )

    builder.add_edge("retrieve_context", "generate_answer")
    builder.add_edge("generate_answer", "validate_response")
    builder.add_edge("validate_response", END)

    _graph_cache = builder.compile()
    return _graph_cache
