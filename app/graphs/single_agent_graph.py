# -*- coding: utf-8 -*-
"""
graphs/single_agent_graph.py - Grafo LangGraph V2 (Single-Agent) — Producao

Arquitetura:

  START
    analyze_query       - classifica intencao, resolve anafora (sem LLM quando possivel)
    retrieve_context    - RAG multi-tabela + regulatorio complementar em paralelo
    evaluate_chunks     - LLM leve verifica se chunks respondem a pergunta;
                          se insuficientes, faz segunda busca com query expandida
    generate_answer     - 1 chamada LLM com contexto validado e hierarquizado
    validate_response   - strip de frases proibidas + re-geracao se qualidade baixa
  END

Camadas de qualidade:
  1. Relevance gate     — chunks com score < threshold descartados antes do LLM
  2. evaluate_chunks    — gpt-4o-mini verifica cobertura; dispara fallback se necessario
  3. generate_answer    — prompt hierarquico tecnico/regulatorio + R1-R9
  4. validate_response  — quality classifier + re-geracao em LOW/UNUSABLE
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
    SINGLE_AGENT_CLASSIFIER_MODEL,
    SINGLE_AGENT_CHUNK_EVAL_ENABLED,
    SINGLE_AGENT_MIN_QUALITY_FOR_REGEN,
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
    ResponseQuality,
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
# Constantes do pipeline de qualidade
# ---------------------------------------------------------------------------

# Score minimo para chunks especialistas entrarem no contexto (hybrid_rrf).
# hybrid_rrf RRF scores tipicos: 0.01-0.07. Threshold baixo para nao filtrar demais.
# Aumentar so se houver muito ruido de chunks irrelevantes confirmado em prod.
_SPECIALIST_MIN_SCORE_HYBRID = float(0.008)
# Score minimo para chunks especialistas entrarem no contexto (vector/cosine)
_SPECIALIST_MIN_SCORE_VECTOR = float(0.22)

# Minimo de chunks especialistas validos para considerar retrieve bem-sucedido
_MIN_SPECIALIST_CHUNKS_OK = 2

# Timeout do avaliador de chunks (segundos)
_CHUNK_EVAL_TIMEOUT_SEC = 6.0

# Timeout da re-geracao em validate_response (segundos)
_REGEN_TIMEOUT_SEC = 25.0

# Quantos chunks usar na segunda busca (mais ampla)
_FALLBACK_K_MULTIPLIER = 2


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
    chunks_evaluated: bool
    chunks_sufficient: bool
    retrieval_attempts: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_user_query(state: SingleAgentState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            raw = msg.content or ""
            stripped = _strip_profile_suffix(raw)
            return _extract_current_user_segment(stripped).strip() or stripped
    return ""


def _extract_context_lines(state: SingleAgentState) -> List[str]:
    messages = state.get("messages") or []
    lines: List[str] = []
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
            lines.append(f"Usuario: {(msg.content or '').strip()[:300]}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Dairy AI: {(msg.content or '').strip()[:300]}")
    return lines[-8:]


def _format_chunks_as_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    parts: List[str] = []
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _apply_relevance_gate(
    chunks: List[Dict[str, Any]],
    search_type: str,
    is_regulatory: bool = False,
) -> List[Dict[str, Any]]:
    """Descarta chunks com score abaixo do threshold dinamico por tipo de busca."""
    if is_regulatory:
        # Regulatorio usa SINGLE_AGENT_REGULATORY_MIN_SCORE (ja aplicado no retrieve)
        return chunks

    if search_type == "hybrid_rrf":
        min_score = _SPECIALIST_MIN_SCORE_HYBRID
    elif search_type in ("vector", "hybrid_union"):
        min_score = _SPECIALIST_MIN_SCORE_VECTOR
    else:
        # text search: sem gate por score
        return chunks

    filtered = [c for c in chunks if float(c.get("score") or 0.0) >= min_score]
    dropped = len(chunks) - len(filtered)
    if dropped > 0:
        _log.debug("relevance_gate: descartou %d/%d chunks (score < %.4f)", dropped, len(chunks), min_score)
    return filtered


def _reduce_chunks_for_prompt(
    *,
    question: str,
    chunks: List[Dict[str, Any]],
    agent_id: int,
    max_sentences: Optional[int] = None,
) -> tuple[str, Dict[str, Any]]:
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


def _get_llm(state: SingleAgentState, max_tokens: int = 1200) -> ChatOpenAI:
    model = state.get("llm_model") or LLM_MODEL
    return ChatOpenAI(
        model=model,
        temperature=AGENT_TEMPERATURE,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# No 1: analyze_query
# ---------------------------------------------------------------------------

async def analyze_query(state: SingleAgentState) -> dict:
    """Classifica intencao, resolve anafora. Sem LLM na maioria dos casos."""
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

        return {
            "query_intent": intent,
            "retrieval_attempts": 0,
        }


# ---------------------------------------------------------------------------
# No 2: retrieve_context
# ---------------------------------------------------------------------------

async def retrieve_context(state: SingleAgentState) -> dict:
    """Busca chunks nas tabelas do intent + regulatorio complementar em paralelo.

    Na segunda tentativa (retrieval_attempts >= 1), usa k ampliado e query
    expandida para aumentar recall quando a primeira busca foi insuficiente.
    """
    intent: Optional[QueryIntent] = state.get("query_intent")

    if not intent or intent.is_greeting:
        return {"retrieved_chunks": [], "context_text": ""}

    tables = intent.search_tables[:SINGLE_AGENT_MAX_TABLES]
    if not tables:
        return {"retrieved_chunks": [], "context_text": ""}

    retrieval_attempts = state.get("retrieval_attempts") or 0
    raw_query = _extract_user_query(state)
    context_lines = _extract_context_lines(state)
    resolved_query = contextualize_query_for_rag(raw_query, context_lines)

    # Na segunda tentativa: expande k e usa todas as tabelas de dominio
    k_multiplier = _FALLBACK_K_MULTIPLIER if retrieval_attempts >= 1 else 1
    search_type = SINGLE_AGENT_SEARCH_TYPE or DEFAULT_SEARCH_TYPE
    k = (SINGLE_AGENT_K_PER_TABLE or DEFAULT_K) * k_multiplier
    k_reg = SINGLE_AGENT_REGULATORY_K * k_multiplier

    # hybrid_rrf usa scores RRF (max ~0.04) — threshold cosine nao se aplica
    effective_threshold = None if search_type == "hybrid_rrf" else MATCH_THRESHOLD

    async with NodeTimer("retrieve_context"):
        precomputed: Optional[List[float]] = None
        embedding_failed = False
        if search_type != "text":
            try:
                precomputed = embed_query(resolved_query)
                _log.info("retrieve_context: embedding OK — %d dims", len(precomputed) if precomputed else 0)
            except Exception as exc:
                embedding_failed = True
                _log.error("retrieve_context: FALHA no embedding — fallback para text search: %s", exc, exc_info=True)

        # Se embedding falhou, usa text search como fallback seguro
        if embedding_failed:
            search_type = "text"
            effective_threshold = None

        loop = asyncio.get_event_loop()

        def _search(table: str, k_override: int) -> List[Dict[str, Any]]:
            try:
                results = search_knowledge_base(
                    query=resolved_query,
                    table_name=table,
                    search_type=search_type,
                    k=k_override,
                    threshold=effective_threshold,
                    precomputed_embedding=precomputed,
                )
                _log.debug(
                    "retrieve_context: tabela=%s k=%d resultados=%d top_score=%s",
                    table, k_override, len(results),
                    f"{float(results[0].get('score') or 0):.4f}" if results else "N/A",
                )
                return results
            except Exception as exc:
                _log.error("retrieve_context: FALHA na tabela %s: %s", table, exc, exc_info=True)
                return []

        regulatory_is_primary = _TABLE_REGULATORIOS in tables
        all_tables_to_search = list(tables)
        if not regulatory_is_primary:
            all_tables_to_search.append(_TABLE_REGULATORIOS)

        futures = []
        for table in all_tables_to_search:
            is_reg_complement = (table == _TABLE_REGULATORIOS and not regulatory_is_primary)
            k_for_table = k_reg if is_reg_complement else k
            futures.append(loop.run_in_executor(None, _search, table, k_for_table))

        results_per_table = await asyncio.gather(*futures)

        specialist_raw: List[Dict[str, Any]] = []
        regulatory_raw: List[Dict[str, Any]] = []
        reg_included = 0
        reg_skipped = 0

        for table, chunks in zip(all_tables_to_search, results_per_table):
            is_reg_complement = (table == _TABLE_REGULATORIOS and not regulatory_is_primary)
            if is_reg_complement:
                for chunk in chunks:
                    score = float(chunk.get("score") or 0.0)
                    if score >= SINGLE_AGENT_REGULATORY_MIN_SCORE:
                        regulatory_raw.append(chunk)
                        reg_included += 1
                    else:
                        reg_skipped += 1
            elif table == _TABLE_REGULATORIOS:
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

        _log.info(
            "retrieve_context: antes do gate — specialist=%d regulatory=%d search_type=%s",
            len(specialist_chunks), len(regulatory_chunks), search_type,
        )

        # Aplica gate de relevancia por score antes de ordenar
        specialist_chunks = _apply_relevance_gate(specialist_chunks, search_type, is_regulatory=False)
        # Regulatorio ja foi filtrado por SINGLE_AGENT_REGULATORY_MIN_SCORE acima

        specialist_chunks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        regulatory_chunks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

        specialist_chunks = specialist_chunks[:k]
        regulatory_chunks = regulatory_chunks[:k_reg]

        all_final = specialist_chunks + [
            c for c in regulatory_chunks if c not in specialist_chunks
        ]
        context_text = _format_chunks_as_context(all_final)

        log_event("retrieve_context_complete",
                  attempt=retrieval_attempts,
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
            "chunks_evaluated": False,
            "chunks_sufficient": False,
            "retrieval_attempts": retrieval_attempts + 1,
        }


# ---------------------------------------------------------------------------
# No 3: evaluate_chunks
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = (
    "Voce avalia se trechos de conhecimento sao suficientes para responder uma pergunta tecnica. "
    "Responda APENAS com JSON valido: {\"sufficient\": true/false, \"reason\": \"<max 15 palavras>\"}. "
    "sufficient=true quando os trechos contem dados diretos que respondem a pergunta (valores, limites, processos). "
    "sufficient=false quando os trechos sao vagos, falam de outro produto/tema ou nao tem o dado pedido."
)


async def evaluate_chunks(state: SingleAgentState) -> dict:
    """Avalia se os chunks recuperados respondem a pergunta.

    Usa gpt-4o-mini com max_tokens=80 para minimizar latencia (~300ms).
    Se insufficient: marca para re-retrieval com k ampliado.
    Se sufficient ou segunda tentativa: segue para generate_answer.
    """
    intent: Optional[QueryIntent] = state.get("query_intent")
    if not intent or intent.is_greeting or state.get("final_response"):
        return {"chunks_evaluated": True, "chunks_sufficient": True}

    specialist_chunks = state.get("specialist_chunks") or []
    regulatory_chunks = state.get("regulatory_chunks") or []
    retrieval_attempts = state.get("retrieval_attempts") or 0

    # Se avaliador desabilitado via config, pula direto
    if not SINGLE_AGENT_CHUNK_EVAL_ENABLED:
        return {"chunks_evaluated": True, "chunks_sufficient": True}

    # retrieval_attempts ja foi incrementado: >= 2 significa segunda busca concluida
    if retrieval_attempts >= 2 or (not specialist_chunks and not regulatory_chunks):
        log_event("evaluate_chunks_skip",
                  reason="second_attempt_or_no_chunks",
                  specialist=len(specialist_chunks),
                  regulatory=len(regulatory_chunks))
        return {"chunks_evaluated": True, "chunks_sufficient": True}

    # Gate rapido: se temos chunks suficientes com score alto, pula o LLM
    if len(specialist_chunks) >= _MIN_SPECIALIST_CHUNKS_OK:
        top_score = max(float(c.get("score") or 0.0) for c in specialist_chunks)
        search_type = SINGLE_AGENT_SEARCH_TYPE or DEFAULT_SEARCH_TYPE
        good_threshold = 0.04 if search_type == "hybrid_rrf" else 0.45
        if top_score >= good_threshold:
            log_event("evaluate_chunks_fastpath",
                      specialist=len(specialist_chunks),
                      top_score=top_score)
            return {"chunks_evaluated": True, "chunks_sufficient": True}

    raw_query = _extract_user_query(state)

    # Monta preview dos chunks (max 600 chars para economizar tokens)
    all_chunks = specialist_chunks[:3] + regulatory_chunks[:1]
    preview_parts = []
    total_chars = 0
    for chunk in all_chunks:
        content = (chunk.get("content") or "").strip()[:200]
        if content:
            preview_parts.append(content)
            total_chars += len(content)
            if total_chars >= 600:
                break

    chunks_preview = "\n---\n".join(preview_parts) if preview_parts else "(nenhum trecho recuperado)"

    human_msg = (
        f"PERGUNTA: {raw_query}\n\n"
        f"TRECHOS RECUPERADOS:\n{chunks_preview}\n\n"
        f"Os trechos respondem diretamente a pergunta?"
    )

    async with NodeTimer("evaluate_chunks"):
        try:
            llm = ChatOpenAI(
                model=SINGLE_AGENT_CLASSIFIER_MODEL,
                temperature=0,
                max_tokens=80,
            )
            async with LLMSlot():
                response = await asyncio.wait_for(
                    llm.ainvoke([
                        SystemMessage(content=_EVAL_SYSTEM),
                        HumanMessage(content=human_msg),
                    ]),
                    timeout=_CHUNK_EVAL_TIMEOUT_SEC,
                )
            raw = (response.content or "").strip()
            # Extrai JSON mesmo se vier com texto extra
            json_match = re.search(r"\{[^}]+\}", raw)
            if json_match:
                parsed = json.loads(json_match.group(0))
                sufficient = bool(parsed.get("sufficient", True))
                reason = str(parsed.get("reason", ""))
            else:
                sufficient = True
                reason = "parse_failed"

        except asyncio.TimeoutError:
            _log.warning("evaluate_chunks: timeout — assumindo sufficient=True")
            sufficient = True
            reason = "timeout"
        except Exception as exc:
            _log.warning("evaluate_chunks: erro — assumindo sufficient=True: %s", exc)
            sufficient = True
            reason = f"error: {exc}"

    log_event("evaluate_chunks_complete",
              sufficient=sufficient,
              reason=reason,
              specialist=len(specialist_chunks),
              regulatory=len(regulatory_chunks),
              retrieval_attempts=retrieval_attempts)

    return {
        "chunks_evaluated": True,
        "chunks_sufficient": sufficient,
    }


def _should_retry_retrieval(state: SingleAgentState) -> str:
    """Decide se faz segundo retrieval ou segue para generate_answer.

    retrieval_attempts ja foi incrementado em retrieve_context, entao:
      1 = completou primeira busca, pode tentar segunda
      2 = completou segunda busca, nao tenta mais
    """
    if not state.get("chunks_sufficient", True):
        retrieval_attempts = state.get("retrieval_attempts") or 0
        if retrieval_attempts < 2:
            return "retrieve_context"
    return "generate_answer"


# ---------------------------------------------------------------------------
# No 4: generate_answer
# ---------------------------------------------------------------------------

async def generate_answer(state: SingleAgentState) -> dict:
    """Gera a resposta final com base nos chunks validados.

    Prompt hierarquico: tecnico lidera, regulatorio complementa.
    Sem evidencia: web fallback → mensagem fixa.
    """
    intent: Optional[QueryIntent] = state.get("query_intent")

    if not intent or intent.is_greeting or state.get("final_response"):
        return {}

    raw_query = _extract_user_query(state)
    specialist_chunks = state.get("specialist_chunks") or []
    regulatory_chunks = state.get("regulatory_chunks") or []

    # Fallback para context_text (compatibilidade)
    if not specialist_chunks and not regulatory_chunks:
        context_text = state.get("context_text") or ""
        if context_text:
            specialist_chunks = [{"content": context_text, "score": 1.0}]

    if not specialist_chunks and not regulatory_chunks:
        log_event("generate_answer_no_kb_evidence", query_len=len(raw_query))
        web_text, web_sources, _ = await _fetch_web_fallback_evidence(raw_query)
        if web_text:
            answer = web_text
            sources_block = _render_web_sources_block(web_sources)
            if sources_block:
                answer = f"{answer.rstrip()}\n\n{sources_block}"
            return {"final_response": answer}
        return {"final_response": _ZERO_EVIDENCE_MSG}

    question_type = detect_question_type(raw_query)

    # Texto completo dos chunks — LLM recebe tudo, sem filtro de sentenca.
    # O evidence_reducer era um gargalo: descartava frases relevantes antes
    # do LLM ver. Como os chunks ja foram rankeados por relevancia no retrieval
    # e temos max 8 chunks (~6000 chars), o custo de contexto e aceitavel.
    specialist_text = _format_chunks_as_context(specialist_chunks)
    regulatory_text = _format_chunks_as_context(regulatory_chunks)

    specialist_reduction = {
        "agent_id": 1,
        "input_chunks": len(specialist_chunks),
        "used_reducer": False,
        "input_chars": len(specialist_text),
        "output_chars": len(specialist_text),
    }
    regulatory_reduction = {
        "agent_id": 3,
        "input_chunks": len(regulatory_chunks),
        "used_reducer": False,
        "input_chars": len(regulatory_text),
        "output_chars": len(regulatory_text),
    }

    log_event("evidence_reduction_complete",
              specialist_used=False,
              specialist_input_chars=len(specialist_text),
              specialist_output_chars=len(specialist_text),
              regulatory_used=False,
              regulatory_input_chars=len(regulatory_text),
              regulatory_output_chars=len(regulatory_text))

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
# No 5: validate_response
# ---------------------------------------------------------------------------

_REGEN_SYSTEM = (
    "Voce e o Dairy AI. A resposta anterior foi classificada como insuficiente. "
    "Reescreva usando APENAS as evidencias fornecidas. "
    "Seja direto: comece pelo dado ou limite principal. "
    "Se as evidencias trouxerem numeros, use-os. Nao invente parametros ausentes."
)


async def validate_response(state: SingleAgentState) -> dict:
    """Pos-processamento + re-geracao automatica se qualidade LOW/UNUSABLE.

    Fluxo:
      1. Strip de frases proibidas
      2. Classifica qualidade
      3. Se LOW/UNUSABLE e houver chunks validos: tenta re-geracao com instrucao diferente
      4. Retorna melhor versao disponivel
    """
    final_response = state.get("final_response") or ""

    if not final_response:
        final_response = "Nao foi possivel gerar uma resposta. Tente novamente."

    raw_query = _extract_user_query(state)

    async with NodeTimer("validate_response"):
        cleaned = strip_prohibited_phrases(final_response)
        cleaned = _postprocess_consolidated_answer(raw_query, cleaned)
        quality = classify_response_quality(cleaned)

        log_event("validate_response_initial",
                  quality=quality,
                  response_chars=len(cleaned))

        # Limiar de qualidade minima para disparar re-geracao
        _quality_rank = {
            ResponseQuality.HIGH: 3,
            ResponseQuality.MEDIUM: 2,
            ResponseQuality.LOW: 1,
            ResponseQuality.UNUSABLE: 0,
        }
        _regen_threshold = _quality_rank.get(SINGLE_AGENT_MIN_QUALITY_FOR_REGEN, 2)
        _needs_regen = _quality_rank.get(quality, 0) < _regen_threshold

        # Re-geracao se qualidade abaixo do limiar e temos evidencias validas
        if _needs_regen:
            specialist_chunks = state.get("specialist_chunks") or []
            regulatory_chunks = state.get("regulatory_chunks") or []

            has_evidence = bool(specialist_chunks or regulatory_chunks)
            specialist_text = state.get("reduced_specialist_text") or ""
            regulatory_text = state.get("reduced_regulatory_text") or ""

            # Usa o texto reduzido ja calculado, ou reconstroi se nao tiver
            if not specialist_text and specialist_chunks:
                specialist_text = _format_chunks_as_context(specialist_chunks[:4])
            if not regulatory_text and regulatory_chunks:
                regulatory_text = _format_chunks_as_context(regulatory_chunks[:2])

            if has_evidence and (specialist_text or regulatory_text):
                question_type = detect_question_type(raw_query)
                human_content = build_synthesis_prompt(
                    question=raw_query,
                    question_type=question_type,
                    specialist_text=specialist_text,
                    regulatory_text=regulatory_text,
                )

                regen_messages = [
                    SystemMessage(content=_REGEN_SYSTEM),
                    HumanMessage(content=human_content),
                ]

                try:
                    async with LLMSlot():
                        llm = ChatOpenAI(
                            model=state.get("llm_model") or LLM_MODEL,
                            temperature=0.1,
                            max_tokens=1200,
                        )
                        regen_response = await asyncio.wait_for(
                            llm.ainvoke(regen_messages),
                            timeout=_REGEN_TIMEOUT_SEC,
                        )
                        regen_text = (regen_response.content or "").strip()
                        regen_text = strip_prohibited_phrases(regen_text)
                        regen_text = _postprocess_consolidated_answer(raw_query, regen_text)
                        regen_quality = classify_response_quality(regen_text)

                        log_event("validate_response_regen",
                                  original_quality=quality,
                                  regen_quality=regen_quality,
                                  regen_chars=len(regen_text))

                        # Aceita a re-geracao se for melhor ou igual
                        if _quality_rank.get(regen_quality, 0) >= _quality_rank.get(quality, 0):
                            cleaned = regen_text
                            quality = regen_quality

                except asyncio.TimeoutError:
                    _log.warning("validate_response: regen timeout — mantendo resposta original")
                except Exception as exc:
                    _log.warning("validate_response: regen falhou: %s", exc)

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
    intent: Optional[QueryIntent] = state.get("query_intent")
    if intent and intent.is_greeting:
        return "validate_response"
    return "retrieve_context"


# ---------------------------------------------------------------------------
# Construcao do grafo
# ---------------------------------------------------------------------------

_graph_cache: Optional[Any] = None


def get_single_agent_graph():
    """Retorna o grafo compilado (lazy, singleton)."""
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache

    builder = StateGraph(SingleAgentState)

    builder.add_node("analyze_query", analyze_query)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("evaluate_chunks", evaluate_chunks)
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

    builder.add_edge("retrieve_context", "evaluate_chunks")

    builder.add_conditional_edges(
        "evaluate_chunks",
        _should_retry_retrieval,
        {
            "retrieve_context": "retrieve_context",
            "generate_answer": "generate_answer",
        },
    )

    builder.add_edge("generate_answer", "validate_response")
    builder.add_edge("validate_response", END)

    _graph_cache = builder.compile()
    return _graph_cache
