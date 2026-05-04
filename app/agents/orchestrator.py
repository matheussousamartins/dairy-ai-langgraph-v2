"""
agents/orchestrator.py â€" Orquestrador multi-agente com execuÃ§Ã£o paralela

Fluxo do grafo:
  classify â†’ route â†’ execute (paralelo) â†’ consolidate â†’ END
                â†˜ respond_direct â†’ consolidate â†’ END

Agente 3 (Regulatórios) é incluído em toda pergunta de laticínios.
Agente 0 (Base Geral) é incluído apenas para glossário e terminologia —
sua base não cobre queries técnicas ou regulatórias genéricas.

Execução paralela:
  Todos os agentes rodam ao mesmo tempo via asyncio.gather + ainvoke.
  Latência total = tempo do agente mais lento (não a soma).
"""

import asyncio
import os
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END

from app.config import (
    LLM_MODEL,
    CONSOLIDATION_TIMEOUT_SEC,
    ORCHESTRATOR_FASTPATH,
    CLASSIFICATION_CACHE_SIZE,
    ENABLE_GENERAL_INDEX_FALLBACK,
    GENERAL_INDEX_FALLBACK_SEARCH_TYPE,
    GENERAL_INDEX_FALLBACK_PER_TABLE_K,
    GENERAL_INDEX_FALLBACK_FINAL_K,
    GENERAL_INDEX_FALLBACK_MIN_RESULTS,
    GENERAL_INDEX_FALLBACK_MAX_TABLES,
    GENERAL_INDEX_FALLBACK_ONLY_ON_WEAK,
    GENERAL_INDEX_FALLBACK_REQUIRE_DAIRY_SIGNAL,
    ENABLE_WEB_FALLBACK,
    WEB_FALLBACK_PROVIDER,
    WEB_FALLBACK_TIMEOUT_SEC,
    WEB_FALLBACK_MAX_RESULTS,
    WEB_FALLBACK_MAX_SOURCES,
    WEB_FALLBACK_ONLY_ON_WEAK,
    WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL,
    WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST,
    WEB_FALLBACK_FETCH_FULLTEXT,
    WEB_FALLBACK_MAX_PAGE_CHARS,
    WEB_FALLBACK_MAX_SNIPPET_CHARS,
    WEB_FALLBACK_ALLOWED_DOMAINS,
    MATCH_THRESHOLD,
)
from app.agents.prompts import get_orchestrator_prompt
from app.agents.agent_config import AGENTS, get_agent_by_id
from app.agents.base_agent import get_agent_graph
from app.rag.search import embed_query, search_general_knowledge_base
from app.tools.web_fallback import (
    search_web_duckduckgo,
    enrich_results_with_page_content,
    build_web_fallback_evidence,
)
from app.agents.orch_schema import OrchestratorState, ClassificationResult
from app.agents.orch_text import (
    _normalize_text,
    _normalize_mul_symbols,
    _strip_profile_suffix,
    _extract_current_user_segment,
    _extract_recent_context_block,
    _has_recent_context_block,
    _build_contextual_search_query,
    _is_objective_question,
    _is_conversation_recap_request,
    _sanitize_math_for_ui,
    _dedupe_paragraphs,
    _enforce_dornic_canonical_formula,
    _looks_uncertain,
    _strip_uncertainty_tail,
    _strip_leading_uncertainty_prefix,
    _extract_factual_candidate,
    _postprocess_consolidated_answer,
)
from app.agents.orch_models import (
    _resolve_state_model,
    _get_classifier,
    _get_consolidation_model,
    _get_direct_model,
)
from app.agents.orch_signals import (
    _ROUTING_RULES_PATH,
    _GREETINGS,
    _DAIRY_TERMS,
    _REGULATORY_STRONG_TERMS,
    _LEGAL_REQUIREMENT_DIRECT_PHRASES,
    _CHEESE_STRONG_TERMS,
    _INTENT_PATTERNS_BY_AGENT,
    _LOW_PRECISION_KEYWORDS,
    _HINT_NOISE_TERMS,
    _HINT_NOISE_TOKENS,
    _SPECIALIST_STRONG_HINTS_DEFAULT,
    _contains_any_phrase,
    _looks_like_greeting_only,
    _contains_dairy_signal,
    _is_strong_regulatory_signal,
    _is_strong_cheese_signal,
    _is_labeling_regulatory_signal,
    _is_legal_requirement_regulatory_signal,
    _is_normative_regulatory_signal,
)
from app.agents.orch_routing import (
    _REGULATORY_BASELINE_ID,
    _ROUTING_BASELINE_IDS,
    _AGENTS_WITHOUT_KB,
    _SPECIALISTS_PER_BUCKET,
    _ROUTING_CONFIDENCE_THRESHOLDS,
    _FALLBACK_MAX_ATTEMPTS,
    _FALLBACK_EXTRA_SPECIALISTS,
    _AGENT_DOMAIN_LABELS,
    _NEAREST_SPECIALIST_MAP,
    _AGENT_TABLE_BY_ID,
    _ALL_AGENT_TABLES,
    _CLASSIFICATION_CACHE,
    _MAX_CLASSIFICATION_CACHE,
    _SPECIALIST_STRONG_HINTS,
    _SPECIALIST_KEYWORDS,
    _cache_get,
    _cache_set,
    _contains_keyword,
    _keyword_weight,
    _sanitize_agent_ids,
    _rule_based_route,
    _clamp_confidence,
    _confidence_to_bucket,
    _estimate_fastpath_confidence,
    _recalibrate_confidence,
    _apply_dairy_hard_constraints,
    _apply_domain_guardrails,
    _build_execution_plan,
    _infer_domain_primary_from_text,
    _choose_primary_agent_id,
)

# Tempo máximo de espera por agente (segundos)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "12"))
_SPECIALISTS_DESC = "".join(
    f"  {agent['agent_id']} = {agent['name']}\n"
    for agent in AGENTS
    if agent["agent_id"] not in (0, 3, 5, 6)
)

def _prefer_direct_fact_response(
    user_text: str,
    responses: List[Dict[str, Any]],
) -> Optional[str]:
    """When question is objective, prefer a direct factual specialist answer."""
    if not _is_objective_question(user_text):
        return None

    direct = []
    for r in responses:
        if not r.get("success") or not r.get("response"):
            continue
        factual = _extract_factual_candidate(str(r.get("response", "")))
        if factual:
            item = dict(r)
            item["factual_response"] = factual
            direct.append(item)
    if len(direct) != 1:
        return None

    # Prefer domain specialist over transversal agents for objective facts.
    chosen = direct[0]
    aid = int(chosen.get("agent_id", -1))
    if aid in (0, 3):
        specialists = [r for r in direct if int(r.get("agent_id", -1)) not in (0, 3)]
        if len(specialists) == 1:
            chosen = specialists[0]
    return _sanitize_math_for_ui(str(chosen.get("factual_response", "")))


def _prefer_regulatory_requirement_response(
    user_text: str,
    responses: List[Dict[str, Any]],
) -> Optional[str]:
    """Para requisito legal explícito, prioriza a resposta factual do Agente 3."""
    if not _is_legal_requirement_regulatory_signal(_normalize_text(user_text)):
        return None

    for r in responses:
        if int(r.get("agent_id", -1)) != 3:
            continue
        if not r.get("success") or not r.get("response"):
            continue
        factual = _extract_factual_candidate(str(r.get("response", "")))
        if factual:
            return _sanitize_math_for_ui(factual)
    return None


def _merge_agent_responses(
    previous: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Mescla respostas por agent_id, preservando melhor evidência."""
    merged: Dict[int, Dict[str, Any]] = {}
    for item in previous:
        aid = int(item.get("agent_id", -1))
        if aid >= 0:
            merged[aid] = dict(item)
    for item in current:
        aid = int(item.get("agent_id", -1))
        if aid < 0:
            continue
        old = merged.get(aid)
        if old is None:
            merged[aid] = dict(item)
            continue
        old_ok = bool(old.get("success")) and bool(old.get("response"))
        new_ok = bool(item.get("success")) and bool(item.get("response"))
        # Prioriza resposta nova se ela melhora sucesso/evidencia.
        if (new_ok and not old_ok) or (new_ok and old_ok):
            merged[aid] = dict(item)
    # Mantem ordem: respostas novas primeiro, depois remanescentes.
    current_order = [int(i.get("agent_id", -1)) for i in current]
    previous_order = [int(i.get("agent_id", -1)) for i in previous]
    ordered_ids: List[int] = []
    for aid in current_order + previous_order:
        if aid >= 0 and aid in merged and aid not in ordered_ids:
            ordered_ids.append(aid)
    return [merged[aid] for aid in ordered_ids]


def _collect_fallback_candidates(state: OrchestratorState) -> List[int]:
    chosen = _sanitize_agent_ids(state.get("chosen_agent_ids", []))
    plan = _sanitize_agent_ids(state.get("execution_plan", chosen))
    alternatives = _sanitize_agent_ids(state.get("routing_alternatives", []))

    seed_specialists = [aid for aid in plan if aid not in _ROUTING_BASELINE_IDS]
    if not seed_specialists:
        seed_specialists = [aid for aid in chosen if aid not in _ROUTING_BASELINE_IDS]

    candidates: List[int] = []
    for aid in seed_specialists:
        for near in _NEAREST_SPECIALIST_MAP.get(aid, []):
            if near not in _ROUTING_BASELINE_IDS and near not in candidates:
                candidates.append(near)

    for aid in alternatives:
        if aid not in _ROUTING_BASELINE_IDS and aid not in candidates:
            candidates.append(aid)

    already_planned = set(plan)
    raw = [aid for aid in candidates if aid not in already_planned]
    return _sanitize_agent_ids(raw)


def _has_weak_or_conflicting_evidence(responses: List[Dict[str, Any]]) -> bool:
    successful = [
        r for r in responses
        if r.get("success") and str(r.get("response", "")).strip()
    ]
    if not successful:
        return True

    factual_count = 0
    uncertain_count = 0
    for item in successful:
        txt = str(item.get("response", ""))
        if _extract_factual_candidate(txt):
            factual_count += 1
        if _looks_uncertain(txt):
            uncertain_count += 1

    if factual_count == 0:
        return True
    if uncertain_count == len(successful):
        return True
    return False


def _has_specialist_factual_evidence(responses: List[Dict[str, Any]]) -> bool:
    for item in responses:
        aid = int(item.get("agent_id", -1))
        if aid in _ROUTING_BASELINE_IDS:
            continue
        if not item.get("success"):
            continue
        txt = str(item.get("response", "")).strip()
        if not txt:
            continue
        if _extract_factual_candidate(txt):
            return True
    return False


def _requires_specialist_primary_evidence(user_text: str) -> bool:
    text_norm = _normalize_text(_strip_profile_suffix(user_text))
    if not text_norm:
        return False
    if _is_legal_requirement_regulatory_signal(text_norm):
        return False
    return _contains_dairy_signal(text_norm)


def _get_specialist_primary_with_regulatory_context(
    user_text: str,
    responses: List[Dict[str, Any]],
    preferred_agent_id: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if _is_legal_requirement_regulatory_signal(_normalize_text(user_text)):
        return None, None

    specialist_candidates: List[Dict[str, Any]] = []
    regulatory_response: Optional[Dict[str, Any]] = None

    for item in responses:
        if not item.get("success") or not str(item.get("response", "")).strip():
            continue
        aid = int(item.get("agent_id", -1))
        factual = _extract_factual_candidate(str(item.get("response", "")))
        if aid == 3:
            if factual:
                regulatory_response = dict(item)
                regulatory_response["response"] = factual
            continue
        if aid in _ROUTING_BASELINE_IDS:
            continue
        if factual:
            row = dict(item)
            row["response"] = factual
            specialist_candidates.append(row)

    if not specialist_candidates:
        return None, regulatory_response

    primary = next(
        (item for item in specialist_candidates if int(item.get("agent_id", -1)) == preferred_agent_id),
        specialist_candidates[0],
    )
    return primary, regulatory_response


def _should_trigger_fallback(state: OrchestratorState) -> Tuple[bool, str]:
    attempts = int(state.get("fallback_attempts", 0) or 0)
    if attempts >= _FALLBACK_MAX_ATTEMPTS:
        return False, "max_attempts_reached"

    responses = state.get("agent_responses", []) or []
    candidates = _collect_fallback_candidates(state)
    if not candidates:
        return False, "no_candidates"

    bucket = str(state.get("routing_bucket", "medium"))
    if bucket == "low":
        return True, "low_confidence_bucket"

    user_text = _get_last_user_text(state.get("messages", []))

    if bucket == "medium":
        # Conservador por padrão, mas dispara se especialista não trouxe evidência factual.
        if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(responses):
            return True, "medium_no_specialist_factual_evidence"
        return False, "medium_conservative"

    if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(responses):
        return True, "no_specialist_factual_evidence"

    if _has_weak_or_conflicting_evidence(responses):
        return True, "weak_or_conflicting_evidence"

    return False, "sufficient_evidence"


def _collect_general_fallback_tables(state: "OrchestratorState") -> List[str]:
    """Seleciona tabelas para fallback geral priorizando contexto atual."""
    ordered_agent_ids: List[int] = []

    for aid in _sanitize_agent_ids(state.get("execution_plan", [])):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)
    for aid in _sanitize_agent_ids(state.get("chosen_agent_ids", [])):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)
    for aid in _sanitize_agent_ids(state.get("routing_alternatives", [])):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)

    # Completa com todos os agentes para realmente virar "base geral".
    for aid in sorted(_AGENT_TABLE_BY_ID):
        if aid not in ordered_agent_ids:
            ordered_agent_ids.append(aid)

    tables: List[str] = []
    for aid in ordered_agent_ids:
        table = _AGENT_TABLE_BY_ID.get(aid, "").strip()
        if table and table not in tables:
            tables.append(table)

    if not tables:
        tables = list(_ALL_AGENT_TABLES)

    cap = max(1, int(GENERAL_INDEX_FALLBACK_MAX_TABLES))
    return tables[:cap]


def _should_use_general_index_fallback(
    state: "OrchestratorState",
    successful_responses: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not ENABLE_GENERAL_INDEX_FALLBACK:
        return False, "general_index_disabled"

    # Evita uso em saudação/off-topic.
    if not state.get("chosen_agent_ids"):
        return False, "no_dairy_route"

    user_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
    text_norm = _normalize_text(user_text)
    if not text_norm:
        return False, "empty_query"

    if GENERAL_INDEX_FALLBACK_REQUIRE_DAIRY_SIGNAL and not _contains_dairy_signal(text_norm):
        return False, "no_dairy_signal"

    if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(successful_responses):
        return True, "no_specialist_factual_evidence"

    if not successful_responses:
        return True, "no_successful_specialist_response"

    if not GENERAL_INDEX_FALLBACK_ONLY_ON_WEAK:
        return True, "enabled_always"

    weak = _has_weak_or_conflicting_evidence(successful_responses)
    if weak:
        return True, "weak_or_conflicting_specialist_evidence"
    return False, "specialist_evidence_sufficient"


def _render_general_fallback_evidence(results: List[Dict[str, Any]], top_n: int = 4) -> str:
    snippets: List[str] = []
    for item in (results or [])[: max(1, top_n)]:
        metadata = item.get("metadata") or {}
        source_table = str(metadata.get("source_table", "")).strip() or "base_geral_unificada"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        content = re.sub(r"\s+", " ", content)
        if len(content) > 520:
            content = content[:520].rstrip() + "..."
        snippets.append(f"[{source_table}] {content}")
    return "\n".join(snippets).strip()


async def _fetch_general_index_fallback_evidence(
    state: "OrchestratorState",
) -> Tuple[str, str]:
    """Busca evidências em índice geral unificado.

    Retorna (evidence_text, reason). evidence_text vazio indica sem evidência útil.
    """
    user_text = _build_contextual_search_query(_get_last_user_text(state.get("messages", [])))
    tables = _collect_general_fallback_tables(state)
    if not user_text or not tables:
        return "", "general_index_no_query_or_tables"

    try:
        rows = await asyncio.to_thread(
            search_general_knowledge_base,
            user_text,
            tables,
            GENERAL_INDEX_FALLBACK_SEARCH_TYPE,
            max(1, int(GENERAL_INDEX_FALLBACK_PER_TABLE_K)),
            max(1, int(GENERAL_INDEX_FALLBACK_FINAL_K)),
            MATCH_THRESHOLD,
        )
    except Exception:
        return "", "general_index_search_error"

    min_results = max(1, int(GENERAL_INDEX_FALLBACK_MIN_RESULTS))
    if len(rows) < min_results:
        return "", "general_index_insufficient_results"

    evidence = _render_general_fallback_evidence(rows, top_n=min(6, len(rows)))
    if not evidence:
        return "", "general_index_empty_evidence"
    return evidence, "general_index_evidence_collected"


def _append_reason(reason: str, marker: str) -> str:
    base = (reason or "").strip()
    mark = (marker or "").strip()
    if not mark:
        return base
    return f"{base} | {mark}" if base else mark


def _should_use_web_fallback(
    state: "OrchestratorState",
    successful_responses: List[Dict[str, Any]],
    general_attempted: bool,
) -> Tuple[bool, str]:
    if not ENABLE_WEB_FALLBACK:
        return False, "web_fallback_disabled"

    if WEB_FALLBACK_PROVIDER != "duckduckgo":
        return False, "web_provider_not_supported"

    user_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
    text_norm = _normalize_text(user_text)
    if not text_norm:
        return False, "web_empty_query"

    if WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL and not _contains_dairy_signal(text_norm):
        return False, "web_no_dairy_signal"

    # Se o índice geral está desabilitado, o "require general first" não faz sentido
    # pois general_used nunca será True — nesse caso, ignora a restrição.
    if WEB_FALLBACK_REQUIRE_GENERAL_FALLBACK_FIRST and ENABLE_GENERAL_INDEX_FALLBACK and not general_attempted:
        return False, "web_requires_general_fallback_first"

    if _requires_specialist_primary_evidence(user_text) and not _has_specialist_factual_evidence(successful_responses):
        return True, "web_no_specialist_factual_evidence"

    if not successful_responses:
        return True, "web_no_specialist_evidence"

    if WEB_FALLBACK_ONLY_ON_WEAK and not _has_weak_or_conflicting_evidence(successful_responses):
        return False, "web_specialist_evidence_sufficient"

    return True, "web_weak_or_conflicting_evidence"


async def _fetch_web_fallback_evidence(
    state: "OrchestratorState",
) -> Tuple[str, List[Dict[str, str]], str]:
    user_text = _build_contextual_search_query(_get_last_user_text(state.get("messages", [])))
    if not user_text:
        return "", [], "web_no_query"

    try:
        rows = await asyncio.to_thread(
            search_web_duckduckgo,
            user_text,
            WEB_FALLBACK_ALLOWED_DOMAINS,
            max(1, int(WEB_FALLBACK_MAX_RESULTS)),
            float(WEB_FALLBACK_TIMEOUT_SEC),
            max(120, int(WEB_FALLBACK_MAX_SNIPPET_CHARS)),
        )
    except Exception:
        return "", [], "web_search_error"

    if not rows:
        return "", [], "web_no_whitelisted_results"

    if WEB_FALLBACK_FETCH_FULLTEXT:
        try:
            rows = await asyncio.to_thread(
                enrich_results_with_page_content,
                rows,
                float(WEB_FALLBACK_TIMEOUT_SEC),
                max(600, int(WEB_FALLBACK_MAX_PAGE_CHARS)),
            )
        except Exception:
            # Mantém snippets mesmo se enriquecimento falhar.
            pass

    evidence_text, sources = build_web_fallback_evidence(
        rows,
        max_sources=max(1, int(WEB_FALLBACK_MAX_SOURCES)),
    )
    if not evidence_text or not sources:
        return "", [], "web_insufficient_evidence"
    return evidence_text, sources, "web_evidence_collected"


def _render_web_sources_block(sources: List[Dict[str, str]]) -> str:
    domains: List[str] = []
    for item in sources or []:
        domain = str(item.get("domain", "")).strip()
        if domain and domain not in domains:
            domains.append(domain)
    if not domains:
        return ""
    return "_Fonte: " + ", ".join(domains) + "_"


def _looks_like_unusable_consolidation_answer(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    markers = (
        "nao foi possivel consolidar",
        "nao foi possivel obter uma resposta",
        "nao foi possivel responder",
        "resposta confiavel no momento",
        "tente reformular sua pergunta",
    )
    return any(marker in normalized for marker in markers)


def _build_evidence_grounded_fallback_answer(
    user_text: str,
    specialist_responses: List[Dict[str, Any]],
    general_evidence_text: str,
    web_evidence_text: str,
    web_sources: List[Dict[str, str]],
) -> str:
    factual_blocks: List[str] = []
    for item in specialist_responses:
        factual = _extract_factual_candidate(str(item.get("response", "")))
        if factual:
            factual_blocks.append(factual)

    if factual_blocks:
        joined = "\n\n".join(factual_blocks[:2]).strip()
        return _postprocess_consolidated_answer(user_text, joined)

    bullets: List[str] = []
    for raw in (general_evidence_text or "").splitlines():
        line = re.sub(r"^\[[^\]]+\]\s*", "", raw.strip())
        if len(line) < 40:
            continue
        bullets.append(line)
        if len(bullets) >= 3:
            break

    if not bullets:
        for raw in (web_evidence_text or "").splitlines():
            line = re.sub(r"^\[Fonte \d+\]\s*", "", raw.strip())
            if len(line) < 40:
                continue
            bullets.append(line)
            if len(bullets) >= 3:
                break

    if bullets:
        intro = (
            "Encontrei evidências relacionadas ao tema, mas a consolidação automática falhou. "
            "O que a base trouxe de mais útil foi:"
        )
        answer = intro + "\n\n- " + "\n- ".join(bullets)
        sources_block = _render_web_sources_block(web_sources) if web_sources else ""
        if sources_block:
            answer = (answer.strip() + "\n\n" + sources_block).strip()
        return _postprocess_consolidated_answer(user_text, answer)

    return _postprocess_consolidated_answer(
        user_text,
        "Não foi possível consolidar a resposta automaticamente, mas houve falha técnica durante a etapa final. "
        "Tente reformular a pergunta ou consultar o agente especialista direto.",
    )


async def _build_last_resort_response(
    state: OrchestratorState,
    user_text: str,
    web_should_use: bool,
    web_used: bool,
    web_evidence_text: str,
    web_sources: List[Dict[str, str]],
) -> OrchestratorState:
    """Último recurso quando KB e fallbacks normais não trouxeram conteúdo.

    Garante que web é tentada antes de desistir: se web_should_use=False mas
    ENABLE_WEB_FALLBACK=True, força a busca web aqui como última camada.
    web_used=True significa que web já foi tentada — não re-tenta.
    """
    forced_text = web_evidence_text
    forced_sources = web_sources

    # web_should_use=False: condições normais não acionaram web.
    # Força aqui como último recurso se o provider estiver habilitado.
    if not web_should_use and not web_used and ENABLE_WEB_FALLBACK:
        try:
            forced_text, forced_sources, _ = await _fetch_web_fallback_evidence(state)
        except Exception:
            forced_text, forced_sources = "", []

    if forced_text and forced_sources:
        prompt = (
            "Você é o assistente geral do DairyApp AI. "
            "Responda à pergunta do usuário com base SOMENTE nas evidências abaixo. "
            "Se a evidência não cobre o ponto exato, informe o que foi encontrado de forma objetiva. "
            "Não mencione agentes, bases de dados ou ferramentas internas.\n\n"
            f"PERGUNTA: {user_text}\n\n"
            f"EVIDÊNCIAS:\n{forced_text}\n\n"
            "Resposta:"
        )
        try:
            response_text = await _ainvoke_consolidation_with_timeout(state, prompt)
            final_text = _postprocess_consolidated_answer(user_text, response_text)
            if _looks_like_unusable_consolidation_answer(final_text):
                final_text = _build_evidence_grounded_fallback_answer(
                    user_text, [], "", forced_text, forced_sources
                )
        except Exception:
            final_text = _build_evidence_grounded_fallback_answer(
                user_text, [], "", forced_text, forced_sources
            )
        sources_block = _render_web_sources_block(forced_sources)
        if sources_block:
            final_text = (final_text.strip() + "\n\n" + sources_block).strip()
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
            "web_fallback_used": True,
            "web_fallback_sources": forced_sources,
            "fallback_used": True,
            "fallback_trigger": "web_last_resort",
            "routing_reason": _append_reason(
                str(state.get("routing_reason", "")), "web_last_resort"
            ),
        }

    # Web também falhou ou está desabilitada — mensagem informativa de último recurso.
    final_text = (
        "Não encontrei informações suficientes nas fontes internas sobre esse tema. "
        "Tente reformular com mais detalhes técnicos — por exemplo, especificando o produto, "
        "o processo ou o parâmetro que deseja entender."
    )
    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


async def _ainvoke_consolidation_with_timeout(
    state: "OrchestratorState",
    prompt: str,
) -> str:
    response = await asyncio.wait_for(
        _get_consolidation_model(_resolve_state_model(state)).ainvoke(
            [HumanMessage(content=prompt)]
        ),
        timeout=float(CONSOLIDATION_TIMEOUT_SEC),
    )
    return str(response.content or "")



# ============================================================
# Nó CLASSIFY
# ============================================================

def _get_last_user_text(messages: List[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _build_classification_state(
    route_text: str,
    agent_ids: List[int],
    confidence: float = 0.50,
    reason: str = "",
    alternatives: Optional[List[int]] = None,
) -> OrchestratorState:
    confidence = _recalibrate_confidence(route_text, agent_ids, confidence)
    bucket = _confidence_to_bucket(confidence)
    alternatives_ids = _sanitize_agent_ids(alternatives or [])

    if not agent_ids:
        return {
            "chosen_agent_ids": [],
            "chosen_agent_names": [],
            "execution_plan": [],
            "primary_agent_id": 0,
            "primary_agent_name": "Assistente Geral",
            "agent_responses": [],
            "final_response": "",
            "routing_confidence": confidence,
            "routing_bucket": bucket,
            "routing_reason": reason or "sem_agentes",
            "routing_alternatives": alternatives_ids,
            "fallback_used": False,
            "fallback_attempts": 0,
            "fallback_trigger": "",
            "previous_agent_responses": [],
            "general_index_fallback_used": False,
            "web_fallback_used": False,
            "web_fallback_sources": [],
        }

    sanitized_ids = _sanitize_agent_ids(agent_ids)
    execution_plan = _build_execution_plan(
        route_text=route_text,
        chosen_ids=sanitized_ids,
        alternatives=alternatives_ids,
        bucket=bucket,
    )

    agent_names = []
    for aid in sanitized_ids:
        cfg = get_agent_by_id(aid)
        agent_names.append(cfg["name"] if cfg else f"Agente {aid}")

    primary_agent_id = _choose_primary_agent_id(execution_plan or sanitized_ids, route_text=route_text)
    primary_cfg = get_agent_by_id(primary_agent_id)
    primary_agent_name = (
        primary_cfg["name"]
        if primary_cfg
        else next(
            (name for aid, name in zip(sanitized_ids, agent_names) if aid == primary_agent_id),
            "Assistente Geral",
        )
    )

    return {
        "chosen_agent_ids": sanitized_ids,
        "chosen_agent_names": agent_names,
        "execution_plan": execution_plan,
        "primary_agent_id": primary_agent_id,
        "primary_agent_name": primary_agent_name,
        "agent_responses": [],
        "final_response": "",
        "routing_confidence": confidence,
        "routing_bucket": bucket,
        "routing_reason": reason or "classificacao_llm",
        "routing_alternatives": alternatives_ids,
        "fallback_used": False,
        "fallback_attempts": 0,
        "fallback_trigger": "",
        "previous_agent_responses": [],
        "general_index_fallback_used": False,
        "web_fallback_used": False,
        "web_fallback_sources": [],
    }


async def classify(state: OrchestratorState) -> OrchestratorState:
    """Identifica quais agentes devem ser consultados.

    Agent 3 (Regulatórios) é incluído em toda pergunta de laticínios.
    Agent 1 (Queijos) é incluído quando há sinal de tecnologia de queijo.
    """
    messages = state.get("messages", [])
    user_text = _get_last_user_text(messages)

    if not user_text:
        return _build_classification_state(route_text="", agent_ids=[], confidence=1.0, reason="mensagem_vazia")

    route_text = _strip_profile_suffix(user_text)
    current_question_norm = _normalize_text(_extract_current_user_segment(route_text))
    if _is_conversation_recap_request(current_question_norm):
        return _build_classification_state(
            route_text=route_text,
            agent_ids=[],
            confidence=0.98 if _has_recent_context_block(route_text) else 0.80,
            reason="conversation_recap",
        )

    cache_key = _normalize_text(route_text)

    cached_ids = _cache_get(cache_key)
    if cached_ids is not None:
        return _build_classification_state(
            route_text=route_text,
            agent_ids=cached_ids,
            confidence=0.95,
            reason="cache_hit",
        )

    if ORCHESTRATOR_FASTPATH:
        fast_ids = _rule_based_route(route_text)
        if fast_ids is not None:
            if fast_ids:
                _cache_set(cache_key, fast_ids)
            return _build_classification_state(
                route_text=route_text,
                agent_ids=fast_ids,
                confidence=_estimate_fastpath_confidence(route_text, fast_ids),
                reason="fastpath_rule_based",
            )

    system_prompt = get_orchestrator_prompt()

    classification_instruction = f"""

Com base na pergunta do usuário, identifique quais agentes devem ser consultados.

REGRAS DE INCLUSÃO:
- Agente 3 (Regulatórios): incluir em TODA pergunta de laticínios.
- Agente 0 (Base Geral): incluir SOMENTE se a pergunta envolver glossário,
  padronização de termos ("qual termo usar", "como chamar"), marcas/fabricantes/
  distribuidores/equipamentos específicos, ou saudação/off-topic. NÃO incluir
  agente 0 em perguntas puramente técnicas, analíticas, de processo ou regulatórias
  — a base do agente 0 não cobre esses temas e só adiciona ruído.
- Especialistas 1-4: adicionar apenas se a pergunta for claramente desse domínio.

ESPECIALISTAS DISPONÍVEIS:
{_SPECIALISTS_DESC}
FORMATO DA RESPOSTA:
- Saudação / off-topic → []
- Pergunta de glossário/terminologia → [0, 3]
- Pergunta regulatória/técnica → [3] ou [3, X]
- Pergunta de glossário + especialidade → [0, 3, X]
- Máx 5 IDs. Ordene por relevância: agente mais relevante primeiro.

ALÉM DOS IDs, informe:
- confidence: número entre 0.0 e 1.0
- reason: justificativa curta
- alternatives: IDs alternativos relevantes (sem repetir os principais)

REGRAS DE DESEMPATE (OBRIGATÓRIAS):
- Se a pergunta for de glossário, padronização de termos, "qual termo usar" ou "significado esperado",
  priorize [0,3] e NÃO escolha especialista como primário.
- Se a pergunta envolver rotulagem/denominação/embalagem de produto lácteo,
  priorize [0,3] (regulatório), mesmo que cite nome de queijo.
- Se a pergunta mencionar "norma", "regulamento", "IN", "RDC", "decreto" ou "artigo",
  priorize [0,3] e evite priorizar formulação (6) como agente principal.
- Se a pergunta pedir requisito mínimo/obrigatório/exigido ("período mínimo exigido",
  "mínimo legal", "deve sofrer maturação"), trate como regulatória e priorize [0,3].
- Se a pergunta for de método analítico/laboratorial (Dornic, titulação, HCl, NaOH, IN 68,
  absorbância, comprimento de onda, centrifugação, m/m, m/v etc.),
  inclua 4 e priorize 4 como especialista — IN 68 é documento de métodos do Agente 4, não regulatório.
- Se a pergunta citar fermentação em queijo/coalhada de processo (corte de coalhada, pH de corte),
  priorize 1 (queijos) e não 2.
- Se a pergunta for de padronização de termo/glossário ("qual termo usar", "significado esperado"),
  priorize [0,3] e evite especialistas como primários.
- Evite super-especializar perguntas institucionais ou terminológicas.

 {_CLASSIFIER_FEW_SHOTS}
 """

    classifier = _get_classifier(_resolve_state_model(state))
    result = await classifier.ainvoke([
        SystemMessage(content=system_prompt + classification_instruction),
        HumanMessage(content=user_text),
    ])

    # Valida IDs (0-6), preserva ordem, remove duplicatas
    agent_ids = _sanitize_agent_ids(result.agent_ids)
    alternatives = _sanitize_agent_ids(getattr(result, "alternatives", []) or [])
    confidence = _clamp_confidence(getattr(result, "confidence", 0.50))
    reason = (getattr(result, "reason", "") or getattr(result, "reasoning", "") or "").strip()

    # Hard constraints para sinais claros de dominio lacteo.
    agent_ids = _apply_dairy_hard_constraints(route_text, agent_ids)
    alternatives = [aid for aid in alternatives if aid not in agent_ids]
    agent_ids, alternatives = _apply_domain_guardrails(route_text, agent_ids, alternatives)

    # Agente 0 sem KB — _sanitize_agent_ids em _apply_dairy_hard_constraints
    # já o remove automaticamente via _AGENTS_WITHOUT_KB.
    if not agent_ids:
        return _build_classification_state(
            route_text=route_text,
            agent_ids=[],
            confidence=confidence,
            reason=reason or "sem_dominio_relevante",
            alternatives=alternatives,
        )

    _cache_set(cache_key, agent_ids)
    classification = _build_classification_state(
        route_text=route_text,
        agent_ids=agent_ids,
        confidence=confidence,
        reason=reason or "classificacao_llm",
        alternatives=alternatives,
    )

    if _should_ask_clarification(
        chosen_ids=classification.get("chosen_agent_ids", []),
        bucket=classification.get("routing_bucket", "medium"),
        confidence=classification.get("routing_confidence", 0.5),
        reason=classification.get("routing_reason", ""),
        messages=messages,
    ):
        classification["needs_clarification"] = True

    return classification

# ============================================================
# Roteamento condicional
# ============================================================

def route(state: OrchestratorState) -> str:
    if state.get("needs_clarification"):
        return "ask_clarification"
    planned = state.get("execution_plan")
    if planned is not None:
        return "respond_direct" if not planned else "execute"
    return "respond_direct" if not state.get("chosen_agent_ids") else "execute"


def route_after_execute(state: OrchestratorState) -> str:
    should_fallback, _ = _should_trigger_fallback(state)
    return "fallback_reclassify" if should_fallback else "consolidate"


def route_after_fallback(state: OrchestratorState) -> str:
    trigger = str(state.get("fallback_trigger", ""))
    if trigger == "fallback_no_plan_change":
        return "consolidate"
    return "execute"


# ============================================================
# NÃ³ EXECUTE â€" execuÃ§Ã£o paralela
# ============================================================

async def execute(state: OrchestratorState) -> OrchestratorState:
    """Invoca todos os agentes em PARALELO via asyncio.gather.

    LatÃªncia total â‰ˆ tempo do agente mais lento (nÃ£o a soma).
    Cada agente tem timeout individual de AGENT_TIMEOUT segundos.
    """
    agent_ids = state.get("execution_plan") or state.get("chosen_agent_ids", [])
    agent_names = [
        (get_agent_by_id(aid) or {}).get("name", f"Agente {aid}")
        for aid in agent_ids
    ]

    user_text = _get_last_user_text(state.get("messages", []))
    search_query = _build_contextual_search_query(user_text)

    if not user_text or not search_query:
        return {"agent_responses": []}

    # Computa o embedding da query UMA vez para todos os agentes paralelos.
    # Sem isso, cada agente chamaria embed_query() independentemente para a mesma string
    # — (N-1) chamadas duplicadas à OpenAI (~150ms cada desperdiçadas).
    shared_embedding: Optional[List[float]] = None
    try:
        shared_embedding = await asyncio.to_thread(
            embed_query, search_query
        )
    except Exception:
        pass  # Fallback: cada agente computa seu próprio embedding normalmente.

    async def call_one(agent_id: int, agent_name: str) -> Dict[str, Any]:
        try:
            graph = get_agent_graph(agent_id, _resolve_state_model(state))
            result = await asyncio.wait_for(
                graph.ainvoke({
                    "messages": [HumanMessage(content=user_text)],
                    "llm_model": _resolve_state_model(state),
                    "precomputed_embedding": shared_embedding,
                }),
                timeout=AGENT_TIMEOUT,
            )
            agent_msgs = result.get("messages", [])
            agent_text = ""
            for msg in reversed(agent_msgs):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if isinstance(content, list):
                        agent_text = "\n".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    elif isinstance(content, str):
                        agent_text = content
                    if agent_text:
                        break
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": agent_text,
                "success": bool(agent_text),
            }
        except asyncio.TimeoutError:
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": f"{agent_name}: timeout ao consultar base de conhecimento.",
                "success": False,
            }
        except Exception as e:
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": f"Erro ao consultar {agent_name}: {e}",
                "success": False,
            }

    # Dispara todos os agentes ao mesmo tempo
    current_responses = await asyncio.gather(
        *[call_one(aid, name) for aid, name in zip(agent_ids, agent_names)]
    )

    previous_responses = state.get("previous_agent_responses", []) or []
    responses = _merge_agent_responses(previous_responses, list(current_responses))

    successful_ids = [r["agent_id"] for r in responses if r.get("success")]
    candidate_ids = successful_ids or agent_ids
    primary_agent_id = _choose_primary_agent_id(candidate_ids, route_text=user_text)
    primary_cfg = get_agent_by_id(primary_agent_id)
    primary_agent_name = (
        primary_cfg["name"] if primary_cfg else "Assistente Geral"
    )

    return {
        "agent_responses": list(responses),
        "primary_agent_id": primary_agent_id,
        "primary_agent_name": primary_agent_name,
        "previous_agent_responses": [],
    }


# ============================================================
# NÃ³ FALLBACK_RECLASSIFY â€" segunda passada inteligente
# ============================================================

async def fallback_reclassify(state: OrchestratorState) -> OrchestratorState:
    route_text = _strip_profile_suffix(_get_last_user_text(state.get("messages", [])))
    current_chosen = _sanitize_agent_ids(state.get("chosen_agent_ids", []))
    current_plan = _sanitize_agent_ids(state.get("execution_plan", current_chosen))
    current_bucket = str(state.get("routing_bucket", "medium"))
    current_conf = _clamp_confidence(state.get("routing_confidence", 0.50))
    current_alts = _sanitize_agent_ids(state.get("routing_alternatives", []))
    current_reason = str(state.get("routing_reason", ""))
    attempts = int(state.get("fallback_attempts", 0) or 0)

    should_fallback, trigger = _should_trigger_fallback(state)
    if not should_fallback:
        return {"fallback_trigger": trigger}

    fallback_candidates = _collect_fallback_candidates(state)
    if not fallback_candidates:
        return {"fallback_trigger": "no_candidates"}

    extra_cap = _FALLBACK_EXTRA_SPECIALISTS.get(current_bucket, 2)
    selected_extra = fallback_candidates[:extra_cap]

    # Expande escolhidos e recalcula plano mantendo bucket original.
    new_chosen = _sanitize_agent_ids(current_chosen + selected_extra)
    new_alts = _sanitize_agent_ids(current_alts + fallback_candidates[extra_cap:])
    new_conf = max(current_conf, 0.65)
    new_reason = (
        f"{current_reason} | fallback_second_pass:{trigger}"
        if current_reason
        else f"fallback_second_pass:{trigger}"
    )

    rebuilt = _build_classification_state(
        route_text=route_text,
        agent_ids=new_chosen,
        confidence=new_conf,
        reason=new_reason,
        alternatives=new_alts,
    )
    # Mantem respostas da primeira passada para mescla posterior.
    rebuilt["previous_agent_responses"] = list(state.get("agent_responses", []) or [])
    rebuilt["fallback_used"] = True
    rebuilt["fallback_attempts"] = attempts + 1
    rebuilt["fallback_trigger"] = trigger
    # Garante que o novo plano realmente evoluiu.
    if rebuilt.get("execution_plan") == current_plan:
        rebuilt["fallback_trigger"] = "fallback_no_plan_change"
        return {"fallback_trigger": "fallback_no_plan_change"}
    return rebuilt


# ============================================================
# Clarificação — detecção e nó
# ============================================================


def _should_ask_clarification(
    chosen_ids: List[int],
    bucket: str,
    confidence: float,
    reason: str,
    messages: List[AnyMessage],
) -> bool:
    """Retorna True quando a pergunta é genuinamente ambígua e merece clarificação.

    Critérios conservadores — só dispara quando bucket=low E a query
    é curta/vaga OU há múltiplos especialistas com confiança muito baixa.
    Fast-path e cache hits são sempre decisivos e nunca chegam aqui.
    """
    if bucket != "low":
        return False

    specialists = [aid for aid in chosen_ids if aid not in _ROUTING_BASELINE_IDS]
    if not specialists:
        return False

    # Fast-path e cache já resolveram — não perguntar
    if "fastpath" in reason or "cache_hit" in reason:
        return False

    # Anti-loop: se o assistente já fez uma pergunta recentemente, não repetir
    recent_ai = [m for m in messages[-6:] if isinstance(m, AIMessage)]
    for msg in recent_ai:
        txt = (msg.content or "") if isinstance(msg.content, str) else ""
        if "?" in txt:
            return False

    user_text = _get_last_user_text(messages)
    current = _extract_current_user_segment(_strip_profile_suffix(user_text))
    current_norm = _normalize_text(current)
    words = current_norm.split()

    # Sinal de laticínio obrigatório — não perguntar em off-topic
    if not _contains_dairy_signal(current_norm):
        return False

    # Query muito curta e vaga
    if len(words) < 5:
        return True

    # Múltiplos especialistas com confiança muito baixa
    if len(specialists) >= 2 and confidence < 0.50:
        return True

    return False


async def ask_clarification(state: OrchestratorState) -> OrchestratorState:
    """Gera uma pergunta de clarificação direcionada ao usuário.

    Chamado quando o orquestrador tem baixa confiança e a pergunta é ambígua
    entre especialistas. A resposta do usuário na próxima mensagem irá
    resolver a ambiguidade naturalmente pelo contexto da conversa.
    """
    user_text = _get_last_user_text(state.get("messages", []))
    current = _extract_current_user_segment(_strip_profile_suffix(user_text))
    chosen = state.get("chosen_agent_ids", [])
    specialists = [aid for aid in chosen if aid not in _ROUTING_BASELINE_IDS]

    domain_options = [
        _AGENT_DOMAIN_LABELS[aid]
        for aid in specialists
        if aid in _AGENT_DOMAIN_LABELS
    ]

    if domain_options:
        options_str = " ou ".join(domain_options)
        system = (
            "Você é o assistente do DairyApp AI, especializado em tecnologia de laticínios. "
            "O usuário fez uma pergunta um pouco ampla ou ambígua. "
            "Faça UMA pergunta curta e direta para entender melhor o que ele precisa, "
            f"considerando que pode ser sobre: {options_str}. "
            "Seja cordial, não explique os agentes internamente, apenas pergunte o que "
            "o usuário quer saber. Responda em português brasileiro. Máximo 2 frases."
        )
    else:
        system = (
            "Você é o assistente do DairyApp AI, especializado em tecnologia de laticínios. "
            "O usuário fez uma pergunta um pouco ampla. Peça para ele detalhar melhor "
            "o que precisa saber, com uma pergunta curta e cordial. "
            "Responda em português brasileiro. Máximo 2 frases."
        )

    response = await _get_direct_model(_resolve_state_model(state)).ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=current or user_text),
    ])

    question = _sanitize_math_for_ui(response.content or "")
    if not question:
        question = "Poderia detalhar um pouco mais sua dúvida? Assim consigo te direcionar melhor."

    return {
        "final_response": question,
        "messages": [AIMessage(content=question)],
        "agent_responses": [],
        "needs_clarification": True,
    }


# ============================================================
# NÃ³ RESPOND_DIRECT â€" saudaÃ§Ãµes e off-topic
# ============================================================

async def respond_direct(state: OrchestratorState) -> OrchestratorState:
    """Resposta direta para saudações e mensagens off-topic (sem RAG)."""
    user_text = _get_last_user_text(state.get("messages", []))
    current_text = _extract_current_user_segment(user_text)
    current_norm = _normalize_text(current_text)

    if _is_conversation_recap_request(current_norm):
        if _has_recent_context_block(user_text):
            system = (
                "Voce e o assistente geral do Dairy AI (DairyApp). "
                "O usuario pediu um resumo do que acabou de ser discutido e voce recebeu "
                "um bloco [Contexto recente da conversa]. Resuma APENAS o que esta nesse "
                "contexto. Priorize sintese executiva em 3 a 5 bullets curtos ou um "
                "paragrafo objetivo. Destaque fatos tecnicos, conclusoes e pendencias. "
                "Nao consulte RAG, nao invente fatos e nao reabra a classificacao por dominio. "
                "Se houver contradicao no proprio contexto, aponte isso explicitamente."
            )
        else:
            system = (
                "Voce e o assistente geral do Dairy AI (DairyApp). "
                "O usuario pediu um resumo da conversa, mas nenhum contexto recente foi fornecido. "
                "Explique isso de forma curta e educada e convide o usuario a retomar o tema "
                "com uma nova pergunta objetiva."
            )
    else:
        system = (
            "Voce e o assistente geral do Dairy AI (DairyApp), especializado em tecnologia "
            "de laticinios. Em saudacoes e primeira interacao, apresente-se de forma curta "
            "como Dairy AI e diga em uma frase como pode ajudar. Depois disso, evite repetir "
            "apresentacoes e va direto ao ponto. Quando pertinente, sugira perguntas tecnicas "
            "sobre queijos, fermentados, regulatorios, qualidade do leite, diagnostico de "
            "defeitos ou formulacao. Responda em portugues brasileiro."
        )

    response = await _get_direct_model(_resolve_state_model(state)).ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_text),
    ])

    final_text = _sanitize_math_for_ui(response.content or "")
    return {
        "agent_responses": [],
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


# ============================================================
# NÃ³ CONSOLIDATE â€" fusÃ£o das respostas
# ============================================================

async def consolidate(state: OrchestratorState) -> OrchestratorState:
    """Funde as respostas dos agentes em uma resposta coerente.

    1 agente bem-sucedido â†’ repassa direto (sem chamada LLM extra).
    2+ agentes â†’ LLM funde preservando todos os dados tÃ©cnicos.
    """
    # Veio de respond_direct: já tem final_response
    if not state.get("chosen_agent_ids") and state.get("final_response"):
        final_text = state.get("final_response") or ""
        user_text = _get_last_user_text(state.get("messages", []))
        final_text = _postprocess_consolidated_answer(user_text, final_text)
        if final_text:
            msgs = state.get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage) and (msgs[-1].content or "") == final_text:
                return {}
            return {"messages": [AIMessage(content=final_text)]}
        return {}

    successful = [
        r for r in state.get("agent_responses", [])
        if r.get("success") and r.get("response")
    ]
    user_text = _get_last_user_text(state.get("messages", []))

    # Fallback final em base geral unificada (controlado por flag).
    # Só entra quando a evidência dos especialistas está ausente/fraca.
    general_should_use, general_trigger = _should_use_general_index_fallback(state, successful)
    general_evidence_text = ""
    general_used = False
    if general_should_use:
        general_evidence_text, evidence_status = await _fetch_general_index_fallback_evidence(state)
        if general_evidence_text:
            general_used = True
            general_trigger = _append_reason(general_trigger, evidence_status)

    # Última camada: web fallback com whitelist de domínios confiáveis.
    web_should_use, web_trigger = _should_use_web_fallback(state, successful, general_should_use)
    web_evidence_text = ""
    web_sources: List[Dict[str, str]] = []
    web_used = False
    if web_should_use:
        web_evidence_text, web_sources, web_status = await _fetch_web_fallback_evidence(state)
        if web_evidence_text and web_sources:
            web_used = True
            web_trigger = _append_reason(web_trigger, web_status)

    if not successful:
        if general_used or web_used:
            evidence_blocks = []
            if general_used and general_evidence_text:
                evidence_blocks.append(f"BASE GERAL UNIFICADA:\n{general_evidence_text}")
            if web_used and web_evidence_text:
                evidence_blocks.append(f"WEB (DOMÍNIOS CONFIÁVEIS):\n{web_evidence_text}")
            evidence_blob = "\n\n".join(evidence_blocks).strip()
            prompt = (
                "Você é o assistente geral do DairyApp AI. "
                "Responda à pergunta do usuário com base SOMENTE nas evidências abaixo. "
                "Se a evidência não for suficiente para afirmar algo com segurança, diga isso explicitamente.\n\n"
                f"PERGUNTA: {user_text}\n\n"
                f"EVIDÊNCIAS:\n{evidence_blob}\n\n"
                "Resposta:"
            )
            try:
                response_text = await _ainvoke_consolidation_with_timeout(state, prompt)
                final_text = _postprocess_consolidated_answer(user_text, response_text)
                if _looks_like_unusable_consolidation_answer(final_text):
                    final_text = _build_evidence_grounded_fallback_answer(
                        user_text,
                        successful,
                        general_evidence_text,
                        web_evidence_text,
                        web_sources,
                    )
            except Exception:
                final_text = _build_evidence_grounded_fallback_answer(
                    user_text,
                    successful,
                    general_evidence_text,
                    web_evidence_text,
                    web_sources,
                )
            sources_block = _render_web_sources_block(web_sources) if web_used else ""
            if sources_block:
                final_text = (final_text.strip() + "\n\n" + sources_block).strip()
            fallback_marker = []
            if general_used:
                fallback_marker.append(f"general_index_fallback:{general_trigger}")
            if web_used:
                fallback_marker.append(f"web_fallback:{web_trigger}")
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
                "general_index_fallback_used": general_used,
                "web_fallback_used": web_used,
                "web_fallback_sources": web_sources if web_used else [],
                "fallback_used": True,
                "fallback_trigger": " | ".join(fallback_marker),
                "routing_reason": _append_reason(
                    str(state.get("routing_reason", "")),
                    "web_fallback_used" if web_used else "general_index_fallback_used",
                ),
            }

        return await _build_last_resort_response(
            state, user_text, web_should_use, web_used, web_evidence_text, web_sources
        )

    # 1 agente: repassa direto (econômico), exceto se fallback geral ou web foi acionado.
    # Só faz fast-path se a resposta tem conteúdo factual — caso contrário cai no _all_uncertain
    # abaixo, evitando propagar "não tenho informação" diretamente ao usuário.
    if len(successful) == 1 and not general_used and not web_used:
        single_fact = _extract_factual_candidate(str(successful[0]["response"]))
        if single_fact:
            final_text = _postprocess_consolidated_answer(user_text, single_fact)
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
            }

    # Todos os agentes retornaram apenas incerteza (sem conteúdo factual).
    # Neste caso, não misturamos as respostas "sem info" com a evidência de fallback
    # pois isso produziria respostas duplicadas e confusas.
    _all_uncertain = successful and not any(
        _extract_factual_candidate(str(r.get("response", ""))) for r in successful
    )
    if _all_uncertain:
        if general_used or web_used:
            # Consolida usando SOMENTE a evidência de fallback — ignora respostas "sem info".
            evidence_blocks = []
            if general_used and general_evidence_text:
                evidence_blocks.append(f"BASE GERAL UNIFICADA:\n{general_evidence_text}")
            if web_used and web_evidence_text:
                evidence_blocks.append(f"WEB (DOMÍNIOS CONFIÁVEIS):\n{web_evidence_text}")
            evidence_blob = "\n\n".join(evidence_blocks).strip()
            fallback_only_prompt = (
                "Você é o assistente geral do DairyApp AI. "
                "Responda à pergunta do usuário com base SOMENTE nas evidências abaixo. "
                "Se a evidência não cobre o dado exato, informe de forma objetiva o que foi encontrado e o que não estava disponível. "
                "Não mencione agentes internos, bases de conhecimento ou ferramentas internas.\n\n"
                f"PERGUNTA: {user_text}\n\n"
                f"EVIDÊNCIAS:\n{evidence_blob}\n\n"
                "Resposta:"
            )
            try:
                response_text = await _ainvoke_consolidation_with_timeout(state, fallback_only_prompt)
                final_text = _postprocess_consolidated_answer(user_text, response_text)
                if _looks_like_unusable_consolidation_answer(final_text):
                    final_text = _build_evidence_grounded_fallback_answer(
                        user_text, [], general_evidence_text, web_evidence_text, web_sources
                    )
            except Exception:
                final_text = _build_evidence_grounded_fallback_answer(
                    user_text, [], general_evidence_text, web_evidence_text, web_sources
                )
            sources_block = _render_web_sources_block(web_sources) if web_used else ""
            if sources_block:
                final_text = (final_text.strip() + "\n\n" + sources_block).strip()
            fallback_marker = []
            if general_used:
                fallback_marker.append(f"general_index_fallback:{general_trigger}")
            if web_used:
                fallback_marker.append(f"web_fallback:{web_trigger}")
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
                "general_index_fallback_used": general_used,
                "web_fallback_used": web_used,
                "web_fallback_sources": web_sources if web_used else [],
                "fallback_used": True,
                "fallback_trigger": " | ".join(fallback_marker),
                "routing_reason": _append_reason(
                    str(state.get("routing_reason", "")),
                    "web_fallback_used" if web_used else "general_index_fallback_used",
                ),
            }
        else:
            return await _build_last_resort_response(
                state, user_text, web_should_use, web_used, web_evidence_text, web_sources
            )

    # 2+ agentes (ou 1 + fallback geral): consolida com LLM.

    regulatory_preferred = (
        _prefer_regulatory_requirement_response(user_text, successful)
        if not general_used and not web_used
        else None
    )
    if regulatory_preferred:
        regulatory_preferred = _postprocess_consolidated_answer(user_text, regulatory_preferred)
        return {
            "final_response": regulatory_preferred,
            "messages": [AIMessage(content=regulatory_preferred)],
        }

    # Em perguntas objetivas, quando houver um único especialista com
    # resposta factual direta, devolve essa resposta sem adicionar ressalvas.
    preferred = _prefer_direct_fact_response(user_text, successful) if not general_used and not web_used else None
    if preferred:
        preferred = _postprocess_consolidated_answer(user_text, preferred)
        return {
            "final_response": preferred,
            "messages": [AIMessage(content=preferred)],
        }

    # Se existe pelo menos uma resposta factual, remove respostas que são
    # apenas ressalva/ausência para não "contaminar" a consolidação.
    factual_responses = []
    for r in successful:
        factual = _extract_factual_candidate(str(r.get("response", "")))
        if factual:
            item = dict(r)
            item["response"] = factual
            factual_responses.append(item)
    if factual_responses:
        successful = factual_responses

    # Otimização: única fonte de evidência factual → repassa direto, sem LLM de consolidação.
    # Consolidar uma única resposta não agrega valor e custa 800ms–2s desnecessários.
    if len(successful) == 1 and not general_used and not web_used:
        final_text = _postprocess_consolidated_answer(user_text, successful[0]["response"])
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # Separa respostas por papel: especialistas de domínio (1,2,4,5,6) vs baseline (0,3).
    # Especialistas = conteúdo técnico principal; Agent 3 = complemento regulatório;
    # Agent 0 = terminologia/glossário (contexto de suporte).
    _specialist_resps = [
        r for r in successful if int(r.get("agent_id", -1)) not in _ROUTING_BASELINE_IDS
    ]
    _regulatory_resp = next(
        (r for r in successful if int(r.get("agent_id", -1)) == 3), None
    )
    _general_resp = next(
        (r for r in successful if int(r.get("agent_id", -1)) == 0), None
    )

    _specialist_block = "".join(
        f"\n--- {r['agent_name']} ---\n{r['response']}\n"
        for r in _specialist_resps
    )
    _regulatory_block = (
        f"\n--- {_regulatory_resp['agent_name']} ---\n{_regulatory_resp['response']}\n"
        if _regulatory_resp else ""
    )
    _general_block = (
        f"\n--- {_general_resp['agent_name']} ---\n{_general_resp['response']}\n"
        if _general_resp else ""
    )
    _fallback_block = ""
    if general_used and general_evidence_text:
        _fallback_block += f"\n--- Base Geral Unificada (fallback) ---\n{general_evidence_text}\n"
    if web_used and web_evidence_text:
        _fallback_block += f"\n--- Web (domínios confiáveis) ---\n{web_evidence_text}\n"

    if _specialist_resps:
        # Caminho hierárquico: especialistas como base, regulatório como complemento.
        _prompt_body = f"PERGUNTA: {user_text}\n\nCONTEÚDO TÉCNICO PRINCIPAL:{_specialist_block}"
        if _regulatory_block:
            _prompt_body += f"\nCONTEXTO REGULATÓRIO COMPLEMENTAR:{_regulatory_block}"
        if _general_block:
            _prompt_body += f"\nTERMINOLOGIA / BASE GERAL:{_general_block}"
        if _fallback_block:
            _prompt_body += f"\nEVIDÊNCIA ADICIONAL:{_fallback_block}"

        consolidation_prompt = (
            "Você é o assistente geral do DairyApp AI.\n"
            "Regras de composição da resposta:\n"
            "1. Use o CONTEÚDO TÉCNICO PRINCIPAL como base da resposta\n"
            "2. Acrescente do CONTEXTO REGULATÓRIO COMPLEMENTAR apenas o que for diretamente "
            "relevante para a pergunta — não repita o que o técnico já cobriu\n"
            "3. Se técnico e norma divergirem em um ponto específico, a norma prevalece naquele ponto\n"
            "4. Se a pergunta for sobre requisito mínimo/obrigatório/exigido, trate o contexto "
            "regulatório como critério definitivo; não transforme prática técnica em exigência legal\n"
            "5. Preserve todos os dados técnicos (temperaturas, pHs, prazos, concentrações)\n"
            "6. NÃO invente fatos além das evidências fornecidas\n"
            "7. NÃO adicione ressalvas genéricas se a pergunta principal já foi respondida\n"
            "8. NÃO mencione agentes internos, bases de conhecimento ou ferramentas\n"
            "9. Tom técnico e profissional em português brasileiro\n\n"
            + _prompt_body
            + "\n\nResposta final:"
        )
    else:
        # Sem especialistas de domínio — apenas baseline (regulatório + geral + fallback).
        _all_block = _regulatory_block + _general_block + _fallback_block
        consolidation_prompt = (
            "Você é o assistente geral do DairyApp AI. "
            "Responda com base SOMENTE nas evidências abaixo. "
            "Preserve todos os dados. NÃO invente fatos. "
            "NÃO mencione agentes ou ferramentas internas. "
            "Tom técnico em português brasileiro.\n\n"
            f"PERGUNTA: {user_text}\n\n"
            f"EVIDÊNCIAS:{_all_block}\n"
            "Resposta:"
        )

    try:
        response_text = await _ainvoke_consolidation_with_timeout(state, consolidation_prompt)
        final_text = _postprocess_consolidated_answer(user_text, response_text)
        if _looks_like_unusable_consolidation_answer(final_text):
            final_text = _build_evidence_grounded_fallback_answer(
                user_text,
                successful,
                general_evidence_text,
                web_evidence_text,
                web_sources,
            )
    except Exception:
        final_text = _build_evidence_grounded_fallback_answer(
            user_text,
            successful,
            general_evidence_text,
            web_evidence_text,
            web_sources,
        )

    if web_used:
        sources_block = _render_web_sources_block(web_sources)
        if sources_block:
            final_text = (final_text.strip() + "\n\n" + sources_block).strip()

    payload: OrchestratorState = {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }
    if general_used or web_used:
        payload["general_index_fallback_used"] = general_used
        payload["web_fallback_used"] = web_used
        payload["web_fallback_sources"] = web_sources if web_used else []
        marker = []
        if general_used:
            marker.append(f"general_index_fallback:{general_trigger}")
        if web_used:
            marker.append(f"web_fallback:{web_trigger}")
        payload.update(
            {
                "fallback_used": True,
                "fallback_trigger": " | ".join(marker),
                "routing_reason": _append_reason(
                    str(state.get("routing_reason", "")),
                    "web_fallback_used" if web_used else "general_index_fallback_used",
                ),
            }
        )
    return payload


# ============================================================
# Montagem e compilação do grafo
# ============================================================

def build_orchestrator_graph() -> Any:
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("execute", execute)
    graph.add_node("fallback_reclassify", fallback_reclassify)
    graph.add_node("respond_direct", respond_direct)
    graph.add_node("consolidate", consolidate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route,
        {"ask_clarification": "ask_clarification", "execute": "execute", "respond_direct": "respond_direct"},
    )

    graph.add_edge("ask_clarification", END)

    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"fallback_reclassify": "fallback_reclassify", "consolidate": "consolidate"},
    )
    graph.add_conditional_edges(
        "fallback_reclassify",
        route_after_fallback,
        {"execute": "execute", "consolidate": "consolidate"},
    )
    graph.add_edge("respond_direct", "consolidate")
    graph.add_edge("consolidate", END)

    return graph.compile()


# ============================================================
# Instância global (lazy cache)
# ============================================================

_orchestrator_graph = None


def get_orchestrator_graph() -> Any:
    global _orchestrator_graph
    if _orchestrator_graph is None:
        _orchestrator_graph = build_orchestrator_graph()
    return _orchestrator_graph
