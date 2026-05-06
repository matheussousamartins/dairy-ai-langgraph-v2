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
import ast
import json
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
    ToolMessage,
)
from langgraph.graph import StateGraph, END

from app.config import (
    LLM_MODEL,
    CONSOLIDATION_TIMEOUT_SEC,
    ORCHESTRATOR_FASTPATH,
    CLASSIFICATION_CACHE_SIZE,
    MATCH_THRESHOLD,
    ENABLE_WEB_FALLBACK,
    WEB_FALLBACK_PROVIDER,
    WEB_FALLBACK_TIMEOUT_SEC,
    WEB_FALLBACK_MAX_RESULTS,
    WEB_FALLBACK_MAX_SOURCES,
    WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL,
    WEB_FALLBACK_FETCH_FULLTEXT,
    WEB_FALLBACK_MAX_PAGE_CHARS,
    WEB_FALLBACK_MAX_SNIPPET_CHARS,
    WEB_FALLBACK_ALLOWED_DOMAINS,
)
from app.agents.prompts import get_orchestrator_prompt
from app.agents.agent_config import AGENTS, get_agent_by_id
from app.agents.base_agent import (
    get_agent_graph,
    _build_contextual_search_query as _build_rag_search_query,
)
from app.rag.search import embed_query, search_vector
from app.tools.web_fallback import (
    build_web_fallback_evidence,
    enrich_results_with_page_content,
    search_web_duckduckgo,
)
from app.agents.orch_schema import OrchestratorState, ClassificationResult
from app.agents.orch_text import (
    _normalize_text,
    _normalize_mul_symbols,
    _strip_profile_suffix,
    _extract_current_user_segment,
    _extract_recent_context_block,
    _has_recent_context_block,
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
    _is_out_of_scope_response,
    _is_general_knowledge_response,
)
from app.agents.orch_models import (
    _resolve_state_model,
    _resolve_consolidation_model,
    _get_classifier,
    _get_consolidation_model,
    _get_direct_model,
)
from app.agents.orch_fewshot import build_few_shot_block
from app.agents.orch_quality import (
    detect_question_type,
    strip_prohibited_phrases,
    classify_response_quality,
    ResponseQuality,
)
from app.agents.synthesis_rules import build_synthesis_prompt
from app.agents.evidence_reducer import (
    build_direct_answer_from_candidates,
    reduce_evidence_for_question,
)
from app.observability import (
    new_trace_id,
    get_trace_id,
    log_event,
    NodeTimer,
    LLMSlot,
)
from app.resilience import get_circuit_breaker, get_timeout_tracker
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


def _requires_synthesis_response(user_text: str) -> bool:
    """Perguntas amplas precisam de síntese, não de extração factual direta."""
    norm = _normalize_text(_strip_profile_suffix(user_text))
    if not norm:
        return False
    broad_markers = (
        "processo", "processo basico", "composicao", "composicao e processo",
        "etapas", "fabricacao", "como fabricar", "como produzir", "indicado",
        "indicados", "descreva", "explique",
    )
    if any(marker in norm for marker in broad_markers):
        return True
    question_type = detect_question_type(user_text)
    return question_type in {"process", "comparative", "troubleshooting"}


def _prefer_direct_fact_response(
    user_text: str,
    responses: List[Dict[str, Any]],
) -> Optional[str]:
    """When question is objective, prefer a direct factual specialist answer."""
    if _requires_synthesis_response(user_text):
        return None
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
    if not (
        _is_legal_requirement_regulatory_signal(_normalize_text(user_text))
        or _is_regulatory_primary_question(user_text)
    ):
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


def _is_regulatory_primary_question(user_text: str) -> bool:
    norm = _normalize_text(_strip_profile_suffix(user_text))
    return (
        _is_strong_regulatory_signal(norm)
        or _is_legal_requirement_regulatory_signal(norm)
        or _is_normative_regulatory_signal(norm)
        or _is_labeling_regulatory_signal(norm)
        or "legislacao" in norm
        or "regulatorio" in norm
        or "regulatoria" in norm
        or "mapa" in norm
        or "anvisa" in norm
        or "riispoa" in norm
    )


def _build_primary_locked_direct_answer(
    user_text: str,
    candidates: List[Dict[str, Any]],
) -> Optional[str]:
    """Preserva a hierarquia: especialista responde; regulatorio complementa."""
    if _is_regulatory_primary_question(user_text):
        return None
    if _requires_synthesis_response(user_text):
        return None

    specialist_candidates = [
        item for item in candidates
        if int(item.get("agent_id", -1)) not in _ROUTING_BASELINE_IDS
        and item.get("answer_source") in {"rag", "rag_evidence"}
    ]
    if not specialist_candidates:
        return None

    specialist_candidates.sort(
        key=lambda item: float(item.get("rag_top_score") or 0.0),
        reverse=True,
    )
    primary = specialist_candidates[0]
    primary_fact = str(primary.get("direct_answer_candidate") or "").strip()
    if not primary_fact:
        primary_fact = _extract_factual_candidate(str(primary.get("response", ""))) or ""
    primary_fact = _sanitize_math_for_ui(primary_fact).strip()
    if not primary_fact:
        return None

    answer = primary_fact
    for regulatory in candidates:
        if int(regulatory.get("agent_id", -1)) != _REGULATORY_BASELINE_ID:
            continue
        answer = _append_missing_regulatory_numeric_complement(
            answer,
            user_text,
            primary_fact,
            str(regulatory.get("response", "")),
        )

    return _sanitize_math_for_ui(answer.strip())


_REGULATORY_COMPLEMENT_STOPWORDS = {
    "sobre", "qual", "quais", "quanto", "ponto", "vista", "legislacao",
    "legislativo", "legal", "norma", "normativo", "riispoa", "decreto",
    "art", "artigo", "para", "com", "uma", "por", "dos", "das", "que",
    "deve", "devem", "pode", "podem", "limite", "maximo", "minimo",
    "recomendado", "recomendada", "aproximadamente", "celulas", "ml",
}


def _meaningful_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]+", _normalize_text(text), flags=re.UNICODE)
        if len(token) >= 4 and token not in _REGULATORY_COMPLEMENT_STOPWORDS
    }


def _evidence_overlap_terms(user_text: str, evidence_text: str) -> set[str]:
    query_terms = _meaningful_terms(_strip_profile_suffix(user_text))
    evidence_terms = _meaningful_terms(evidence_text)
    return query_terms & evidence_terms


def _evidence_has_numbers(text: str) -> bool:
    return bool(_numeric_markers(text))


def _required_regulatory_topic_terms(user_text: str) -> set[str]:
    norm = _normalize_text(_strip_profile_suffix(user_text))
    required: set[str] = set()
    if (
        "ccs" in norm
        or "celulas somaticas" in norm
        or "celula somatica" in norm
        or "contagem de celulas" in norm
    ):
        required.update({"ccs", "celulas somaticas", "celula somatica"})
    if "contagem bacteriana" in norm or "cbt" in norm or "contagem padrao" in norm:
        required.update({"contagem bacteriana", "cbt", "contagem padrao"})
    if "maturacao" in norm or "maturado" in norm or "maturados" in norm:
        required.update({"maturacao", "maturado", "maturados", "sessenta dias", "60 dias"})
    if "rotulag" in norm or "rotulo" in norm:
        required.update({"rotulagem", "rotulo", "rotulos"})
    if "gordura" in norm:
        required.add("gordura")
    if "proteina" in norm:
        required.add("proteina")
    if "lactose" in norm:
        required.add("lactose")
    if "umidade" in norm:
        required.add("umidade")
    return required


def _is_raw_milk_storage_treatment_question(user_text: str) -> bool:
    norm = _normalize_text(_strip_profile_suffix(user_text))
    return (
        ("leite cru" in norm or "leite refrigerado" in norm)
        and any(t in norm for t in ("dois dias", "2 dias", "48 horas", "armazen", "estoc"))
        and any(t in norm for t in ("pasteurizacao", "pasteurização", "tratamento"))
    )


def _matches_required_regulatory_topic(user_text: str, evidence_text: str) -> bool:
    evidence_norm = _normalize_text(evidence_text)
    if _is_raw_milk_storage_treatment_question(user_text):
        return any(
            term in evidence_norm
            for term in ("dois dias", "2 dias", "48 horas", "armazen", "estoc", "termiz")
        )

    required = _required_regulatory_topic_terms(user_text)
    if not required:
        return True
    return any(term in evidence_norm for term in required)


_DAIRY_PRODUCT_TERMS = {
    "parmesao", "parmigiano", "reggiano", "reggianito", "grana", "sbrinz",
    "sardo", "mussarela", "mozzarella", "pizza cheese", "prato", "gouda",
    "edam", "reino", "mimolette", "estepe", "muenster", "queijo",
    "leite", "iogurte", "kefir", "ricota", "provolone", "cream cheese",
    "minas",
}


def _product_terms_in_text(text: str) -> set[str]:
    norm = _normalize_text(text)
    return {term for term in _DAIRY_PRODUCT_TERMS if term in norm}


def _is_relevant_regulatory_complement(
    user_text: str,
    evidence_text: str,
    *,
    regulatory_query: bool,
) -> bool:
    """Allows Agent 3 to complement technical answers without polluting them."""
    if regulatory_query:
        return _matches_required_regulatory_topic(user_text, evidence_text)

    if not _matches_required_regulatory_topic(user_text, evidence_text):
        return False

    user_products = _product_terms_in_text(user_text)
    evidence_products = _product_terms_in_text(evidence_text)
    generic_products = {"queijo", "leite"}
    specific_user_products = user_products - generic_products
    if specific_user_products:
        product_overlap = specific_user_products & evidence_products
    else:
        product_overlap = user_products & evidence_products
    if not product_overlap:
        return False

    overlap = _evidence_overlap_terms(user_text, evidence_text)
    if len(overlap) >= 2:
        return True

    if _has_regulatory_limit_signal(evidence_text) and _evidence_has_numbers(evidence_text):
        return True

    return False


def _is_relevant_evidence_text(
    user_text: str,
    evidence_text: str,
    *,
    agent_id: int,
    top_score: float = 0.0,
    raw_rag: bool = False,
) -> bool:
    """Decide se um trecho/resposta deve entrar na consolidação.

    A regra prioriza recall: evidências com sinal de domínio passam para o LLM
    que decide se usa — apenas lixo claro (fora de escopo, ausência explícita,
    sem nenhum sinal dairy) é descartado antes.
    """
    text = str(evidence_text or "").strip()
    if not text:
        return False
    if _is_out_of_scope_response(text):
        return False
    if not raw_rag and _looks_uncertain(text):
        return False

    user_norm = _normalize_text(_strip_profile_suffix(user_text))
    text_norm = _normalize_text(text)

    if agent_id == _REGULATORY_BASELINE_ID:
        regulatory_query = (
            _is_strong_regulatory_signal(user_norm)
            or _is_legal_requirement_regulatory_signal(user_norm)
            or _is_normative_regulatory_signal(user_norm)
            or _is_labeling_regulatory_signal(user_norm)
            or "legislacao" in user_norm
            or "regulator" in user_norm
        )
        if not _is_relevant_regulatory_complement(
            user_text,
            text,
            regulatory_query=regulatory_query,
        ):
            return False
        # Regulatório passou o complement check — ainda verifica overlap ou sinal legal+número
        overlap = _evidence_overlap_terms(user_text, text)
        if overlap:
            return True
        if _has_regulatory_limit_signal(text) and _evidence_has_numbers(text):
            if regulatory_query and (_numeric_markers(user_norm) & _numeric_markers(text)):
                return True
        return False

    # Agentes especialistas (não regulatório)
    overlap = _evidence_overlap_terms(user_text, text)
    if overlap:
        return True

    # Score RAG alto: o modelo de embeddings avaliou relevância — confiamos nele.
    # Threshold baixado de 0.075 para 0.04 para recuperar casos de sinonímia técnica
    # (ex: "reduções microbiológicas" vs "inativação de patógenos").
    if top_score >= 0.04 and _contains_dairy_signal(user_norm):
        return True

    # Para especialistas: qualquer sinal dairy na evidência é suficiente quando
    # a pergunta também tem sinal dairy — o LLM filtra irrelevância interna.
    if _contains_dairy_signal(user_norm) and _contains_dairy_signal(text_norm):
        return True

    return False


def _numeric_markers(text: str) -> set[str]:
    markers = set()
    for raw in re.findall(r"\d[\d.,]*", text or ""):
        digits = re.sub(r"\D", "", raw)
        if digits:
            markers.add(digits)
    return markers


def _has_regulatory_limit_signal(text: str) -> bool:
    norm = _normalize_text(text)
    return any(
        signal in norm
        for signal in (
            "limite", "maximo", "maxima", "minimo", "minima", "deve atender",
            "deve apresentar", "exigido", "exigida", "obrigatorio", "obrigatoria",
            "legislacao", "riispoa", "decreto", "instrucao normativa", "portaria",
        )
    )


_LOCAL_REGULATORY_IN76_CCS_EVIDENCE = (
    "Instrucao Normativa MAPA no 76, de 26 de novembro de 2018 - Art. 7o. "
    "O leite cru refrigerado de tanque individual ou de uso comunitario deve "
    "apresentar medias geometricas trimestrais de Contagem Padrao em Placas "
    "de no maximo 300.000 UFC/mL e de Contagem de Celulas Somaticas (CCS) "
    "de no maximo 500.000 CS/mL. As medias geometricas consideram analises "
    "de tres meses consecutivos e ininterruptos, com no minimo uma amostra "
    "mensal de cada tanque."
)


def _build_local_primary_regulatory_evidence(user_text: str) -> Optional[str]:
    """Fonte primaria versionada para fatos legais curtos e criticos."""
    norm = _normalize_text(_strip_profile_suffix(user_text))
    asks_ccs = (
        "ccs" in norm
        or "celulas somaticas" in norm
        or "celula somatica" in norm
        or "contagem de celulas" in norm
    )
    if not asks_ccs:
        return None

    asks_requirement = any(
        signal in norm
        for signal in (
            "qual",
            "quanto",
            "limite",
            "legisl",
            "legal",
            "norma",
            "regulator",
            "do ponto de vista",
            "recomendada",
            "recomendado",
        )
    )
    if not asks_requirement:
        return None
    return _LOCAL_REGULATORY_IN76_CCS_EVIDENCE


def _append_missing_regulatory_numeric_complement(
    final_text: str,
    user_text: str,
    specialist_block: str,
    regulatory_text: str,
) -> str:
    """Garante que limites regulatórios relevantes não sumam na consolidação."""
    regulatory_fact = _extract_factual_candidate(regulatory_text)
    if not regulatory_fact or not _has_regulatory_limit_signal(regulatory_fact):
        return final_text
    if not _matches_required_regulatory_topic(user_text, regulatory_fact):
        return final_text

    regulatory_numbers = _numeric_markers(regulatory_fact)
    if not regulatory_numbers:
        return final_text
    if regulatory_numbers.issubset(_numeric_markers(final_text)):
        return final_text

    topic_terms = _meaningful_terms(user_text) | _meaningful_terms(specialist_block) | _meaningful_terms(final_text)
    regulatory_terms = _meaningful_terms(regulatory_fact)
    if not (topic_terms & regulatory_terms):
        return final_text

    regulatory_fact = _sanitize_math_for_ui(regulatory_fact).rstrip()
    if regulatory_fact in final_text:
        return final_text
    return f"{final_text.rstrip()}\n\nDo ponto de vista regulatório, {regulatory_fact}"


def _build_factual_response_candidates(
    responses: List[Dict[str, Any]],
    user_text: str = "",
) -> List[Dict[str, Any]]:
    """Normaliza respostas de agentes em evidências utilizáveis.

    O contrato de produção é: agentes não decidem fallback. Eles podem redigir
    respostas, mas o orquestrador só consolida evidência RAG/factual relacionada
    à pergunta. [CONHECIMENTO GERAL] puro, ausência e fora de escopo ficam fora.
    """
    candidates: List[Dict[str, Any]] = []
    for item in responses:
        has_response = bool(str(item.get("response", "")).strip())
        has_evidence = bool(str(item.get("rag_evidence_text", "")).strip())
        if not item.get("success") or not (has_response or has_evidence):
            continue

        raw_response = str(item.get("response", ""))
        if _is_out_of_scope_response(raw_response) and not has_evidence:
            continue

        aid = int(item.get("agent_id", -1))
        factual = _extract_factual_candidate(raw_response)
        is_general = _is_general_knowledge_response(raw_response)
        rag_evidence = str(item.get("rag_evidence_text", "") or "").strip()
        rag_top_score = float(item.get("rag_top_score") or 0.0)

        selected_text = ""
        answer_source = ""
        requires_consolidation = False

        reduced = None
        if rag_evidence and user_text:
            reduced = reduce_evidence_for_question(
                user_text,
                rag_evidence,
                agent_id=aid,
                top_score=rag_top_score,
            )

        reduced_text = (reduced.text if reduced and reduced.text else "").strip()
        if rag_evidence and (
            not user_text
            or (
                reduced_text
                and _is_relevant_evidence_text(
                    user_text,
                    reduced_text,
                    agent_id=aid,
                    top_score=rag_top_score,
                    raw_rag=True,
                )
            )
        ):
            selected_text = reduced_text if user_text else rag_evidence
            answer_source = "rag_evidence"
            requires_consolidation = True
        elif factual and not is_general and (
            not user_text
            or _is_relevant_evidence_text(
                user_text,
                factual,
                agent_id=aid,
                top_score=rag_top_score,
            )
        ):
            selected_text = factual
            answer_source = "rag"

        if not selected_text:
            continue

        row = dict(item)
        row["response"] = selected_text
        row["answer_source"] = answer_source
        if reduced and reduced.text:
            row["evidence_snippets"] = reduced.snippets
            row["direct_answer_candidate"] = reduced.direct_answer
            row["reduced_evidence"] = True
        row["evidence_quality"] = (
            "usable_regulatory_evidence"
            if aid == _REGULATORY_BASELINE_ID
            else "usable_specialist_evidence"
        )
        row["requires_consolidation"] = requires_consolidation
        candidates.append(row)

    def _priority(row: Dict[str, Any]) -> Tuple[int, int]:
        source = row.get("answer_source")
        source_rank = 0 if source in {"rag", "rag_evidence", "local_primary_regulatory_source"} else 1
        aid = int(row.get("agent_id", -1))
        role_rank = 1 if aid in _ROUTING_BASELINE_IDS else 0
        return source_rank, role_rank

    return sorted(candidates, key=_priority)


def _unwrap_tool_rows(payload: Any) -> List[Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            try:
                payload = ast.literal_eval(payload)
            except Exception:
                return []

    if isinstance(payload, list):
        flattened: List[Any] = []
        for item in payload:
            if isinstance(item, dict):
                nested = None
                for key in ("results", "chunks", "items", "data", "documents", "matches"):
                    value = item.get(key)
                    if isinstance(value, list):
                        nested = value
                        break
                if nested is not None:
                    flattened.extend(nested)
                    continue
            flattened.append(item)
        return flattened

    if isinstance(payload, dict):
        for key in ("results", "chunks", "items", "data", "documents", "matches"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        content = payload.get("content")
        if isinstance(content, list):
            return content

        if any(
            key in payload
            for key in ("content", "text", "page_content", "snippet", "chunk")
        ):
            return [payload]

    return []


def _coerce_tool_result_rows(content: Any) -> List[Dict[str, Any]]:
    """Normaliza payloads de ToolMessage da busca RAG.

    A causa raiz do bug crônico era tratar apenas a resposta textual do agente
    como evidência. Esta função preserva os chunks recuperados como artefato de
    primeira classe para o orquestrador.
    """
    rows: List[Dict[str, Any]] = []
    for item in _unwrap_tool_rows(content):
        if isinstance(item, str):
            text = item.strip()
            metadata: Dict[str, Any] = {}
            score = 0.0
        elif isinstance(item, dict):
            text = str(
                item.get("content")
                or item.get("text")
                or item.get("page_content")
                or item.get("snippet")
                or item.get("chunk")
                or ""
            ).strip()
            metadata = item.get("metadata") or item.get("meta") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            source = (
                item.get("source")
                or item.get("filename")
                or item.get("doc_id")
                or item.get("path")
            )
            if source and not any(
                metadata.get(key) for key in ("source", "filename", "doc_id", "path")
            ):
                metadata = dict(metadata)
                metadata["source"] = source
            try:
                score = float(
                    item.get("score")
                    or item.get("similarity")
                    or item.get("relevance_score")
                    or 0.0
                )
            except Exception:
                score = 0.0
        else:
            continue

        if not text:
            continue
        rows.append({
            "content": text,
            "score": score,
            "metadata": metadata,
        })
    return rows


def _format_rag_rows(rows: List[Dict[str, Any]], top_n: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    """Formata rows brutas do search_vector no mesmo formato de _extract_rag_evidence_from_messages."""
    if not rows:
        return "", []
    sorted_rows = sorted(rows, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    selected = sorted_rows[:max(1, top_n)]
    blocks: List[str] = []
    for idx, item in enumerate(selected, start=1):
        metadata = item.get("metadata") or {}
        source = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("doc_id")
            or metadata.get("path")
            or "fonte interna"
        )
        score = float(item.get("score", 0.0) or 0.0)
        content = re.sub(r"\s+", " ", str(item.get("content", "")).strip())
        blocks.append(f"Trecho {idx} — score {score:.4f} — {source}\n{content}")
    return "\n\n".join(blocks).strip(), selected


def _extract_rag_evidence_from_messages(messages: List[AnyMessage], top_n: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        msg_rows = _coerce_tool_result_rows(msg.content)
        if not msg_rows:
            artifact = getattr(msg, "artifact", None)
            if artifact is not None:
                msg_rows = _coerce_tool_result_rows(artifact)
        rows.extend(msg_rows)

    if not rows:
        return "", []

    rows.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    selected = rows[: max(1, top_n)]
    blocks: List[str] = []
    for idx, item in enumerate(selected, start=1):
        metadata = item.get("metadata") or {}
        source = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("doc_id")
            or metadata.get("path")
            or "fonte interna"
        )
        score = float(item.get("score", 0.0) or 0.0)
        content = re.sub(r"\s+", " ", str(item.get("content", "")).strip())
        blocks.append(f"Trecho {idx} — score {score:.4f} — {source}\n{content}")
    return "\n\n".join(blocks).strip(), selected


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
        old_ok = bool(old.get("success")) and bool(old.get("response") or old.get("rag_evidence_text"))
        new_ok = bool(item.get("success")) and bool(item.get("response") or item.get("rag_evidence_text"))
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






def _append_reason(reason: str, marker: str) -> str:
    base = (reason or "").strip()
    mark = (marker or "").strip()
    if not mark:
        return base
    return f"{base} | {mark}" if base else mark


def _looks_like_chunk_dump(text: str) -> bool:
    """Detecta vazamento de artefatos RAG na resposta final."""
    raw = str(text or "")
    if not raw.strip():
        return False

    normalized = _normalize_text(raw)
    if "source_table" in normalized or "embeddings_agente" in normalized:
        return True
    if re.search(r"\b(?:trecho|chunk)\s+\d+\s*[—-]\s*score\b", normalized):
        return True
    if re.search(r"\|[-:\s|]{3,}\|", raw):
        return True
    if len(re.findall(r"\b(?:Trecho|chunk)\s+\d+\b", raw, flags=re.IGNORECASE)) >= 2:
        return True
    return False


def _looks_like_unusable_consolidation_answer(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if _looks_like_chunk_dump(text):
        return True
    if _looks_uncertain(text):
        return True
    markers = (
        "nao foi possivel consolidar",
        "nao foi possivel obter uma resposta",
        "nao foi possivel responder",
        "resposta confiavel no momento",
        "tente reformular sua pergunta",
        "nao encontrei informacoes",
        "nao foram encontradas informacoes",
        "nao foram encontrados dados",
        "nao ha dados especificos",
        "nao ha evidencias",
    )
    return any(marker in normalized for marker in markers)


_ZERO_EVIDENCE_MSG = (
    "Não encontrei informações suficientes para responder a essa pergunta. "
    "Tente reformular com termos mais específicos sobre o produto, processo ou parâmetro desejado."
)


async def _ainvoke_consolidation_with_timeout(
    state: "OrchestratorState",
    prompt: str,
) -> str:
    async with LLMSlot():
        response = await asyncio.wait_for(
            _get_consolidation_model(_resolve_consolidation_model(state)).ainvoke(
                [HumanMessage(content=prompt)]
            ),
            timeout=float(CONSOLIDATION_TIMEOUT_SEC),
        )
    return str(response.content or "")


async def _consolidation_llm_retry(
    user_text: str,
    original_prompt: str,
    successful: List[Dict[str, Any]],
) -> str:
    """Retry de consolidação com gpt-4o-mini quando o consolidador principal falha.

    Retorna _ZERO_EVIDENCE_MSG apenas como sentinela interno — o chamador
    (consolidate) deve escalar para web fallback quando receber esse valor.
    """
    evidence_parts: List[str] = []
    for r in successful:
        resp = str(r.get("response") or "").strip()
        if resp and not _is_general_knowledge_response(resp):
            evidence_parts.append(resp[:1200])

    evidence_blob = "\n\n".join(evidence_parts).strip()
    if not evidence_blob:
        return _ZERO_EVIDENCE_MSG

    retry_prompt = (
        "Você é um assistente técnico de laticínios. "
        "Responda à pergunta abaixo com base EXCLUSIVAMENTE nas evidências fornecidas.\n\n"
        "ATENÇÃO: se as evidências contiverem QUALQUER valor numérico relacionado à pergunta "
        "(contagens, temperaturas, limites, concentrações), esse número DEVE aparecer na resposta. "
        "Proibido dizer 'não há especificação' quando a evidência contém um número.\n\n"
        "Seja direto e técnico. Não mencione ferramentas ou bases de dados internas.\n\n"
        f"PERGUNTA: {user_text}\n\n"
        f"EVIDÊNCIAS:\n{evidence_blob}\n\n"
        "Resposta:"
    )

    try:
        async with LLMSlot():
            response = await asyncio.wait_for(
                _get_consolidation_model("gpt-4o-mini").ainvoke(
                    [HumanMessage(content=retry_prompt)]
                ),
                timeout=15.0,
            )
        result = _postprocess_consolidated_answer(user_text, str(response.content or ""))
        if result and not _looks_like_unusable_consolidation_answer(result):
            return result
    except Exception:
        pass

    return _ZERO_EVIDENCE_MSG


def _render_web_sources_block(sources: List[Dict[str, str]]) -> str:
    clean_sources: List[str] = []
    for item in (sources or [])[: max(1, int(WEB_FALLBACK_MAX_SOURCES or 1))]:
        title = str(item.get("title") or item.get("domain") or "fonte").strip()
        url = str(item.get("url") or "").strip()
        domain = str(item.get("domain") or "").strip()
        if not url:
            continue
        label = title
        if domain and domain.lower() not in title.lower():
            label = f"{title} ({domain})"
        clean_sources.append(f"[{label}]({url})")
    if not clean_sources:
        return ""
    return f"Fonte consultada: {'; '.join(clean_sources)}."


def _web_fallback_enabled_for_question(user_text: str) -> bool:
    if not ENABLE_WEB_FALLBACK:
        return False
    if WEB_FALLBACK_PROVIDER != "duckduckgo":
        return False
    if not (user_text or "").strip():
        return False
    if WEB_FALLBACK_REQUIRE_DAIRY_SIGNAL and not _contains_dairy_signal(
        _normalize_text(_strip_profile_suffix(user_text))
    ):
        return False
    return True


async def _fetch_web_fallback_evidence(
    user_text: str,
) -> Tuple[str, List[Dict[str, str]], str]:
    if not ENABLE_WEB_FALLBACK or WEB_FALLBACK_PROVIDER != "duckduckgo":
        return "", [], "web_fallback_disabled"

    query = _build_rag_search_query(user_text)

    async def _search(allowed_domains: List[str]) -> List[Dict]:
        try:
            return await asyncio.to_thread(
                search_web_duckduckgo,
                query,
                allowed_domains,
                WEB_FALLBACK_MAX_RESULTS,
                WEB_FALLBACK_TIMEOUT_SEC,
                WEB_FALLBACK_MAX_SNIPPET_CHARS,
            )
        except Exception:
            return []

    # Tentativa 1: domínios confiáveis (whitelist)
    results = await _search(WEB_FALLBACK_ALLOWED_DOMAINS)

    # Tentativa 2: sem restrição de domínio — last resort garantido
    if not results:
        results = await _search([])

    if not results:
        return "", [], "web_fallback_search_error"

    if WEB_FALLBACK_FETCH_FULLTEXT:
        try:
            results = await asyncio.to_thread(
                enrich_results_with_page_content,
                results,
                WEB_FALLBACK_TIMEOUT_SEC,
                WEB_FALLBACK_MAX_PAGE_CHARS,
            )
        except Exception:
            pass

    evidence_text, sources = build_web_fallback_evidence(
        results,
        max_sources=WEB_FALLBACK_MAX_SOURCES,
    )
    if not evidence_text or not sources:
        return "", [], "web_fallback_no_results"
    status = "web_fallback_whitelist" if results else "web_fallback_open"
    return evidence_text, sources, status


async def _build_web_last_resort_response(
    state: OrchestratorState,
    user_text: str,
    *,
    trigger: str,
) -> OrchestratorState:
    evidence_text, sources, status = await _fetch_web_fallback_evidence(user_text)
    if not evidence_text:
        return {
            "final_response": _ZERO_EVIDENCE_MSG,
            "messages": [AIMessage(content=_ZERO_EVIDENCE_MSG)],
            "web_fallback_used": False,
            "web_fallback_sources": [],
            "fallback_used": False,
            "fallback_trigger": status,
        }

    prompt = (
        "Voce e o assistente tecnico do DairyApp AI.\n"
        "A base interna nao trouxe evidencia suficiente para responder. "
        "Responda com base SOMENTE nas evidencias web abaixo, vindas de dominios permitidos. "
        "Nao mencione que a base interna falhou. Nao invente fatos. "
        "Se a evidencia web for parcial, responda somente o que ela sustenta. "
        "Use linguagem profissional, objetiva e em portugues brasileiro. "
        "Nao inclua lista de fontes no corpo da resposta; as fontes serao anexadas depois.\n\n"
        f"PERGUNTA: {user_text}\n\n"
        f"EVIDENCIAS WEB:\n{evidence_text}\n\n"
        "Resposta:"
    )

    try:
        response_text = await _ainvoke_consolidation_with_timeout(state, prompt)
        final_text = _postprocess_consolidated_answer(user_text, response_text)
    except Exception:
        final_text = ""

    if not final_text or _looks_like_unusable_consolidation_answer(final_text):
        final_text = _ZERO_EVIDENCE_MSG

    if final_text != _ZERO_EVIDENCE_MSG:
        sources_block = _render_web_sources_block(sources)
        if sources_block:
            final_text = f"{final_text.rstrip()}\n\n{sources_block}"

    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
        "web_fallback_used": final_text != _ZERO_EVIDENCE_MSG,
        "web_fallback_sources": sources if final_text != _ZERO_EVIDENCE_MSG else [],
        "fallback_used": final_text != _ZERO_EVIDENCE_MSG,
        "fallback_trigger": f"web_last_resort:{trigger}:{status}",
        "routing_reason": _append_reason(str(state.get("routing_reason", "")), "web_last_resort"),
    }


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

    Agent 3 (Regulatórios) é incluído como complemento em perguntas de laticínios.
    Agent 1 (Queijos) lidera quando há sinal de tecnologia de queijo.
    """
    # Inicia tracing para o request
    trace_id = new_trace_id()

    messages = state.get("messages", [])
    user_text = _get_last_user_text(messages)

    if not user_text:
        log_event("classify_empty", node="classify", trace_id=trace_id)
        return _build_classification_state(route_text="", agent_ids=[], confidence=1.0, reason="mensagem_vazia")

    async with NodeTimer("classify") as timer:
        route_text = _strip_profile_suffix(user_text)
        current_question_norm = _normalize_text(_extract_current_user_segment(route_text))
        if _is_conversation_recap_request(current_question_norm):
            timer.add(path="recap")
            return _build_classification_state(
                route_text=route_text,
                agent_ids=[],
                confidence=0.98 if _has_recent_context_block(route_text) else 0.80,
                reason="conversation_recap",
            )

        cache_key = _normalize_text(route_text)

        cached_ids = _cache_get(cache_key)
        if cached_ids is not None:
            timer.add(path="cache_hit", agents=cached_ids)
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
                timer.add(path="fastpath", agents=fast_ids)
                return _build_classification_state(
                    route_text=route_text,
                    agent_ids=fast_ids,
                    confidence=_estimate_fastpath_confidence(route_text, fast_ids),
                    reason="fastpath_rule_based",
                )

        system_prompt = get_orchestrator_prompt()
        few_shot_block = build_few_shot_block()

        classification_instruction = f"""

Com base na pergunta do usuário, identifique quais agentes devem ser consultados.

REGRAS DE INCLUSÃO:
- Agente 3 (Regulatórios): incluir como complemento em toda pergunta de laticínios
  técnica ou regulatória, exceto saudação/off-topic. Ele não substitui o especialista:
  o especialista deve liderar quando houver domínio técnico claro; o Agente 3 apenas
  complementa se houver evidência regulatória aderente ao mesmo tema/produto.
- Agente 0 (Base Geral): incluir SOMENTE se a pergunta envolver glossário,
  padronização de termos ("qual termo usar", "como chamar"), marcas/fabricantes/
  distribuidores/equipamentos específicos, ou saudação/off-topic. NÃO incluir
  agente 0 em perguntas puramente técnicas, analíticas, de processo ou regulatórias
  — a base do agente 0 não cobre esses temas e só adiciona ruído.
- Especialistas 1-4: adicionar apenas se a pergunta for claramente desse domínio.

ESPECIALISTAS DISPONÍVEIS:
- Agente 1 (Queijos): queijos duros, semiduros, pasta filata. Processos, defeitos, rendimento, culturas.
- Agente 3 (Regulatórios): normas MAPA/ANVISA, INs, RDCs, RIISPOA, Codex, rotulagem.

FORMATO DA RESPOSTA:
- Saudação / off-topic → []
- Pergunta de glossário/terminologia → [0, 3]
- Pergunta técnica de queijo → [1, 3] (1 lidera, 3 complementa se houver aderência)
- Pergunta regulatória pura → [3]
- Pergunta regulatória + técnica → [3, X] ou [X, 3], com o agente mais relevante primeiro
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

{few_shot_block}"""

        async with LLMSlot():
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
            timer.add(path="llm_no_domain", confidence=confidence)
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

        timer.add(
            path="llm",
            agents=agent_ids,
            confidence=confidence,
            bucket=classification.get("routing_bucket", ""),
        )

        return classification

# ============================================================
# Roteamento condicional
# ============================================================

def route(state: OrchestratorState) -> str:
    planned = state.get("execution_plan")
    if planned is not None:
        return "respond_direct" if not planned else "execute"
    return "respond_direct" if not state.get("chosen_agent_ids") else "execute"


# ============================================================
# NÃ³ EXECUTE â€" execuÃ§Ã£o paralela
# ============================================================

async def execute(state: OrchestratorState) -> OrchestratorState:
    """Invoca todos os agentes em PARALELO via asyncio.gather.

    Latência total ≈ tempo do agente mais lento (não a soma).
    Cada agente tem timeout individual de AGENT_TIMEOUT segundos.
    """
    log_event("execute_start", node="execute",
              agents=state.get("execution_plan") or state.get("chosen_agent_ids", []))
    agent_ids = state.get("execution_plan") or state.get("chosen_agent_ids", [])
    agent_names = [
        (get_agent_by_id(aid) or {}).get("name", f"Agente {aid}")
        for aid in agent_ids
    ]

    user_text = _get_last_user_text(state.get("messages", []))
    search_query = _build_rag_search_query(user_text)

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
        cb = get_circuit_breaker(agent_id)
        tt = get_timeout_tracker(agent_id)

        if cb.is_open():
            log_event(
                "circuit_open_skip",
                node="execute",
                agent_id=agent_id,
                circuit_state=cb.state,
            )
            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": "",
                "success": False,
                "circuit_skipped": True,
            }

        adaptive_timeout = tt.get_timeout()
        t_start = asyncio.get_event_loop().time()

        try:
            graph = get_agent_graph(agent_id, _resolve_state_model(state))
            result = await asyncio.wait_for(
                graph.ainvoke({
                    "messages": [HumanMessage(content=user_text)],
                    "llm_model": _resolve_state_model(state),
                    "search_query": search_query,
                    "precomputed_embedding": shared_embedding,
                }),
                timeout=adaptive_timeout,
            )
            duration = asyncio.get_event_loop().time() - t_start
            tt.record(duration)
            cb.record_success()

            agent_msgs = result.get("messages", [])
            rag_evidence_text, rag_evidence_rows = _extract_rag_evidence_from_messages(agent_msgs)
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
                "success": bool(agent_text or rag_evidence_text),
                "rag_evidence_text": rag_evidence_text,
                "rag_evidence_count": len(rag_evidence_rows),
                "rag_evidence_rows": rag_evidence_rows,
                "rag_top_score": (
                    float(rag_evidence_rows[0].get("score", 0.0) or 0.0)
                    if rag_evidence_rows else 0.0
                ),
            }
        except asyncio.TimeoutError:
            duration = asyncio.get_event_loop().time() - t_start
            tt.record(duration)
            cb.record_failure(is_infra_timeout=True)

            log_event(
                "agent_timeout",
                node="execute",
                agent_id=agent_id,
                timeout_sec=adaptive_timeout,
                duration_ms=int(duration * 1000),
            )

            # O LLM do subagente deu timeout, mas o RAG (Supabase) geralmente
            # completa em <500ms. Refaz a busca vetorial diretamente — sem LLM —
            # para entregar os chunks ao consolidador, que sintetiza no lugar.
            if shared_embedding:
                agent_cfg = get_agent_by_id(agent_id) or {}
                table_name = agent_cfg.get("table_name", "")
                if table_name:
                    try:
                        fallback_rows = await asyncio.to_thread(
                            search_vector,
                            shared_embedding,
                            table_name,
                            5,
                            None,
                        )
                        if fallback_rows:
                            fb_text, fb_rows = _format_rag_rows(fallback_rows)
                            if fb_rows:
                                top_score = float(fb_rows[0].get("score", 0.0) or 0.0)
                                log_event(
                                    "agent_timeout_rag_salvaged",
                                    node="execute",
                                    agent_id=agent_id,
                                    salvaged_chunks=len(fb_rows),
                                )
                                return {
                                    "agent_id": agent_id,
                                    "agent_name": agent_name,
                                    "response": "",
                                    "success": True,
                                    "rag_evidence_text": fb_text,
                                    "rag_evidence_count": len(fb_rows),
                                    "rag_evidence_rows": fb_rows,
                                    "rag_top_score": top_score,
                                    "timeout_llm": True,
                                }
                    except Exception:
                        pass

            return {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "response": f"{agent_name}: timeout ao consultar base de conhecimento.",
                "success": False,
            }
        except Exception as e:
            duration = asyncio.get_event_loop().time() - t_start
            cb.record_failure()
            log_event(
                "agent_error",
                node="execute",
                agent_id=agent_id,
                error=str(e),
            )
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

    log_event(
        "execute_complete", node="execute",
        agents_called=agent_ids,
        agents_successful=successful_ids,
        primary_agent=primary_agent_id,
    )

    return {
        "agent_responses": list(responses),
        "primary_agent_id": primary_agent_id,
        "primary_agent_name": primary_agent_name,
        "previous_agent_responses": [],
    }


# ============================================================
# No RESPOND_DIRECT — saudacoes e off-topic
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

def _build_consolidation_prompt(
    user_text: str,
    question_type: str,
    specialist_block: str,
    regulatory_block: str,
    has_specialists: bool,
) -> str:
    """Monta o prompt de síntese do consolidador V1.

    Delega inteiramente a synthesis_rules.build_synthesis_prompt() —
    fonte única de verdade para R1–R9, instruções de formato e hierarquia
    técnico/regulatório. V1 e V2 produzem prompts idênticos para o mesmo input.
    """
    specialist_text = specialist_block.strip() if has_specialists else ""
    regulatory_text = regulatory_block.strip()
    return build_synthesis_prompt(
        question=user_text,
        question_type=question_type,
        specialist_text=specialist_text,
        regulatory_text=regulatory_text,
    )


async def consolidate(state: OrchestratorState) -> OrchestratorState:
    """Funde as respostas dos agentes em uma resposta coerente via LLM.

    Fluxo único e determinístico:
      1. Coleta evidências dos agentes bem-sucedidos (especialista + regulatório).
      2. Monta blocos de input para o LLM de síntese.
      3. Chama o LLM de síntese em TODOS os caminhos com evidência.
      4. Fallback estruturado apenas quando evidência real é zero.

    Não há atalhos de retorno antecipado sem LLM — toda resposta ao usuário
    passa pela síntese, garantindo coerência e evitando dumps de chunks.
    """
    log_event("consolidate_start", node="consolidate",
              agents_responded=len(state.get("agent_responses", [])))

    # Veio de respond_direct (saudação/off-topic): já tem final_response pronto.
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

    user_text = _get_last_user_text(state.get("messages", []))
    successful = [
        r for r in state.get("agent_responses", [])
        if r.get("success") and (r.get("response") or r.get("rag_evidence_text"))
    ]

    if not successful:
        return await _build_web_last_resort_response(
            state, user_text, trigger="no_agent_evidence",
        )

    # -----------------------------------------------------------------------
    # Coleta e filtragem de candidatos
    # -----------------------------------------------------------------------
    _factual_candidates = _build_factual_response_candidates(successful, user_text)

    # Injeta evidência regulatória hardcoded para fatos legais críticos (ex: IN 76 CCS).
    _local_regulatory_evidence = _build_local_primary_regulatory_evidence(user_text)
    if _local_regulatory_evidence and not any(
        int(c.get("agent_id", -1)) == _REGULATORY_BASELINE_ID
        for c in _factual_candidates
    ):
        _factual_candidates.append({
            "agent_id": _REGULATORY_BASELINE_ID,
            "agent_name": "Regulatorios por Pais",
            "success": True,
            "response": _local_regulatory_evidence,
            "answer_source": "local_primary_regulatory_source",
            "evidence_quality": "usable_regulatory_evidence",
            "requires_consolidation": True,
        })

    # Se não há evidência filtrada, não promovemos texto bruto/chunk para resposta.
    # O caminho correto é fallback controlado ou mensagem de evidência insuficiente.
    if not _factual_candidates:
        return await _build_web_last_resort_response(
            state, user_text, trigger="no_kb_evidence",
        )

    direct_answer = _build_primary_locked_direct_answer(user_text, _factual_candidates)
    if direct_answer:
        final_text = _postprocess_consolidated_answer(user_text, direct_answer)
        log_event(
            "consolidate_complete",
            node="consolidate",
            specialists=sum(
                1 for c in _factual_candidates
                if int(c.get("agent_id", -1)) not in _ROUTING_BASELINE_IDS
            ),
            has_regulatory=any(
                int(c.get("agent_id", -1)) == _REGULATORY_BASELINE_ID
                for c in _factual_candidates
            ),
            synthesis_path="deterministic_direct_answer",
        )
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # -----------------------------------------------------------------------
    # Separa papéis: especialista (base) vs regulatório (complemento)
    # -----------------------------------------------------------------------
    specialist_candidates = [
        r for r in _factual_candidates
        if int(r.get("agent_id", -1)) not in _ROUTING_BASELINE_IDS
    ]
    regulatory_candidate = next(
        (r for r in _factual_candidates if int(r.get("agent_id", -1)) == _REGULATORY_BASELINE_ID),
        None,
    )

    # Descarta conhecimento geral de especialistas quando há RAG real de outro especialista,
    # mas preserva sempre o regulatório como complemento (mesmo quando [CONHECIMENTO GERAL]).
    specialist_rag = [
        r for r in specialist_candidates
        if r.get("answer_source") in {"rag", "rag_evidence"}
    ]
    if specialist_rag:
        specialist_candidates = specialist_rag

    # -----------------------------------------------------------------------
    # Monta blocos de input para o LLM
    # -----------------------------------------------------------------------
    specialist_block = "".join(
        f"\n--- {r.get('agent_name', 'Especialista')} ---\n{r['response']}\n"
        for r in specialist_candidates
    )
    regulatory_block = (
        f"\n--- {regulatory_candidate.get('agent_name', 'Regulatorios')} ---\n"
        f"{regulatory_candidate['response']}\n"
        if regulatory_candidate else ""
    )

    question_type = detect_question_type(user_text)

    # -----------------------------------------------------------------------
    # Prompt de síntese — construído por função dedicada
    # -----------------------------------------------------------------------
    consolidation_prompt = _build_consolidation_prompt(
        user_text=user_text,
        question_type=question_type,
        specialist_block=specialist_block,
        regulatory_block=regulatory_block,
        has_specialists=bool(specialist_candidates),
    )

    # -----------------------------------------------------------------------
    # Chamada ao LLM de síntese (com retry automático em caso de falha)
    # -----------------------------------------------------------------------
    all_candidates_for_retry = specialist_candidates + ([regulatory_candidate] if regulatory_candidate else [])
    try:
        response_text = await _ainvoke_consolidation_with_timeout(state, consolidation_prompt)
        final_text = _postprocess_consolidated_answer(user_text, response_text)
        if _looks_like_unusable_consolidation_answer(final_text):
            final_text = await _consolidation_llm_retry(
                user_text, consolidation_prompt, all_candidates_for_retry
            )
    except Exception:
        final_text = await _consolidation_llm_retry(
            user_text, consolidation_prompt, all_candidates_for_retry
        )

    # Se a síntese falhou mesmo com evidência presente, escala para web fallback
    # antes de devolver qualquer mensagem de ausência ao usuário.
    if final_text == _ZERO_EVIDENCE_MSG or _looks_like_unusable_consolidation_answer(final_text):
        return await _build_web_last_resort_response(
            state, user_text, trigger="synthesis_failed",
        )

    # Garante que limites regulatórios numéricos críticos não sumam na síntese.
    if regulatory_candidate:
        final_text = _append_missing_regulatory_numeric_complement(
            final_text,
            user_text,
            specialist_block,
            str(regulatory_candidate.get("response", "")),
        )

    log_event(
        "consolidate_complete",
        node="consolidate",
        specialists=len(specialist_candidates),
        has_regulatory=bool(regulatory_candidate),
        synthesis_path="llm",
    )

    return {
        "final_response": final_text,
        "messages": [AIMessage(content=final_text)],
    }


# ============================================================
# Montagem e compilação do grafo
# ============================================================

def build_orchestrator_graph() -> Any:
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify)
    graph.add_node("execute", execute)
    graph.add_node("respond_direct", respond_direct)
    graph.add_node("consolidate", consolidate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route,
        {"execute": "execute", "respond_direct": "respond_direct"},
    )

    graph.add_edge("execute", "consolidate")
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
