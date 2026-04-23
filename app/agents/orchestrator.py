"""
agents/orchestrator.py â€” Orquestrador multi-agente com execuÃ§Ã£o paralela

Fluxo do grafo:
  classify â†’ route â†’ execute (paralelo) â†’ consolidate â†’ END
                â†˜ respond_direct â†’ consolidate â†’ END

Agentes 0 (Base Geral) e 3 (Regulatórios) são SEMPRE incluídos
para qualquer pergunta sobre laticÃ­nios â€” o classificador Ã© instruÃ­do
a retorná-los obrigatoriamente.

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
from typing import Any, Dict, List, Optional, Annotated, Tuple
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.config import (
    LLM_MODEL,
    CLASSIFIER_TEMPERATURE,
    CONSOLIDATION_TEMPERATURE,
    DIRECT_TEMPERATURE,
    ORCHESTRATOR_FASTPATH,
    CLASSIFICATION_CACHE_SIZE,
)
from app.agents.prompts import get_orchestrator_prompt
from app.agents.agent_config import AGENTS, get_agent_by_id
from app.agents.base_agent import get_agent_graph

# Tempo máximo de espera por agente (segundos)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "12"))
_SPECIALISTS_DESC = "".join(
    f"  {agent['agent_id']} = {agent['name']}\n"
    for agent in AGENTS
    if agent["agent_id"] not in (0, 3)
)

_CLASSIFICATION_CACHE: "OrderedDict[str, List[int]]" = OrderedDict()
_MAX_CLASSIFICATION_CACHE = max(0, CLASSIFICATION_CACHE_SIZE)
_GREETINGS = {
    "oi", "ola", "olá", "bom dia", "boa tarde", "boa noite",
    "e ai", "e aí", "tudo bem", "blz", "beleza",
}
_DAIRY_TERMS = {
    "leite", "lacteo", "laticinio", "laticinios", "queijo",
    "iogurte", "fermentado", "ricota", "requeijao", "mussarela",
    "coalhada", "soro", "pasteurizacao", "ccs", "cbt", "rtiq", "rdc",
}
_QUALITY_LAB_TERMS = {
    "laboratorio", "analise", "analitico", "amostra", "coleta",
    "controle de qualidade", "qualidade", "bpl", "boas praticas",
    "incendio", "emergencia", "evacuacao", "extintor", "epi", "brigada",
}

_ROUTING_BASELINE_IDS = [0, 3]
_ROUTING_CONFIDENCE_THRESHOLDS = {
    "high": 0.80,
    "medium": 0.55,
}
_SPECIALISTS_PER_BUCKET = {
    "high": 1,
    "medium": 2,
    "low": 3,
}
_FALLBACK_MAX_ATTEMPTS = 1
_FALLBACK_EXTRA_SPECIALISTS = {
    "high": 1,
    "medium": 2,
    "low": 2,
}
# Mapa de vizinhanca entre especialistas (extraido da taxonomia day1).
_NEAREST_SPECIALIST_MAP: Dict[int, List[int]] = {
    1: [6, 3, 5],
    2: [4, 6, 5],
    3: [1, 4, 6],
    4: [2, 3, 5],
    5: [1, 4, 6],
    6: [1, 2, 4],
}


def _load_taxonomy_nearest_map() -> Dict[int, List[int]]:
    """Carrega vizinhanca de dominios a partir da taxonomia (se disponivel)."""
    taxonomy_path = Path("docs/orchestrator/day1/AGENT_ROUTING_TAXONOMY.yaml")
    if not taxonomy_path.exists():
        return dict(_NEAREST_SPECIALIST_MAP)
    try:
        import yaml  # type: ignore
    except Exception:
        return dict(_NEAREST_SPECIALIST_MAP)

    try:
        raw = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8")) or {}
        agents = raw.get("agents", {}) or {}
        loaded: Dict[int, List[int]] = {}
        for k, info in agents.items():
            try:
                aid = int(k)
            except (TypeError, ValueError):
                continue
            if aid in _ROUTING_BASELINE_IDS:
                continue
            confusion = info.get("confusion_with", []) if isinstance(info, dict) else []
            near: List[int] = []
            for raw_id in confusion or []:
                try:
                    nid = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if nid not in _ROUTING_BASELINE_IDS and 0 <= nid <= 6 and nid not in near:
                    near.append(nid)
            if near:
                loaded[aid] = near
        if loaded:
            return loaded
    except Exception:
        pass
    return dict(_NEAREST_SPECIALIST_MAP)


_NEAREST_SPECIALIST_MAP = _load_taxonomy_nearest_map()


def _choose_primary_agent_id(agent_ids: List[int]) -> int:
    """Escolhe o agente principal para exibicao ao cliente.

    Regra:
    1) Preferir especialistas de dominio (1..6, exceto 3 se houver 1..2/4..6).
    2) Se nao houver especialista de dominio, preferir 3 (Regulatorios).
    3) Por fim, usar 0 (Base Geral).
    """
    if not agent_ids:
        return 0

    # Especialistas de dominio (na ordem de relevancia original)
    for aid in agent_ids:
        if aid not in (0, 3):
            return aid

    if 3 in agent_ids:
        return 3
    if 0 in agent_ids:
        return 0
    return agent_ids[0]


def _sanitize_math_for_ui(text: str) -> str:
    """Converte trechos matematicos em LaTeX para texto simples amigavel ao front."""
    if not text:
        return text

    out = str(text)
    # Delimitadores comuns de math mode
    out = re.sub(r"\\\[(.*?)\\\]", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\\\((.*?)\\\)", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$\$(.*?)\$\$", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\$(.*?)\$", r"\1", out, flags=re.DOTALL)

    # Comandos latex usuais em respostas de calculo
    out = out.replace(r"\times", "x")
    out = out.replace(r"\cdot", "x")
    out = out.replace(r"\,", " ")
    out = out.replace("\\n", "\n")
    out = out.replace("\t", " ")
    out = re.sub(r"\\text\{([^}]*)\}", r"\1", out)

    # Limpeza de comandos LaTeX residuais
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("{", "").replace("}", "")

    # Remove wrappers visuais do tipo [ ... ] quando estiverem sozinhos na linha
    cleaned_lines = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]") and len(s) >= 2:
            s = s[1:-1].strip()
        cleaned_lines.append(s if s else line.strip())
    out = "\n".join(cleaned_lines)

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _dedupe_paragraphs(text: str) -> str:
    """Remove paragrafos duplicados mantendo a primeira ocorrencia."""
    if not text:
        return text
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen = set()
    kept: List[str] = []
    for p in parts:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(p)
    return "\n\n".join(kept).strip()


def _normalize_mul_symbols(text: str) -> str:
    return (
        text.replace("×", "x")
        .replace("*", "x")
        .replace("X", "x")
    )


def _enforce_dornic_canonical_formula(user_text: str, text: str) -> str:
    """Garante forma canonica da formula Dornic quando a pergunta for desse tema.

    Formula canonica (IN 68): Acidez (Dornic) = V x f x 0,9 x 10
    """
    if not text:
        return text
    q = _normalize_text(user_text)
    if "dornic" not in q and not ("acidez" in q and "titul" in q):
        return text

    out = text
    # Corrige forma incompleta observada em consolidacoes conflitantes.
    patterns = [
        r"Acidez\s*\(?°?\s*D(?:ornic)?\)?\s*=\s*V\s*[x\*]\s*f\s*[x\*]\s*10",
        r"Acidez\s*\('?\s*Dornic\s*'?\)\s*=\s*V\s*[x\*]\s*f\s*[x\*]\s*10",
    ]
    for pat in patterns:
        out = re.sub(
            pat,
            "Acidez (Dornic) = V x f x 0,9 x 10",
            out,
            flags=re.IGNORECASE,
        )

    # Se houver duas secoes de formula Dornic, mantem so a primeira ocorrencia.
    lines = out.splitlines()
    new_lines: List[str] = []
    seen_formula_line = False
    for line in lines:
        ln_norm = _normalize_mul_symbols(line)
        is_dornic_formula = (
            "acidez" in ln_norm.lower()
            and "dornic" in ln_norm.lower()
            and "=" in ln_norm
            and "v" in ln_norm.lower()
            and "f" in ln_norm.lower()
            and "10" in ln_norm
        )
        if is_dornic_formula:
            if seen_formula_line:
                continue
            seen_formula_line = True
        new_lines.append(line)
    out = "\n".join(new_lines).strip()
    return out


def _postprocess_consolidated_answer(user_text: str, text: str) -> str:
    out = _sanitize_math_for_ui(text or "")
    out = _enforce_dornic_canonical_formula(user_text, out)
    out = _dedupe_paragraphs(out)
    out = _sanitize_math_for_ui(out)
    return out


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _strip_profile_suffix(text: str) -> str:
    if "\n[Perfil" in text:
        return text.split("\n[Perfil", 1)[0]
    return text


def _is_objective_question(text: str) -> bool:
    q = _normalize_text(text)
    if not q:
        return False
    patterns = (
        r"^(quem e|qual e|quais sao|quanto e|onde fica|onde e|quando|como se chama)\b",
        r"^(quem|qual|quais|quanto|onde|quando)\b",
    )
    return any(re.search(p, q) for p in patterns)


def _looks_uncertain(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return True
    uncertainty_markers = (
        "nao encontrei informacao suficiente",
        "nao ha evidencia suficiente",
        "faltam evidencias",
        "pode ser",
        "talvez",
        "recomenda-se verificar",
        "aconselhavel verificar",
        "consultar fontes adicionais",
        "com o meu conhecimento atual",
        "com o seu conhecimento atual",
    )
    return any(marker in t for marker in uncertainty_markers)


def _strip_uncertainty_tail(text: str) -> str:
    """Remove cauda de ressalva genérica quando houver fato já respondido.

    Ex.: "X é Y. No entanto, ..." -> "X é Y."
    """
    if not text:
        return ""
    out = str(text).strip()

    # Corta no início de conectores de ressalva.
    m = re.search(r"\b(No entanto|Por[ée]m|Contudo)\b", out, flags=re.IGNORECASE)
    if m:
        out = out[: m.start()].strip()

    # Corta no início de frases de falta de evidência genérica.
    m2 = re.search(
        r"\b(a base atual n[ãa]o trouxe|nao trouxe informacao suficiente|"
        r"faltam evidencias|com o meu conhecimento atual|"
        r"recomenda-se verificar|aconselh[aá]vel verificar)\b",
        out,
        flags=re.IGNORECASE,
    )
    if m2:
        out = out[: m2.start()].strip()

    # Limpeza de pontuação residual
    out = re.sub(r"[;,:\-–—]+$", "", out).strip()
    return out


def _extract_factual_candidate(text: str) -> Optional[str]:
    """Extrai parte factual útil de uma resposta mista (fato + ressalva)."""
    cleaned = _sanitize_math_for_ui(text or "")
    if not cleaned:
        return None
    head = _strip_uncertainty_tail(cleaned)
    if not head:
        return None
    if len(head.split()) < 3:
        return None
    if _looks_uncertain(head):
        return None
    return head


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


def _build_keyword_sets() -> Dict[int, set]:
    keyword_sets: Dict[int, set] = {}
    for agent in AGENTS:
        aid = agent["agent_id"]
        if aid in (0, 3):
            continue
        raw_keywords = agent.get("keywords", []) or []
        words = {
            _normalize_text(str(k))
            for k in raw_keywords
            if isinstance(k, str) and len(_normalize_text(k)) >= 4
        }
        keyword_sets[aid] = words
    return keyword_sets


_SPECIALIST_KEYWORDS = _build_keyword_sets()


def _cache_get(cache_key: str) -> Optional[List[int]]:
    if _MAX_CLASSIFICATION_CACHE <= 0:
        return None
    cached = _CLASSIFICATION_CACHE.get(cache_key)
    if cached is None:
        return None
    _CLASSIFICATION_CACHE.move_to_end(cache_key)
    return list(cached)


def _cache_set(cache_key: str, agent_ids: List[int]) -> None:
    if _MAX_CLASSIFICATION_CACHE <= 0:
        return
    _CLASSIFICATION_CACHE[cache_key] = list(agent_ids)
    _CLASSIFICATION_CACHE.move_to_end(cache_key)
    while len(_CLASSIFICATION_CACHE) > _MAX_CLASSIFICATION_CACHE:
        _CLASSIFICATION_CACHE.popitem(last=False)


def _looks_like_greeting_only(text_norm: str) -> bool:
    if not text_norm:
        return False
    if text_norm in _GREETINGS:
        return True
    if len(text_norm.split()) <= 4 and any(text_norm.startswith(g) for g in _GREETINGS):
        return True
    return False


def _contains_dairy_signal(text_norm: str) -> bool:
    if any(term in text_norm for term in _DAIRY_TERMS):
        return True
    if re.search(r"\b(in|rdc|rtiq)\s*\d{1,4}\b", text_norm):
        return True
    return False


def _rule_based_route(user_text: str) -> Optional[List[int]]:
    text = _normalize_text(_strip_profile_suffix(user_text))
    if not text:
        return []

    if _looks_like_greeting_only(text):
        return []

    # Perguntas de laboratorio/controle de qualidade (inclui seguranca em lab)
    # devem consultar Qualidade do Leite mesmo sem termos "dairy" explicitos.
    if any(term in text for term in _QUALITY_LAB_TERMS):
        return [0, 3, 4]

    specialist_scores: List[tuple[int, int]] = []
    for aid, keywords in _SPECIALIST_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw and kw in text)
        if score > 0:
            specialist_scores.append((aid, score))

    specialist_scores.sort(key=lambda x: x[1], reverse=True)

    # Alta confiança: 2+ keywords do mesmo especialista.
    high_conf = [aid for aid, score in specialist_scores if score >= 2]
    if high_conf:
        ids = [0, 3] + high_conf[:3]
        return ids

    # Domínio dairy evidente, mas sem especialista forte -> baseline [0, 3].
    if _contains_dairy_signal(text):
        return [0, 3]

    # Baixa confiança: deixar o classificador LLM decidir.
    return None


def _sanitize_agent_ids(raw_ids: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for aid in raw_ids:
        if 0 <= aid <= 6 and aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def _clamp_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = 0.50
    return max(0.0, min(1.0, conf))


def _confidence_to_bucket(confidence: float) -> str:
    if confidence >= _ROUTING_CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if confidence >= _ROUTING_CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _estimate_fastpath_confidence(route_text: str, agent_ids: List[int]) -> float:
    if not agent_ids:
        return 0.98
    if any(aid not in _ROUTING_BASELINE_IDS for aid in agent_ids):
        return 0.92
    if _contains_dairy_signal(_normalize_text(route_text)):
        return 0.72
    return 0.60


def _apply_dairy_hard_constraints(route_text: str, agent_ids: List[int]) -> List[int]:
    if not agent_ids:
        return []
    text_norm = _normalize_text(route_text)
    if not _contains_dairy_signal(text_norm):
        return agent_ids
    out = list(agent_ids)
    # 0 e 3 sao obrigatorios para perguntas com sinal de lacteos.
    if 0 not in out:
        out.insert(0, 0)
    if 3 not in out:
        insert_at = 1 if 0 in out else 0
        out.insert(insert_at, 3)
    # Mantem ordenacao com baseline no topo.
    baseline = [aid for aid in _ROUTING_BASELINE_IDS if aid in out]
    tail = [aid for aid in out if aid not in baseline]
    return _sanitize_agent_ids(baseline + tail)


def _build_execution_plan(
    route_text: str,
    chosen_ids: List[int],
    alternatives: Optional[List[int]],
    bucket: str,
) -> List[int]:
    chosen = _sanitize_agent_ids(chosen_ids)
    alts = _sanitize_agent_ids(alternatives or [])

    if not chosen:
        return []

    text_norm = _normalize_text(route_text)
    has_dairy_signal = _contains_dairy_signal(text_norm)
    has_specialist = any(aid not in _ROUTING_BASELINE_IDS for aid in chosen)
    has_baseline_pair = all(aid in chosen for aid in _ROUTING_BASELINE_IDS)
    is_dairy_route = has_dairy_signal or has_specialist or has_baseline_pair

    if is_dairy_route:
        # Se a classificação já apontou domínio lácteo (baseline e/ou especialista),
        # mantém o par obrigatório 0+3 no plano de execução.
        if 0 not in chosen:
            chosen.insert(0, 0)
        if 3 not in chosen:
            insert_at = 1 if 0 in chosen else 0
            chosen.insert(insert_at, 3)
        chosen = _sanitize_agent_ids(chosen)

        base = [aid for aid in _ROUTING_BASELINE_IDS if aid in chosen]
        specialists = [aid for aid in chosen if aid not in _ROUTING_BASELINE_IDS]

        # Completa especialistas com alternativas relevantes.
        for aid in alts:
            if aid not in _ROUTING_BASELINE_IDS and aid not in specialists:
                specialists.append(aid)

        max_specialists = _SPECIALISTS_PER_BUCKET.get(bucket, 3)

        # Se houver especialista na classificação/alternativas, garante ao menos 1.
        if specialists and max_specialists < 1:
            max_specialists = 1

        plan = base + specialists[:max_specialists]
    else:
        max_agents = _SPECIALISTS_PER_BUCKET.get(bucket, 3)
        merged = chosen + [aid for aid in alts if aid not in chosen]
        plan = merged[:max_agents]

    return _sanitize_agent_ids(plan)[:5]


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
    return [aid for aid in candidates if aid not in already_planned]


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

    if _has_weak_or_conflicting_evidence(responses):
        return True, "weak_or_conflicting_evidence"

    return False, "sufficient_evidence"


# ============================================================
# Estado do orquestrador
# ============================================================

class OrchestratorState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    chosen_agent_ids: List[int]
    chosen_agent_names: List[str]
    execution_plan: List[int]
    agent_responses: List[Dict[str, Any]]
    final_response: str
    primary_agent_id: int
    primary_agent_name: str
    user_profile: Optional[Dict[str, Any]]
    routing_confidence: float
    routing_bucket: str
    routing_reason: str
    routing_alternatives: List[int]
    fallback_used: bool
    fallback_attempts: int
    fallback_trigger: str
    previous_agent_responses: List[Dict[str, Any]]


# ============================================================
# Schema de classificação
# ============================================================

class ClassificationResult(BaseModel):
    """
    agent_ids: Lista de IDs relevantes, ordenada por relevância.
               Deve SEMPRE incluir 0 e 3 para perguntas de laticínios.
               [] apenas para saudações ou tópicos fora do setor.
    confidence: Grau de confiança do roteamento (0.0 a 1.0).
    reason/reasoning: Justificativa breve (para debug).
    alternatives: IDs alternativos relevantes para fallback/planner.
    """
    agent_ids: List[int]
    confidence: float = 0.50
    reason: str = ""
    alternatives: List[int] = []
    reasoning: str = ""


# ============================================================
# Lazy init dos modelos
# ============================================================

_classifier_model = None
_consolidation_model = None
_direct_model = None


def _get_classifier():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = ChatOpenAI(model=LLM_MODEL, temperature=CLASSIFIER_TEMPERATURE).with_structured_output(
            ClassificationResult
        )
    return _classifier_model


def _get_consolidation_model():
    global _consolidation_model
    if _consolidation_model is None:
        _consolidation_model = ChatOpenAI(model=LLM_MODEL, temperature=CONSOLIDATION_TEMPERATURE)
    return _consolidation_model


def _get_direct_model():
    global _direct_model
    if _direct_model is None:
        _direct_model = ChatOpenAI(model=LLM_MODEL, temperature=DIRECT_TEMPERATURE)
    return _direct_model


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
    confidence = _clamp_confidence(confidence)
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

    primary_agent_id = _choose_primary_agent_id(execution_plan or sanitized_ids)
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
    }


async def classify(state: OrchestratorState) -> OrchestratorState:
    """Identifica quais agentes devem ser consultados.

    Agentes 0 e 3 são SEMPRE obrigatórios para qualquer pergunta
    de laticÃ­nios â€” o prompt instrui o LLM explicitamente.
    """
    messages = state.get("messages", [])
    user_text = _get_last_user_text(messages)

    if not user_text:
        return _build_classification_state(route_text="", agent_ids=[], confidence=1.0, reason="mensagem_vazia")

    route_text = _strip_profile_suffix(user_text)
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

REGRA OBRIGATÃ“RIA:
- Para QUALQUER pergunta relacionada a laticínios (produtos, processos,
  ingredientes, fabricantes, distribuidores, equipamentos, normas, qualidade,
  defeitos, formulação, legislação), SEMPRE inclua os agentes 0 e 3 na lista.
- Agente 0 (Base Geral Dairy): glossário, produtos, fabricantes, ingredientes,
  distribuidores, equipamentos â€” base de conhecimento transversal.
- Agente 3 (Regulatórios por País): normas, legislação, requisitos legais.

ESPECIALISTAS (adicione apenas se a pergunta for claramente desse domínio):
{_SPECIALISTS_DESC}
 FORMATO DA RESPOSTA:
 - SaudaÃ§Ã£o / off-topic (sem relaÃ§Ã£o com laticÃ­nios) â†’ []
 - Pergunta de laticÃ­nios sem especialidade clara â†’ [0, 3]
 - Pergunta com especialidade clara â†’ [0, 3, X]
 - Pergunta com mÃºltiplas especialidades â†’ [0, 3, X, Y] (mÃ¡x 5 IDs)
 - Ordene por relevância: o agente mais relevante primeiro.
 
 ALÃ‰M DOS IDs, informe:
 - confidence: nÃºmero entre 0.0 e 1.0
 - reason: justificativa curta
 - alternatives: IDs alternativos relevantes (sem repetir os principais)
 """

    classifier = _get_classifier()
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

    if not agent_ids:
        return _build_classification_state(
            route_text=route_text,
            agent_ids=[],
            confidence=confidence,
            reason=reason or "sem_dominio_relevante",
            alternatives=alternatives,
        )

    _cache_set(cache_key, agent_ids)
    return _build_classification_state(
        route_text=route_text,
        agent_ids=agent_ids,
        confidence=confidence,
        reason=reason or "classificacao_llm",
        alternatives=alternatives,
    )

# ============================================================
# Roteamento condicional
# ============================================================

def route(state: OrchestratorState) -> str:
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
# NÃ³ EXECUTE â€” execuÃ§Ã£o paralela
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

    if not user_text:
        return {"agent_responses": []}

    async def call_one(agent_id: int, agent_name: str) -> Dict[str, Any]:
        try:
            graph = get_agent_graph(agent_id)
            result = await asyncio.wait_for(
                graph.ainvoke({"messages": [HumanMessage(content=user_text)]}),
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
    primary_agent_id = _choose_primary_agent_id(candidate_ids)
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
# NÃ³ FALLBACK_RECLASSIFY â€” segunda passada inteligente
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
# NÃ³ RESPOND_DIRECT â€” saudaÃ§Ãµes e off-topic
# ============================================================

async def respond_direct(state: OrchestratorState) -> OrchestratorState:
    """Resposta direta para saudações e mensagens off-topic (sem RAG)."""
    user_text = _get_last_user_text(state.get("messages", []))

    system = (
        "Voce e o assistente geral do Dairy AI (DairyApp), especializado em tecnologia "
        "de laticinios. Em saudacoes e primeira interacao, apresente-se de forma curta "
        "como Dairy AI e diga em uma frase como pode ajudar. Depois disso, evite repetir "
        "apresentacoes e va direto ao ponto. Quando pertinente, sugira perguntas tecnicas "
        "sobre queijos, fermentados, regulatorios, qualidade do leite, diagnostico de "
        "defeitos ou formulacao. Responda em portugues brasileiro."
    )

    response = await _get_direct_model().ainvoke([
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
# NÃ³ CONSOLIDATE â€” fusÃ£o das respostas
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

    if not successful:
        final_text = (
            "Não foi possível obter uma resposta no momento. "
            "Por favor, tente reformular sua pergunta."
        )
        user_text = _get_last_user_text(state.get("messages", []))
        final_text = _postprocess_consolidated_answer(user_text, final_text)
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # 1 agente: repassa direto (econômico)
    if len(successful) == 1:
        single_fact = _extract_factual_candidate(str(successful[0]["response"]))
        user_text = _get_last_user_text(state.get("messages", []))
        final_text = _postprocess_consolidated_answer(
            user_text,
            single_fact or successful[0]["response"],
        )
        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }

    # 2+ agentes: consolida com LLM
    user_text = _get_last_user_text(state.get("messages", []))

    # Em perguntas objetivas, quando houver um único especialista com
    # resposta factual direta, devolve essa resposta sem adicionar ressalvas.
    preferred = _prefer_direct_fact_response(user_text, successful)
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

    responses_text = "".join(
        f"\n--- {r['agent_name']} ---\n{r['response']}\n"
        for r in successful
    )

    consolidation_prompt = (
        "Você é o assistente geral do DairyApp AI. Recebeu respostas de múltiplos "
        "especialistas para a pergunta do usuário. Sua tarefa:\n"
        "- Fundir em UMA resposta coerente e completa\n"
        "- Preservar TODOS os dados técnicos (temperaturas, pHs, normas, prazos)\n"
        "- Não perder informação de nenhum especialista\n"
        "- NÃO adicionar fatos novos que não estejam nas respostas dos especialistas\n"
        "- Se houver lacuna de evidência, diga explicitamente que a base atual não trouxe informação suficiente\n"
        "- Evite misturar produtos/rotinas diferentes sem indicar a diferença\n"
        "- Não mencionar que consultou múltiplos agentes internos\n"
        "- Tom técnico e profissional em português brasileiro\n\n"
        f"PERGUNTA: {user_text}\n\n"
        f"RESPOSTAS DOS ESPECIALISTAS:{responses_text}\n"
        "Resposta unificada:"
    )

    try:
        response = await _get_consolidation_model().ainvoke(
            [HumanMessage(content=consolidation_prompt)]
        )
        final_text = _postprocess_consolidated_answer(user_text, response.content or "")
    except Exception:
        final_text = _postprocess_consolidated_answer(
            user_text,
            "\n\n".join(r["response"] for r in successful),
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
    graph.add_node("fallback_reclassify", fallback_reclassify)
    graph.add_node("respond_direct", respond_direct)
    graph.add_node("consolidate", consolidate)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route,
        {"execute": "execute", "respond_direct": "respond_direct"},
    )

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

