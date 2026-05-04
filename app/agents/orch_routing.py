"""
orch_routing.py — Motor de roteamento do fast-path e construção do plano de execução.

Responsabilidade: dado um texto de usuário, produzir uma lista ordenada de
agentes para executar e um nível de confiança associado.

Estado atual dos agentes:
  Ativos (com KB ingerida):
    1 — Tecnologia de Queijos   (especialista)
    3 — Regulatórios por País   (baseline de todo laticínio)

  Aguardando ingestão de KB (não roteados):
    0 — Base Geral Dairy
    2 — Fermentados
    4 — Qualidade do Leite
    5 — Diagnóstico de Defeitos
    6 — Formulação e Desenvolvimento

Para ativar um agente: remova-o de _AGENTS_WITHOUT_KB e adicione suas
funções de sinal em orch_signals.py.
"""

import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import CLASSIFICATION_CACHE_SIZE
from app.agents.agent_config import AGENTS
from app.agents.orch_text import (
    _normalize_text,
    _is_objective_question,
    _strip_profile_suffix,
)
from app.agents.orch_signals import (
    _INTENT_PATTERNS_BY_AGENT,
    _LOW_PRECISION_KEYWORDS,
    _HINT_NOISE_TERMS,
    _HINT_NOISE_TOKENS,
    _SPECIALIST_STRONG_HINTS_DEFAULT,
    _contains_dairy_signal,
    _contains_any_phrase,
    _looks_like_greeting_only,
    _is_strong_regulatory_signal,
    _is_strong_cheese_signal,
    _is_labeling_regulatory_signal,
    _is_legal_requirement_regulatory_signal,
    _is_normative_regulatory_signal,
)


# ---------------------------------------------------------------------------
# Constantes de agentes
# ---------------------------------------------------------------------------

# Agente regulatório presente em toda pergunta de laticínios.
_REGULATORY_BASELINE_ID: int = 3

# Mantém compatibilidade com código que itera sobre baseline_ids.
_ROUTING_BASELINE_IDS: List[int] = [_REGULATORY_BASELINE_ID]

# Agentes sem KB: removidos de qualquer rota. Remova daqui ao ingerir KB.
_AGENTS_WITHOUT_KB: Set[int] = {0, 2, 4, 5, 6}

# Número máximo de especialistas por bucket de confiança.
_SPECIALISTS_PER_BUCKET: Dict[str, int] = {
    "high": 1,
    "medium": 2,
    "low": 2,
}

# Limites de confiança para classificação em buckets.
_ROUTING_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "high": 0.86,
    "medium": 0.55,
}

# Número máximo de tentativas de fallback de roteamento.
_FALLBACK_MAX_ATTEMPTS: int = 1

# Especialistas extras permitidos por bucket no fallback.
_FALLBACK_EXTRA_SPECIALISTS: Dict[str, int] = {
    "high": 1,
    "medium": 1,
    "low": 2,
}

# Rótulos de domínio para o nó de clarificação (apenas agentes ativos).
_AGENT_DOMAIN_LABELS: Dict[int, str] = {
    1: "fabricação e tecnologia de queijos",
}


# ---------------------------------------------------------------------------
# Mapa de vizinhança entre especialistas (para fallback de reclassificação)
# ---------------------------------------------------------------------------

# Mapa padrão: com apenas agentes 1 e 3 ativos, não há vizinhos úteis para Agent 1
# (Agent 3 é baseline e sempre incluído). Mantém a estrutura para extensão futura.
_NEAREST_SPECIALIST_MAP: Dict[int, List[int]] = {
    1: [],   # sem vizinhos especialistas ativos no momento
    3: [],   # baseline regulatório; não é expandido via fallback
}


def _load_taxonomy_nearest_map() -> Dict[int, List[int]]:
    """Carrega mapa de vizinhança de domínios a partir da taxonomia (se disponível).

    Filtra automaticamente agentes sem KB para evitar expansão de fallback
    em agentes sem conteúdo.
    """
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
            if aid in _ROUTING_BASELINE_IDS or aid in _AGENTS_WITHOUT_KB:
                continue
            confusion = info.get("confusion_with", []) if isinstance(info, dict) else []
            near: List[int] = []
            for raw_id in (confusion or []):
                try:
                    nid = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if (
                    nid not in _ROUTING_BASELINE_IDS
                    and nid not in _AGENTS_WITHOUT_KB
                    and 0 <= nid <= 6
                    and nid not in near
                ):
                    near.append(nid)
            if near:
                loaded[aid] = near
        return loaded if loaded else dict(_NEAREST_SPECIALIST_MAP)
    except Exception:
        return dict(_NEAREST_SPECIALIST_MAP)


_NEAREST_SPECIALIST_MAP = _load_taxonomy_nearest_map()


# ---------------------------------------------------------------------------
# Tabelas de agentes (para buscas gerais e fallback de índice)
# ---------------------------------------------------------------------------

_AGENT_TABLE_BY_ID: Dict[int, str] = {
    int(agent["agent_id"]): str(agent.get("table_name", ""))
    for agent in AGENTS
    if str(agent.get("table_name", "")).strip()
}

_ALL_AGENT_TABLES: List[str] = []
for _a in AGENTS:
    _t = str(_a.get("table_name", "")).strip()
    if _t and _t not in _ALL_AGENT_TABLES:
        _ALL_AGENT_TABLES.append(_t)


# ---------------------------------------------------------------------------
# Cache de classificação (LRU em memória)
# ---------------------------------------------------------------------------

_CLASSIFICATION_CACHE: "OrderedDict[str, List[int]]" = OrderedDict()
_MAX_CLASSIFICATION_CACHE: int = max(0, CLASSIFICATION_CACHE_SIZE)


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


# ---------------------------------------------------------------------------
# Hints fortes de especialistas
# ---------------------------------------------------------------------------

def _load_specialist_strong_hints() -> Dict[int, set]:
    """Carrega e mescla hints fortes do Agente 1 a partir do ROUTING_SPECIALIST_HINTS.yaml.

    Aplica filtros de qualidade: aceita apenas hints multi-palavra ou presentes
    nos defaults, descartando termos de baixa precisão e ruído.
    """
    def _norm(value: str) -> str:
        v = (value or "").lower().strip()
        v = unicodedata.normalize("NFKD", v)
        v = "".join(ch for ch in v if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", v)

    # Tokens de keywords do Agente 1 para validação de relevância
    agent_1_tokens: set = set()
    for agent in AGENTS:
        if agent.get("agent_id") != 1:
            continue
        for raw_kw in (agent.get("keywords", []) or []):
            kw_norm = _norm(str(raw_kw))
            for tk in kw_norm.split():
                if len(tk) >= 4 and tk not in _LOW_PRECISION_KEYWORDS:
                    agent_1_tokens.add(tk)

    # Inicializa com defaults do YAML (alta confiança)
    merged: Dict[int, set] = {
        1: {_norm(str(h)) for h in _SPECIALIST_STRONG_HINTS_DEFAULT.get(1, set()) if str(h).strip()}
    }

    hints_path = Path("docs/orchestrator/day1/ROUTING_SPECIALIST_HINTS.yaml")
    if not hints_path.exists():
        return merged

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(hints_path.read_text(encoding="utf-8")) or {}
        specialists = raw.get("specialists", {}) or {}
        info_1 = specialists.get(1) or specialists.get("1") or {}
        if not isinstance(info_1, dict):
            return merged

        hints = info_1.get("strong_hints_normalized", [])
        if not isinstance(hints, list):
            return merged

        default_bucket = merged[1]
        bucket = merged[1]

        for hint in hints:
            normalized = _norm(str(hint))
            if not normalized:
                continue
            if normalized in _HINT_NOISE_TERMS:
                continue
            if any(ch.isdigit() for ch in normalized):
                continue
            if any(tok in _HINT_NOISE_TOKENS for tok in normalized.split()):
                continue
            # Aceita apenas hints multi-palavra ou já presentes nos defaults
            if normalized not in default_bucket and " " not in normalized:
                continue
            if normalized not in default_bucket and len(normalized.split()) > 2:
                continue
            if normalized in _LOW_PRECISION_KEYWORDS:
                continue
            # Valida relevância por tokens de keyword
            if normalized not in default_bucket:
                tokens = [t for t in normalized.split() if len(t) >= 4]
                if tokens and agent_1_tokens and not any(t in agent_1_tokens for t in tokens):
                    continue
            bucket.add(normalized)

    except Exception:
        pass

    return merged


_SPECIALIST_STRONG_HINTS: Dict[int, set] = _load_specialist_strong_hints()


# ---------------------------------------------------------------------------
# Keywords por especialista (scoring do fast-path)
# ---------------------------------------------------------------------------

def _build_keyword_sets() -> Dict[int, set]:
    """Constrói conjuntos de keywords normalizadas para o Agente 1."""
    keyword_sets: Dict[int, set] = {}
    for agent in AGENTS:
        aid = agent["agent_id"]
        if aid in _AGENTS_WITHOUT_KB or aid == _REGULATORY_BASELINE_ID:
            continue
        raw_keywords = agent.get("keywords", []) or []
        words = {
            _normalize_text(str(k))
            for k in raw_keywords
            if isinstance(k, str) and len(_normalize_text(k)) >= 4
        }
        keyword_sets[aid] = words
    return keyword_sets


_SPECIALIST_KEYWORDS: Dict[int, set] = _build_keyword_sets()


def _contains_keyword(text_norm: str, keyword_norm: str) -> bool:
    """Match por fronteira de palavra para evitar falso-positivo por substring."""
    if not keyword_norm:
        return False
    pattern = rf"(?<!\w){re.escape(keyword_norm)}(?!\w)"
    return re.search(pattern, text_norm) is not None


def _keyword_weight(aid: int, keyword_norm: str) -> int:
    """Peso de uma keyword: 3=hint forte, 2=multi-palavra ou termo longo, 1=padrão, 0=ruído."""
    if keyword_norm in _SPECIALIST_STRONG_HINTS.get(aid, set()):
        return 3
    if keyword_norm in _LOW_PRECISION_KEYWORDS:
        return 0
    if " " in keyword_norm or len(keyword_norm) >= 9:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Sanitização de IDs de agentes
# ---------------------------------------------------------------------------

def _sanitize_agent_ids(raw_ids: List[int]) -> List[int]:
    """Remove duplicatas, IDs inválidos e agentes sem KB. Preserva a ordem."""
    seen: set = set()
    out: List[int] = []
    for aid in raw_ids:
        if 0 <= aid <= 6 and aid not in seen and aid not in _AGENTS_WITHOUT_KB:
            seen.add(aid)
            out.append(aid)
    return out


# ---------------------------------------------------------------------------
# Fast-path determinístico (sem LLM)
# ---------------------------------------------------------------------------

def _rule_based_route(user_text: str) -> Optional[List[int]]:
    """Entry-point público do fast-path. Retorna lista de agentes ou None."""
    return _rule_based_route_impl(user_text)


def _rule_based_route_impl(user_text: str) -> Optional[List[int]]:
    """Motor de regras hierárquicas para agentes 1 e 3.

    Precedência (da mais específica para a mais genérica):
      1. Saudação/off-topic → []
      2. Requisito legal mínimo → [3]
      3. Contexto normativo explícito → [3] ou [1, 3]
      4. Rotulagem/denominação → [3] ou [1, 3]
      5. Sinal regulatório forte → [3] ou [1, 3]
      6. Sinal forte de queijo → [1, 3]
      7. Padrões regex de intenção → [1, 3] ou [3]
      8. Scoring de keywords do Agente 1 → [1, 3]
      9. Dairy genérico → [3] (baseline seguro)
     10. Baixa confiança → None (delega ao LLM)
    """
    text = _normalize_text(_strip_profile_suffix(user_text))
    if not text:
        return []

    if _looks_like_greeting_only(text):
        return []

    # Requisito mínimo legal → puramente regulatório, sem especialistas
    if _is_legal_requirement_regulatory_signal(text):
        return _sanitize_agent_ids([3])

    # Contexto normativo explícito + domínio de queijo → ambos
    if _is_normative_regulatory_signal(text):
        if _is_strong_cheese_signal(text):
            return _sanitize_agent_ids([1, 3])
        return _sanitize_agent_ids([3])

    # Rotulagem/denominação de produto lácteo
    if _is_labeling_regulatory_signal(text):
        if _is_strong_cheese_signal(text):
            return _sanitize_agent_ids([1, 3])
        return _sanitize_agent_ids([3])

    # Sinal regulatório forte
    if _is_strong_regulatory_signal(text):
        if _is_strong_cheese_signal(text):
            return _sanitize_agent_ids([1, 3])
        return _sanitize_agent_ids([3])

    # Sinal forte de tecnologia de queijo
    if _is_strong_cheese_signal(text):
        return _sanitize_agent_ids([1, 3])

    # Padrões de intenção de alta precisão (regex)
    for aid, patterns in _INTENT_PATTERNS_BY_AGENT.items():
        if any(re.search(pat, text) for pat in patterns):
            if aid == 1:
                return _sanitize_agent_ids([1, 3])
            if aid == 3:
                return _sanitize_agent_ids([3])

    # Scoring de keywords do Agente 1
    agent_1_keywords = _SPECIALIST_KEYWORDS.get(1, set())
    hits = weighted = 0
    for kw in agent_1_keywords:
        if kw and _contains_keyword(text, kw):
            w = _keyword_weight(1, kw)
            if w > 0:
                weighted += w
                hits += 1

    if hits >= 2 or weighted >= 3:
        return _sanitize_agent_ids([1, 3])
    if weighted >= 2 and hits >= 1:
        return _sanitize_agent_ids([1, 3])

    # Dairy genérico objetivo → delega ao LLM para escolher especialista
    if _contains_dairy_signal(text) and _is_objective_question(text):
        return None

    # Dairy genérico não-objetivo → baseline regulatório seguro
    if _contains_dairy_signal(text):
        return _sanitize_agent_ids([3])

    # Sem sinal dairy → delega ao LLM classifier
    return None


# ---------------------------------------------------------------------------
# Calibração de confiança
# ---------------------------------------------------------------------------

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
    """Estima confiança para rotas produzidas pelo fast-path determinístico."""
    text_norm = _normalize_text(route_text)
    if not agent_ids:
        return 0.98  # off-topic: alta certeza de nenhum agente
    if _is_strong_regulatory_signal(text_norm) or _is_strong_cheese_signal(text_norm):
        return 0.86  # sinal de domínio claro
    if any(aid != _REGULATORY_BASELINE_ID for aid in agent_ids):
        return 0.82  # especialista identificado
    if _contains_dairy_signal(text_norm):
        return 0.70  # dairy genérico
    return 0.60


def _recalibrate_confidence(
    route_text: str,
    agent_ids: List[int],
    raw_confidence: float,
) -> float:
    """Ajusta a confiança bruta para evitar excesso de bucket 'high' em casos ambíguos."""
    conf = _clamp_confidence(raw_confidence)
    ids = _sanitize_agent_ids(agent_ids)
    text_norm = _normalize_text(route_text)

    if not ids:
        return conf  # off-topic — confiança mantida como está

    # Domínios de alta especificidade elevam confiança mínima
    if _is_strong_regulatory_signal(text_norm) and _REGULATORY_BASELINE_ID in ids:
        conf = max(conf, 0.84)
    if _is_strong_cheese_signal(text_norm) and 1 in ids:
        conf = max(conf, 0.82)

    # Dois ou mais especialistas → aumenta ambiguidade → rebaixa teto
    specialists = [aid for aid in ids if aid != _REGULATORY_BASELINE_ID]
    if len(specialists) > 1:
        conf = min(conf, 0.74)

    # Dairy genérico + pergunta objetiva → evita 'high' artificial
    if (
        set(ids) <= {_REGULATORY_BASELINE_ID}
        and _contains_dairy_signal(text_norm)
        and _is_objective_question(text_norm)
    ):
        conf = min(conf, 0.69)

    return _clamp_confidence(conf)


# ---------------------------------------------------------------------------
# Hard constraints e guardrails
# ---------------------------------------------------------------------------

def _apply_dairy_hard_constraints(route_text: str, agent_ids: List[int]) -> List[int]:
    """Garante que Agent 3 (regulatório) está presente em toda pergunta de laticínios."""
    if not agent_ids:
        return []
    text_norm = _normalize_text(route_text)
    if not _contains_dairy_signal(text_norm):
        return agent_ids
    out = list(agent_ids)
    if _REGULATORY_BASELINE_ID not in out:
        out.append(_REGULATORY_BASELINE_ID)
    return _sanitize_agent_ids(out)


def _apply_domain_guardrails(
    route_text: str,
    agent_ids: List[int],
    alternatives: List[int],
) -> Tuple[List[int], List[int]]:
    """Filtra alternativas incompatíveis com o domínio detectado."""
    text_norm = _normalize_text(route_text)
    ids = _sanitize_agent_ids(agent_ids)
    alts = _sanitize_agent_ids(alternatives)

    if not _contains_dairy_signal(text_norm):
        return ids, alts

    # Requisito legal: somente Agent 3, sem especialistas concorrentes
    if _is_legal_requirement_regulatory_signal(text_norm):
        ids = _sanitize_agent_ids([_REGULATORY_BASELINE_ID])
        alts = []
        return ids, alts

    # Regulatório forte sem contexto de queijo: remove especialistas das alternativas
    if _is_strong_regulatory_signal(text_norm) and not _is_strong_cheese_signal(text_norm):
        alts = [aid for aid in alts if aid == _REGULATORY_BASELINE_ID]

    return ids, alts


# ---------------------------------------------------------------------------
# Construção do plano de execução
# ---------------------------------------------------------------------------

def _build_execution_plan(
    route_text: str,
    chosen_ids: List[int],
    alternatives: Optional[List[int]],
    bucket: str,
) -> List[int]:
    """Gera a lista ordenada de agentes a executar.

    Ordenação: especialista(s) primeiro, Agent 3 por último como copiloto.
    O especialista lidera a resposta; Agent 3 complementa com contexto regulatório.
    """
    chosen = _sanitize_agent_ids(chosen_ids)
    alts = _sanitize_agent_ids(alternatives or [])

    if not chosen:
        return []

    text_norm = _normalize_text(route_text)
    has_dairy_signal = _contains_dairy_signal(text_norm)
    has_specialist = any(aid != _REGULATORY_BASELINE_ID for aid in chosen)
    is_dairy_route = has_dairy_signal or has_specialist or _REGULATORY_BASELINE_ID in chosen

    if is_dairy_route:
        # Agent 3 sempre presente em rotas de laticínios
        if _REGULATORY_BASELINE_ID not in chosen:
            chosen.append(_REGULATORY_BASELINE_ID)
        chosen = _sanitize_agent_ids(chosen)

        specialists = [aid for aid in chosen if aid != _REGULATORY_BASELINE_ID]

        # Complementa com alternativas relevantes não redundantes
        for aid in alts:
            if aid != _REGULATORY_BASELINE_ID and aid not in specialists:
                specialists.append(aid)

        # Guardrail: regulatório puro (sem sinal de queijo) → sem especialistas
        if _is_strong_regulatory_signal(text_norm) and not _is_strong_cheese_signal(text_norm):
            specialists = []

        max_specialists = _SPECIALISTS_PER_BUCKET.get(bucket, 2)
        if specialists and max_specialists < 1:
            max_specialists = 1

        selected = specialists[:max_specialists]
        # Especialista lidera; Agent 3 fecha o plano como copiloto regulatório
        plan = selected + [_REGULATORY_BASELINE_ID] if selected else [_REGULATORY_BASELINE_ID]
    else:
        max_agents = _SPECIALISTS_PER_BUCKET.get(bucket, 2)
        plan = (chosen + [aid for aid in alts if aid not in chosen])[:max_agents]

    return _sanitize_agent_ids(plan)[:4]


# ---------------------------------------------------------------------------
# Inferência de agente primário
# ---------------------------------------------------------------------------

def _infer_domain_primary_from_text(text: str, candidate_ids: List[int]) -> Optional[int]:
    """Identifica qual agente deve liderar a resposta com base nos sinais de domínio."""
    text_norm = _normalize_text(text)
    ids = _sanitize_agent_ids(candidate_ids)
    if not ids:
        return None

    # Requisito legal → Agent 3 definitivamente primário
    if _is_legal_requirement_regulatory_signal(text_norm) and _REGULATORY_BASELINE_ID in ids:
        return _REGULATORY_BASELINE_ID

    # Contexto normativo com sinal de queijo → Agent 1 é o especialista primário
    if _is_normative_regulatory_signal(text_norm):
        if _is_strong_cheese_signal(text_norm) and 1 in ids:
            return 1
        if _REGULATORY_BASELINE_ID in ids:
            return _REGULATORY_BASELINE_ID

    # Rotulagem → Agent 3 como primário regulatório
    if _is_labeling_regulatory_signal(text_norm) and _REGULATORY_BASELINE_ID in ids:
        return _REGULATORY_BASELINE_ID

    # Sinal regulatório forte → Agent 3
    if _is_strong_regulatory_signal(text_norm) and _REGULATORY_BASELINE_ID in ids:
        return _REGULATORY_BASELINE_ID

    # Sinal forte de queijo → Agent 1
    if _is_strong_cheese_signal(text_norm) and 1 in ids:
        return 1

    # Hints fortes do especialista (Agent 1)
    hints_1 = _SPECIALIST_STRONG_HINTS.get(1, set())
    if any(_contains_keyword(text_norm, h) for h in hints_1) and 1 in ids:
        return 1

    return None


def _choose_primary_agent_id(agent_ids: List[int], route_text: str = "") -> int:
    """Escolhe o agente a ser exibido como responsável pela resposta."""
    if not agent_ids:
        return _REGULATORY_BASELINE_ID

    domain_primary = _infer_domain_primary_from_text(route_text, agent_ids)
    if domain_primary is not None:
        return domain_primary

    # Especialista (Agent 1) antes do baseline regulatório
    for aid in agent_ids:
        if aid != _REGULATORY_BASELINE_ID:
            return aid

    return _REGULATORY_BASELINE_ID
