"""
orch_signals.py — Detecção de sinais de domínio do fast-path.

Responsabilidade única: dado um texto normalizado, dizer qual domínio ele
pertence. Sem imports de LangChain, sem state, sem side-effects.

Agentes ativos:
  1 — Tecnologia de Queijos
  3 — Regulatórios por País

Agentes sem KB (não roteados até ingestão):
  0, 2, 4, 5, 6
Para adicionar um agente: remova-o de _AGENTS_WITHOUT_KB em orch_routing.py
e adicione as funções de sinal correspondentes aqui.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict

from app.agents.orch_text import _normalize_text

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Carregamento de regras do YAML
# ---------------------------------------------------------------------------

_ROUTING_RULES_PATH = Path("docs/orchestrator/routing_rules.yaml")


def _load_routing_rules() -> Dict[str, Any]:
    """Carrega regras de sinalização de docs/orchestrator/routing_rules.yaml.

    Retorna dicionário com frozensets e dicts para uso das funções de sinal.
    Se o arquivo não existir ou falhar, opera com sets vazios — o sistema
    ainda funciona via LLM classifier com menor precisão no fast-path.
    """
    _empty: Dict[str, Any] = {
        "greetings": frozenset(),
        "dairy_signal_terms": frozenset(),
        "regulatory_strong_terms": frozenset(),
        "legal_requirement_phrases": frozenset(),
        "cheese_strong_terms": frozenset(),
        "intent_patterns_by_agent": {},
        "low_precision_keywords": frozenset(),
        "hint_noise_terms": frozenset(),
        "hint_noise_tokens": frozenset(),
        "specialist_strong_hints_default": {},
    }

    if not _ROUTING_RULES_PATH.exists():
        _log.warning(
            "routing_rules.yaml não encontrado em %s — fast-path operando com "
            "sets vazios. Roteamento delegado ao LLM classifier.",
            _ROUTING_RULES_PATH,
        )
        return _empty

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(_ROUTING_RULES_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        _log.warning("Erro ao carregar routing_rules.yaml: %s — sets vazios.", exc)
        return _empty

    def _fs(data: Any) -> frozenset:
        if isinstance(data, list):
            return frozenset(str(x).strip() for x in data if str(x).strip())
        return frozenset()

    domain = raw.get("domain_signals", {}) or {}
    regulatory = domain.get("regulatory", {}) or {}
    cheese = domain.get("cheese", {}) or {}
    noise = raw.get("noise_control", {}) or {}

    # Padrões de intenção (apenas agentes 1 e 3 são ativos)
    intent_patterns: Dict[int, tuple] = {}
    for k, v in (raw.get("intent_patterns_by_agent", {}) or {}).items():
        try:
            aid = int(k)
        except (TypeError, ValueError):
            continue
        if aid not in (1, 3):
            continue
        if isinstance(v, list):
            intent_patterns[aid] = tuple(str(p) for p in v if str(p).strip())

    # Hints padrão por especialista (apenas agente 1 é ativo)
    specialist_hints: Dict[int, set] = {}
    for k, v in (raw.get("specialist_strong_hints", {}) or {}).items():
        try:
            aid = int(k)
        except (TypeError, ValueError):
            continue
        if aid != 1:
            continue
        specialist_hints[aid] = (
            {str(h).strip() for h in v if str(h).strip()} if isinstance(v, list) else set()
        )

    return {
        "greetings": _fs(raw.get("greetings", [])),
        "dairy_signal_terms": _fs(raw.get("dairy_signal_terms", [])),
        "regulatory_strong_terms": _fs(regulatory.get("strong_terms", [])),
        "legal_requirement_phrases": _fs(regulatory.get("legal_requirement_phrases", [])),
        "cheese_strong_terms": _fs(cheese.get("strong_terms", [])),
        "intent_patterns_by_agent": intent_patterns,
        "low_precision_keywords": _fs(raw.get("low_precision_keywords", [])),
        "hint_noise_terms": _fs(noise.get("hint_noise_terms", [])),
        "hint_noise_tokens": _fs(noise.get("hint_noise_tokens", [])),
        "specialist_strong_hints_default": specialist_hints,
    }


_ROUTING_RULES = _load_routing_rules()

_GREETINGS: frozenset                        = _ROUTING_RULES["greetings"]
_DAIRY_TERMS: frozenset                      = _ROUTING_RULES["dairy_signal_terms"]
_REGULATORY_STRONG_TERMS: frozenset          = _ROUTING_RULES["regulatory_strong_terms"]
_LEGAL_REQUIREMENT_DIRECT_PHRASES: frozenset = _ROUTING_RULES["legal_requirement_phrases"]
_CHEESE_STRONG_TERMS: frozenset              = _ROUTING_RULES["cheese_strong_terms"]
_INTENT_PATTERNS_BY_AGENT: Dict[int, tuple]  = _ROUTING_RULES["intent_patterns_by_agent"]
_LOW_PRECISION_KEYWORDS: frozenset           = _ROUTING_RULES["low_precision_keywords"]
_HINT_NOISE_TERMS: frozenset                 = _ROUTING_RULES["hint_noise_terms"]
_HINT_NOISE_TOKENS: frozenset                = _ROUTING_RULES["hint_noise_tokens"]
_SPECIALIST_STRONG_HINTS_DEFAULT: Dict[int, set] = _ROUTING_RULES["specialist_strong_hints_default"]


# ---------------------------------------------------------------------------
# Funções de sinal genérico
# ---------------------------------------------------------------------------

def _contains_any_phrase(text_norm: str, phrases: frozenset) -> bool:
    """Retorna True se qualquer frase do conjunto aparecer no texto normalizado."""
    for phrase in phrases:
        p = _normalize_text(str(phrase))
        if not p:
            continue
        if " " in p:
            if p in text_norm:
                return True
        else:
            if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text_norm):
                return True
    return False


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
    return bool(re.search(r"\b(in|rdc|rtiq)\s*\d{1,4}\b", text_norm))


# ---------------------------------------------------------------------------
# Sinais regulatórios — Agente 3
# ---------------------------------------------------------------------------

def _is_legal_requirement_regulatory_signal(text_norm: str) -> bool:
    """Detecta perguntas sobre requisito mínimo legal (→ puramente Agent 3)."""
    if not text_norm:
        return False
    if _contains_any_phrase(text_norm, _LEGAL_REQUIREMENT_DIRECT_PHRASES):
        return True

    has_requirement_marker = (
        "exigid" in text_norm
        or "obrigat" in text_norm
        or "minimo legal" in text_norm
        or "mínimo legal" in text_norm
    )
    if not has_requirement_marker:
        return False

    requirement_subjects = (
        "periodo minimo", "período mínimo",
        "prazo minimo", "prazo mínimo",
        "tempo minimo", "tempo mínimo",
        "maturacao", "maturação",
        "limite minimo", "limite mínimo",
        "limite maximo", "limite máximo",
        "teor minimo", "teor mínimo",
        "deve sofrer maturacao", "deve sofrer maturação",
    )
    return any(subject in text_norm for subject in requirement_subjects)


def _is_labeling_regulatory_signal(text_norm: str) -> bool:
    """Detecta perguntas sobre rotulagem, denominação de venda, alegações nutricionais."""
    if not text_norm:
        return False

    labeling_terms = (
        "rotular", "rotulado", "rotulagem",
        "denominacao", "denominação",
        "denominacao de venda", "denominação de venda",
        "embalagem",
    )
    if any(t in text_norm for t in labeling_terms):
        if "denominacao" in text_norm or "denominação" in text_norm:
            return True
        dairy_products = (
            "provolone", "ricota", "minas", "queijo", "iogurte",
            "mussarela", "muçarela", "requeijao", "requeijão",
            "cream cheese", "bebida lactea", "bebida láctea",
            "sobremesa lactea", "sobremesa láctea",
        )
        if any(p in text_norm for p in dairy_products):
            return True

    # Alegações de ausência/composição sem a palavra "rotulagem"
    absence_claims = (
        "nao contem", "não contém", "isento de",
        "sem adicao de", "valor nulo",
    )
    nutritional_terms = (
        "gordura", "acucar", "açúcar", "sodio", "sódio",
        "calorias", "energetico", "energético",
    )
    if any(ac in text_norm for ac in absence_claims) and any(nt in text_norm for nt in nutritional_terms):
        return True

    return False


def _is_normative_regulatory_signal(text_norm: str) -> bool:
    """Detecta referência explícita a norma, instrução normativa ou decreto."""
    if not text_norm:
        return False
    if _is_legal_requirement_regulatory_signal(text_norm):
        return True
    markers = (
        "norma", "normas", "regulamento", "requisito legal", "requisitos legais",
        "instrucao normativa", "instrução normativa",
        "rdc", "riispoa", "decreto", "art.",
    )
    if any(m in text_norm for m in markers):
        return True
    return bool(re.search(r"\b(in|rdc)\s*\d{1,4}\b", text_norm))


def _is_strong_regulatory_signal(text_norm: str) -> bool:
    """Sinal forte de domínio regulatório (Agent 3)."""
    if not text_norm:
        return False
    if _is_legal_requirement_regulatory_signal(text_norm):
        return True
    if _contains_any_phrase(text_norm, _REGULATORY_STRONG_TERMS):
        return True
    return bool(re.search(r"\b(in|rdc)\s*\d{1,4}\b", text_norm))


# ---------------------------------------------------------------------------
# Sinais de tecnologia de queijos — Agente 1
# ---------------------------------------------------------------------------

def _is_strong_cheese_signal(text_norm: str) -> bool:
    """Sinal forte de domínio de tecnologia de queijos (Agent 1)."""
    if not text_norm:
        return False
    return _contains_any_phrase(text_norm, _CHEESE_STRONG_TERMS)
