"""Resolucao conversacional pre-RAG.

Esta camada decide se a mensagem atual depende do historico antes de montar a
query de busca. Ela e intencionalmente deterministica: o LLM continua sendo
usado apenas na etapa seguinte, para reescrever a query quando ha contexto.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class ConversationResolution:
    depends_on_previous: bool
    intent: str
    reason: str


_ANAPHORA_TOKENS = {
    "isso",
    "isto",
    "esse",
    "essa",
    "esses",
    "essas",
    "dessa",
    "desse",
    "desses",
    "dessas",
    "nisso",
    "nele",
    "nela",
    "anterior",
}

_FOLLOWUP_PREFIXES = (
    "e no caso",
    "e quanto",
    "e para",
    "e se",
    "e no ",
    "e do ",
    "e da ",
    "e de ",
    "e em ",
    "e sob",
    "e com",
    "e qual",
    "e quais",
    "e como",
    "e o que",
    "e a ",
    "e os ",
    "e as ",
    "agora",
    "entao",
    "então",
    "nesse caso",
    "neste caso",
    "sobre isso",
    "compare",
    "comparando",
)

_RECAP_PHRASES = (
    "no caso anterior",
    "no contexto anterior",
    "do que falamos",
    "o que falamos antes",
    "o que conversamos",
    "sobre o que conversamos",
    "conversamos recentemente",
    "falamos recentemente",
    "me explique sobre o que conversamos",
    "me lembre do que conversamos",
    "resuma o que conversamos",
    "resuma o que falamos",
    "retome o que falamos",
    "continue de onde paramos",
    "que voce falou",
    "que você falou",
    "em relacao ao anterior",
    "em relação ao anterior",
    "sobre o anterior",
    "mesmo caso",
    "mesma coisa",
)

_DEEPENING_TERMS = (
    "aprofund",
    "detalh",
    "explique melhor",
    "explica melhor",
    "por que",
    "porque",
    "qual impacto",
    "quais impactos",
    "impacto tecnologico",
    "impacto tecnológico",
    "consequencia",
    "consequência",
    "efeito",
    "efeitos",
    "isso muda",
    "isso vale",
    "isso se aplica",
)

_COMPARISON_TERMS = (
    "compare",
    "comparando",
    "comparacao",
    "comparação",
    "diferenca",
    "diferença",
    "versus",
    " vs ",
)

_DAIRY_TOPIC_TERMS = (
    "leite",
    "queijo",
    "parmesao",
    "parmesão",
    "mussarela",
    "ccs",
    "celulas somaticas",
    "células somáticas",
    "ph",
    "maturacao",
    "maturação",
    "legislacao",
    "legislação",
    "riispoa",
)


def _strip_profile_suffix(text: str) -> str:
    if "\n[Perfil" in text:
        return text.split("\n[Perfil", 1)[0]
    return text


def _normalize(text: str) -> str:
    cleaned = _strip_profile_suffix(text or "")
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _has_recent_history(history: Optional[Iterable[Mapping[str, str]]]) -> bool:
    if history is None:
        return False
    for item in history:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"human", "ai"} and content:
            return True
    return False


def resolve_conversation_turn(
    message: str,
    history: Optional[Iterable[Mapping[str, str]]] = None,
    *,
    max_autonomous_chars: int = 220,
) -> ConversationResolution:
    """Classifica a mensagem atual quanto ao uso de contexto recente."""
    text = _normalize(message)
    if not text:
        return ConversationResolution(False, "empty", "empty_message")

    words = text.split()
    token_set = _tokens(text)
    has_history = _has_recent_history(history)
    has_anaphora = bool(token_set & _ANAPHORA_TOKENS)
    starts_followup = any(text.startswith(prefix) for prefix in _FOLLOWUP_PREFIXES)
    has_recap = any(phrase in text for phrase in _RECAP_PHRASES)
    has_deepening = any(term in text for term in _DEEPENING_TERMS)
    has_comparison = any(term in text for term in _COMPARISON_TERMS)
    has_dairy_topic = any(term in text for term in _DAIRY_TOPIC_TERMS)

    if has_recap:
        return ConversationResolution(True, "recap", "explicit_recap_reference")
    if has_comparison:
        return ConversationResolution(True, "comparison", "comparison_reference")
    if starts_followup:
        return ConversationResolution(True, "followup", "followup_prefix")
    if words and words[0] == "e" and len(words) <= 12:
        return ConversationResolution(True, "followup", "short_e_prefix")
    if has_anaphora and len(words) <= 24:
        return ConversationResolution(True, "followup", "anaphora_token")
    if has_deepening and (has_anaphora or has_history):
        return ConversationResolution(True, "deepening", "deepening_with_context")
    if len(text) <= max_autonomous_chars and has_deepening and not has_dairy_topic:
        return ConversationResolution(True, "deepening", "underspecified_deepening")

    return ConversationResolution(False, "standalone", "standalone_or_topic_shift")


def should_use_conversation_context(
    message: str,
    history: Optional[Iterable[Mapping[str, str]]] = None,
    *,
    max_autonomous_chars: int = 220,
) -> bool:
    return resolve_conversation_turn(
        message,
        history,
        max_autonomous_chars=max_autonomous_chars,
    ).depends_on_previous
