from typing import Optional

from app.config import ALLOWED_CHAT_MODELS, LLM_MODEL


def get_allowed_chat_models() -> list[str]:
    allowed = list(ALLOWED_CHAT_MODELS)
    if LLM_MODEL not in allowed:
        allowed.insert(0, LLM_MODEL)
    return allowed


def resolve_chat_model(requested_model: Optional[str]) -> str:
    normalized = (requested_model or "").strip()
    if not normalized:
        return LLM_MODEL

    allowed = get_allowed_chat_models()
    if normalized not in allowed:
        allowed_str = ", ".join(allowed)
        raise ValueError(f"Modelo '{normalized}' não permitido. Use um destes: {allowed_str}")
    return normalized
