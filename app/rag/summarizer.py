"""
rag/summarizer.py — Sumarizador de histórico de conversa

Comprime mensagens antigas em um parágrafo de contexto técnico para evitar
perda silenciosa de informação quando a sessão excede MEMORY_WINDOW.

Filosofia:
  - O resumo deve preservar fatos técnicos específicos (pH, temperatura,
    normativas, produtos mencionados) para que follow-ups implícitos
    ainda possam ser respondidos com precisão.
  - Sumarização cumulativa: se já existe um resumo anterior, ele é incluído
    como entrada, garantindo que sessões muito longas nunca percam contexto.
  - Fail-safe absoluto: qualquer falha retorna None — o pipeline principal
    nunca é bloqueado por uma falha de sumarização.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.config import (
    MEMORY_SUMMARIZATION_ENABLED,
    MEMORY_SUMMARIZATION_MODEL,
    MEMORY_SUMMARIZATION_TIMEOUT_SEC,
)

_log = logging.getLogger(__name__)

_summarization_model: Optional[ChatOpenAI] = None


def _get_summarization_model() -> ChatOpenAI:
    global _summarization_model
    if _summarization_model is None:
        _summarization_model = ChatOpenAI(
            model=MEMORY_SUMMARIZATION_MODEL,
            temperature=0,
            max_tokens=500,
            timeout=MEMORY_SUMMARIZATION_TIMEOUT_SEC,
        )
    return _summarization_model


def _format_dialogue_for_summary(messages: List[Dict[str, str]]) -> str:
    """Formata a lista de mensagens em texto legível para o prompt de sumarização."""
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content", "") or "").strip()
        if not content:
            continue
        if role == "summary":
            snippet = content[:400] + "..." if len(content) > 400 else content
            lines.append(f"[Resumo anterior]\n{snippet}")
        elif role == "human":
            snippet = content[:280] + "..." if len(content) > 280 else content
            lines.append(f"Usuário: {snippet}")
        elif role == "ai":
            snippet = content[:280] + "..." if len(content) > 280 else content
            lines.append(f"Assistente: {snippet}")
    return "\n".join(lines)


def summarize_conversation(messages: List[Dict[str, str]]) -> Optional[str]:
    """Comprime uma lista de mensagens em um parágrafo de contexto técnico.

    Aceita mensagens com role "summary" (resumo acumulativo anterior) para
    que o novo resumo seja cumulativo e não perca contexto de sessões longas.

    Fail-safe garantido: retorna None em qualquer exceção.
    Nunca levanta exceção para o chamador.

    Args:
        messages: Lista de dicts com "role" e "content".
                  Roles aceitos: "human", "ai", "summary".

    Returns:
        Texto do resumo (≤ 600 chars), ou None em caso de falha.
    """
    if not MEMORY_SUMMARIZATION_ENABLED:
        return None
    if not messages:
        return None

    dialogue = _format_dialogue_for_summary(messages)
    if not dialogue.strip():
        return None

    prompt = (
        "Você é especialista em tecnologia de laticínios. "
        "Abaixo está parte de uma conversa técnica. "
        "Escreva um resumo conciso (2-4 frases) que preserve:\n"
        "- Os tópicos técnicos discutidos (produtos, processos, parâmetros)\n"
        "- Dados quantitativos importantes (pH, temperatura, tempo, percentuais)\n"
        "- Normativas ou regulamentos citados\n"
        "- Contexto necessário para responder perguntas de follow-up\n\n"
        "Escreva apenas o resumo, em português brasileiro. "
        "Sem introdução, sem explicação, sem marcadores de lista.\n\n"
        f"Conversa:\n{dialogue}\n\n"
        "Resumo:"
    )

    try:
        llm = _get_summarization_model()
        resp = llm.invoke(prompt)
        raw = (resp.content or "").strip()
        if not raw:
            return None

        raw = re.sub(r"\s+", " ", raw).strip()
        # Trunca resposta muito longa em sentença completa para não cortar no meio de uma ideia.
        if len(raw) > 600:
            truncated = raw[:597]
            last_period = truncated.rfind(".")
            if last_period > 300:
                raw = truncated[: last_period + 1]
            else:
                raw = truncated + "..."

        _log.info(
            "summarize_conversation: resumo gerado (%d msgs → %d chars)",
            len(messages),
            len(raw),
        )
        return raw

    except Exception as exc:
        _log.warning("summarize_conversation: falha silenciosa — %s", exc)
        return None
