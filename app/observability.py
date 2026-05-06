"""
observability.py — Structured logging e tracing para o pipeline do orquestrador.

Cada request recebe um trace_id único propagado por todos os nós do grafo.
Os logs são emitidos em JSON estruturado para fácil ingestão em ferramentas
de observabilidade (Datadog, CloudWatch, Loki, etc.).

Também expõe o semáforo global de concorrência LLM para prevenir 429.
"""

import asyncio
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from app.config import MAX_CONCURRENT_LLM_CALLS

_log = logging.getLogger("dairyapp")

# ---------------------------------------------------------------------------
# Trace ID — propagado via ContextVar (async-safe)
# ---------------------------------------------------------------------------

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    """Gera e registra um novo trace_id para o request atual."""
    tid = uuid.uuid4().hex[:16]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    """Retorna o trace_id do contexto atual (vazio se não inicializado)."""
    return _trace_id_var.get()


def set_trace_id(tid: str) -> None:
    """Define o trace_id no contexto atual (para propagação entre nós)."""
    _trace_id_var.set(tid or "")


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_event(
    event: str,
    *,
    node: str = "",
    duration_ms: Optional[float] = None,
    level: str = "info",
    **extra: Any,
) -> None:
    """Emite um log estruturado com trace_id, nó e métricas.

    Args:
        event: Nome do evento (ex: "classify_complete", "agent_timeout").
        node: Nome do nó do grafo (ex: "classify", "execute", "consolidate").
        duration_ms: Duração da operação em milissegundos.
        level: Nível do log ("debug", "info", "warning", "error").
        **extra: Campos adicionais (routing_bucket, agents_chosen, etc.).
    """
    payload: Dict[str, Any] = {
        "event": event,
        "trace_id": get_trace_id(),
    }
    if node:
        payload["node"] = node
    if duration_ms is not None:
        payload["duration_ms"] = round(duration_ms, 1)
    payload.update(extra)

    log_fn = getattr(_log, level, _log.info)
    log_fn(
        "%s",
        payload,
        extra={"structured": payload},
    )


class NodeTimer:
    """Context manager para medir e logar duração de um nó do grafo.

    Uso:
        async with NodeTimer("classify") as timer:
            result = await do_classification(...)
            timer.add(routing_bucket="high", agents=[1, 3])
    """

    def __init__(self, node: str) -> None:
        self.node = node
        self._start: float = 0.0
        self._extra: Dict[str, Any] = {}

    def add(self, **kwargs: Any) -> None:
        """Adiciona campos extras ao log de conclusão."""
        self._extra.update(kwargs)

    async def __aenter__(self) -> "NodeTimer":
        self._start = time.perf_counter()
        log_event(f"{self.node}_start", node=self.node, level="debug")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        if exc_type is not None:
            log_event(
                f"{self.node}_error",
                node=self.node,
                duration_ms=elapsed,
                level="error",
                error=str(exc_val),
                error_type=exc_type.__name__ if exc_type else "",
                **self._extra,
            )
        else:
            log_event(
                f"{self.node}_complete",
                node=self.node,
                duration_ms=elapsed,
                **self._extra,
            )


# ---------------------------------------------------------------------------
# Semáforo global de concorrência LLM
# ---------------------------------------------------------------------------

_llm_semaphore: Optional[asyncio.Semaphore] = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Lazy init do semáforo (precisa ser criado dentro de um event loop)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    return _llm_semaphore


async def acquire_llm_slot() -> None:
    """Adquire um slot no semáforo de concorrência LLM.

    Bloqueia se MAX_CONCURRENT_LLM_CALLS já estiverem em uso.
    """
    await _get_llm_semaphore().acquire()


def release_llm_slot() -> None:
    """Libera um slot no semáforo de concorrência LLM."""
    _get_llm_semaphore().release()


class LLMSlot:
    """Context manager async para controlar concorrência de chamadas LLM.

    Uso:
        async with LLMSlot():
            response = await llm.ainvoke(messages)
    """

    async def __aenter__(self) -> "LLMSlot":
        await acquire_llm_slot()
        return self

    async def __aexit__(self, *args: Any) -> None:
        release_llm_slot()
