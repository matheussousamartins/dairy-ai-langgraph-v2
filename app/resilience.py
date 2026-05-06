"""
resilience.py — Circuit breaker e adaptive timeout por agente.

Circuit breaker protege contra agentes instáveis: após N falhas consecutivas,
abre o circuito por recovery_sec, depois tenta half-open (1 probe). Se o probe
passar, fecha novamente. Se falhar, reabre.

Adaptive timeout calcula p95 de latência real por agente via janela deslizante,
clampado entre AGENT_TIMEOUT_MIN_SEC e AGENT_TIMEOUT_MAX_SEC. Isso evita que
um agente lento sempre consuma o timeout fixo de agentes rápidos.
"""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

_log = logging.getLogger("dairyapp.resilience")


# ---------------------------------------------------------------------------
# Config (lida lazy para não importar config no módulo-level em circular risk)
# ---------------------------------------------------------------------------

def _cfg():
    from app.config import (
        CIRCUIT_BREAKER_ENABLED,
        CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        CIRCUIT_BREAKER_RECOVERY_SEC,
        AGENT_TIMEOUT_MIN_SEC,
        AGENT_TIMEOUT_MAX_SEC,
        AGENT_TIMEOUT_WINDOW,
    )
    return {
        "enabled": CIRCUIT_BREAKER_ENABLED,
        "failure_threshold": CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        "recovery_sec": CIRCUIT_BREAKER_RECOVERY_SEC,
        "timeout_min": AGENT_TIMEOUT_MIN_SEC,
        "timeout_max": AGENT_TIMEOUT_MAX_SEC,
        "timeout_window": AGENT_TIMEOUT_WINDOW,
    }


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class _State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-agent circuit breaker (thread-safe for asyncio single-thread use)."""

    agent_id: int
    _state: _State = field(default=_State.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _probe_in_flight: bool = field(default=False, init=False)

    @property
    def state(self) -> str:
        return self._state.value

    def is_open(self) -> bool:
        """Returns True if request should be rejected (circuit is open)."""
        cfg = _cfg()
        if not cfg["enabled"]:
            return False

        if self._state == _State.CLOSED:
            return False

        if self._state == _State.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= cfg["recovery_sec"]:
                self._state = _State.HALF_OPEN
                self._probe_in_flight = False
                _log.info(
                    "circuit_breaker agent=%d OPEN→HALF_OPEN after %.1fs",
                    self.agent_id, elapsed,
                )
                return False  # allow the probe through
            return True  # still open

        # HALF_OPEN: allow one probe at a time
        if self._probe_in_flight:
            return True  # second call rejected while probe is running
        self._probe_in_flight = True
        return False

    def record_success(self) -> None:
        if self._state in (_State.HALF_OPEN, _State.OPEN):
            _log.info(
                "circuit_breaker agent=%d %s→CLOSED (success)",
                self.agent_id, self._state.value,
            )
        self._state = _State.CLOSED
        self._failures = 0  # reseta contagem consecutiva
        self._probe_in_flight = False

    def record_failure(self, is_infra_timeout: bool = False) -> None:
        """Registra falha. Timeouts de infra (pool/rede) não abrem o circuito —
        apenas erros reais do agente (LLM, código) contam para o threshold."""
        cfg = _cfg()
        self._probe_in_flight = False

        if is_infra_timeout:
            # Timeout de infra: reseta contagem consecutiva (não é falha do agente)
            # mas não fecha o circuito se já estava em HALF_OPEN.
            if self._state == _State.HALF_OPEN:
                # Probe não confirma nem nega — volta para OPEN conservadoramente
                self._state = _State.OPEN
                self._opened_at = time.monotonic()
                _log.warning(
                    "circuit_breaker agent=%d HALF_OPEN→OPEN (infra timeout during probe)",
                    self.agent_id,
                )
            return

        self._failures += 1  # falha real consecutiva

        if self._state == _State.HALF_OPEN:
            # Probe com falha real → reopen
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
            _log.warning(
                "circuit_breaker agent=%d HALF_OPEN→OPEN (probe failed, failures=%d)",
                self.agent_id, self._failures,
            )
            return

        if self._state == _State.CLOSED and self._failures >= cfg["failure_threshold"]:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
            _log.warning(
                "circuit_breaker agent=%d CLOSED→OPEN (failures=%d >= threshold=%d)",
                self.agent_id, self._failures, cfg["failure_threshold"],
            )

    def to_dict(self) -> dict:
        cfg = _cfg()
        d: dict = {"state": self._state.value, "failures": self._failures}
        if self._state == _State.OPEN:
            remaining = max(0.0, cfg["recovery_sec"] - (time.monotonic() - self._opened_at))
            d["recovery_remaining_sec"] = round(remaining, 1)
        return d


# ---------------------------------------------------------------------------
# Adaptive Timeout Tracker
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveTimeoutTracker:
    """Tracks rolling latency window per agent and computes adaptive p95 timeout."""

    agent_id: int
    _latencies: deque = field(default_factory=lambda: deque(maxlen=50), init=False)

    def record(self, duration_sec: float) -> None:
        self._latencies.append(duration_sec)

    def get_timeout(self) -> float:
        cfg = _cfg()
        t_min = cfg["timeout_min"]
        t_max = cfg["timeout_max"]
        window = cfg["timeout_window"]

        if len(self._latencies) < window:
            return t_max  # not enough data yet → use max (conservative)

        sorted_lats = sorted(self._latencies)
        p95_idx = math.ceil(0.95 * len(sorted_lats)) - 1
        p95 = sorted_lats[max(0, p95_idx)]
        # Add 30% headroom over p95
        timeout = min(t_max, max(t_min, p95 * 1.3))
        return round(timeout, 1)

    def to_dict(self) -> dict:
        if not self._latencies:
            return {"samples": 0, "p95_sec": None, "current_timeout_sec": _cfg()["timeout_max"]}
        sorted_lats = sorted(self._latencies)
        p95_idx = math.ceil(0.95 * len(sorted_lats)) - 1
        p95 = sorted_lats[max(0, p95_idx)]
        return {
            "samples": len(self._latencies),
            "p95_sec": round(p95, 2),
            "current_timeout_sec": self.get_timeout(),
        }


# ---------------------------------------------------------------------------
# Global registries
# ---------------------------------------------------------------------------

_circuit_breakers: Dict[int, CircuitBreaker] = {}
_timeout_trackers: Dict[int, AdaptiveTimeoutTracker] = {}


def get_circuit_breaker(agent_id: int) -> CircuitBreaker:
    if agent_id not in _circuit_breakers:
        _circuit_breakers[agent_id] = CircuitBreaker(agent_id=agent_id)
    return _circuit_breakers[agent_id]


def get_timeout_tracker(agent_id: int) -> AdaptiveTimeoutTracker:
    if agent_id not in _timeout_trackers:
        _timeout_trackers[agent_id] = AdaptiveTimeoutTracker(agent_id=agent_id)
    return _timeout_trackers[agent_id]


def all_circuit_states() -> Dict[int, dict]:
    """Returns current state of all circuit breakers (for health check)."""
    return {aid: cb.to_dict() for aid, cb in _circuit_breakers.items()}


def all_timeout_stats() -> Dict[int, dict]:
    """Returns current adaptive timeout stats (for health check)."""
    return {aid: tt.to_dict() for aid, tt in _timeout_trackers.items()}
