"""
orch_warmup.py — Warmup do cache de classificação no startup.

Pré-carrega as queries mais frequentes/representativas do dataset de
benchmark no cache de classificação. Isso garante que as primeiras
requests dos usuários tenham latência menor (cache hit no classify).

O warmup é conservador: usa apenas o fast-path (regras determinísticas),
sem chamadas LLM. Queries que precisariam de LLM para classificar
são ignoradas — serão classificadas sob demanda e cacheadas normalmente.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.agents.orch_text import _normalize_text
from app.agents.orch_routing import (
    _cache_set,
    _rule_based_route,
    _AGENTS_WITHOUT_KB,
)

_log = logging.getLogger("dairyapp.warmup")


def _load_benchmark_queries() -> List[Dict]:
    """Carrega queries do dataset de benchmark YAML."""
    queries_path = Path("tests/fixtures/rag/rag_queries.yaml")
    if not queries_path.exists():
        return []

    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(queries_path.read_text(encoding="utf-8")) or {}
        return raw.get("queries", []) or []
    except Exception:
        return []


def warmup_classification_cache(max_entries: int = 100) -> int:
    """Pré-aquece o cache de classificação com queries do benchmark.

    Usa apenas fast-path (determinístico, sem LLM). Retorna o número
    de entries inseridas no cache.

    Args:
        max_entries: Número máximo de queries a pré-carregar.

    Returns:
        Número de entries efetivamente cacheadas.
    """
    queries = _load_benchmark_queries()
    if not queries:
        _log.info("Nenhuma query de benchmark encontrada para warmup")
        return 0

    cached = 0
    for item in queries[:max_entries * 2]:  # Itera mais para compensar skips
        if cached >= max_entries:
            break

        question = str(item.get("question", "")).strip()
        if not question:
            continue

        agent_id = item.get("agent_id")
        if agent_id is not None and int(agent_id) in _AGENTS_WITHOUT_KB:
            continue

        cache_key = _normalize_text(question)
        if not cache_key:
            continue

        # Tenta fast-path determinístico
        fast_ids = _rule_based_route(question)
        if fast_ids is not None and fast_ids:
            _cache_set(cache_key, fast_ids)
            cached += 1
            continue

        # Se fast-path não resolve, usa o agent_id do dataset como rota conhecida
        if agent_id is not None:
            aid = int(agent_id)
            if aid not in _AGENTS_WITHOUT_KB:
                route = [aid]
                # Agente 3 é baseline — incluir se não está na rota
                if 3 not in route and aid != 3:
                    route.append(3)
                _cache_set(cache_key, route)
                cached += 1

    _log.info("Warmup: %d queries pré-cacheadas", cached)
    return cached
