"""
Benchmark simples de latência do fluxo agêntico.

Mede p50/p95/p99 de chamadas diretas aos grafos (sem HTTP), útil para
comparar mudanças de roteamento/prompt/retrieval.

Uso:
  python scripts/benchmark_latency.py --target orchestrator --repeats 3
  python scripts/benchmark_latency.py --target agent --agent-id 3 --repeats 5
  python scripts/benchmark_latency.py --target orchestrator --queries-file tests/fixtures/rag/queries.txt
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

from langchain_core.messages import AIMessage, HumanMessage

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.connection import close_pools, init_pools
from app.agents.base_agent import get_agent_graph
from app.agents.orchestrator import get_orchestrator_graph


DEFAULT_QUERIES = [
    "Qual a temperatura de filagem da mussarela?",
    "A IN 65 permite adicionar leite na fabricação de ricota?",
    "Como reduzir sinérese em iogurte batido?",
    "Quais causas prováveis de estufamento tardio em queijo?",
    "Quais análises básicas de qualidade do leite cru devo monitorar?",
]


def _load_queries(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_QUERIES
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo de queries não encontrado: {p}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _extract_text(result: dict) -> str:
    if isinstance(result, dict):
        final_response = result.get("final_response")
        if isinstance(final_response, str) and final_response.strip():
            return final_response
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
                return msg.content
    return ""


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


async def _invoke_once(
    target: str,
    query: str,
    agent_id: int,
) -> Tuple[float, bool]:
    if target == "orchestrator":
        graph = get_orchestrator_graph()
        payload = {"messages": [HumanMessage(content=query)]}
    else:
        graph = get_agent_graph(agent_id)
        payload = {"messages": [HumanMessage(content=query)]}

    t0 = time.perf_counter()
    try:
        result = await graph.ainvoke(payload)
        ok = bool(_extract_text(result))
    except Exception:
        ok = False
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, ok


async def _run_benchmark(
    target: str,
    queries: List[str],
    repeats: int,
    agent_id: int,
    concurrency: int,
    warmup: int,
) -> None:
    total_calls = max(1, repeats) * len(queries)
    schedule = [queries[i % len(queries)] for i in range(total_calls)]

    # Warmup para reduzir efeito de cold-start.
    for _ in range(max(0, warmup)):
        for q in queries[: min(len(queries), 2)]:
            await _invoke_once(target, q, agent_id)

    latencies: List[float] = []
    success = 0

    sem = asyncio.Semaphore(max(1, concurrency))

    async def worker(query: str) -> None:
        nonlocal success
        async with sem:
            ms, ok = await _invoke_once(target, query, agent_id)
            latencies.append(ms)
            if ok:
                success += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker(q) for q in schedule))
    elapsed = time.perf_counter() - started

    if not latencies:
        print("Nenhuma chamada executada.")
        return

    print("\n=== Benchmark de Latência ===")
    print(f"target: {target}")
    if target == "agent":
        print(f"agent_id: {agent_id}")
    print(f"calls: {len(latencies)}")
    print(f"success: {success}/{len(latencies)}")
    print(f"wall_time_s: {elapsed:.2f}")
    print(f"throughput_rps: {len(latencies) / max(elapsed, 1e-9):.2f}")
    print(f"mean_ms: {statistics.mean(latencies):.1f}")
    print(f"min_ms: {min(latencies):.1f}")
    print(f"p50_ms: {_percentile(latencies, 0.50):.1f}")
    print(f"p90_ms: {_percentile(latencies, 0.90):.1f}")
    print(f"p95_ms: {_percentile(latencies, 0.95):.1f}")
    print(f"p99_ms: {_percentile(latencies, 0.99):.1f}")
    print(f"max_ms: {max(latencies):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["orchestrator", "agent"], default="orchestrator")
    parser.add_argument("--agent-id", type=int, default=3)
    parser.add_argument("--queries-file", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    queries = _load_queries(args.queries_file)
    if not queries:
        raise ValueError("Nenhuma query disponível para benchmark.")

    init_pools()
    try:
        asyncio.run(
            _run_benchmark(
                target=args.target,
                queries=queries,
                repeats=args.repeats,
                agent_id=args.agent_id,
                concurrency=args.concurrency,
                warmup=args.warmup,
            )
        )
    finally:
        close_pools()


if __name__ == "__main__":
    main()
