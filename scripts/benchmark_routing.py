"""Benchmark fixo de roteamento do orquestrador.

Mede qualidade de roteamento por iteracao, com foco em:
- Routing@1
- Routing@3
- fallback_rate
- p95_latency

Dataset padrao: tests/fixtures/rag/rag_queries.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from langchain_core.messages import HumanMessage

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agents.orchestrator import get_orchestrator_graph
from app.db.connection import close_pools, init_pools


@dataclass
class RoutingCase:
    case_id: int
    expected_agent_id: int
    question: str
    group: str


@dataclass
class RoutingResult:
    case_id: int
    expected_agent_id: int
    question: str
    group: str
    primary_agent_id: int
    chosen_agent_ids: list[int]
    execution_plan: list[int]
    routing_bucket: str
    routing_confidence: float
    fallback_used: bool
    fallback_attempts: int
    fallback_trigger: str
    latency_ms: float
    error: str = ""

    @property
    def top3(self) -> list[int]:
        source = self.execution_plan if self.execution_plan else self.chosen_agent_ids
        return source[:3]

    @property
    def routing_at_1(self) -> bool:
        return self.primary_agent_id == self.expected_agent_id

    @property
    def routing_at_3(self) -> bool:
        return self.expected_agent_id in self.top3


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    out: list[int] = []
    for v in values:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv not in out:
            out.append(iv)
    return out


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("rate limit" in text) or ("rate_limit_exceeded" in text) or ("error code: 429" in text)


def _read_dataset(path: Path) -> list[RoutingCase]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get("queries") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset invalido: campo 'queries' ausente ou vazio.")

    out: list[RoutingCase] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("pergunta") or "").strip()
        if not question:
            continue
        expected_agent_id = _safe_int(item.get("agent_id"), -1)
        if expected_agent_id < 0:
            continue
        out.append(
            RoutingCase(
                case_id=idx,
                expected_agent_id=expected_agent_id,
                question=question,
                group=str(item.get("group") or ""),
            )
        )
    if not out:
        raise ValueError("Nenhum caso valido encontrado no dataset.")
    return out


def _filter_cases(
    cases: list[RoutingCase],
    allowed_agents: set[int] | None,
    limit_total: int | None,
    limit_per_agent: int | None,
    seed: int,
) -> list[RoutingCase]:
    filtered = [c for c in cases if not allowed_agents or c.expected_agent_id in allowed_agents]
    if not filtered:
        return []

    rng = random.Random(seed)
    grouped: dict[int, list[RoutingCase]] = defaultdict(list)
    for c in filtered:
        grouped[c.expected_agent_id].append(c)
    for lst in grouped.values():
        rng.shuffle(lst)

    selected: list[RoutingCase] = []
    if limit_per_agent and limit_per_agent > 0:
        for aid in sorted(grouped):
            selected.extend(grouped[aid][:limit_per_agent])
    else:
        for aid in sorted(grouped):
            selected.extend(grouped[aid])

    rng.shuffle(selected)
    if limit_total and limit_total > 0:
        selected = selected[:limit_total]
    return selected


async def _run_case(
    case: RoutingCase,
    max_retries: int,
    retry_base_seconds: float,
) -> RoutingResult:
    graph = get_orchestrator_graph()
    last_exc: Exception | None = None

    for attempt in range(max(0, max_retries) + 1):
        t0 = time.perf_counter()
        try:
            output = await graph.ainvoke({"messages": [HumanMessage(content=case.question)]})
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return RoutingResult(
                case_id=case.case_id,
                expected_agent_id=case.expected_agent_id,
                question=case.question,
                group=case.group,
                primary_agent_id=_safe_int(output.get("primary_agent_id"), 0),
                chosen_agent_ids=_to_int_list(output.get("chosen_agent_ids")),
                execution_plan=_to_int_list(output.get("execution_plan")),
                routing_bucket=str(output.get("routing_bucket") or "unknown"),
                routing_confidence=_safe_float(output.get("routing_confidence"), 0.0),
                fallback_used=bool(output.get("fallback_used", False)),
                fallback_attempts=_safe_int(output.get("fallback_attempts"), 0),
                fallback_trigger=str(output.get("fallback_trigger") or ""),
                latency_ms=latency_ms,
                error="",
            )
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries and _is_rate_limit_error(exc):
                delay = retry_base_seconds * (2 ** attempt)
                await asyncio.sleep(delay)
                continue
            break

    return RoutingResult(
        case_id=case.case_id,
        expected_agent_id=case.expected_agent_id,
        question=case.question,
        group=case.group,
        primary_agent_id=-1,
        chosen_agent_ids=[],
        execution_plan=[],
        routing_bucket="error",
        routing_confidence=0.0,
        fallback_used=False,
        fallback_attempts=0,
        fallback_trigger="",
        latency_ms=0.0,
        error=str(last_exc) if last_exc else "unknown_error",
    )


async def _run_all(
    cases: list[RoutingCase],
    concurrency: int,
    max_retries: int,
    retry_base_seconds: float,
    inter_request_sleep: float,
) -> list[RoutingResult]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[RoutingResult] = []

    async def worker(case: RoutingCase) -> None:
        async with sem:
            results.append(
                await _run_case(
                    case,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                )
            )
            if inter_request_sleep > 0:
                await asyncio.sleep(inter_request_sleep)

    await asyncio.gather(*(worker(c) for c in cases))
    return results


def _rate(ok_count: int, total: int) -> float:
    return (100.0 * ok_count / total) if total else 0.0


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _print_summary(results: list[RoutingResult]) -> None:
    total = len(results)
    ok_rows = [r for r in results if not r.error]
    failed_rows = [r for r in results if r.error]
    measured = len(ok_rows)
    r1_ok = sum(1 for r in ok_rows if r.routing_at_1)
    r3_ok = sum(1 for r in ok_rows if r.routing_at_3)
    fallback_count = sum(1 for r in ok_rows if r.fallback_used)
    lat = [r.latency_ms for r in ok_rows]

    print("\n=== Routing Benchmark ===")
    print(f"total_cases: {total}")
    print(f"measured_cases: {measured}")
    print(f"failed_cases: {len(failed_rows)}")
    print(f"routing_at_1: {_fmt_pct(_rate(r1_ok, measured))} ({r1_ok}/{measured})")
    print(f"routing_at_3: {_fmt_pct(_rate(r3_ok, measured))} ({r3_ok}/{measured})")
    print(
        f"fallback_rate: {_fmt_pct(_rate(fallback_count, measured))} "
        f"({fallback_count}/{measured})"
    )
    print(f"latency_p50_ms: {_percentile(lat, 0.50):.1f}" if lat else "latency_p50_ms: 0.0")
    print(f"latency_p95_ms: {_percentile(lat, 0.95):.1f}" if lat else "latency_p95_ms: 0.0")
    print(f"latency_p99_ms: {_percentile(lat, 0.99):.1f}" if lat else "latency_p99_ms: 0.0")
    print(f"latency_mean_ms: {statistics.mean(lat):.1f}" if lat else "latency_mean_ms: 0.0")

    by_bucket = Counter(r.routing_bucket for r in ok_rows)
    print("\nBucket distribution:")
    for bucket, count in sorted(by_bucket.items(), key=lambda x: x[0]):
        print(f"- {bucket}: {count}")

    print("\nPer expected agent:")
    grouped: dict[int, list[RoutingResult]] = defaultdict(list)
    for r in ok_rows:
        grouped[r.expected_agent_id].append(r)
    for aid in sorted(grouped):
        rows = grouped[aid]
        n = len(rows)
        a1 = sum(1 for r in rows if r.routing_at_1)
        a3 = sum(1 for r in rows if r.routing_at_3)
        fb = sum(1 for r in rows if r.fallback_used)
        print(
            f"- agent {aid}: n={n} | "
            f"R@1={_fmt_pct(_rate(a1, n))} | "
            f"R@3={_fmt_pct(_rate(a3, n))} | "
            f"fallback={_fmt_pct(_rate(fb, n))}"
        )

    confusion = Counter((r.expected_agent_id, r.primary_agent_id) for r in ok_rows if not r.routing_at_1)
    if confusion:
        print("\nTop confusions (expected -> predicted):")
        for (exp, got), count in confusion.most_common(10):
            print(f"- {exp} -> {got}: {count}")
    if failed_rows:
        by_error = Counter(r.error.splitlines()[0][:120] for r in failed_rows)
        print("\nTop errors:")
        for msg, count in by_error.most_common(5):
            print(f"- {count}x {msg}")


def _results_to_json(results: list[RoutingResult]) -> dict[str, Any]:
    total = len(results)
    ok_rows = [r for r in results if not r.error]
    failed_rows = [r for r in results if r.error]
    measured = len(ok_rows)
    r1_ok = sum(1 for r in ok_rows if r.routing_at_1)
    r3_ok = sum(1 for r in ok_rows if r.routing_at_3)
    fallback_count = sum(1 for r in ok_rows if r.fallback_used)
    lat = [r.latency_ms for r in ok_rows]

    grouped: dict[int, list[RoutingResult]] = defaultdict(list)
    for r in ok_rows:
        grouped[r.expected_agent_id].append(r)

    per_agent = {}
    for aid in sorted(grouped):
        rows = grouped[aid]
        n = len(rows)
        per_agent[str(aid)] = {
            "n": n,
            "routing_at_1": _rate(sum(1 for r in rows if r.routing_at_1), n),
            "routing_at_3": _rate(sum(1 for r in rows if r.routing_at_3), n),
            "fallback_rate": _rate(sum(1 for r in rows if r.fallback_used), n),
        }

    confusion = Counter((r.expected_agent_id, r.primary_agent_id) for r in ok_rows if not r.routing_at_1)
    confusion_json = [
        {"expected_agent_id": exp, "predicted_agent_id": got, "count": count}
        for (exp, got), count in confusion.most_common(20)
    ]

    return {
        "total_cases": total,
        "measured_cases": measured,
        "failed_cases": len(failed_rows),
        "routing_at_1": _rate(r1_ok, measured),
        "routing_at_3": _rate(r3_ok, measured),
        "fallback_rate": _rate(fallback_count, measured),
        "latency_ms": {
            "p50": _percentile(lat, 0.50),
            "p95": _percentile(lat, 0.95),
            "p99": _percentile(lat, 0.99),
            "mean": statistics.mean(lat) if lat else 0.0,
        },
        "bucket_distribution": dict(Counter(r.routing_bucket for r in ok_rows)),
        "per_expected_agent": per_agent,
        "top_confusions": confusion_json,
        "top_errors": [
            {"error": msg, "count": count}
            for msg, count in Counter(r.error.splitlines()[0][:120] for r in failed_rows).most_common(20)
        ],
        "cases": [
            {
                "case_id": r.case_id,
                "expected_agent_id": r.expected_agent_id,
                "primary_agent_id": r.primary_agent_id,
                "chosen_agent_ids": r.chosen_agent_ids,
                "execution_plan": r.execution_plan,
                "routing_bucket": r.routing_bucket,
                "routing_confidence": r.routing_confidence,
                "fallback_used": r.fallback_used,
                "fallback_attempts": r.fallback_attempts,
                "fallback_trigger": r.fallback_trigger,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "question": r.question,
                "group": r.group,
            }
            for r in results
        ],
    }


def _parse_agent_filter(value: str | None) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out or None


def _iter_warmup(cases: Iterable[RoutingCase], n: int) -> list[RoutingCase]:
    out: list[RoutingCase] = []
    for case in cases:
        out.append(case)
        if len(out) >= n:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/rag/rag_queries.yaml",
        help="Arquivo com dataset de queries.",
    )
    parser.add_argument(
        "--agents",
        default=None,
        help="Filtro por agent_id esperado, ex: 1,2,4",
    )
    parser.add_argument(
        "--limit-total",
        type=int,
        default=0,
        help="Limite total de casos apos amostragem (0 = sem limite).",
    )
    parser.add_argument(
        "--limit-per-agent",
        type=int,
        default=0,
        help="Limite por agent_id esperado (0 = sem limite).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para amostragem deterministica.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concorrencia de execucao.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Quantidade de casos de aquecimento antes da medicao.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Caminho opcional para salvar resultado em JSON.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Tentativas adicionais para erros de rate limit por caso.",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=2.5,
        help="Base do backoff exponencial para retry de rate limit.",
    )
    parser.add_argument(
        "--inter-request-sleep",
        type=float,
        default=0.15,
        help="Pausa entre requests para reduzir pico de TPM.",
    )
    args = parser.parse_args()

    allowed_agents = _parse_agent_filter(args.agents)
    cases = _read_dataset(Path(args.dataset))
    selected = _filter_cases(
        cases=cases,
        allowed_agents=allowed_agents,
        limit_total=args.limit_total if args.limit_total > 0 else None,
        limit_per_agent=args.limit_per_agent if args.limit_per_agent > 0 else None,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("Nenhum caso selecionado para benchmark.")

    print("Dataset:", args.dataset)
    print("Selected cases:", len(selected))
    print("Expected agents:", sorted({c.expected_agent_id for c in selected}))
    print("Concurrency:", args.concurrency)

    init_pools()
    try:
        warmup_cases = _iter_warmup(selected, max(0, args.warmup))
        if warmup_cases:
            print(f"Warmup: {len(warmup_cases)} case(s)")
            asyncio.run(
                _run_all(
                    warmup_cases,
                    concurrency=1,
                    max_retries=args.max_retries,
                    retry_base_seconds=args.retry_base_seconds,
                    inter_request_sleep=args.inter_request_sleep,
                )
            )

        started = time.perf_counter()
        results = asyncio.run(
            _run_all(
                selected,
                concurrency=args.concurrency,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                inter_request_sleep=args.inter_request_sleep,
            )
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
    finally:
        close_pools()

    _print_summary(results)
    print(f"\nwall_time_ms: {wall_ms:.1f}")

    if args.output_json:
        out = _results_to_json(results)
        out["wall_time_ms"] = wall_ms
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json_saved: {output_path}")


if __name__ == "__main__":
    main()
