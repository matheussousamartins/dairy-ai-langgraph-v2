"""Consome o dataset rag_queries.yaml e mede cobertura de retrieval.

Uso:
  python scripts/consume_rag_queries.py
  python scripts/consume_rag_queries.py --search-type text --k 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.connection import init_pools, close_pools
from app.rag.search import search_knowledge_base


def build_query(item: dict[str, Any]) -> str:
    expected_all = item.get("expected_all") or []
    if expected_all:
        return " ".join(str(x) for x in expected_all).strip()
    expected = str(item.get("expected") or "").strip()
    if expected:
        return expected
    return str(item.get("pergunta") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/rag/rag_queries.yaml",
        help="Caminho do arquivo rag_queries.yaml",
    )
    parser.add_argument(
        "--search-type",
        default="text",
        choices=["text", "vector", "hybrid_rrf"],
        help="Estratégia de busca para o teste rápido.",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k da busca.")
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="Exibe as perguntas sem resultado.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    data = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    queries = data.get("queries") or []
    if not queries:
        print("Nenhuma query encontrada no dataset.")
        return

    stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "hit": 0, "miss": 0}
    )
    misses: list[tuple[int, str]] = []

    init_pools()
    try:
        for item in queries:
            agent_id = int(item["agent_id"])
            table_name = str(item["table_name"])
            query = build_query(item)

            stats[agent_id]["total"] += 1
            results = search_knowledge_base(
                query=query,
                table_name=table_name,
                search_type=args.search_type,
                k=args.k,
            )
            if results:
                stats[agent_id]["hit"] += 1
            else:
                stats[agent_id]["miss"] += 1
                misses.append((agent_id, str(item.get("pergunta", ""))))
    finally:
        close_pools()

    total = len(queries)
    total_hits = sum(s["hit"] for s in stats.values())
    total_miss = total - total_hits
    coverage = (100.0 * total_hits / total) if total else 0.0

    print(f"Dataset: {dataset_path}")
    print(f"Search type: {args.search_type} | k={args.k}")
    print(f"Total: {total} | Hits: {total_hits} | Misses: {total_miss}")
    print(f"Coverage: {coverage:.1f}%")
    print("\nPor agente:")
    for agent_id in sorted(stats):
        s = stats[agent_id]
        agent_cov = (100.0 * s["hit"] / s["total"]) if s["total"] else 0.0
        print(
            f"- agente {agent_id}: "
            f"{s['hit']}/{s['total']} ({agent_cov:.1f}%)"
        )

    if args.show_misses and misses:
        print("\nPerguntas sem resultado:")
        for agent_id, pergunta in misses:
            print(f"- [{agent_id}] {pergunta}")


if __name__ == "__main__":
    main()
