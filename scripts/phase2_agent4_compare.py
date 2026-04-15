"""Compara vector vs hybrid_rrf para o agente 4 usando rag_queries.yaml.

Uso (PowerShell):
  $env:PYTHONPATH='.'
  python scripts/phase2_agent4_compare.py
"""

import unicodedata
from pathlib import Path

import yaml

from app.db.connection import init_pools, close_pools
from app.rag.search import search_knowledge_base


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def _hit(blob: str, expected: str | None, expected_all: list[str] | None) -> bool:
    b = _norm(blob)
    if expected_all:
        return all(_norm(term) in b for term in expected_all)
    if expected:
        return _norm(expected) in b
    return bool(blob.strip())


def main() -> None:
    ds_path = Path("tests/fixtures/rag/rag_queries.yaml")
    data = yaml.safe_load(ds_path.read_text(encoding="utf-8")) or {}
    queries = [q for q in (data.get("queries") or []) if q.get("agent_id") == 4]

    if not queries:
        print("Nenhuma query do agente 4 encontrada no dataset.")
        return

    search_types = ["vector", "hybrid_rrf"]

    init_pools()
    try:
        print("Comparando agente 4 (vector vs hybrid_rrf)")
        print("-" * 52)
        for st in search_types:
            hits = 0
            total = 0
            for q in queries:
                total += 1
                results = search_knowledge_base(
                    query=q["pergunta"],
                    table_name=q["table_name"],
                    search_type=st,
                    k=5,
                    threshold=0.0,
                )
                blob = "\n".join(r.get("content", "") for r in results)
                ok = _hit(blob, q.get("expected"), q.get("expected_all"))
                hits += 1 if ok else 0

            rate = hits / max(total, 1)
            print(f"{st:>10}: {hits:>2}/{total:<2}  ({rate:.0%})")
    finally:
        close_pools()


if __name__ == "__main__":
    main()
