"""
Phase 3 single-agent evaluation.

This test treats RAG_ARCHITECTURE=single_agent as its own product surface:
question -> V2 graph -> final answer. It complements the specialist-agent
phase3 test and gives the simplified pipeline a dedicated quality gate.
"""

import asyncio
import os
import re

import psycopg
import pytest
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.phase3


_FORBIDDEN_ANSWER_FRAGMENTS = (
    "base de conhecimento",
    "trechos recuperados",
    "meu conhecimento atual",
    "agente especialista",
    "embeddings_agente",
    "source_table",
)


def _row_question(row: dict) -> str:
    return str(row.get("question") or row.get("pergunta") or "").strip()


def _row_expected(row: dict) -> str:
    expected_all = row.get("expected_all")
    if expected_all:
        return ", ".join(str(item) for item in expected_all)
    return str(row.get("answer") or row.get("expected") or "").strip()


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text or ""))


def _table_has_data(db_url: str, table_name: str) -> bool:
    try:
        with psycopg.connect(db_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                return int(cur.fetchone()[0]) > 0
    except Exception:
        return False


def test_phase3_single_agent_e2e(
    db_supabase,
    require_openai,
    rag_dataset,
    max_queries,
    llm_judge_yesno,
):
    from app.graphs.single_agent_graph import get_single_agent_graph

    candidates = [
        row for row in rag_dataset
        if _row_question(row) and _row_expected(row)
    ]

    table_names = sorted({str(row.get("table_name") or "") for row in candidates})
    table_cache = {
        table_name: _table_has_data(db_supabase, table_name)
        for table_name in table_names
        if table_name
    }
    eligible = [
        row for row in candidates
        if table_cache.get(str(row.get("table_name") or ""), False)
    ]

    if max_queries:
        eligible = eligible[:max_queries]

    if not eligible:
        pytest.skip("No single-agent dataset rows with populated KB tables")

    graph = get_single_agent_graph()
    answer_timeout_sec = float(os.getenv("SINGLE_AGENT_E2E_TIMEOUT_SEC", "60"))
    total = 0
    judge_yes = 0
    numeric_total = 0
    numeric_preserved = 0
    forbidden_hits = []
    errors = []

    print("\n" + "=" * 60)
    print("FASE 3 - SINGLE AGENT E2E")
    print("=" * 60)

    for row in eligible:
        question = _row_question(row)
        expected = _row_expected(row)
        total += 1

        try:
            result = asyncio.run(asyncio.wait_for(
                graph.ainvoke({"messages": [HumanMessage(content=question)]}),
                timeout=answer_timeout_sec,
            ))
            answer = str((result or {}).get("final_response") or "").strip()
        except asyncio.TimeoutError:
            errors.append((question, f"timeout after {answer_timeout_sec}s"))
            continue
        except Exception as exc:
            errors.append((question, str(exc)))
            continue

        lowered = answer.lower()
        for fragment in _FORBIDDEN_ANSWER_FRAGMENTS:
            if fragment in lowered:
                forbidden_hits.append((question, fragment))

        expected_numbers = _extract_numbers(expected)
        if expected_numbers:
            numeric_total += 1
            answer_numbers = _extract_numbers(answer)
            if expected_numbers & answer_numbers:
                numeric_preserved += 1

        try:
            ok = llm_judge_yesno(question, expected, [answer])
        except Exception:
            ok = False
        judge_yes += 1 if ok else 0

    approval_rate = judge_yes / max(total, 1)
    numeric_rate = numeric_preserved / max(numeric_total, 1)

    print(f"Perguntas: {total}")
    print(f"LLM approved: {approval_rate:.0%} ({judge_yes}/{total})")
    if numeric_total:
        print(f"Numeros preservados: {numeric_rate:.0%} ({numeric_preserved}/{numeric_total})")
    if errors:
        print(f"Erros: {len(errors)}")

    assert not errors[:3], f"Single-agent errors: {errors[:3]}"
    assert not forbidden_hits[:3], f"Forbidden internal wording leaked: {forbidden_hits[:3]}"
    assert approval_rate >= 0.55, (
        f"Single-agent approval rate ({approval_rate:.0%}) below minimum (55%)"
    )
    if numeric_total:
        assert numeric_rate >= 0.70, (
            f"Numeric preservation rate ({numeric_rate:.0%}) below minimum (70%)"
        )
