from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.agents.agent_config import AGENTS


ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "tests" / "fixtures" / "rag" / "rag_queries.yaml"
OUT_DIR = ROOT / "docs" / "orchestrator" / "day1"
OUT_FILE = OUT_DIR / "ROUTING_BASELINE_REPORT.md"


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries") or []
    return [q for q in queries if isinstance(q, dict)]


def _agent_map() -> dict[int, dict[str, Any]]:
    return {int(a["agent_id"]): a for a in AGENTS}


def _fmt_pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


def build_report() -> str:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    queries = _load_dataset(DATASET_FILE)
    amap = _agent_map()

    total = len(queries)
    by_agent = Counter()
    by_table = Counter()
    by_group = Counter()
    group_by_agent = defaultdict(Counter)
    multi_expected = 0
    empty_expected = 0

    for q in queries:
        aid = int(q.get("agent_id", -1))
        group = str(q.get("group", "unknown"))
        table = str(q.get("table_name", "unknown"))
        expected_all = q.get("expected_all") or []
        expected_one = q.get("expected")

        by_agent[aid] += 1
        by_table[table] += 1
        by_group[group] += 1
        group_by_agent[aid][group] += 1

        if isinstance(expected_all, list) and len(expected_all) > 1:
            multi_expected += 1
        if not expected_all and not expected_one:
            empty_expected += 1

    covered_agents = sorted(a for a in by_agent if a in amap)
    missing_agents = sorted(set(amap.keys()) - set(covered_agents))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines: list[str] = []
    lines.append("# Routing Baseline Report (Day 1)")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{generated_at}`")
    lines.append(f"- Dataset file: `{DATASET_FILE.as_posix()}`")
    lines.append(f"- Total queries: **{total}**")
    lines.append(f"- Agents covered in dataset: **{len(covered_agents)} / {len(amap)}**")
    lines.append(f"- Queries with multi-term expected_all: **{multi_expected}**")
    lines.append(f"- Queries with no expected field: **{empty_expected}**")
    lines.append("")
    lines.append("## Agent Distribution")
    lines.append("")
    lines.append("| agent_id | agent_name | queries | share |")
    lines.append("|---:|---|---:|---:|")
    for aid in sorted(by_agent):
        name = amap.get(aid, {}).get("name", "unknown")
        count = by_agent[aid]
        lines.append(f"| {aid} | {name} | {count} | {_fmt_pct(count, total)} |")
    lines.append("")
    if missing_agents:
        lines.append(f"- Missing agent_ids in dataset: `{missing_agents}`")
    else:
        lines.append("- Missing agent_ids in dataset: `[]`")
    lines.append("")
    lines.append("## Table Distribution")
    lines.append("")
    lines.append("| table_name | queries | share |")
    lines.append("|---|---:|---:|")
    for t, count in by_table.most_common():
        lines.append(f"| {t} | {count} | {_fmt_pct(count, total)} |")
    lines.append("")
    lines.append("## Top Query Groups")
    lines.append("")
    lines.append("| group | queries |")
    lines.append("|---|---:|")
    for g, count in by_group.most_common(25):
        lines.append(f"| {g} | {count} |")
    lines.append("")
    lines.append("## Per-Agent Group Spread")
    lines.append("")
    for aid in sorted(group_by_agent):
        name = amap.get(aid, {}).get("name", "unknown")
        lines.append(f"### Agent {aid} - {name}")
        lines.append("")
        lines.append("| group | queries |")
        lines.append("|---|---:|")
        for g, count in group_by_agent[aid].most_common(15):
            lines.append(f"| {g} | {count} |")
        lines.append("")
    lines.append("## Baseline Metric Contract (Day 1)")
    lines.append("")
    lines.append("Track these as official routing KPIs from Day 2 onward:")
    lines.append("")
    lines.append("- `Routing@1`: selected primary agent matches expected primary domain.")
    lines.append("- `Routing@3`: expected domain appears in top-3 selected agents.")
    lines.append("- `Fallback Rate`: % requests that needed secondary routing attempt.")
    lines.append("- `Cross-Agent Conflict Rate`: % responses with conflicting specialist claims.")
    lines.append("- `Answer Accuracy`: judged against dataset expected answer.")
    lines.append("- `P95 Latency`: end-to-end response time.")
    lines.append("- `Cost per Request`: model + retrieval cost.")
    lines.append("")
    lines.append("Initial target bands (enterprise-ready, to validate in Day 2):")
    lines.append("")
    lines.append("- `Routing@1 >= 90%`")
    lines.append("- `Routing@3 >= 97%`")
    lines.append("- `Fallback Rate <= 12%`")
    lines.append("- `Cross-Agent Conflict Rate <= 3%`")
    lines.append("- `P95 Latency <= 4.5s` (orchestrator stream)")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUT_FILE.write_text(report, encoding="utf-8")
    print(f"[ok] Wrote: {OUT_FILE}")


if __name__ == "__main__":
    main()

