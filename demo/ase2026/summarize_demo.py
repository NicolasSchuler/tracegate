from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an ASE 2026 TraceGate demo run.")
    parser.add_argument("--run-id", required=True, help="Run id under evals/results and evals/artifacts.")
    parser.add_argument(
        "--results-dir",
        default="evals/results",
        help="Directory containing <run-id>.jsonl and policy logs.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="evals/artifacts",
        help="Directory containing <run-id>/contexts.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    artifacts_dir = Path(args.artifacts_dir)
    run_id = str(args.run_id)
    results_path = results_dir / f"{run_id}.jsonl"
    helper_events_path = results_dir / f"{run_id}.adaptive_policy.jsonl"
    budget_events_path = results_dir / f"{run_id}.adaptive_budget_policy.jsonl"

    if not results_path.exists():
        print(f"Missing results file: {results_path}")
        return 2

    rows = _load_jsonl(results_path)
    if not rows:
        print(f"No attempt records found in {results_path}")
        return 1

    terminal = _terminal_by_case(rows)
    successes = sum(1 for row in terminal.values() if bool(row.get("success")))
    print(f"Run: {run_id}")
    print(f"Cases: {len(terminal)} terminal, {successes} success, {len(terminal) - successes} failed")

    helper_counts: Counter[str] = Counter()
    helper_outcomes: Counter[str] = Counter()
    budget_counts: Counter[str] = Counter()
    prompt_paths = 0
    evidence_paths = 0

    print()
    print(
        "case_id                                  att ok note                  helper -> executed / outcome                  budget x   tokens"
    )
    print(
        "---------------------------------------- --- -- --------------------- -------------------------------------------- ---------- ------"
    )
    for row in rows:
        md = _metadata(row)
        helper = str(md.get("adaptive_helper_action_selected") or "none")
        executed = str(md.get("adaptive_helper_action_executed") or "none")
        outcome = str(md.get("adaptive_helper_outcome") or "none")
        budget = str(md.get("adaptive_budget_action_selected") or "none")
        multiplier = md.get("adaptive_budget_applied_multiplier")
        tokens = md.get("tokens_total")
        note = str(row.get("note") or "")
        ok = "Y" if row.get("success") else "N"
        helper_counts[helper] += 1
        helper_outcomes[outcome] += 1
        budget_counts[budget] += 1
        prompt_paths += 1 if md.get("prompt_path") else 0
        evidence_paths += 1 if md.get("evidence_path") else 0
        print(
            f"{str(row.get('case_id') or '')[:40]:40} "
            f"{int(row.get('attempt') or 0):3d} "
            f"{ok:2} "
            f"{note[:21]:21} "
            f"{helper[:18]:18} -> {executed[:12]:12} / {outcome[:12]:12} "
            f"{budget[:10]:10} "
            f"{_fmt_float(multiplier):>4} "
            f"{_fmt_int(tokens):>6}"
        )

    print()
    print("Helper actions:", _format_counter(helper_counts))
    print("Helper outcomes:", _format_counter(helper_outcomes))
    print("Budget actions:", _format_counter(budget_counts))
    print(f"Recorded prompt paths: {prompt_paths}/{len(rows)}")
    print(f"Recorded evidence paths: {evidence_paths}/{len(rows)}")

    helper_events = _safe_count_jsonl(helper_events_path)
    budget_events = _safe_count_jsonl(budget_events_path)
    print(f"Policy events: helper={helper_events}, budget={budget_events}")

    first_success = next((row for row in rows if row.get("success")), None)
    if first_success is not None:
        attempt_dir = _attempt_dir(
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            case_id=str(first_success.get("case_id") or ""),
            condition=str(first_success.get("condition") or "layered_adaptive_llm"),
            attempt=int(first_success.get("attempt") or 0),
        )
        print()
        print("First successful attempt artifacts:")
        print(f"  {attempt_dir}")
        for name in (
            "snapshot.json",
            "evidence.json",
            "prompt.txt",
            "patch.diff",
            "pytest_before_combined.txt",
            "pytest_after_combined.txt",
        ):
            marker = "ok" if (attempt_dir / name).exists() else "missing"
            print(f"  [{marker}] {attempt_dir / name}")

    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _terminal_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    terminal: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        attempt = int(row.get("attempt") or 0)
        if case_id not in terminal or attempt >= int(terminal[case_id].get("attempt") or 0):
            terminal[case_id] = row
    return terminal


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("attempt_metadata")
    return value if isinstance(value, dict) else {}


def _safe_count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _attempt_dir(
    *,
    artifacts_dir: Path,
    run_id: str,
    case_id: str,
    condition: str,
    attempt: int,
) -> Path:
    return (
        artifacts_dir
        / run_id
        / "contexts"
        / case_id
        / condition
        / f"attempt_{attempt:02d}"
    )


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common())


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
