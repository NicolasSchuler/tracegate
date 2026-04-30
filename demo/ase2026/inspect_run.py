from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARTIFACT_NAMES = (
    "snapshot.json",
    "evidence.json",
    "prompt.txt",
    "patch.diff",
    "pytest_before_combined.txt",
    "pytest_after_combined.txt",
)
PREVIEW_NAMES = (
    ("Patch Diff", "patch.diff"),
    ("Validation Output", "pytest_after_combined.txt"),
    ("Prompt Head", "prompt.txt"),
    ("Evidence Head", "evidence.json"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an ASE 2026 TraceGate demo run for screencast navigation."
    )
    parser.add_argument("--run-id", required=True, help="Run id under evals/results.")
    parser.add_argument("--case", dest="case_id", default="", help="Optional case id.")
    parser.add_argument("--attempt", type=int, default=None, help="Optional attempt number.")
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
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=12,
        help="Lines to show for each artifact preview.",
    )
    parser.add_argument(
        "--timeline-limit",
        type=int,
        default=40,
        help="Maximum attempt rows to show in the timeline; 0 means all.",
    )
    args = parser.parse_args()

    run_id = str(args.run_id)
    results_dir = Path(args.results_dir)
    artifacts_dir = Path(args.artifacts_dir)
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

    helper_events = _load_jsonl_if_exists(helper_events_path)
    budget_events = _load_jsonl_if_exists(budget_events_path)
    selected = _select_attempt(
        rows=rows,
        case_id=str(args.case_id or ""),
        attempt=args.attempt,
    )
    if selected is None:
        print("No matching attempt found.")
        return 1

    attempt_dir = _resolve_attempt_dir(
        row=selected,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
    )

    _print_overview(
        run_id=run_id,
        results_path=results_path,
        artifacts_dir=artifacts_dir,
        rows=rows,
        helper_events=helper_events,
        budget_events=budget_events,
    )
    timeline_rows = rows
    if args.case_id:
        timeline_rows = [row for row in rows if str(row.get("case_id") or "") == args.case_id]
    _print_timeline(rows=timeline_rows, limit=max(0, int(args.timeline_limit)))
    _print_selected_attempt(row=selected, attempt_dir=attempt_dir)
    _print_policy_decisions(
        row=selected,
        helper_events=helper_events,
        budget_events=budget_events,
    )
    _print_artifact_previews(
        attempt_dir=attempt_dir,
        preview_lines=max(0, int(args.preview_lines)),
    )
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_jsonl(path)


def _select_attempt(
    *,
    rows: list[dict[str, Any]],
    case_id: str,
    attempt: int | None,
) -> dict[str, Any] | None:
    candidates = [row for row in rows if not case_id or str(row.get("case_id") or "") == case_id]
    if not candidates:
        return None

    if attempt is not None:
        exact = [row for row in candidates if int(row.get("attempt") or 0) == int(attempt)]
        if exact:
            return exact[0]
        return None

    materialized_success = [
        row
        for row in candidates
        if bool(row.get("success"))
        and str(_metadata(row).get("adaptive_helper_outcome") or "") == "materialized"
    ]
    if materialized_success:
        return materialized_success[0]

    success = [row for row in candidates if bool(row.get("success"))]
    if success:
        return success[0]

    first_case_id = str(candidates[0].get("case_id") or "")
    same_case = [row for row in candidates if str(row.get("case_id") or "") == first_case_id]
    return max(same_case, key=lambda row: int(row.get("attempt") or 0), default=candidates[-1])


def _resolve_attempt_dir(*, row: dict[str, Any], artifacts_dir: Path, run_id: str) -> Path:
    md = _metadata(row)
    for key in ("prompt_path", "evidence_path"):
        raw = md.get(key)
        if isinstance(raw, str) and raw:
            parent = Path(raw).expanduser().parent
            if parent.exists():
                return parent

    case_id = str(row.get("case_id") or "")
    condition = str(row.get("condition") or "layered_adaptive_llm")
    attempt = int(row.get("attempt") or 0)
    return artifacts_dir / run_id / "contexts" / case_id / condition / f"attempt_{attempt:02d}"


def _print_overview(
    *,
    run_id: str,
    results_path: Path,
    artifacts_dir: Path,
    rows: list[dict[str, Any]],
    helper_events: list[dict[str, Any]],
    budget_events: list[dict[str, Any]],
) -> None:
    terminal = _terminal_by_case(rows)
    successes = sum(1 for row in terminal.values() if bool(row.get("success")))
    prompt_recorded, prompt_existing = _path_coverage(rows, key="prompt_path")
    evidence_recorded, evidence_existing = _path_coverage(rows, key="evidence_path")

    print(_section("Run Overview"))
    print(f"run_id:          {run_id}")
    print(f"results:         {results_path}")
    print(f"artifacts:       {artifacts_dir / run_id}")
    print(f"attempt rows:    {len(rows)}")
    print(f"terminal cases:  {len(terminal)}")
    print(f"success/failure: {successes}/{len(terminal) - successes}")
    print(f"prompt paths:    {prompt_recorded}/{len(rows)} recorded, {prompt_existing} existing")
    print(f"evidence paths:  {evidence_recorded}/{len(rows)} recorded, {evidence_existing} existing")
    print(f"policy events:   helper={len(helper_events)}, budget={len(budget_events)}")


def _print_timeline(*, rows: list[dict[str, Any]], limit: int) -> None:
    print()
    print(_section("Attempt Timeline"))
    print(
        "case_id                                  att ok note                  "
        "helper -> executed / outcome                  budget x   tokens reward"
    )
    print(
        "---------------------------------------- --- -- --------------------- "
        "-------------------------------------------- ---------- ------ ------"
    )
    visible = rows if limit == 0 else rows[:limit]
    for row in visible:
        md = _metadata(row)
        helper = _short(md.get("adaptive_helper_action_selected"), 18)
        executed = _short(md.get("adaptive_helper_action_executed"), 12)
        outcome = _short(md.get("adaptive_helper_outcome"), 12)
        budget = _short(md.get("adaptive_budget_action_selected"), 10)
        multiplier = _fmt_float(md.get("adaptive_budget_applied_multiplier"), digits=1)
        tokens = _fmt_int(md.get("tokens_total"))
        reward = _fmt_float(md.get("adaptive_online_reward"), digits=2)
        note = _short(row.get("note"), 21)
        ok = "Y" if row.get("success") else "N"
        print(
            f"{_short(row.get('case_id'), 40):40} "
            f"{int(row.get('attempt') or 0):3d} "
            f"{ok:2} "
            f"{note:21} "
            f"{helper:18} -> {executed:12} / {outcome:12} "
            f"{budget:10} "
            f"{multiplier:>4} "
            f"{tokens:>6} "
            f"{reward:>6}"
        )
    if limit and len(rows) > limit:
        print(f"... {len(rows) - limit} more attempt row(s); use --timeline-limit 0 to show all.")


def _print_selected_attempt(*, row: dict[str, Any], attempt_dir: Path) -> None:
    md = _metadata(row)
    print()
    print(_section("Selected Attempt"))
    print(f"case_id:         {row.get('case_id')}")
    print(f"condition:       {row.get('condition')}")
    print(f"attempt:         {row.get('attempt')}")
    print(f"success:         {row.get('success')}")
    print(f"note:            {row.get('note')}")
    print(f"artifact_dir:    {attempt_dir}")
    print(f"helper:          {md.get('adaptive_helper_action_selected')} -> "
          f"{md.get('adaptive_helper_action_executed')} / {md.get('adaptive_helper_outcome')}")
    print(f"budget:          {md.get('adaptive_budget_action_selected')} "
          f"x{_fmt_float(md.get('adaptive_budget_applied_multiplier'), digits=1)}")
    print(f"tokens:          {_fmt_int(md.get('tokens_total'))}")
    print(f"prompt_path:     {md.get('prompt_path') or 'none'}")
    print(f"evidence_path:   {md.get('evidence_path') or 'none'}")
    print()
    print("Artifacts:")
    for name in ARTIFACT_NAMES:
        path = attempt_dir / name
        marker = "ok" if path.exists() else "missing"
        size = f", {path.stat().st_size} bytes" if path.exists() else ""
        print(f"  [{marker}] {path}{size}")


def _print_policy_decisions(
    *,
    row: dict[str, Any],
    helper_events: list[dict[str, Any]],
    budget_events: list[dict[str, Any]],
) -> None:
    case_id = str(row.get("case_id") or "")
    attempt = int(row.get("attempt") or 0)
    helper = _matching_event(helper_events, case_id=case_id, attempt=attempt)
    budget = _matching_event(budget_events, case_id=case_id, attempt=attempt)

    print()
    print(_section("Policy Decisions"))
    _print_policy_event("Helper policy", helper)
    _print_policy_event("Budget policy", budget)


def _print_policy_event(label: str, event: Mapping[str, Any] | None) -> None:
    print(f"{label}:")
    if event is None:
        print("  missing")
        return
    print(f"  mode:       {event.get('mode')}")
    print(f"  algorithm:  {event.get('algorithm')}")
    if event.get("architecture"):
        print(f"  arch:       {event.get('architecture')}")
    print(f"  selected:   {event.get('selected_action')}")
    print(f"  applied:    {event.get('applied_action')}")
    print(f"  behavior:   {event.get('behavior_action')}")
    print(f"  strategy:   {event.get('decision_strategy')}")
    print(f"  reward:     {_fmt_float(event.get('reward'), digits=3)}")
    if event.get("selected_multiplier") is not None:
        print(f"  multiplier: {_fmt_float(event.get('selected_multiplier'), digits=1)} -> "
              f"{_fmt_float(event.get('applied_multiplier'), digits=1)}")
    changed = event.get("applied_controls_changed") or event.get("selected_controls_changed")
    if isinstance(changed, dict):
        active = [key for key, value in changed.items() if bool(value)]
        print(f"  changed:    {', '.join(active) if active else 'none'}")


def _print_artifact_previews(*, attempt_dir: Path, preview_lines: int) -> None:
    print()
    print(_section("Artifact Previews"))
    if preview_lines == 0:
        print("previews disabled (--preview-lines 0)")
        return
    for title, name in PREVIEW_NAMES:
        print()
        print(f"[{title}] {attempt_dir / name}")
        _print_preview(attempt_dir / name, max_lines=preview_lines)


def _print_preview(path: Path, *, max_lines: int) -> None:
    if not path.exists():
        print("  missing")
        return
    shown = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if shown >= max_lines:
                print("  ...")
                return
            print("  " + _short(line.rstrip(), 160))
            shown += 1
    if shown == 0:
        print("  <empty>")


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


def _path_coverage(rows: list[dict[str, Any]], *, key: str) -> tuple[int, int]:
    recorded = 0
    existing = 0
    for row in rows:
        raw = _metadata(row).get(key)
        if not isinstance(raw, str) or not raw:
            continue
        recorded += 1
        if Path(raw).expanduser().exists():
            existing += 1
    return recorded, existing


def _matching_event(
    events: list[dict[str, Any]],
    *,
    case_id: str,
    attempt: int,
) -> dict[str, Any] | None:
    for event in events:
        if str(event.get("case_id") or "") != case_id:
            continue
        if int(event.get("attempt") or 0) == attempt:
            return event
    return None


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("attempt_metadata")
    return value if isinstance(value, dict) else {}


def _section(title: str) -> str:
    return f"== {title} =="


def _short(value: Any, width: int) -> str:
    text = str(value if value is not None else "")
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "~"


def _fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def _fmt_float(value: Any, *, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
