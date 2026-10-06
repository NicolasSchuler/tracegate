"""Select rerun case IDs from eval JSONL outputs.

Reads attempt-level JSONL records, identifies terminal rows per case for a
given condition, and writes a newline-delimited list of case IDs matching
the chosen selection mode (failed, attempt-ge, or failed-or-attempt-ge).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import _json as json
from .case_outcome import case_outcomes


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of dicts.

    Raises ``ValueError`` on malformed lines or non-object rows.
    """
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object row")
        rows.append(payload)
    return rows


def _coerce_attempt(value: Any) -> int:
    """Coerce *value* to a non-negative attempt number (0 on failure)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _terminal_by_case(
    *,
    rows: list[dict[str, Any]],
    condition: str,
) -> dict[str, dict[str, Any]]:
    """Return the terminal (highest-attempt) row per case for *condition*.

    Outcomes come from :func:`evals.case_outcome.case_outcomes` (STA-19, DAT-24), so a
    case listed here as failed is also a failure in ``analyze_results``. The returned
    row's ``success`` is the case outcome. With several runs in one file, a case
    succeeded only if it succeeded in every run, and the highest terminal attempt is
    kept.
    """
    terminal: dict[str, dict[str, Any]] = {}
    for (_run_id, case_id, _condition), outcome in case_outcomes(rows, condition=condition).items():
        row = {**outcome.terminal, "success": outcome.success}
        previous = terminal.get(case_id)
        if previous is None:
            terminal[case_id] = row
            continue
        terminal[case_id] = {
            **(
                row
                if _coerce_attempt(row.get("attempt")) >= _coerce_attempt(previous.get("attempt"))
                else previous
            ),
            "success": bool(previous.get("success")) and outcome.success,
        }
    return terminal


def _select_case_ids(
    *,
    terminal_rows: dict[str, dict[str, Any]],
    mode: str,
    attempt_threshold: int,
    max_attempt: int,
) -> list[str]:
    """Filter terminal rows by *mode* and return sorted selected case IDs.

    Modes:
      * ``failed`` — select cases that did not succeed.
      * ``attempt-ge`` — select cases with attempt >= *attempt_threshold*.
      * ``failed-or-attempt-ge`` — union of both criteria.
    """
    selected: list[str] = []
    threshold = max(1, int(attempt_threshold))
    effective_max_attempt = max(0, int(max_attempt))
    for case_id, row in terminal_rows.items():
        success = bool(row.get("success"))
        attempt = _coerce_attempt(row.get("attempt"))
        if effective_max_attempt > 0 and attempt > effective_max_attempt:
            continue
        is_failed = not success
        is_attempt_ge = attempt >= threshold
        include = False
        if mode == "failed":
            include = is_failed
        elif mode == "attempt-ge":
            include = is_attempt_ge
        elif mode == "failed-or-attempt-ge":
            include = is_failed or is_attempt_ge
        if include:
            selected.append(case_id)
    return sorted(selected)


def _is_request_budget_exceeded_failure(row: dict[str, Any]) -> bool:
    """Check whether *row* failed due to ``request_budget_exceeded``."""
    if bool(row.get("success")):
        return False
    note = str(row.get("note") or "").strip()
    if note and note != "patcher_request_failed":
        return False

    patch_meta = row.get("patch_meta")
    if isinstance(patch_meta, dict):
        error = str(
            patch_meta.get("patcher_request_failure_error") or patch_meta.get("error") or ""
        ).strip()
        if error == "request_budget_exceeded":
            return True

    attempt_meta = row.get("attempt_metadata")
    if isinstance(attempt_meta, dict):
        error = str(
            attempt_meta.get("patcher_request_failure_error") or attempt_meta.get("error") or ""
        ).strip()
        if error == "request_budget_exceeded":
            return True

    return False


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: select rerun case IDs and write them to a file."""
    parser = argparse.ArgumentParser(description="Select rerun case IDs from eval JSONL outputs.")
    parser.add_argument(
        "--results-jsonl",
        required=True,
        help="Path to eval JSONL file (attempt-level records).",
    )
    parser.add_argument(
        "--condition",
        default="base_repair",
        help="Condition to read terminal rows from (default: base_repair).",
    )
    parser.add_argument(
        "--mode",
        choices=["failed", "attempt-ge", "failed-or-attempt-ge"],
        default="failed",
        help="Selection mode (default: failed).",
    )
    parser.add_argument(
        "--attempt-threshold",
        type=int,
        default=2,
        help="Attempt threshold for attempt-ge modes (default: 2).",
    )
    parser.add_argument(
        "--max-attempt",
        type=int,
        default=0,
        help=(
            "Optional terminal attempt cap (0 disables). "
            "Rows with attempt > max-attempt are excluded."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for newline-delimited case IDs.",
    )
    args = parser.parse_args(argv)

    input_path = Path(str(args.results_jsonl)).expanduser()
    if not input_path.exists() or not input_path.is_file():
        print(f"Results JSONL not found or not a file: {input_path}")
        return 2

    output_path = Path(str(args.out)).expanduser()
    try:
        rows = _load_jsonl(input_path)
    except ValueError as exc:
        print(str(exc))
        return 2

    condition = str(args.condition).strip()
    if not condition:
        print("Invalid --condition: must be a non-empty string")
        return 2

    terminal_rows = _terminal_by_case(rows=rows, condition=condition)
    if not terminal_rows:
        print(f"No rows found for condition={condition!r} in {input_path}")
        return 2

    selected_case_ids = _select_case_ids(
        terminal_rows=terminal_rows,
        mode=str(args.mode),
        attempt_threshold=int(args.attempt_threshold),
        max_attempt=int(args.max_attempt),
    )
    terminal_budget_failed = {
        case_id
        for case_id, row in terminal_rows.items()
        if _is_request_budget_exceeded_failure(row)
    }
    selected_set = set(selected_case_ids)
    selected_budget_failed = len(terminal_budget_failed.intersection(selected_set))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{case_id}\n" for case_id in selected_case_ids),
        encoding="utf-8",
    )

    success_count = sum(1 for row in terminal_rows.values() if bool(row.get("success")))
    print(
        "Selected "
        f"{len(selected_case_ids)} case(s) from {len(terminal_rows)} terminal row(s) "
        f"for condition={condition!r}, mode={args.mode!r}, "
        f"attempt_threshold={int(args.attempt_threshold)}, "
        f"max_attempt={int(args.max_attempt)} (terminal successes={success_count})."
    )
    print(
        "Terminal request_budget_exceeded failures: "
        f"{len(terminal_budget_failed)} (selected={selected_budget_failed})."
    )
    print(f"Wrote case list: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
