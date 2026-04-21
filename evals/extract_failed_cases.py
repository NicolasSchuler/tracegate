"""Extract unsolved case IDs from a single eval JSONL run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import select_cases


def _default_output_path(results_jsonl: Path) -> Path:
    """Return the default newline-delimited case-list path for *results_jsonl*."""
    if results_jsonl.suffix == ".jsonl":
        return results_jsonl.with_suffix(".failed.txt")
    return results_jsonl.with_name(f"{results_jsonl.name}.failed.txt")


def _resolve_condition(*, rows: list[dict[str, Any]], requested: str) -> str:
    """Resolve the target condition, inferring it when the file is unambiguous."""
    condition = str(requested).strip()
    if condition:
        return condition

    conditions = {
        str(row.get("condition") or "").strip()
        for row in rows
        if str(row.get("record_type") or "attempt").strip() != "infra_event"
        and str(row.get("condition") or "").strip()
    }
    if len(conditions) == 1:
        return next(iter(conditions))
    if not conditions:
        raise ValueError("No attempt rows with a non-empty condition were found.")
    available = ", ".join(sorted(conditions))
    raise ValueError(
        "Multiple conditions found in results JSONL; pass --condition explicitly. "
        f"Available conditions: {available}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for extracting unsolved case IDs from one eval run."""
    parser = argparse.ArgumentParser(
        description="Extract unsolved case IDs from a single eval JSONL run."
    )
    parser.add_argument(
        "results_jsonl",
        help="Path to eval JSONL file (attempt-level records).",
    )
    parser.add_argument(
        "--condition",
        default="",
        help="Condition to extract. If omitted, infer it when the file has only one condition.",
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
        default="",
        help=(
            "Output path for newline-delimited failed case IDs "
            "(default: alongside the input as <run>.failed.txt)."
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(str(args.results_jsonl)).expanduser()
    if not input_path.exists() or not input_path.is_file():
        print(f"Results JSONL not found or not a file: {input_path}")
        return 2

    try:
        rows = select_cases._load_jsonl(input_path)
        condition = _resolve_condition(rows=rows, requested=str(args.condition))
    except ValueError as exc:
        print(str(exc))
        return 2

    terminal_rows = select_cases._terminal_by_case(rows=rows, condition=condition)
    if not terminal_rows:
        print(f"No rows found for condition={condition!r} in {input_path}")
        return 2

    selected_case_ids = select_cases._select_case_ids(
        terminal_rows=terminal_rows,
        mode="failed",
        attempt_threshold=1,
        max_attempt=int(args.max_attempt),
    )
    output_path = (
        Path(str(args.out)).expanduser()
        if str(args.out).strip()
        else _default_output_path(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{case_id}\n" for case_id in selected_case_ids),
        encoding="utf-8",
    )

    success_count = sum(1 for row in terminal_rows.values() if bool(row.get("success")))
    print(
        "Extracted "
        f"{len(selected_case_ids)} failed case(s) from {len(terminal_rows)} terminal row(s) "
        f"for condition={condition!r} (terminal successes={success_count})."
    )
    print(f"Wrote case list: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
