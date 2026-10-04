"""One definition of a case's outcome (2026-10-04 review, STA-19 and DAT-24).

``analyze_results`` counted a case as solved when any attempt succeeded, while
``select_cases`` and ``extract_failed_cases`` read the highest attempt and dropped
runtime-error rows, and ``report_run_metrics`` mixed both. The same case could succeed
in the summary and appear in ``<run>.failed.txt``. Every reader now uses
:func:`case_outcomes` or :func:`case_outcome`:

- ``infra_event`` rows are telemetry, not attempts.
- One row per attempt number. When a resumed unit wrote the same attempt twice, a real
  outcome beats a resume-eligible ``unit_runtime_error`` row, and otherwise the later
  row wins. Readers load files in sorted path order, so the choice is deterministic.
- A case succeeded when any scored attempt succeeded. A ``unit_runtime_error`` row is a
  failed attempt and stays in the denominator (ATT-08).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .eval_types import EVAL_RECORD_TYPE_INFRA_EVENT, normalize_condition_name

CaseKey = tuple[str, str, str]
"""``(run_id, case_id, condition)``."""


def coerce_attempt(value: Any) -> int:
    """Attempt number as a non-negative int (0 when missing or malformed)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def is_infra_event(record: Mapping[str, Any]) -> bool:
    return str(record.get("record_type") or "").strip() == EVAL_RECORD_TYPE_INFRA_EVENT


def is_resume_eligible(record: Mapping[str, Any]) -> bool:
    patch_meta = record.get("patch_meta")
    return isinstance(patch_meta, Mapping) and bool(patch_meta.get("infra_resume_eligible"))


def scored_attempts(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The scored attempt rows of one case, one per attempt number, in attempt order."""
    by_attempt: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if is_infra_event(record):
            continue
        attempt = coerce_attempt(record.get("attempt"))
        current = by_attempt.get(attempt)
        if current is not None and is_resume_eligible(record) and not is_resume_eligible(current):
            continue
        by_attempt[attempt] = record
    return [by_attempt[attempt] for attempt in sorted(by_attempt)]


@dataclass(frozen=True)
class CaseOutcome:
    """The outcome of one ``(run_id, case_id, condition)``."""

    success: bool
    attempts: tuple[Mapping[str, Any], ...]
    success_attempt: int | None

    @property
    def terminal(self) -> Mapping[str, Any]:
        """The highest scored attempt."""
        return self.attempts[-1]

    @property
    def terminal_attempt(self) -> int:
        return coerce_attempt(self.terminal.get("attempt"))

    @property
    def resume_eligible(self) -> bool:
        """The case failed on a unit runtime error that ``--resume`` would retry."""
        return not self.success and is_resume_eligible(self.terminal)


def case_outcome(records: Iterable[Mapping[str, Any]]) -> CaseOutcome | None:
    """Outcome of one case's records, or ``None`` when it has no scored attempt."""
    attempts = scored_attempts(records)
    if not attempts:
        return None
    success_attempt = next(
        (coerce_attempt(row.get("attempt")) for row in attempts if bool(row.get("success"))),
        None,
    )
    return CaseOutcome(
        success=success_attempt is not None,
        attempts=tuple(attempts),
        success_attempt=success_attempt,
    )


def case_outcomes(
    records: Iterable[Mapping[str, Any]], *, condition: str | None = None
) -> dict[CaseKey, CaseOutcome]:
    """Group attempt records by ``(run_id, case_id, condition)`` and resolve each outcome.

    Records without a case ID or condition are skipped. With *condition*, only that
    condition's records are read.
    """
    wanted = normalize_condition_name(condition) if condition is not None else None
    grouped: dict[CaseKey, list[Mapping[str, Any]]] = {}
    for record in records:
        case_id = str(record.get("case_id") or "").strip()
        record_condition = normalize_condition_name(record.get("condition"))
        if not case_id or not record_condition:
            continue
        if wanted is not None and record_condition != wanted:
            continue
        key = (str(record.get("run_id") or ""), case_id, record_condition)
        grouped.setdefault(key, []).append(record)
    outcomes: dict[CaseKey, CaseOutcome] = {}
    for key in sorted(grouped):
        outcome = case_outcome(grouped[key])
        if outcome is not None:
            outcomes[key] = outcome
    return outcomes
