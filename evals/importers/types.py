"""Shared importer data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImportRunSummary:
    """Common importer summary payload."""

    source: str
    attempted: int
    imported: int
    skipped: int
    skip_reasons: dict[str, int]
    imported_case_ids: list[str]
    notes: list[str] = field(default_factory=list)
