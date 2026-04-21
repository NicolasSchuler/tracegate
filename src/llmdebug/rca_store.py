"""SQLite-backed RCA persistence helpers."""

from __future__ import annotations

from typing import Any

from .state_db import (
    append_rca_record,
    append_rca_records_bulk,
    count_rca_records,
    fetch_latest_rca_record,
    fetch_rca_history_records,
)


def store_rca_record(record: dict[str, Any], *, out_dir: str, max_records: int) -> None:
    """Persist one RCA record to SQLite."""
    append_rca_record(out_dir, record, max_records=max_records)


def store_rca_records_bulk(
    records: list[dict[str, Any]],
    *,
    out_dir: str,
    max_records: int,
) -> None:
    """Persist multiple RCA records to SQLite."""
    append_rca_records_bulk(out_dir, records, max_records=max_records)


def has_any_records(*, out_dir: str) -> bool:
    """Return True if the SQLite RCA table already has rows."""
    return count_rca_records(out_dir) > 0


def latest_record(*, out_dir: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Fetch latest RCA row, optionally for one session."""
    return fetch_latest_rca_record(out_dir, session_id=session_id)


def history(
    *,
    out_dir: str,
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch RCA history, newest first."""
    return fetch_rca_history_records(out_dir, session_id=session_id, limit=limit)
