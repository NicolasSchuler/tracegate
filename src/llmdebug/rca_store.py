"""SQLite-backed RCA persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .state_db import (
    append_rca_record,
    append_rca_records_bulk,
    count_rca_records,
    fetch_latest_rca_record,
    fetch_rca_history_records,
    stored_rca_attempt_ids,
)


def store_rca_record(record: Mapping[str, Any], *, out_dir: str, max_records: int) -> None:
    """Persist one RCA record to SQLite.

    Args:
        record: RCA attempt record; it replaces any stored row with the same
            ``attempt_id``.
        out_dir: Snapshot directory that holds the state database.
        max_records: Retention limit applied after the write; ``0`` or less keeps every
            row.
    """
    append_rca_record(out_dir, record, max_records=max_records)


def store_rca_records_bulk(
    records: list[dict[str, Any]],
    *,
    out_dir: str,
    max_records: int,
    keep_existing: bool = False,
) -> None:
    """Persist multiple RCA records to SQLite.

    Args:
        records: RCA attempt records to write.
        out_dir: Snapshot directory that holds the state database.
        max_records: Retention limit applied after the write; ``0`` or less keeps every
            row.
        keep_existing: When true, records whose ``attempt_id`` is already stored are
            skipped so the stored rows win; otherwise they replace the stored rows.
    """
    append_rca_records_bulk(out_dir, records, max_records=max_records, keep_existing=keep_existing)


def has_any_records(*, out_dir: str) -> bool:
    """Return True if the SQLite RCA table already has rows.

    Args:
        out_dir: Snapshot directory that holds the state database.

    Returns:
        True when at least one RCA attempt is stored; False otherwise, including when
        the database does not exist.
    """
    return count_rca_records(out_dir) > 0


def latest_record(*, out_dir: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Return the latest stored RCA record, optionally for one session.

    Args:
        out_dir: Snapshot directory that holds the state database.
        session_id: Session to filter on; ``None`` or empty considers every session.

    Returns:
        The newest record, or ``None`` when nothing is stored or the newest row's
        payload cannot be decoded.
    """
    return fetch_latest_rca_record(out_dir, session_id=session_id)


def history(
    *,
    out_dir: str,
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return stored RCA history, newest first.

    Args:
        out_dir: Snapshot directory that holds the state database.
        session_id: Session to filter on; ``None`` or empty considers every session.
        limit: Maximum number of records; ``0`` or less returns all of them.

    Returns:
        Decoded RCA records ordered from newest to oldest; rows with malformed JSON
        are skipped.
    """
    return fetch_rca_history_records(out_dir, session_id=session_id, limit=limit)


def stored_attempt_ids(*, out_dir: str, attempt_ids: list[str]) -> set[str]:
    """Return which of *attempt_ids* are already stored in SQLite.

    Args:
        out_dir: Snapshot directory that holds the state database.
        attempt_ids: Candidate attempt IDs; empty strings are ignored.

    Returns:
        The subset of *attempt_ids* that already have a stored row.
    """
    return stored_rca_attempt_ids(out_dir, attempt_ids)
