"""RCA state tracking facade for MCP/debug workflows.

This module preserves the public RCA API while delegating internals to focused
identity/transition/backend helpers.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from . import rca_identity as _rca_identity
from .rca_backend import (
    get_rca_history as _get_rca_history_backend,
)
from .rca_backend import (
    get_rca_status as _get_rca_status_backend,
)
from .rca_backend import (
    persist_rca_record,
)
from .rca_engine import build_rca_attempt_record
from .rca_transitions import (
    DEFAULT_STATE,
)
from .rca_types import RCAAttemptRecord, RCAAttemptRequest

RCABackend = Literal["sqlite", "jsonl"]

# advance_rca_attempt reads the previous record and writes the next one; MCP tools run in
# worker threads (RCA-09), so concurrent advances in one process are serialized here.
_ADVANCE_LOCK = threading.Lock()


def _now_utc() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string ending in ``Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_attempt_id() -> str:
    """Return a new attempt ID: epoch milliseconds plus a random 8-hex-digit suffix."""
    return f"{int(datetime.now(timezone.utc).timestamp() * 1000)}-{uuid.uuid4().hex[:8]}"


def build_failure_signature(
    snapshot: dict[str, Any] | None,
    *,
    stdout: str = "",
    stderr: str = "",
    normalize: bool = True,
) -> str:
    """Build a stable failure signature from snapshot (preferred) or textual output.

    Delegates to :func:`llmdebug.rca_identity.build_failure_signature`.

    Args:
        snapshot: Captured snapshot; when present, the signature comes from its
            exception, crash frame, and pytest node ID.
        stdout: Captured standard output, used only without a snapshot.
        stderr: Captured standard error, preferred over *stdout* without a snapshot.
        normalize: Whether to replace run-specific values (addresses, paths, IDs,
            numbers) before hashing.

    Returns:
        ``<ExceptionType>:<hash>`` for a snapshot, or ``TextFailure:<hash>`` for text
        output.
    """
    return _rca_identity.build_failure_signature(
        snapshot, stdout=stdout, stderr=stderr, normalize=normalize
    )


def derive_session_id(
    snapshot: dict[str, Any] | None,
    *,
    out_dir: str,
    fallback: str = "global",
) -> str:
    """Derive a stable session ID from *out_dir* and the pytest node ID or name.

    Delegates to :func:`llmdebug.rca_identity.derive_session_id`.

    Args:
        snapshot: Snapshot whose pytest node ID, or else its name, scopes the session.
        out_dir: Snapshot directory; a hash of its resolved path prefixes the ID.
        fallback: Scope used when the snapshot provides neither node ID nor name.

    Returns:
        A session ID of the form ``<path-hash>::<scope>``.
    """
    return _rca_identity.derive_session_id(snapshot, out_dir=out_dir, fallback=fallback)


def get_rca_history(
    *,
    out_dir: str,
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Load RCA history, newest-first.

    Args:
        out_dir: Snapshot directory that holds the RCA store.
        session_id: Session to filter on; ``None`` or empty includes every session.
        limit: Maximum number of records; ``0`` or less returns all of them.

    Returns:
        RCA attempt records ordered from newest to oldest.
    """
    return _get_rca_history_backend(out_dir=out_dir, session_id=session_id, limit=limit)


def get_rca_status(*, out_dir: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Return latest RCA attempt for a session.

    Args:
        out_dir: Snapshot directory that holds the RCA store.
        session_id: Session to look up; ``None`` or empty returns the latest attempt
            across all sessions.

    Returns:
        The latest attempt record, or ``None`` when none is stored.
    """
    return _get_rca_status_backend(out_dir=out_dir, session_id=session_id)


def advance_rca_attempt(
    *,
    out_dir: str,
    request: RCAAttemptRequest,
) -> RCAAttemptRecord:
    """Advance RCA state machine and persist attempt row.

    Reads the session's latest record, builds the next one with
    :func:`llmdebug.rca_engine.build_rca_attempt_record`, and persists it. Advances in
    one process are serialized so concurrent MCP tool calls cannot interleave the read
    and the write.

    Args:
        out_dir: Snapshot directory that holds the RCA store.
        request: Normalized attempt inputs, including the session and the action.

    Returns:
        The new attempt record as persisted.

    Raises:
        ValueError: If ``request.action`` is not a valid RCA action.
    """
    with _ADVANCE_LOCK:
        now = _now_utc()
        previous = get_rca_status(out_dir=out_dir, session_id=request.session_id)
        record = build_rca_attempt_record(
            previous=previous,
            attempt_id=_new_attempt_id(),
            request=request,
            now=now,
        )
        persist_rca_record(out_dir=out_dir, record=record)
    return record


# Backward-compatible alias for callers/tests that use this constant.
_DEFAULT_STATE = DEFAULT_STATE
