"""RCA state tracking facade for MCP/debug workflows.

This module preserves the public RCA API while delegating internals to focused
identity/transition/backend helpers.
"""

from __future__ import annotations

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


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_attempt_id() -> str:
    return f"{int(datetime.now(timezone.utc).timestamp() * 1000)}-{uuid.uuid4().hex[:8]}"


def build_failure_signature(
    snapshot: dict[str, Any] | None,
    *,
    stdout: str = "",
    stderr: str = "",
    normalize: bool = True,
) -> str:
    """Build a stable failure signature from snapshot (preferred) or textual output."""
    return _rca_identity.build_failure_signature(
        snapshot, stdout=stdout, stderr=stderr, normalize=normalize
    )


def derive_session_id(
    snapshot: dict[str, Any] | None,
    *,
    out_dir: str,
    fallback: str = "global",
) -> str:
    """Derive stable session id from out_dir + pytest nodeid/name when available."""
    return _rca_identity.derive_session_id(snapshot, out_dir=out_dir, fallback=fallback)


def get_rca_history(
    *,
    out_dir: str,
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Load RCA history, newest-first."""
    return _get_rca_history_backend(out_dir=out_dir, session_id=session_id, limit=limit)


def get_rca_status(*, out_dir: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Return latest RCA attempt for a session."""
    return _get_rca_status_backend(out_dir=out_dir, session_id=session_id)


def advance_rca_attempt(
    *,
    out_dir: str,
    request: RCAAttemptRequest,
) -> RCAAttemptRecord:
    """Advance RCA state machine and persist attempt row."""
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
