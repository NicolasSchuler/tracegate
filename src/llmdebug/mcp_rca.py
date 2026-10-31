"""RCA helpers and handlers for MCP tools.

Centralizes RCA-related logic for the MCP server:
- Input normalization and context resolution for manual advancement
- RCA state advancement helpers (gate mode, evidence extraction)
- Handler implementations for rca_status, rca_history, rca_advance
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, cast

from . import _json as json
from ._coerce import coerce_int
from ._debug_session import coerce_crash_frame_index
from ._snapshot_loader import load_snapshot
from .mcp_contract import (
    render_invalid_response_format_error,
    render_tool_error,
    render_tool_response,
    validate_evidence_schema,
    validate_response_format,
)
from .rca_state import (
    advance_rca_attempt,
    build_failure_signature,
    derive_session_id,
    get_rca_history,
    get_rca_status,
)
from .rca_types import RCAAttemptRecord, RCAAttemptRequest, RCASnapshotAdvanceRequest
from .snapshot_types import FrameDict, SnapshotDict

_logger = logging.getLogger("llmdebug.mcp")


@dataclass(frozen=True)
class RCAAdvanceContext:
    """Normalized inputs for one manual RCA advancement request.

    Attributes:
        snapshot: Snapshot that identity was derived from, or ``None`` when the caller
            supplied both session and signature or no snapshot could be loaded.
        session_id: Resolved RCA session ID.
        failure_signature: Resolved failure signature.
        evidence_refs: Evidence refs split from the comma-separated input.
        verification_result: Verification status and checks, or ``None`` when no
            status was given.
        hypothesis: Stripped hypothesis, or ``None`` when blank.
        experiment_plan: Stripped experiment plan, or ``None`` when blank.
        proposed_change_summary: Stripped change summary, or ``None`` when blank.
        override_reason: Stripped strict-gate override reason, or ``None`` when blank.
    """

    snapshot: dict[str, Any] | None
    session_id: str
    failure_signature: str
    evidence_refs: list[str]
    verification_result: dict[str, Any] | None
    hypothesis: str | None
    experiment_plan: str | None
    proposed_change_summary: str | None
    override_reason: str | None


@dataclass(frozen=True)
class RCAAdvanceToolRequest:
    """Raw MCP tool inputs for one manual RCA advancement request.

    Attributes:
        action: RCA action to perform.
        out_dir: Snapshot directory that holds the RCA store.
        session_id: RCA session; derived from the snapshot when empty.
        snapshot_path: Snapshot file used to derive the session and signature.
        failure_signature: Stable failure fingerprint; derived when empty.
        evidence_refs: Comma-separated evidence references.
        hypothesis: Root-cause hypothesis; empty keeps the previous one.
        experiment_plan: Experiment plan; empty keeps the previous one.
        proposed_change_summary: Summary of the proposed change; empty keeps the
            previous one.
        verification_status: Verification outcome such as ``pass`` or ``fail``; empty
            records no new result.
        verification_checks: Comma-separated verification check descriptions.
        gate_mode: Gate mode override; empty uses ``LLMDEBUG_RCA_GATE_MODE`` or
            ``soft``.
        exploratory: Whether the step is exploratory, which allows a strict-gate
            override.
        override_reason: Justification for a strict-gate override.
        response_format: ``text`` or ``json``.
        evidence_schema: ``summary`` or ``full`` JSON payload detail.
    """

    action: str
    out_dir: str = ".llmdebug"
    session_id: str = ""
    snapshot_path: str = ""
    failure_signature: str = ""
    evidence_refs: str = ""
    hypothesis: str = ""
    experiment_plan: str = ""
    proposed_change_summary: str = ""
    verification_status: str = ""
    verification_checks: str = ""
    gate_mode: str = ""
    exploratory: bool = False
    override_reason: str = ""
    response_format: str = "text"
    evidence_schema: str = "summary"


class SnapshotLoadError(RuntimeError):
    """Raised when a requested snapshot cannot be loaded for manual RCA flow."""


class RCAInputError(ValueError):
    """Raised when manual RCA inputs are ambiguous or internally inconsistent."""


def _split_csv_values(raw: str) -> list[str]:
    """Split comma-separated input into stripped, non-empty values."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_snapshot(
    *,
    out_dir: str,
    snapshot_path: str,
    session_id: str,
    failure_signature: str,
) -> dict[str, Any] | None:
    """Load the snapshot that identifies a manual RCA advance, when one is needed.

    An explicit *snapshot_path* is loaded; with neither session nor signature the
    latest snapshot is tried; with both, no snapshot is needed.

    Raises:
        SnapshotLoadError: If *snapshot_path* is given but cannot be loaded.
        RCAInputError: If only one of *session_id* and *failure_signature* is given.
    """
    if snapshot_path:
        try:
            return load_snapshot(snapshot_path, out_dir)
        except Exception as exc:  # pragma: no cover - exercised through public API
            raise SnapshotLoadError(f"Error loading snapshot: {type(exc).__name__}: {exc}") from exc

    has_session = bool(session_id.strip())
    has_signature = bool(failure_signature.strip())
    if not has_session and not has_signature:
        try:
            return load_snapshot(None, out_dir)
        except Exception:
            return None
    if has_session != has_signature:
        raise RCAInputError(
            "When snapshot_path is omitted, provide both session_id and failure_signature "
            "together, or leave both empty to derive from the latest snapshot."
        )
    return None


def _verification_result(status: str, checks: str) -> dict[str, Any] | None:
    """Build a verification result from a status and checks; ``None`` without a status."""
    normalized_status = status.strip()
    if not normalized_status:
        return None
    return {
        "status": normalized_status,
        "checks": _split_csv_values(checks),
    }


def _strip_or_none(raw: str) -> str | None:
    """Return the stripped text, or ``None`` when it is blank."""
    normalized = raw.strip()
    return normalized or None


def resolve_rca_advance_context(
    *,
    request: RCAAdvanceToolRequest,
) -> RCAAdvanceContext:
    """Resolve normalized RCA context from MCP tool string inputs.

    Explicit session and signature values win; empty ones are derived from the
    resolved snapshot (session fallback ``manual``).

    Args:
        request: Raw tool inputs.

    Returns:
        The normalized context for the RCA attempt.

    Raises:
        SnapshotLoadError: If ``request.snapshot_path`` is given but cannot be loaded.
        RCAInputError: If, without a snapshot path, only one of session ID and
            failure signature is given.
    """
    snapshot = _resolve_snapshot(
        out_dir=request.out_dir,
        snapshot_path=request.snapshot_path,
        session_id=request.session_id,
        failure_signature=request.failure_signature,
    )
    resolved_session = request.session_id.strip() or derive_session_id(
        snapshot, out_dir=request.out_dir, fallback="manual"
    )
    signature = request.failure_signature.strip() or build_failure_signature(snapshot)
    return RCAAdvanceContext(
        snapshot=snapshot,
        session_id=resolved_session,
        failure_signature=signature,
        evidence_refs=_split_csv_values(request.evidence_refs),
        verification_result=_verification_result(
            request.verification_status, request.verification_checks
        ),
        hypothesis=_strip_or_none(request.hypothesis),
        experiment_plan=_strip_or_none(request.experiment_plan),
        proposed_change_summary=_strip_or_none(request.proposed_change_summary),
        override_reason=_strip_or_none(request.override_reason),
    )


# ---------------------------------------------------------------------------
# RCA state helpers (extracted from mcp_server.py)
# ---------------------------------------------------------------------------


def effective_gate_mode(gate_mode: str | None) -> str:
    """Resolve RCA gate mode from param or env, defaulting to "soft".

    Args:
        gate_mode: Requested gate mode; ``None`` or empty uses
            ``LLMDEBUG_RCA_GATE_MODE``.

    Returns:
        ``off``, ``soft``, or ``strict``; ``soft`` when the resolved value is not one
        of them.
    """
    candidate = (gate_mode or os.getenv("LLMDEBUG_RCA_GATE_MODE", "soft")).strip().lower()
    if candidate in {"off", "soft", "strict"}:
        return candidate
    return "soft"


def extract_evidence_refs(snapshot: dict[str, Any]) -> list[str]:
    """Build evidence references from a snapshot's exception, crash frame, and pytest node ID.

    Args:
        snapshot: Loaded snapshot.

    Returns:
        Refs in the order ``exception:<type>``, ``exception_message:<first 80
        chars>``, ``crash_frame:<file>:<line>:<function>`` (falling back to the first
        frame), and ``pytest_nodeid:<nodeid>``; missing parts are skipped.
    """
    snapshot_t = cast(SnapshotDict, snapshot)
    refs: list[str] = []
    exc = snapshot_t.get("exception", {})
    if isinstance(exc, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        exc_type = str(exc.get("type") or "")
        exc_msg = str(exc.get("message") or "")
        if exc_type:
            refs.append(f"exception:{exc_type}")
        if exc_msg:
            refs.append(f"exception_message:{exc_msg[:80]}")

    frames = snapshot_t.get("frames", [])
    if isinstance(frames, list) and frames:  # pyright: ignore[reportUnnecessaryIsInstance]
        crash_idx = coerce_crash_frame_index(snapshot)
        frame: Any | None = None
        if 0 <= crash_idx < len(frames) and isinstance(frames[crash_idx], dict):
            frame = cast(FrameDict, frames[crash_idx])
        elif isinstance(frames[0], dict):
            frame = cast(FrameDict, frames[0])
        if frame:
            file_rel = str(frame.get("file_rel", frame.get("file", "?")))
            line = coerce_int(frame.get("line"), default=0, minimum=0)
            func = str(frame.get("function") or "?")
            refs.append(f"crash_frame:{file_rel}:{line}:{func}")

    pytest_block = snapshot_t.get("pytest", {})
    if isinstance(pytest_block, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        nodeid = str(pytest_block.get("nodeid") or "")
        if nodeid:
            refs.append(f"pytest_nodeid:{nodeid}")
    return refs


def advance_rca(
    *,
    request: RCASnapshotAdvanceRequest,
) -> RCAAttemptRecord | None:
    """Attempt an RCA state advance, returning None on failure.

    The session ID (fallback ``mcp``) and failure signature are derived from the
    snapshot or fallback output; the values on ``request.attempt`` are ignored. The
    gate mode is resolved with :func:`effective_gate_mode`. Failures are logged.

    Args:
        request: Snapshot, output directory, and attempt inputs.

    Returns:
        The persisted attempt record, or ``None`` when the update failed.
    """
    try:
        session_id = derive_session_id(request.snapshot, out_dir=request.out_dir, fallback="mcp")
        signature = build_failure_signature(
            request.snapshot,
            stdout=request.fallback_stdout,
            stderr=request.fallback_stderr,
        )
        return advance_rca_attempt(
            out_dir=request.out_dir,
            request=RCAAttemptRequest(
                session_id=session_id,
                action=request.attempt.action,
                failure_signature=signature,
                evidence_refs=request.attempt.evidence_refs,
                hypothesis=request.attempt.hypothesis,
                experiment_plan=request.attempt.experiment_plan,
                proposed_change_summary=request.attempt.proposed_change_summary,
                verification_result=request.attempt.verification_result,
                gate_mode=effective_gate_mode(request.attempt.gate_mode),
                exploratory=request.attempt.exploratory,
                override_reason=request.attempt.override_reason,
                stall_threshold=request.attempt.stall_threshold,
                exploration_budget=request.attempt.exploration_budget,
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive path
        _logger.warning("RCA state update failed: %s: %s", type(exc).__name__, exc)
        return None


def advance_rca_with_snapshot_refs(
    *,
    request: RCASnapshotAdvanceRequest,
    extra_evidence_refs: list[str] | None = None,
) -> RCAAttemptRecord | None:
    """Advance RCA like :func:`advance_rca`, auto-extracting evidence refs from the snapshot.

    The extracted refs plus *extra_evidence_refs* replace ``request.attempt.evidence_refs``.

    Args:
        request: Snapshot, output directory, and attempt inputs.
        extra_evidence_refs: Additional refs appended after the snapshot's refs.

    Returns:
        The persisted attempt record, or ``None`` when the update failed.
    """
    refs = extract_evidence_refs(request.snapshot) if isinstance(request.snapshot, dict) else []
    if extra_evidence_refs:
        refs.extend(extra_evidence_refs)
    return advance_rca(
        request=RCASnapshotAdvanceRequest(
            out_dir=request.out_dir,
            snapshot=request.snapshot,
            attempt=RCAAttemptRequest(
                session_id=request.attempt.session_id,
                action=request.attempt.action,
                failure_signature=request.attempt.failure_signature,
                evidence_refs=tuple(refs),
                hypothesis=request.attempt.hypothesis,
                experiment_plan=request.attempt.experiment_plan,
                proposed_change_summary=request.attempt.proposed_change_summary,
                verification_result=request.attempt.verification_result,
                gate_mode=request.attempt.gate_mode,
                exploratory=request.attempt.exploratory,
                override_reason=request.attempt.override_reason,
                stall_threshold=request.attempt.stall_threshold,
                exploration_budget=request.attempt.exploration_budget,
            ),
            fallback_stdout=request.fallback_stdout,
            fallback_stderr=request.fallback_stderr,
        ),
    )


def render_rca_update_failed(
    *,
    tool: str,
    response_format: str,
    evidence_schema: str,
) -> str:
    """Render a standardized RCA update failure error response.

    Args:
        tool: Name of the MCP tool whose RCA update failed.
        response_format: ``json`` for the error envelope, ``text`` for the bare message.
        evidence_schema: Requested evidence schema for the envelope.

    Returns:
        The rendered ``RCA_STATE_UPDATE_ERROR`` response.
    """
    return render_tool_error(
        tool=tool,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=True,
        code="RCA_STATE_UPDATE_ERROR",
        message="Error updating RCA state.",
    )


# ---------------------------------------------------------------------------
# RCA handler implementations (extracted from mcp_server.py)
# ---------------------------------------------------------------------------


def handle_rca_status(
    out_dir: str = ".llmdebug",
    session_id: str = "",
    snapshot_path: str = "",
    response_format: str = "text",
    evidence_schema: str = "summary",
) -> str:
    """Return the latest RCA attempt for the ``llmdebug_rca_status`` MCP tool.

    Args:
        out_dir: Snapshot directory that holds the RCA store.
        session_id: Session to look up; when empty, it is derived from
            *snapshot_path* if given, else the latest attempt overall is returned.
        snapshot_path: Snapshot used to derive the session ID.
        response_format: ``text`` or ``json``.
        evidence_schema: ``summary`` or ``full`` JSON payload detail.

    Returns:
        The latest attempt as JSON text or a JSON envelope, or an error response (a
        bare message in text mode when no attempt exists).
    """
    format_error = validate_response_format(response_format)
    if format_error is not None:
        return render_invalid_response_format_error(
            tool="llmdebug_rca_status",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
        )
    schema_error = validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return render_tool_error(
            tool="llmdebug_rca_status",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )

    resolved_session = session_id.strip()
    if not resolved_session and snapshot_path:
        try:
            snap = load_snapshot(snapshot_path, out_dir)
            resolved_session = derive_session_id(snap, out_dir=out_dir, fallback="mcp")
        except Exception as e:
            _logger.warning("session ID resolution failed: %s: %s", type(e).__name__, e)
            resolved_session = ""

    status = get_rca_status(out_dir=out_dir, session_id=resolved_session or None)
    if status is None:
        if resolved_session:
            message = f"No RCA attempts found for session {resolved_session!r} in {out_dir}."
        else:
            message = f"No RCA attempts found in {out_dir}."
        if response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_rca_status",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="NO_RCA_ATTEMPTS",
            message=message,
        )
    body = json.dumps(status, indent=2, ensure_ascii=False)
    if response_format == "text":
        return body
    return render_tool_response(
        tool="llmdebug_rca_status",
        body=body,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=False,
        evidence_payload={
            "rca_status": status,
            "session_id": resolved_session,
        },
    )


def handle_rca_history(
    out_dir: str = ".llmdebug",
    session_id: str = "",
    limit: int = 20,
    response_format: str = "text",
    evidence_schema: str = "summary",
) -> str:
    """Return RCA history for the ``llmdebug_rca_history`` MCP tool.

    Args:
        out_dir: Snapshot directory that holds the RCA store.
        session_id: Session to filter on; empty includes every session.
        limit: Maximum number of attempts; must be at least 1.
        response_format: ``text`` or ``json``.
        evidence_schema: ``summary`` or ``full`` JSON payload detail.

    Returns:
        The attempts, newest first, as JSON text or a JSON envelope, or an error
        response (a bare message in text mode when no attempt exists).
    """
    format_error = validate_response_format(response_format)
    if format_error is not None:
        return render_invalid_response_format_error(
            tool="llmdebug_rca_history",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
        )
    schema_error = validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return render_tool_error(
            tool="llmdebug_rca_history",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )
    if limit <= 0:
        return render_tool_error(
            tool="llmdebug_rca_history",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=f"limit must be >= 1 (got {limit})",
        )

    history = get_rca_history(out_dir=out_dir, session_id=session_id.strip() or None, limit=limit)
    if not history:
        if session_id.strip():
            message = f"No RCA attempts found for session {session_id!r} in {out_dir}."
        else:
            message = f"No RCA attempts found in {out_dir}."
        if response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_rca_history",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="NO_RCA_ATTEMPTS",
            message=message,
        )
    body = json.dumps(history, indent=2, ensure_ascii=False)
    if response_format == "text":
        return body
    return render_tool_response(
        tool="llmdebug_rca_history",
        body=body,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=False,
        evidence_payload={
            "rca_history": history,
            "session_id": session_id.strip(),
            "limit": int(limit),
        },
    )


def handle_rca_advance(
    *,
    request: RCAAdvanceToolRequest,
) -> str:
    """Advance the RCA state machine for the ``llmdebug_rca_advance`` MCP tool.

    Args:
        request: Raw tool inputs.

    Returns:
        The new attempt record as JSON text or a JSON envelope, or an error response
        when an argument is invalid, the snapshot cannot be loaded, or the RCA update
        fails.
    """
    format_error = validate_response_format(request.response_format)
    if format_error is not None:
        return render_invalid_response_format_error(
            tool="llmdebug_rca_advance",
            response_format=request.response_format,
            evidence_schema=request.evidence_schema,
            with_rca=False,
        )
    schema_error = validate_evidence_schema(request.evidence_schema)
    if schema_error is not None:
        return render_tool_error(
            tool="llmdebug_rca_advance",
            response_format=request.response_format,
            evidence_schema=request.evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )

    try:
        context = resolve_rca_advance_context(request=request)
    except SnapshotLoadError as exc:
        message = str(exc)
        if request.response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_rca_advance",
            response_format=request.response_format,
            evidence_schema=request.evidence_schema,
            with_rca=False,
            code="SNAPSHOT_LOAD_ERROR",
            message=message,
        )
    except RCAInputError as exc:
        message = str(exc)
        if request.response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_rca_advance",
            response_format=request.response_format,
            evidence_schema=request.evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=message,
        )

    try:
        record = advance_rca_attempt(
            out_dir=request.out_dir,
            request=RCAAttemptRequest(
                session_id=context.session_id,
                action=request.action,
                failure_signature=context.failure_signature,
                evidence_refs=tuple(context.evidence_refs),
                hypothesis=context.hypothesis,
                experiment_plan=context.experiment_plan,
                proposed_change_summary=context.proposed_change_summary,
                verification_result=context.verification_result,
                gate_mode=effective_gate_mode(request.gate_mode),
                exploratory=request.exploratory,
                override_reason=context.override_reason,
            ),
        )
    except ValueError as exc:
        message = str(exc)
        if request.response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_rca_advance",
            response_format=request.response_format,
            evidence_schema=request.evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=message,
        )
    except Exception as exc:
        message = f"Error updating RCA state: {type(exc).__name__}: {exc}"
        if request.response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_rca_advance",
            response_format=request.response_format,
            evidence_schema=request.evidence_schema,
            with_rca=False,
            code="RCA_STATE_UPDATE_ERROR",
            message=message,
        )
    body = json.dumps(record, indent=2, ensure_ascii=False)
    if request.response_format == "text":
        return body
    return render_tool_response(
        tool="llmdebug_rca_advance",
        body=body,
        response_format=request.response_format,
        evidence_schema=request.evidence_schema,
        with_rca=False,
        evidence_payload={
            "rca_record": record,
            "action": request.action,
            "session_id": context.session_id,
        },
    )
