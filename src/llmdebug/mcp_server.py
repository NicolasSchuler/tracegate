"""llmdebug MCP server — expose crash snapshots and RCA workflows to IDE agents.

Provides tools for programmatic access to debug snapshots and RCA state:
- llmdebug_diagnose: Concise crash summary optimized for LLM consumption
- llmdebug_show: Full expanded JSON snapshot
- llmdebug_list: List available snapshots
- llmdebug_frame: Detailed view of a specific stack frame
- llmdebug_git_context: On-demand enhanced git metadata for crash triage
- llmdebug_diff: Compare two snapshots with structured delta
- llmdebug_rca_status / llmdebug_rca_history: RCA state visibility
- llmdebug_rca_advance: Manual RCA state-machine advancement

Run with: llmdebug-mcp (stdio transport for IDE integration)
"""

from __future__ import annotations

import functools
import inspect
import logging
import sys
from collections.abc import Callable
from typing import ParamSpec

# Guard import — mcp is optional
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "llmdebug MCP server requires: pip install 'llmdebug[mcp]'",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

from . import _json as json
from ._formatting import (
    format_enhanced_git_context_text,
)
from ._rate_limiter import load_mcp_rate_limiter
from .git_context import get_enhanced_git_context
from .mcp_contract import (
    render_invalid_response_format_error as _render_invalid_response_format_error,
)
from .mcp_contract import (
    render_tool_error as _render_tool_error,
)
from .mcp_contract import (
    render_tool_response as _render_tool_response,
)
from .mcp_contract import (
    validate_evidence_schema as _validate_evidence_schema,
)
from .mcp_contract import (
    validate_response_format as _validate_response_format,
)
from .mcp_diff_tools import handle_diff
from .mcp_git_context import collect_git_evidence_refs, parse_max_commits
from .mcp_paths import resolve_mcp_out_dir
from .mcp_rca import (
    RCAAdvanceToolRequest,
    RCAAttemptRequest,
    RCASnapshotAdvanceRequest,
    handle_rca_advance,
    handle_rca_history,
    handle_rca_status,
)
from .mcp_rca import (
    advance_rca_with_snapshot_refs as _advance_rca_with_snapshot_refs,
)
from .mcp_rca import (
    render_rca_update_failed as _render_rca_update_failed,
)
from .mcp_snapshot_tools import (
    core_evidence_from_snapshot as _core_evidence_from_snapshot,
)
from .mcp_snapshot_tools import (
    diagnose_context_section as _diagnose_context_section,
)
from .mcp_snapshot_tools import (
    diagnose_crash_frame_section as _diagnose_crash_frame_section,
)
from .mcp_snapshot_tools import (
    diagnose_exception_section as _diagnose_exception_section,
)
from .mcp_snapshot_tools import (
    handle_frame,
    handle_list,
)
from .mcp_snapshot_tools import (
    load_snapshot_safe as _load_snapshot_safe,
)
from .mcp_snapshot_tools import (
    normalize_detail as _normalize_detail,
)
from .mcp_snapshot_tools import (
    render_invalid_detail_error as _render_invalid_detail_error,
)
from .mcp_snapshot_tools import (
    snapshot_error_code as _snapshot_error_code,
)

_logger = logging.getLogger("llmdebug.mcp")

_mcp = FastMCP("llmdebug")

# Shared token-bucket rate limiter for all MCP tool calls.
_mcp_limiter = load_mcp_rate_limiter()


def _check_rate_limit(tool: str, response_format: str, evidence_schema: str) -> str | None:
    """Return an error string if the call is rate-limited, else ``None``."""
    if _mcp_limiter.acquire():
        return None
    return _render_tool_error(
        tool=tool,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=False,
        code="RATE_LIMITED",
        message=(
            f"Rate limit exceeded ({_mcp_limiter.rate} calls/s, "
            f"burst {_mcp_limiter.burst}). Please retry after a short delay."
        ),
    )


_P = ParamSpec("_P")


def _confined(tool: Callable[_P, str]) -> Callable[_P, str]:
    """Resolve the tool's ``out_dir`` inside the MCP root, or answer with an error (RCA-02)."""
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> str:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
            bound.arguments["out_dir"] = resolve_mcp_out_dir(str(bound.arguments["out_dir"]))
        except ValueError as exc:
            return _render_tool_error(
                tool=tool.__name__,
                response_format=str(bound.arguments.get("response_format", "json")),
                evidence_schema=str(bound.arguments.get("evidence_schema", "summary")),
                with_rca=False,
                code="INVALID_ARGUMENT",
                message=str(exc),
            )
        return tool(*bound.args, **bound.kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@_mcp.tool()
@_confined
def llmdebug_diagnose(
    out_dir: str = ".llmdebug",
    detail: str = "context",
    gate_mode: str = "",
    exploratory: bool = False,
    response_format: str = "json",
    with_rca: bool = False,
    evidence_schema: str = "summary",
) -> str:
    """Get a concise diagnosis of the latest snapshot.

    Returns a structured summary with: exception type/message,
    crash frame with source and locals, repro command, and git context.
    Optimized for LLM consumption.

    Args:
        out_dir: Snapshot directory (default: .llmdebug)
        detail: Detail level — "crash" (exception + crash frame only),
                "full" (+ all frames), or "context" (+ repro/git, default)
        gate_mode: RCA gate mode override: off|soft|strict (default from env or soft)
        exploratory: Mark this step as exploratory (used by strict gate override logic)
        response_format: "json" (default) or "text"
        with_rca: Include RCA block (default: false)
        evidence_schema: "summary" (default) or "full" for JSON payload detail
    """
    format_error = _validate_response_format(response_format)
    if format_error is not None:
        return _render_invalid_response_format_error(
            tool="llmdebug_diagnose",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
        )
    rate_error = _check_rate_limit("llmdebug_diagnose", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    schema_error = _validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return _render_tool_error(
            tool="llmdebug_diagnose",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )
    normalized_detail = _normalize_detail(detail)
    if normalized_detail is None:
        return _render_invalid_detail_error(
            tool="llmdebug_diagnose",
            detail=detail,
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
        )
    detail = normalized_detail

    result = _load_snapshot_safe(None, out_dir, "llmdebug_diagnose")
    if isinstance(result, str):
        return _render_tool_error(
            tool="llmdebug_diagnose",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code=_snapshot_error_code(result),
            message=result,
        )
    snapshot = result
    core_evidence = _core_evidence_from_snapshot(snapshot)

    sections: list[str | None] = [
        _diagnose_exception_section(snapshot),
        _diagnose_crash_frame_section(snapshot, detail),
        _diagnose_context_section(snapshot, detail),
    ]
    body = "\n\n".join(s for s in sections if s is not None)
    rca_record = (
        _advance_rca_with_snapshot_refs(
            request=RCASnapshotAdvanceRequest(
                out_dir=out_dir,
                snapshot=snapshot,
                attempt=RCAAttemptRequest(
                    session_id="",
                    action="bind_evidence",
                    failure_signature="",
                    gate_mode=gate_mode,
                    exploratory=exploratory,
                ),
            ),
        )
        if with_rca
        else None
    )
    if with_rca and rca_record is None:
        return _render_rca_update_failed(
            tool="llmdebug_diagnose",
            response_format=response_format,
            evidence_schema=evidence_schema,
        )
    return _render_tool_response(
        tool="llmdebug_diagnose",
        body=body,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=with_rca,
        rca_record=rca_record,
        evidence_payload={
            **core_evidence,
            "snapshot": snapshot,
            "detail": detail,
        },
    )


@_mcp.tool()
@_confined
def llmdebug_show(
    out_dir: str = ".llmdebug",
    snapshot_path: str = "",
    detail: str = "context",
    raw_session: bool = False,
    with_rca: bool = False,
    gate_mode: str = "",
    exploratory: bool = False,
    response_format: str = "json",
    evidence_schema: str = "summary",
) -> str:
    """Show the debug snapshot as expanded JSON.

    Returns snapshot data with all frames, locals, and metadata.
    Use llmdebug_diagnose first for a concise summary,
    then llmdebug_show if you need the full picture.

    Args:
        out_dir: Snapshot directory (default: .llmdebug)
        snapshot_path: Path to a specific snapshot file. If empty, uses latest.
        detail: Detail level — "crash", "full", or "context" (default, everything)
        raw_session: Return raw DebugSession envelope when available.
        with_rca: Return JSON envelope with snapshot + RCA block.
        gate_mode: RCA gate mode override: off|soft|strict.
        exploratory: Mark this step as exploratory.
        response_format: "json" (default) or "text".
        evidence_schema: "summary" (default) or "full" for JSON payload detail.
    """
    format_error = _validate_response_format(response_format)
    if format_error is not None:
        return _render_invalid_response_format_error(
            tool="llmdebug_show",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
        )
    rate_error = _check_rate_limit("llmdebug_show", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    schema_error = _validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return _render_tool_error(
            tool="llmdebug_show",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )
    normalized_detail = _normalize_detail(detail)
    if normalized_detail is None:
        return _render_invalid_detail_error(
            tool="llmdebug_show",
            detail=detail,
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
        )
    detail = normalized_detail

    result = _load_snapshot_safe(
        snapshot_path or None,
        out_dir,
        "llmdebug_show",
        raw=raw_session,
    )
    if isinstance(result, str):
        return _render_tool_error(
            tool="llmdebug_show",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code=_snapshot_error_code(result),
            message=result,
        )
    snapshot = result

    # Apply detail filter
    if not raw_session and detail in ("crash", "full", "context"):
        try:
            from .detail_filter import filter_snapshot

            snapshot = filter_snapshot(snapshot, detail)  # type: ignore[arg-type]
        except Exception as e:  # nosec B110 — graceful filter fallback
            _logger.warning(
                "filter_snapshot(%r) failed, returning unfiltered: %s: %s",
                detail,
                type(e).__name__,
                e,
            )

    if response_format == "text" and not with_rca:
        return json.dumps(snapshot, indent=2, ensure_ascii=False, default=str)

    normalized_snapshot = snapshot
    if raw_session:
        try:
            from ._debug_session import normalize_snapshot_view

            normalized_snapshot = normalize_snapshot_view(snapshot)
        except Exception as e:
            _logger.warning(
                "normalize_snapshot_view failed, using raw: %s: %s",
                type(e).__name__,
                e,
            )
            normalized_snapshot = snapshot

    rca_record = (
        _advance_rca_with_snapshot_refs(
            request=RCASnapshotAdvanceRequest(
                out_dir=out_dir,
                snapshot=normalized_snapshot,
                attempt=RCAAttemptRequest(
                    session_id="",
                    action="bind_evidence",
                    failure_signature="",
                    gate_mode=gate_mode,
                    exploratory=exploratory,
                ),
            ),
        )
        if with_rca
        else None
    )
    if with_rca and rca_record is None:
        return _render_rca_update_failed(
            tool="llmdebug_show",
            response_format=response_format,
            evidence_schema=evidence_schema,
        )
    core_evidence = _core_evidence_from_snapshot(normalized_snapshot)
    payload = {
        "snapshot": snapshot,
        "rca": rca_record,
    }
    if response_format == "text":
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    return _render_tool_response(
        tool="llmdebug_show",
        body="Expanded snapshot payload",
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=with_rca,
        rca_record=rca_record,
        evidence_payload={
            **core_evidence,
            "snapshot": snapshot,
            "normalized_snapshot": normalized_snapshot,
            "raw_session": raw_session,
            "detail": detail,
        },
    )


@_mcp.tool()
@_confined
def llmdebug_list(
    out_dir: str = ".llmdebug",
    limit: int = 10,
    response_format: str = "text",
    evidence_schema: str = "summary",
) -> str:
    """List available debug snapshots, newest first.

    Returns snapshot metadata: timestamp, name, exception type,
    message, and file location. Use snapshot_path from the results
    with llmdebug_show to inspect a specific snapshot.

    Args:
        out_dir: Snapshot directory (default: .llmdebug)
        limit: Maximum number of snapshots to return (default: 10)
        response_format: "text" (default) or "json"
        evidence_schema: "summary" (default) or "full" for JSON payload detail
    """
    rate_error = _check_rate_limit("llmdebug_list", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    return handle_list(
        out_dir=out_dir,
        limit=limit,
        response_format=response_format,
        evidence_schema=evidence_schema,
    )


@_mcp.tool()
@_confined
def llmdebug_frame(
    index: int = 0,
    out_dir: str = ".llmdebug",
    snapshot_path: str = "",
    gate_mode: str = "",
    exploratory: bool = False,
    response_format: str = "json",
    with_rca: bool = False,
    evidence_schema: str = "summary",
) -> str:
    """Show a specific stack frame in detail.

    Returns source code, all local variables with types and values,
    and function arguments. Frame 0 is the crash site. Use higher
    indices to inspect callers.

    Args:
        index: Frame index (0 = crash site, higher = callers)
        out_dir: Snapshot directory (default: .llmdebug)
        snapshot_path: Path to a specific snapshot file. If empty, uses latest.
        gate_mode: RCA gate mode override: off|soft|strict.
        exploratory: Mark this step as exploratory.
        response_format: "json" (default) or "text".
        with_rca: Include RCA block (default: false).
        evidence_schema: "summary" (default) or "full" for JSON payload detail.
    """
    rate_error = _check_rate_limit("llmdebug_frame", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    return handle_frame(
        index=index,
        out_dir=out_dir,
        snapshot_path=snapshot_path,
        gate_mode=gate_mode,
        exploratory=exploratory,
        response_format=response_format,
        with_rca=with_rca,
        evidence_schema=evidence_schema,
    )


@_mcp.tool()
@_confined
def llmdebug_git_context(
    out_dir: str = ".llmdebug",
    snapshot_path: str = "",
    max_commits: int = 5,
    gate_mode: str = "",
    exploratory: bool = False,
    response_format: str = "json",
    with_rca: bool = False,
    evidence_schema: str = "summary",
) -> str:
    """Show on-demand enhanced git metadata for a snapshot.

    Returns metadata-only context:
    - crash-line blame metadata
    - recent commit metadata
    - crash-file diffstat metadata

    Use this when the snapshot itself should stay compact but you still need
    git history or diff context around the failing line.
    """
    format_error = _validate_response_format(response_format)
    if format_error is not None:
        return _render_invalid_response_format_error(
            tool="llmdebug_git_context",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
        )
    rate_error = _check_rate_limit("llmdebug_git_context", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    schema_error = _validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return _render_tool_error(
            tool="llmdebug_git_context",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )

    result = _load_snapshot_safe(snapshot_path or None, out_dir, "llmdebug_git_context")
    if isinstance(result, str):
        return _render_tool_error(
            tool="llmdebug_git_context",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code=_snapshot_error_code(result),
            message=result,
        )
    snapshot = result
    core_evidence = _core_evidence_from_snapshot(snapshot)

    try:
        max_commits = parse_max_commits(max_commits)
    except ValueError as exc:
        return _render_tool_error(
            tool="llmdebug_git_context",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="INVALID_ARGUMENT",
            message=str(exc),
        )
    try:
        enhanced = get_enhanced_git_context(snapshot, out_dir=out_dir, max_commits=max_commits)
    except Exception as exc:  # RCA-10: keep failures inside the error envelope
        _logger.warning("enhanced git context failed: %s: %s", type(exc).__name__, exc)
        return _render_tool_error(
            tool="llmdebug_git_context",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="GIT_CONTEXT_ERROR",
            message=f"Enhanced git context failed: {type(exc).__name__}: {exc}",
        )
    if enhanced is None:
        body = "Enhanced git context unavailable (not in a git repository or git is unavailable)."
        rca_record = (
            _advance_rca_with_snapshot_refs(
                request=RCASnapshotAdvanceRequest(
                    out_dir=out_dir,
                    snapshot=snapshot,
                    attempt=RCAAttemptRequest(
                        session_id="",
                        action="bind_evidence",
                        failure_signature="",
                        gate_mode=gate_mode,
                        exploratory=exploratory,
                    ),
                ),
            )
            if with_rca
            else None
        )
        if with_rca and rca_record is None:
            return _render_rca_update_failed(
                tool="llmdebug_git_context",
                response_format=response_format,
                evidence_schema=evidence_schema,
            )
        return _render_tool_response(
            tool="llmdebug_git_context",
            body=body,
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            rca_record=rca_record,
            evidence_payload={**core_evidence, "snapshot": snapshot},
        )

    evidence_refs = collect_git_evidence_refs(enhanced)
    body = format_enhanced_git_context_text(enhanced)
    rca_record = (
        _advance_rca_with_snapshot_refs(
            request=RCASnapshotAdvanceRequest(
                out_dir=out_dir,
                snapshot=snapshot,
                attempt=RCAAttemptRequest(
                    session_id="",
                    action="bind_evidence",
                    failure_signature="",
                    gate_mode=gate_mode,
                    exploratory=exploratory,
                ),
            ),
            extra_evidence_refs=evidence_refs,
        )
        if with_rca
        else None
    )
    if with_rca and rca_record is None:
        return _render_rca_update_failed(
            tool="llmdebug_git_context",
            response_format=response_format,
            evidence_schema=evidence_schema,
        )
    return _render_tool_response(
        tool="llmdebug_git_context",
        body=body,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=with_rca,
        rca_record=rca_record,
        evidence_payload={
            **core_evidence,
            "snapshot": snapshot,
            "enhanced_git_context": enhanced,
        },
    )


@_mcp.tool()
@_confined
def llmdebug_diff(
    old_ref: str = "previous",
    new_ref: str = "latest",
    out_dir: str = ".llmdebug",
    gate_mode: str = "",
    exploratory: bool = False,
    response_format: str = "json",
    with_rca: bool = False,
    evidence_schema: str = "summary",
) -> str:
    """Compare two snapshots to show what changed between debugging iterations.

    Returns changes in: exception type/message, crash location,
    local variable values, array shapes, and git context.
    Default: compares the two most recent snapshots.

    Args:
        old_ref: Older snapshot — "latest", "previous", "#N" (list index), or file path.
        new_ref: Newer snapshot — same format as old_ref.
        out_dir: Snapshot directory (default: .llmdebug)
        gate_mode: RCA gate mode override: off|soft|strict.
        exploratory: Mark this step as exploratory.
        response_format: "json" (default) or "text".
        with_rca: Include RCA block (default: false).
        evidence_schema: "summary" (default) or "full" for JSON payload detail.
    """
    rate_error = _check_rate_limit("llmdebug_diff", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    return handle_diff(
        old_ref=old_ref,
        new_ref=new_ref,
        out_dir=out_dir,
        gate_mode=gate_mode,
        exploratory=exploratory,
        response_format=response_format,
        with_rca=with_rca,
        evidence_schema=evidence_schema,
    )


@_mcp.tool()
@_confined
def llmdebug_rca_status(
    out_dir: str = ".llmdebug",
    session_id: str = "",
    snapshot_path: str = "",
    response_format: str = "text",
    evidence_schema: str = "summary",
) -> str:
    """Return latest RCA state for a session (or latest overall if session unspecified)."""
    rate_error = _check_rate_limit("llmdebug_rca_status", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    return handle_rca_status(
        out_dir=out_dir,
        session_id=session_id,
        snapshot_path=snapshot_path,
        response_format=response_format,
        evidence_schema=evidence_schema,
    )


@_mcp.tool()
@_confined
def llmdebug_rca_history(
    out_dir: str = ".llmdebug",
    session_id: str = "",
    limit: int = 20,
    response_format: str = "text",
    evidence_schema: str = "summary",
) -> str:
    """Return RCA history (newest-first) for a session or globally."""
    rate_error = _check_rate_limit("llmdebug_rca_history", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    return handle_rca_history(
        out_dir=out_dir,
        session_id=session_id,
        limit=limit,
        response_format=response_format,
        evidence_schema=evidence_schema,
    )


@_mcp.tool()
@_confined
def llmdebug_rca_advance(
    action: str,
    out_dir: str = ".llmdebug",
    session_id: str = "",
    snapshot_path: str = "",
    failure_signature: str = "",
    evidence_refs: str = "",
    hypothesis: str = "",
    experiment_plan: str = "",
    proposed_change_summary: str = "",
    verification_status: str = "",
    verification_checks: str = "",
    gate_mode: str = "",
    exploratory: bool = False,
    override_reason: str = "",
    response_format: str = "text",
    evidence_schema: str = "summary",
) -> str:
    """Manually advance RCA state machine for custom agent workflows."""
    rate_error = _check_rate_limit("llmdebug_rca_advance", response_format, evidence_schema)
    if rate_error is not None:
        return rate_error
    return handle_rca_advance(
        request=RCAAdvanceToolRequest(
            action=action,
            out_dir=out_dir,
            session_id=session_id,
            snapshot_path=snapshot_path,
            failure_signature=failure_signature,
            evidence_refs=evidence_refs,
            hypothesis=hypothesis,
            experiment_plan=experiment_plan,
            proposed_change_summary=proposed_change_summary,
            verification_status=verification_status,
            verification_checks=verification_checks,
            gate_mode=gate_mode,
            exploratory=exploratory,
            override_reason=override_reason,
            response_format=response_format,
            evidence_schema=evidence_schema,
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the llmdebug MCP server."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(name)s: %(message)s",
    )
    _mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
