"""Diff MCP tool handler implementation.

Provides the core implementation for llmdebug_diff, comparing two snapshots
with structured delta output.
"""

from __future__ import annotations

import logging
from typing import Any

from . import _json as json
from ._snapshot_loader import resolve_snapshot_ref
from .mcp_contract import (
    render_invalid_response_format_error,
    render_tool_error,
    render_tool_response,
    validate_evidence_schema,
    validate_response_format,
)
from .mcp_rca import (
    advance_rca_with_snapshot_refs,
    extract_evidence_refs,
    render_rca_update_failed,
)
from .mcp_snapshot_tools import core_evidence_from_snapshot
from .rca_types import RCAAttemptRequest, RCASnapshotAdvanceRequest
from .snapshot_diff import diff_snapshots, format_diff_text, structured_diff_summary

_logger = logging.getLogger("llmdebug.mcp")


def handle_diff(
    old_ref: str = "previous",
    new_ref: str = "latest",
    out_dir: str = ".llmdebug",
    gate_mode: str = "",
    exploratory: bool = False,
    response_format: str = "json",
    with_rca: bool = False,
    evidence_schema: str = "summary",
) -> str:
    """Implementation of llmdebug_diff."""
    format_error = validate_response_format(response_format)
    if format_error is not None:
        return render_invalid_response_format_error(
            tool="llmdebug_diff",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
        )
    schema_error = validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return render_tool_error(
            tool="llmdebug_diff",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )

    try:
        old = resolve_snapshot_ref(old_ref, out_dir)
        new = resolve_snapshot_ref(new_ref, out_dir)
    except FileNotFoundError as e:
        message = str(e)
        code = "NO_SNAPSHOTS" if "No snapshots found" in message else "SNAPSHOT_RESOLUTION_ERROR"
        return render_tool_error(
            tool="llmdebug_diff",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code=code,
            message=message,
        )
    except ValueError as e:
        return render_tool_error(
            tool="llmdebug_diff",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="SNAPSHOT_RESOLUTION_ERROR",
            message=str(e),
        )
    except (ImportError, json.JSONDecodeError, OSError) as e:
        return render_tool_error(
            tool="llmdebug_diff",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="SNAPSHOT_LOAD_ERROR",
            message=f"Error loading snapshots: {type(e).__name__}: {e}",
        )
    except Exception as e:
        _logger.error("Unexpected error in llmdebug_diff: %s: %s", type(e).__name__, e)
        return render_tool_error(
            tool="llmdebug_diff",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="SNAPSHOT_LOAD_ERROR",
            message=f"Error loading snapshots: {type(e).__name__}: {e}",
        )

    result = diff_snapshots(old, new)
    structured = structured_diff_summary(result)
    verification_status = "fail" if result.failure_signature_same else "inconclusive"
    verification_result: dict[str, Any] = {
        "status": verification_status,
        "checks": ["llmdebug_diff"],
    }
    previous_refs = [f"previous:{ref}" for ref in extract_evidence_refs(old)]
    core_evidence = core_evidence_from_snapshot(new)
    evidence_ids = list(core_evidence.get("evidence_ids") or [])
    evidence_ids.extend(previous_refs)
    core_evidence["evidence_ids"] = list(dict.fromkeys(evidence_ids))
    rca_record = (
        advance_rca_with_snapshot_refs(
            request=RCASnapshotAdvanceRequest(
                out_dir=out_dir,
                snapshot=new,
                attempt=RCAAttemptRequest(
                    session_id="",
                    action="verify",
                    failure_signature="",
                    verification_result=verification_result,
                    proposed_change_summary=(
                        "Compared previous and latest failures; "
                        "inspected exception/location/locals deltas."
                    ),
                    gate_mode=gate_mode,
                    exploratory=exploratory,
                ),
            ),
            extra_evidence_refs=previous_refs,
        )
        if with_rca
        else None
    )
    if with_rca and rca_record is None:
        return render_rca_update_failed(
            tool="llmdebug_diff",
            response_format=response_format,
            evidence_schema=evidence_schema,
        )
    body = format_diff_text(result)
    return render_tool_response(
        tool="llmdebug_diff",
        body=body,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=with_rca,
        rca_record=rca_record,
        structured_delta=structured,
        evidence_payload={
            **core_evidence,
            "old_snapshot": old,
            "new_snapshot": new,
        },
    )
