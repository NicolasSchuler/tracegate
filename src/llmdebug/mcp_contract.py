"""Shared MCP response contract helpers.

These helpers keep tool response envelopes consistent across MCP adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import _json as json
from .mcp_types import (
    MCPErrorPayload,
    MCPMetadataPayload,
    MCPResponsePayload,
    StableEvidencePayload,
)

VALID_RESPONSE_FORMATS = {"text", "json"}
VALID_EVIDENCE_SCHEMAS = {"summary", "full"}
RCA_PROMPT_CONTRACT = [
    "Observed failure (cite evidence)",
    "Root-cause hypothesis (falsifiable)",
    "Why evidence supports it",
    "Minimal fix plan",
    "Verification plan",
    "Diff vs previous failed attempt",
]


def validate_response_format(response_format: str) -> str | None:
    """Validate the requested MCP response envelope format.

    Returns an error string instead of raising so callers can surface the
    problem in a stable MCP error envelope when needed.

    Args:
        response_format: Requested envelope format; ``text`` and ``json`` are valid.

    Returns:
        An error message for an invalid format, or ``None`` when it is valid.
    """
    if response_format not in VALID_RESPONSE_FORMATS:
        return f"Invalid response_format {response_format!r}. Use 'text' or 'json'."
    return None


def validate_evidence_schema(evidence_schema: str) -> str | None:
    """Validate the requested evidence schema name.

    ``summary`` is the stable cross-tool contract. ``full`` keeps those stable
    fields and may append additive tool-specific details.

    Args:
        evidence_schema: Requested evidence schema; ``summary`` and ``full`` are valid.

    Returns:
        An error message for an invalid schema, or ``None`` when it is valid.
    """
    if evidence_schema not in VALID_EVIDENCE_SCHEMAS:
        return f"Invalid evidence_schema {evidence_schema!r}. Use 'summary' or 'full'."
    return None


def normalized_evidence_schema(evidence_schema: str) -> str:
    """Return a safe schema label for error responses.

    Invalid inputs degrade to ``summary`` so JSON error payloads still expose a
    predictable evidence block.

    Args:
        evidence_schema: Requested evidence schema, possibly invalid.

    Returns:
        *evidence_schema* when it is valid, else ``summary``.
    """
    if evidence_schema in VALID_EVIDENCE_SCHEMAS:
        return evidence_schema
    return "summary"


def default_evidence_fields() -> StableEvidencePayload:
    """Return the guaranteed keys for stable MCP evidence payloads.

    Every JSON success or error envelope starts from this shape so clients can
    rely on these keys being present even when a tool cannot load a snapshot.

    Returns:
        A new payload with empty exception, crash frame (index ``-1``), locals
        summary, repro, evidence IDs, and sections.
    """
    return {
        "exception": {"type": "", "message": ""},
        "crash_frame": {"index": -1, "file_rel": "", "line": 0, "function": "", "code": ""},
        "locals_summary": [],
        "repro": {"argv": [], "nodeid": ""},
        "evidence_ids": [],
        "sections": [],
    }


def rca_view(rca_record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a full RCA record down to the stable MCP-facing subset.

    Args:
        rca_record: Full RCA attempt record.

    Returns:
        The attempt and session IDs, state, failure signature, repeat count, allowed
        actions, next required steps, gate feedback, and gate warnings (under
        ``warnings``).
    """
    return {
        "attempt_id": rca_record.get("attempt_id"),
        "session_id": rca_record.get("session_id"),
        "state": rca_record.get("state"),
        "failure_signature": rca_record.get("failure_signature"),
        "repeat_count": rca_record.get("repeat_count"),
        "allowed_actions": rca_record.get("allowed_actions", []),
        "next_required": rca_record.get("next_required", []),
        "gate_feedback": rca_record.get("gate_feedback", {}),
        "warnings": rca_record.get("gate_warnings", []),
    }


def build_text_response(
    body: str,
    *,
    rca_record: Mapping[str, Any] | None = None,
    structured_delta: dict[str, Any] | None = None,
    with_rca: bool = True,
) -> str:
    """Render the human-readable MCP response form.

    The text format is optimized for direct display. The structured diff is
    appended when one is given, and the RCA view and prompt contract only when
    *with_rca* is set and a record is available.

    Args:
        body: Rendered tool output.
        rca_record: RCA attempt record to summarize, if any.
        structured_delta: Structured diff to append as JSON, if any.
        with_rca: Whether to append the RCA view and prompt contract.

    Returns:
        The body followed by the optional sections, separated by blank lines.
    """
    sections = [body]
    if structured_delta is not None:
        structured_text = json.dumps(
            structured_delta,
            indent=2,
            ensure_ascii=False,
        )
        sections.append("## Structured Diff\n" + structured_text)
    if with_rca and rca_record is not None:
        sections.append("## RCA\n" + json.dumps(rca_view(rca_record), indent=2, ensure_ascii=False))
        sections.append(
            "## RCA Prompt Contract\n"
            + "\n".join(f"{idx}) {line}" for idx, line in enumerate(RCA_PROMPT_CONTRACT, start=1))
        )
    return "\n\n".join(sections)


def build_json_response(
    *,
    tool: str,
    body: str,
    rca_record: Mapping[str, Any] | None,
    structured_delta: dict[str, Any] | None,
    evidence_schema: str,
    with_rca: bool,
    evidence_payload: StableEvidencePayload | dict[str, Any] | None = None,
    evidence_sections: list[dict[str, Any]] | None = None,
) -> str:
    """Render a JSON MCP response envelope.

    ``summary`` schema returns only the stable evidence block. ``full`` keeps
    those stable fields and appends additive details such as structured diffs
    and tool-specific payload fields.

    Args:
        tool: Name of the MCP tool, echoed in the envelope.
        body: Rendered tool output, stored as ``evidence.summary``.
        rca_record: RCA attempt record to embed, if any.
        structured_delta: Structured diff, included as ``evidence.diff`` in ``full``
            mode.
        evidence_schema: ``summary`` or ``full``.
        with_rca: Whether to embed the RCA record, view, and prompt contract.
        evidence_payload: Stable evidence fields, which override the defaults, plus
            tool-specific fields that only ``full`` mode includes.
        evidence_sections: Rendered sections that replace ``evidence.sections``.

    Returns:
        The JSON-encoded response envelope.
    """
    stable = default_evidence_fields()
    evidence: dict[str, Any] = dict(stable)
    evidence["summary"] = body

    payload_dict = evidence_payload if isinstance(evidence_payload, dict) else {}
    for key in stable:
        if key in payload_dict:
            evidence[key] = payload_dict[key]
    if evidence_sections is not None:
        evidence["sections"] = evidence_sections
    if evidence_schema == "full":
        if structured_delta is not None:
            evidence["diff"] = structured_delta
        evidence.update({key: value for key, value in payload_dict.items() if key not in stable})

    metadata: MCPMetadataPayload = {
        "response_format": "json",
        "with_rca": with_rca,
        "evidence_schema": evidence_schema,
        "status": "ok",
    }
    payload: MCPResponsePayload = {
        "tool": tool,
        "format_version": "2.0",
        "evidence": evidence,
        "metadata": metadata,
    }
    if with_rca and rca_record is not None:
        payload["rca"] = {
            "record": rca_record,
            "view": rca_view(rca_record),
            "prompt_contract": list(RCA_PROMPT_CONTRACT),
        }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def render_tool_response(
    *,
    tool: str,
    body: str,
    response_format: str,
    evidence_schema: str,
    with_rca: bool,
    rca_record: Mapping[str, Any] | None = None,
    structured_delta: dict[str, Any] | None = None,
    evidence_payload: StableEvidencePayload | dict[str, Any] | None = None,
    evidence_sections: list[dict[str, Any]] | None = None,
) -> str:
    """Dispatch to the requested MCP response format.

    JSON responses always use the stable envelope defined in ``mcp_types``.
    Any other format renders text: the body plus the optional structured diff and
    RCA sections.

    Args:
        tool: Name of the MCP tool, echoed in JSON envelopes.
        body: Rendered tool output.
        response_format: ``json`` for the envelope; any other value renders text.
        evidence_schema: ``summary`` or ``full`` evidence detail for JSON envelopes.
        with_rca: Whether to include the RCA block.
        rca_record: RCA attempt record to include, if any.
        structured_delta: Structured diff to include, if any.
        evidence_payload: Evidence fields for JSON envelopes.
        evidence_sections: Rendered sections for JSON envelopes.

    Returns:
        The rendered response.
    """
    if response_format == "json":
        return build_json_response(
            tool=tool,
            body=body,
            rca_record=rca_record,
            structured_delta=structured_delta,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            evidence_payload=evidence_payload,
            evidence_sections=evidence_sections,
        )
    return build_text_response(
        body,
        rca_record=rca_record,
        structured_delta=structured_delta,
        with_rca=with_rca,
    )


def render_tool_error(
    *,
    tool: str,
    response_format: str,
    evidence_schema: str,
    with_rca: bool,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> str:
    """Render a tool failure in text or stable JSON form.

    Text mode returns only the message. JSON mode preserves the normal envelope
    shape, marks metadata status as ``error``, and includes a stable empty
    evidence payload so clients can parse failures uniformly.

    Args:
        tool: Name of the MCP tool, echoed in the envelope.
        response_format: ``json`` for the envelope; any other value returns the bare
            message.
        evidence_schema: Requested evidence schema; invalid values are reported as
            ``summary``.
        with_rca: Whether the caller requested RCA, echoed in the metadata.
        code: Machine-readable error code.
        message: Human-readable error description.
        details: Optional structured error context.

    Returns:
        The error message in text mode, or the JSON-encoded error envelope.
    """
    if response_format != "json":
        return message

    normalized_schema = normalized_evidence_schema(evidence_schema)
    evidence: dict[str, Any] = dict(default_evidence_fields())
    evidence["summary"] = message
    metadata: MCPMetadataPayload = {
        "response_format": "json",
        "with_rca": with_rca,
        "evidence_schema": normalized_schema,
        "status": "error",
    }
    error_payload: MCPErrorPayload = {
        "code": code,
        "message": message,
    }
    if details is not None:
        error_payload["details"] = details
    payload: MCPResponsePayload = {
        "tool": tool,
        "format_version": "2.0",
        "evidence": evidence,
        "metadata": metadata,
        "error": error_payload,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def render_invalid_response_format_error(
    *,
    tool: str,
    response_format: str,
    evidence_schema: str,
    with_rca: bool,
) -> str:
    """Render a stable JSON error for an invalid ``response_format`` value.

    Args:
        tool: Name of the MCP tool, echoed in the envelope.
        response_format: The rejected format value.
        evidence_schema: Requested evidence schema.
        with_rca: Whether the caller requested RCA.

    Returns:
        A JSON ``INVALID_ARGUMENT`` error envelope listing the allowed formats.

    Raises:
        ValueError: If *response_format* is actually valid.
    """
    message = validate_response_format(response_format)
    if message is None:
        raise ValueError("response_format is valid; cannot render invalid-format error")
    return render_tool_error(
        tool=tool,
        response_format="json",
        evidence_schema=evidence_schema,
        with_rca=with_rca,
        code="INVALID_ARGUMENT",
        message=message,
        details={
            "argument": "response_format",
            "value": response_format,
            "allowed_values": sorted(VALID_RESPONSE_FORMATS),
        },
    )
