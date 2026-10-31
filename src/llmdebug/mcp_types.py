"""Typed contracts for MCP response payload assembly."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class EvidenceException(TypedDict, total=False):
    """Stable exception summary embedded in MCP evidence payloads.

    Attributes:
        type: Exception class name; empty when the snapshot has no exception.
        message: Exception message text.
    """

    type: str
    message: str


class EvidenceCrashFrame(TypedDict, total=False):
    """Stable crash-frame locator for cross-tool MCP consumers.

    Attributes:
        index: Position of the frame in the snapshot's frame list; ``-1`` when no
            frame is available.
        file_rel: Repository-relative file path, falling back to the recorded path.
        line: Line number of the frame, ``0`` when unknown.
        function: Name of the function executing in the frame.
        code: Source line at the frame's current line.
    """

    index: int
    file_rel: str
    line: int
    function: str
    code: str


class EvidenceLocalSummary(TypedDict, total=False):
    """Compact local-variable preview for stable MCP responses.

    Attributes:
        name: Local variable name.
        type: Type name from the frame's locals metadata; empty when not recorded.
        preview: Short rendering of the value, truncated to 120 characters.
    """

    name: str
    type: str
    preview: str


class EvidenceRepro(TypedDict, total=False):
    """Minimal reproduction metadata exposed in stable MCP responses.

    Attributes:
        argv: Command-line arguments that rerun the failing test.
        nodeid: Pytest node ID of the failing test.
    """

    argv: list[Any]
    nodeid: str


class StableEvidencePayload(TypedDict, total=False):
    """Structured evidence block embedded in stable-schema MCP responses.

    This is the stable, versioned contract that MCP clients parse; fields
    are intentionally kept additive so older clients can ignore new keys.
    Multiple MCP tools can include this payload in their response envelope
    when the stable evidence schema is requested.

    Attributes:
        exception: Exception type and message extracted from the snapshot.
        crash_frame: Index, file, line, function, and source line of the snapshot's
            crash frame (or of the requested frame for ``llmdebug_frame``).
        locals_summary: Brief type-and-preview entries for the frame's locals.
        repro: Reproduction command (argv) and pytest node ID when available.
        evidence_ids: Ordered, deduplicated evidence reference strings such as
            ``exception:<type>`` or ``crash_frame:<file>:<line>:<function>``.
        sections: Ordered rendered sections supplied by the calling tool; empty
            unless a tool provides them.
        summary: Rendered tool output text, or the error message on failures.
    """

    exception: EvidenceException
    crash_frame: EvidenceCrashFrame
    locals_summary: list[EvidenceLocalSummary]
    repro: EvidenceRepro
    evidence_ids: list[str]
    sections: list[dict[str, Any]]
    summary: str


class MCPErrorPayload(TypedDict, total=False):
    """Machine-readable MCP tool error details.

    Attributes:
        code: Stable error code such as ``INVALID_ARGUMENT`` or ``NO_SNAPSHOTS``.
        message: Human-readable description of the failure.
        details: Optional structured context, such as the rejected argument, its
            value, and the allowed values.
    """

    code: str
    message: str
    details: dict[str, Any]


class MCPResponsePayload(TypedDict, total=False):
    """Top-level JSON envelope returned by MCP tool handlers.

    Attributes:
        tool: Name of the MCP tool that produced the response.
        format_version: Version of the envelope format.
        evidence: Stable evidence block and summary text, plus tool-specific fields
            when the ``full`` evidence schema is requested.
        metadata: How the envelope was rendered and whether the call succeeded.
        rca: RCA record, its MCP-facing view, and the prompt contract; present only
            when RCA output was requested and a record is available.
        error: Error details; present only on failures.
    """

    tool: str
    format_version: str
    evidence: dict[str, Any]
    metadata: MCPMetadataPayload
    rca: dict[str, Any]
    error: MCPErrorPayload


class MCPMetadataPayload(TypedDict, total=False):
    """Envelope metadata describing how a JSON MCP payload was rendered.

    Attributes:
        response_format: Format of the rendered envelope (always ``json``).
        with_rca: Whether the caller requested the RCA block.
        evidence_schema: Evidence schema applied, ``summary`` or ``full``.
        status: ``ok`` for successful calls, ``error`` for failures.
    """

    response_format: str
    with_rca: bool
    evidence_schema: str
    status: Literal["ok", "error"]


__all__ = [
    "EvidenceCrashFrame",
    "EvidenceException",
    "EvidenceLocalSummary",
    "EvidenceRepro",
    "MCPErrorPayload",
    "MCPMetadataPayload",
    "MCPResponsePayload",
    "StableEvidencePayload",
]
