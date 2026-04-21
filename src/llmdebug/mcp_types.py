"""Typed contracts for MCP response payload assembly."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class EvidenceException(TypedDict, total=False):
    """Stable exception summary embedded in MCP evidence payloads."""

    type: str
    message: str


class EvidenceCrashFrame(TypedDict, total=False):
    """Stable crash-frame locator for cross-tool MCP consumers."""

    index: int
    file_rel: str
    line: int
    function: str
    code: str


class EvidenceLocalSummary(TypedDict, total=False):
    """Compact local-variable preview for stable MCP responses."""

    name: str
    type: str
    preview: str


class EvidenceRepro(TypedDict, total=False):
    """Minimal reproduction metadata exposed in stable MCP responses."""

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
        crash_frame: File, line, and function of the outermost crash frame.
        locals_summary: Brief type-and-preview entries for crash-frame locals.
        repro: Reproduction command (argv) and pytest node ID when available.
        evidence_ids: Ordered content-addressed IDs used for deduplication.
        sections: Ordered rendered sections ready for LLM prompt assembly.
        summary: Free-text one-paragraph crash summary.
    """

    exception: EvidenceException
    crash_frame: EvidenceCrashFrame
    locals_summary: list[EvidenceLocalSummary]
    repro: EvidenceRepro
    evidence_ids: list[str]
    sections: list[dict[str, Any]]
    summary: str


class MCPErrorPayload(TypedDict, total=False):
    """Machine-readable MCP tool error details."""

    code: str
    message: str
    details: dict[str, Any]


class MCPResponsePayload(TypedDict, total=False):
    """Top-level JSON envelope returned by MCP tool handlers."""

    tool: str
    format_version: str
    evidence: dict[str, Any]
    metadata: MCPMetadataPayload
    rca: dict[str, Any]
    error: MCPErrorPayload


class MCPMetadataPayload(TypedDict, total=False):
    """Envelope metadata describing how a JSON MCP payload was rendered."""

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
