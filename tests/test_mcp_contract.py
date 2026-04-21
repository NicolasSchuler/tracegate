"""Tests for MCP response format validation and evidence schema contracts."""

from __future__ import annotations

import pytest

from llmdebug import _json as json
from llmdebug.mcp_contract import (
    render_invalid_response_format_error,
    render_tool_error,
    render_tool_response,
    validate_evidence_schema,
    validate_response_format,
)


def test_validate_response_format_rejects_unknown_values() -> None:
    message = validate_response_format("yaml")
    assert message is not None
    assert "Invalid response_format" in message


def test_validate_evidence_schema_rejects_unknown_values() -> None:
    message = validate_evidence_schema("raw")
    assert message is not None
    assert "Invalid evidence_schema" in message


def test_render_invalid_response_format_error_uses_json_envelope() -> None:
    result = render_invalid_response_format_error(
        tool="llmdebug_example",
        response_format="yaml",
        evidence_schema="summary",
        with_rca=True,
    )
    payload = json.loads(result)
    assert payload["tool"] == "llmdebug_example"
    assert payload["metadata"]["response_format"] == "json"
    assert payload["metadata"]["status"] == "error"
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert payload["error"]["details"]["argument"] == "response_format"
    assert payload["error"]["details"]["value"] == "yaml"


def test_render_tool_error_text_mode_returns_plain_message() -> None:
    result = render_tool_error(
        tool="llmdebug_example",
        response_format="text",
        evidence_schema="summary",
        with_rca=False,
        code="NO_SNAPSHOTS",
        message="No snapshots found.",
    )
    assert result == "No snapshots found."


def test_render_tool_error_json_normalizes_invalid_schema_to_summary() -> None:
    result = render_tool_error(
        tool="llmdebug_example",
        response_format="json",
        evidence_schema="raw",
        with_rca=False,
        code="SNAPSHOT_LOAD_ERROR",
        message="Failed to load snapshot",
    )
    payload = json.loads(result)
    assert payload["metadata"]["evidence_schema"] == "summary"
    assert payload["metadata"]["status"] == "error"


def test_render_tool_response_json_full_includes_rca_and_structured_diff() -> None:
    result = render_tool_response(
        tool="llmdebug_example",
        body="Diff summary",
        response_format="json",
        evidence_schema="full",
        with_rca=True,
        rca_record={
            "attempt_id": "a1",
            "session_id": "s1",
            "state": "verified",
            "failure_signature": "sig",
            "repeat_count": 1,
            "allowed_actions": ["verify"],
            "next_required": [],
            "gate_feedback": {},
            "gate_warnings": [],
        },
        structured_delta={"same_signature": False},
        evidence_payload={"snapshot": {"id": "snap-1"}},
    )
    payload = json.loads(result)
    assert payload["metadata"]["status"] == "ok"
    assert payload["metadata"]["evidence_schema"] == "full"
    assert payload["evidence"]["diff"] == {"same_signature": False}
    assert payload["evidence"]["snapshot"] == {"id": "snap-1"}
    assert payload["rca"]["view"]["attempt_id"] == "a1"
    assert payload["rca"]["prompt_contract"]


def test_render_tool_response_json_summary_keeps_explicit_sections() -> None:
    result = render_tool_response(
        tool="llmdebug_example",
        body="Summary",
        response_format="json",
        evidence_schema="summary",
        with_rca=False,
        evidence_sections=[{"title": "Crash", "body": "details"}],
    )
    payload = json.loads(result)
    assert payload["evidence"]["sections"] == [{"title": "Crash", "body": "details"}]
    assert "rca" not in payload


def test_render_invalid_response_format_error_raises_when_value_is_valid() -> None:
    with pytest.raises(ValueError, match="response_format is valid"):
        render_invalid_response_format_error(
            tool="llmdebug_example",
            response_format="json",
            evidence_schema="summary",
            with_rca=False,
        )
