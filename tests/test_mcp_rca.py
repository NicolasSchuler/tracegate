"""Tests for MCP root-cause-analysis context resolution and error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llmdebug import _json as json
from llmdebug.config import SnapshotConfig
from llmdebug.mcp_rca import (
    RCAAdvanceToolRequest,
    RCAInputError,
    SnapshotLoadError,
    effective_gate_mode,
    extract_evidence_refs,
    handle_rca_advance,
    handle_rca_history,
    handle_rca_status,
    resolve_rca_advance_context,
)
from llmdebug.output import write_bundle


def _write_snapshot(out_dir: str) -> None:
    payload = {
        "schema_version": "1.0",
        "name": "test_example",
        "timestamp_utc": "2026-02-07T21:00:00Z",
        "llmdebug_version": "2.0.0",
        "exception": {
            "type": "ValueError",
            "qualified_type": "builtins.ValueError",
            "message": "invalid literal for int(): 'abc'",
        },
        "traceback": "Traceback (most recent call last):\nValueError: invalid literal",
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 42,
                "function": "parse_input",
                "locals": {"value": "'abc'"},
                "locals_meta": {"value": {"type": "builtins.str"}},
            }
        ],
        "crash_frame_index": 0,
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }
    cfg = SnapshotConfig(out_dir=out_dir, output_format="json")
    write_bundle(payload, cfg)


def test_resolve_rca_advance_context_parses_manual_fields_without_snapshot(tmp_path) -> None:
    context = resolve_rca_advance_context(
        request=RCAAdvanceToolRequest(
            out_dir=str(tmp_path),
            snapshot_path="",
            session_id=" sess-1 ",
            failure_signature=" sig-1 ",
            evidence_refs="a,b , c",
            hypothesis=" hypothesis ",
            experiment_plan=" plan ",
            proposed_change_summary=" summary ",
            verification_status="pass",
            verification_checks="pytest, lint ",
            override_reason=" explore ",
            action="bind_evidence",
        ),
    )
    assert context.snapshot is None
    assert context.session_id == "sess-1"
    assert context.failure_signature == "sig-1"
    assert context.evidence_refs == ["a", "b", "c"]
    assert context.hypothesis == "hypothesis"
    assert context.experiment_plan == "plan"
    assert context.proposed_change_summary == "summary"
    assert context.override_reason == "explore"
    assert context.verification_result == {"status": "pass", "checks": ["pytest", "lint"]}


def test_resolve_rca_advance_context_rejects_partial_manual_identity(tmp_path) -> None:
    _write_snapshot(str(tmp_path))
    with pytest.raises(RCAInputError, match="provide both session_id and failure_signature"):
        resolve_rca_advance_context(
            request=RCAAdvanceToolRequest(
                out_dir=str(tmp_path),
                snapshot_path="",
                session_id="manual-session",
                failure_signature="",
                evidence_refs="",
                hypothesis="",
                experiment_plan="",
                proposed_change_summary="",
                verification_status="",
                verification_checks="",
                override_reason="",
                action="bind_evidence",
            ),
        )


def test_resolve_rca_advance_context_derives_identity_when_manual_identity_absent(tmp_path) -> None:
    _write_snapshot(str(tmp_path))
    context = resolve_rca_advance_context(
        request=RCAAdvanceToolRequest(
            out_dir=str(tmp_path),
            snapshot_path="",
            session_id="",
            failure_signature="",
            evidence_refs="",
            hypothesis="",
            experiment_plan="",
            proposed_change_summary="",
            verification_status="",
            verification_checks="",
            override_reason="",
            action="bind_evidence",
        ),
    )
    assert context.snapshot is not None
    assert context.session_id
    assert context.failure_signature.startswith("ValueError:")
    assert context.verification_result is None


def test_resolve_rca_advance_context_without_snapshot_uses_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmdebug.mcp_rca.derive_session_id",
        lambda snapshot, out_dir, fallback: "manual-session",
    )
    monkeypatch.setattr(
        "llmdebug.mcp_rca.build_failure_signature",
        lambda snapshot: "manual-signature",
    )

    context = resolve_rca_advance_context(
        request=RCAAdvanceToolRequest(
            out_dir=str(tmp_path),
            snapshot_path="",
            session_id="",
            failure_signature="",
            evidence_refs="",
            hypothesis="",
            experiment_plan="",
            proposed_change_summary="",
            verification_status="",
            verification_checks="",
            override_reason="",
            action="bind_evidence",
        ),
    )
    assert context.snapshot is None
    assert context.session_id == "manual-session"
    assert context.failure_signature == "manual-signature"


def test_resolve_rca_advance_context_raises_snapshot_load_error(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(SnapshotLoadError, match="Error loading snapshot: FileNotFoundError"):
        resolve_rca_advance_context(
            request=RCAAdvanceToolRequest(
                out_dir=str(tmp_path),
                snapshot_path=str(missing_path),
                session_id="",
                failure_signature="",
                evidence_refs="",
                hypothesis="",
                experiment_plan="",
                proposed_change_summary="",
                verification_status="",
                verification_checks="",
                override_reason="",
                action="bind_evidence",
            ),
        )


def test_extract_evidence_refs_handles_non_numeric_line() -> None:
    refs = extract_evidence_refs(
        {
            "exception": {"type": "ValueError", "message": "boom"},
            "frames": [{"file": "/code/app.py", "line": "?"}],
            "crash_frame_index": 0,
            "pytest": {"nodeid": "tests/test_app.py::test_parse"},
        }
    )

    assert "exception:ValueError" in refs
    assert "crash_frame:/code/app.py:0:?" in refs
    assert "pytest_nodeid:tests/test_app.py::test_parse" in refs


def test_extract_evidence_refs_falls_back_to_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llmdebug.mcp_rca.coerce_crash_frame_index", lambda _snapshot: 1)
    refs = extract_evidence_refs(
        {
            "exception": {"type": "ValueError", "message": "boom"},
            "frames": [
                {"file": "/code/app.py", "line": 12, "function": "first"},
                "bad-frame",
            ],
            "pytest": {"nodeid": "tests/test_app.py::test_parse"},
        }
    )

    assert "crash_frame:/code/app.py:12:first" in refs


# ---------------------------------------------------------------------------
# effective_gate_mode
# ---------------------------------------------------------------------------


def test_effective_gate_mode_valid() -> None:
    assert effective_gate_mode("off") == "off"
    assert effective_gate_mode("soft") == "soft"
    assert effective_gate_mode("strict") == "strict"
    assert effective_gate_mode("  SOFT  ") == "soft"


def test_effective_gate_mode_invalid_defaults_to_soft() -> None:
    assert effective_gate_mode("invalid") == "soft"
    assert effective_gate_mode("") == "soft"


def test_effective_gate_mode_none_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_GATE_MODE", "strict")
    assert effective_gate_mode(None) == "strict"


# ---------------------------------------------------------------------------
# handle_rca_status
# ---------------------------------------------------------------------------


def test_handle_rca_status_invalid_response_format(tmp_path: Path) -> None:
    result = handle_rca_status(out_dir=str(tmp_path), response_format="xml")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_rca_status_invalid_evidence_schema(tmp_path: Path) -> None:
    result = handle_rca_status(
        out_dir=str(tmp_path), response_format="json", evidence_schema="verbose"
    )
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_rca_status_no_attempts_text(tmp_path: Path) -> None:
    result = handle_rca_status(out_dir=str(tmp_path), response_format="text")
    assert "No RCA attempts found" in result


def test_handle_rca_status_no_attempts_json(tmp_path: Path) -> None:
    result = handle_rca_status(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "NO_RCA_ATTEMPTS"


def test_handle_rca_status_no_attempts_with_session_id(tmp_path: Path) -> None:
    result = handle_rca_status(out_dir=str(tmp_path), session_id="s1", response_format="text")
    assert "s1" in result


def test_handle_rca_status_with_snapshot_path_session_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    _write_snapshot(str(tmp_path))
    # Use the helper with a snapshot path that will fail gracefully
    result = handle_rca_status(
        out_dir=str(tmp_path),
        snapshot_path=str(tmp_path / "nonexistent.json"),
        response_format="text",
    )
    # Should still return "No RCA attempts" since no actual attempts
    assert "No RCA attempts" in result


def test_handle_rca_status_uses_snapshot_path_to_derive_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = {
        "exception": {"type": "ValueError", "message": "boom"},
        "frames": [],
    }
    monkeypatch.setattr("llmdebug.mcp_rca.load_snapshot", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(
        "llmdebug.mcp_rca.derive_session_id",
        lambda snap, out_dir, fallback: "derived-session",
    )
    monkeypatch.setattr("llmdebug.mcp_rca.get_rca_status", lambda *_a, **_k: None)

    result = handle_rca_status(
        out_dir=str(tmp_path),
        snapshot_path=str(tmp_path / "snapshot.json"),
        response_format="text",
    )

    assert "derived-session" in result


def test_handle_rca_status_success_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    _write_snapshot(str(tmp_path))
    # create one attempt so status is non-None
    from llmdebug.rca_state import advance_rca_attempt
    from llmdebug.rca_types import RCAAttemptRequest

    advance_rca_attempt(
        out_dir=str(tmp_path),
        request=RCAAttemptRequest(
            session_id="s-status",
            action="bind_evidence",
            failure_signature="sig-status",
            evidence_refs=("r1",),
        ),
    )
    result = handle_rca_status(out_dir=str(tmp_path), session_id="s-status", response_format="text")
    assert "s-status" in result or "state" in result


def test_handle_rca_status_success_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    from llmdebug.rca_state import advance_rca_attempt
    from llmdebug.rca_types import RCAAttemptRequest

    advance_rca_attempt(
        out_dir=str(tmp_path),
        request=RCAAttemptRequest(
            session_id="s-status-j",
            action="bind_evidence",
            failure_signature="sig-j",
            evidence_refs=("r1",),
        ),
    )
    result = handle_rca_status(
        out_dir=str(tmp_path), session_id="s-status-j", response_format="json"
    )
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "ok"


# ---------------------------------------------------------------------------
# handle_rca_history
# ---------------------------------------------------------------------------


def test_handle_rca_history_invalid_response_format(tmp_path: Path) -> None:
    result = handle_rca_history(out_dir=str(tmp_path), response_format="xml")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_rca_history_invalid_evidence_schema(tmp_path: Path) -> None:
    result = handle_rca_history(
        out_dir=str(tmp_path), response_format="json", evidence_schema="verbose"
    )
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_rca_history_limit_zero(tmp_path: Path) -> None:
    result = handle_rca_history(out_dir=str(tmp_path), response_format="json", limit=0)
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "INVALID_ARGUMENT"


def test_handle_rca_history_no_history_text(tmp_path: Path) -> None:
    result = handle_rca_history(out_dir=str(tmp_path), response_format="text")
    assert "No RCA attempts" in result


def test_handle_rca_history_no_history_json(tmp_path: Path) -> None:
    result = handle_rca_history(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "NO_RCA_ATTEMPTS"


def test_handle_rca_history_no_history_with_session(tmp_path: Path) -> None:
    result = handle_rca_history(out_dir=str(tmp_path), session_id="sess-x", response_format="text")
    assert "sess-x" in result


def test_handle_rca_history_success_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    from llmdebug.rca_state import advance_rca_attempt
    from llmdebug.rca_types import RCAAttemptRequest

    advance_rca_attempt(
        out_dir=str(tmp_path),
        request=RCAAttemptRequest(
            session_id="s-hist",
            action="bind_evidence",
            failure_signature="sig-hist",
            evidence_refs=("r1",),
        ),
    )
    result = handle_rca_history(out_dir=str(tmp_path), response_format="text")
    # result should be JSON-formatted history list
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) >= 1


def test_handle_rca_history_success_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    from llmdebug.rca_state import advance_rca_attempt
    from llmdebug.rca_types import RCAAttemptRequest

    advance_rca_attempt(
        out_dir=str(tmp_path),
        request=RCAAttemptRequest(
            session_id="s-hist-j",
            action="bind_evidence",
            failure_signature="sig-hj",
            evidence_refs=("r1",),
        ),
    )
    result = handle_rca_history(out_dir=str(tmp_path), response_format="json", limit=5)
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "ok"


# ---------------------------------------------------------------------------
# handle_rca_advance
# ---------------------------------------------------------------------------


def _make_advance_request(**kwargs) -> RCAAdvanceToolRequest:  # type: ignore[no-untyped-def]
    defaults = {
        "action": "bind_evidence",
        "out_dir": "/tmp/fake",
        "session_id": "s1",
        "failure_signature": "sig1",
        "evidence_refs": "r1",
        "response_format": "json",
        "evidence_schema": "summary",
    }
    defaults.update(kwargs)
    return RCAAdvanceToolRequest(**defaults)  # type: ignore[arg-type]


def test_handle_rca_advance_invalid_response_format(tmp_path: Path) -> None:
    req = _make_advance_request(out_dir=str(tmp_path), response_format="xml")
    result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_rca_advance_invalid_evidence_schema(tmp_path: Path) -> None:
    req = _make_advance_request(out_dir=str(tmp_path), evidence_schema="verbose")
    result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_rca_advance_snapshot_load_error_text(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_rca.resolve_rca_advance_context",
        side_effect=SnapshotLoadError("load failed"),
    ):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="text")
        result = handle_rca_advance(request=req)
    assert "load failed" in result


def test_handle_rca_advance_snapshot_load_error_json(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_rca.resolve_rca_advance_context",
        side_effect=SnapshotLoadError("load failed"),
    ):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="json")
        result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "SNAPSHOT_LOAD_ERROR"


def test_handle_rca_advance_input_error_text(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_rca.resolve_rca_advance_context",
        side_effect=RCAInputError("bad input"),
    ):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="text")
        result = handle_rca_advance(request=req)
    assert "bad input" in result


def test_handle_rca_advance_input_error_json(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_rca.resolve_rca_advance_context",
        side_effect=RCAInputError("bad input"),
    ):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="json")
        result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "INVALID_ARGUMENT"


def test_handle_rca_advance_value_error_text(tmp_path: Path) -> None:
    with patch("llmdebug.mcp_rca.advance_rca_attempt", side_effect=ValueError("bad action")):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="text")
        result = handle_rca_advance(request=req)
    assert "bad action" in result


def test_handle_rca_advance_value_error_json(tmp_path: Path) -> None:
    with patch("llmdebug.mcp_rca.advance_rca_attempt", side_effect=ValueError("bad action")):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="json")
        result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "INVALID_ARGUMENT"


def test_handle_rca_advance_generic_exception_text(tmp_path: Path) -> None:
    with patch("llmdebug.mcp_rca.advance_rca_attempt", side_effect=RuntimeError("unexpected")):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="text")
        result = handle_rca_advance(request=req)
    assert "Error updating RCA state" in result


def test_handle_rca_advance_generic_exception_json(tmp_path: Path) -> None:
    with patch("llmdebug.mcp_rca.advance_rca_attempt", side_effect=RuntimeError("unexpected")):
        req = _make_advance_request(out_dir=str(tmp_path), response_format="json")
        result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "RCA_STATE_UPDATE_ERROR"


def test_handle_rca_advance_success_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    req = _make_advance_request(out_dir=str(tmp_path), response_format="text")
    result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert "state" in parsed


def test_handle_rca_advance_success_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    req = _make_advance_request(out_dir=str(tmp_path), response_format="json")
    result = handle_rca_advance(request=req)
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "ok"


# ---------------------------------------------------------------------------
# extract_evidence_refs — edge-case branches
# ---------------------------------------------------------------------------


def test_extract_evidence_refs_non_dict_exception_skips_exc_processing() -> None:
    """Branch [180, 188]: exception is not a dict — exc processing skipped."""
    refs = extract_evidence_refs(
        {
            "exception": "not_a_dict",
            "frames": [{"file": "/code/app.py", "line": 1, "function": "fn"}],
            "crash_frame_index": 0,
        }
    )
    assert not any(r.startswith("exception:") for r in refs)
    assert any(r.startswith("crash_frame:") for r in refs)


def test_extract_evidence_refs_non_dict_frame_at_index_zero() -> None:
    """Branches [194, 196] and [196, 202]: frames[0] is not a dict — frame is None."""
    refs = extract_evidence_refs(
        {
            "exception": {"type": "ValueError", "message": "boom"},
            "frames": ["not_a_dict"],
            "crash_frame_index": 0,
        }
    )
    assert "exception:ValueError" in refs
    assert not any(r.startswith("crash_frame:") for r in refs)


def test_extract_evidence_refs_non_dict_pytest_block_skips_nodeid() -> None:
    """Branch [203, 207]: pytest block is not a dict — nodeid skipped."""
    refs = extract_evidence_refs(
        {
            "exception": {"type": "ValueError", "message": "boom"},
            "frames": [{"file": "/code/app.py", "line": 1, "function": "fn"}],
            "crash_frame_index": 0,
            "pytest": "not_a_dict",
        }
    )
    assert "exception:ValueError" in refs
    assert any(r.startswith("crash_frame:") for r in refs)
    assert not any(r.startswith("pytest_nodeid:") for r in refs)
