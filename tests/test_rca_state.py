"""Tests for RCA state persistence and transitions."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

from llmdebug import _json as json
from llmdebug.rca_state import (
    advance_rca_attempt,
    build_failure_signature,
    derive_session_id,
    get_rca_history,
    get_rca_status,
)
from llmdebug.rca_types import RCAAttemptRequest


def _sample_snapshot(
    *,
    exc_type: str = "ValueError",
    exc_message: str = "invalid literal",
    line: int = 42,
    function: str = "parse_input",
) -> dict[str, Any]:
    return {
        "name": "test_parse",
        "exception": {"type": exc_type, "message": exc_message},
        "frames": [
            {
                "file_rel": "app.py",
                "line": line,
                "function": function,
            }
        ],
        "crash_frame_index": 0,
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }


def test_build_failure_signature_is_stable_for_identical_snapshot() -> None:
    snapshot = _sample_snapshot()
    sig_a = build_failure_signature(snapshot)
    sig_b = build_failure_signature(snapshot)
    assert sig_a == sig_b
    assert sig_a.startswith("ValueError:")


def test_advance_rca_attempt_persists_and_updates_state(tmp_path: Path) -> None:
    out_dir = str(tmp_path)
    snapshot = _sample_snapshot()
    session_id = derive_session_id(snapshot, out_dir=out_dir, fallback="mcp")
    failure_signature = build_failure_signature(snapshot)

    first = advance_rca_attempt(
        out_dir=out_dir,
        request=RCAAttemptRequest(
            session_id=session_id,
            action="bind_evidence",
            failure_signature=failure_signature,
            evidence_refs=("crash_frame:app.py:42:parse_input",),
            gate_mode="soft",
        ),
    )
    second = advance_rca_attempt(
        out_dir=out_dir,
        request=RCAAttemptRequest(
            session_id=session_id,
            action="hypothesize",
            failure_signature=failure_signature,
            evidence_refs=("crash_frame:app.py:42:parse_input",),
            hypothesis="Non-digit input reaches int() conversion.",
            experiment_plan="Validate input before converting.",
            gate_mode="soft",
        ),
    )

    assert first["state"] == "evidence_bound"
    assert second["state"] == "hypothesized"
    status = get_rca_status(out_dir=out_dir, session_id=session_id)
    assert status is not None
    assert status["attempt_id"] == second["attempt_id"]
    history = get_rca_history(out_dir=out_dir, session_id=session_id, limit=10)
    assert len(history) == 2


def test_strict_gate_blocks_transition_when_requirements_missing(tmp_path: Path) -> None:
    out_dir = str(tmp_path)
    snapshot = _sample_snapshot()
    session_id = derive_session_id(snapshot, out_dir=out_dir, fallback="mcp")
    failure_signature = build_failure_signature(snapshot)

    record = advance_rca_attempt(
        out_dir=out_dir,
        request=RCAAttemptRequest(
            session_id=session_id,
            action="plan",
            failure_signature=failure_signature,
            evidence_refs=(),
            hypothesis="",
            gate_mode="strict",
        ),
    )
    assert record["gate_feedback"]["allowed"] is False
    assert record["state"] == "captured"


def test_stalled_state_after_repeated_signature_without_hypothesis_change(tmp_path: Path) -> None:
    out_dir = str(tmp_path)
    snapshot = _sample_snapshot()
    session_id = derive_session_id(snapshot, out_dir=out_dir, fallback="mcp")
    failure_signature = build_failure_signature(snapshot)

    record: dict[str, Any] | None = None
    for _ in range(3):
        record = advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id=session_id,
                action="bind_evidence",
                failure_signature=failure_signature,
                evidence_refs=("crash_frame:app.py:42:parse_input",),
                gate_mode="soft",
                stall_threshold=3,
            ),
        )

    assert record is not None
    assert record["state"] == "stalled"
    assert any("stalled" in warning.lower() for warning in record["gate_warnings"])


def test_rca_history_retention_cap(tmp_path: Path, monkeypatch) -> None:
    out_dir = str(tmp_path)
    snapshot = _sample_snapshot()
    session_id = derive_session_id(snapshot, out_dir=out_dir, fallback="mcp")
    failure_signature = build_failure_signature(snapshot)

    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "2")
    for _ in range(4):
        advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id=session_id,
                action="bind_evidence",
                failure_signature=failure_signature,
                evidence_refs=("crash_frame:app.py:42:parse_input",),
                gate_mode="soft",
            ),
        )

    history = get_rca_history(out_dir=out_dir, session_id=session_id, limit=10)
    assert len(history) == 2

    store = Path(out_dir) / "rca_attempts.jsonl"
    assert len(store.read_text(encoding="utf-8").splitlines()) == 2


def test_concurrent_appends_produce_valid_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    out_dir = str(tmp_path)
    snapshot = _sample_snapshot()
    session_id = derive_session_id(snapshot, out_dir=out_dir, fallback="mcp")
    failure_signature = build_failure_signature(snapshot)
    writes = 64

    def _write_attempt(i: int) -> None:
        advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id=session_id,
                action="bind_evidence",
                failure_signature=failure_signature,
                evidence_refs=(f"crash_frame:app.py:42:parse_input:{i}",),
                gate_mode="soft",
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_write_attempt, range(writes)))

    store = Path(out_dir) / "rca_attempts.jsonl"
    lines = [line for line in store.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == writes
    for line in lines:
        decoded = json.loads(line)
        assert isinstance(decoded, dict)
        assert "attempt_id" in decoded


def test_jsonl_status_tail_reader_skips_malformed_lines(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    out_dir = str(tmp_path)
    store = Path(out_dir) / "rca_attempts.jsonl"
    records = [
        {"attempt_id": "a1", "session_id": "s1", "state": "captured"},
        {"attempt_id": "b1", "session_id": "s2", "state": "captured"},
        {"attempt_id": "a2", "session_id": "s1", "state": "evidence_bound"},
    ]
    store.write_text(
        "\n".join(
            [
                json.dumps(records[0]),
                "{not-valid-json",
                json.dumps(records[1]),
                " ",
                json.dumps(records[2]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    status = get_rca_status(out_dir=out_dir, session_id="s1")
    assert status is not None
    assert status["attempt_id"] == "a2"
