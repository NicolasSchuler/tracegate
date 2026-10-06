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
    # The first call binds new evidence; only the three that add nothing count as repairs.
    for _ in range(4):
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


def _advance(out_dir: str, session_id: str, signature: str, action: str, **kwargs: Any) -> Any:
    request = RCAAttemptRequest(
        session_id=session_id,
        action=action,
        failure_signature=signature,
        evidence_refs=("crash_frame:app.py:42:parse_input",),
        hypothesis="parser rejects the empty string",
        experiment_plan="guard the empty case",
        gate_mode="strict",
        **kwargs,
    )
    return advance_rca_attempt(out_dir=out_dir, request=request)


def test_stale_pass_does_not_resolve_a_different_failure(tmp_path: Path) -> None:
    """RCA-01: a pass recorded for ValueError:AAA must not resolve TypeError:BBB."""
    out_dir = str(tmp_path)
    _advance(out_dir, "s", "ValueError:AAA", "verify", verification_result={"status": "pass"})

    record = _advance(out_dir, "s", "TypeError:BBB", "resolve")

    assert record["gate_feedback"]["allowed"] is False
    assert "verification_required" in record["gate_feedback"]["triggered_checks"]
    assert record["verification_result"] is None
    assert record["state"] != "resolved"


def test_a_new_change_invalidates_the_previous_verification(tmp_path: Path) -> None:
    out_dir = str(tmp_path)
    _advance(out_dir, "s", "ValueError:AAA", "verify", verification_result={"status": "pass"})
    changed = _advance(out_dir, "s", "ValueError:AAA", "change")

    record = _advance(out_dir, "s", "ValueError:AAA", "resolve")

    assert changed["verification_result"] is None
    assert record["gate_feedback"]["allowed"] is False


def test_pass_carries_forward_for_the_same_failure(tmp_path: Path) -> None:
    out_dir = str(tmp_path)
    _advance(out_dir, "s", "ValueError:AAA", "verify", verification_result={"status": "pass"})

    record = _advance(out_dir, "s", "ValueError:AAA", "resolve")

    assert record["gate_feedback"]["allowed"] is True
    assert record["state"] == "resolved"
    assert record["verification_result"] == {"status": "pass"}


def test_denied_action_keeps_a_stalled_session_stalled(tmp_path: Path) -> None:
    """RCA-04: "stalled" isn't an action target, and a denial used to reset it to captured."""
    out_dir = str(tmp_path)
    for _ in range(4):
        stalled = advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id="s",
                action="bind_evidence",
                failure_signature="ValueError:AAA",
                evidence_refs=("crash_frame:app.py:42:parse_input",),
                gate_mode="soft",
            ),
        )
    assert stalled["state"] == "stalled"

    denied = advance_rca_attempt(
        out_dir=out_dir,
        request=RCAAttemptRequest(
            session_id="s", action="plan", failure_signature="ValueError:AAA", gate_mode="strict"
        ),
    )

    assert denied["gate_feedback"]["allowed"] is False
    assert denied["state"] == "stalled"


def test_denied_action_with_an_unknown_previous_state_falls_back_to_captured() -> None:
    from llmdebug.rca_transitions import resolve_target_state

    assert resolve_target_state("not-a-state", "plan", False) == "captured"
    assert resolve_target_state("verified", "resolve", False) == "verified"


def test_strict_gate_enforces_the_transition_matrix(tmp_path: Path) -> None:
    """RCA-06: from evidence_bound, resolve is blocked even with a passing verification."""
    out_dir = str(tmp_path)
    _advance(out_dir, "s", "ValueError:AAA", "bind_evidence")

    record = _advance(
        out_dir, "s", "ValueError:AAA", "resolve", verification_result={"status": "pass"}
    )

    assert record["gate_feedback"]["allowed"] is False
    assert record["gate_feedback"]["triggered_checks"] == ["illegal_transition"]
    assert record["state"] == "evidence_bound"


_STRICT_WALK = (
    ("capture", {}),
    ("bind_evidence", {"evidence_refs": ("crash_frame:app.py:42:parse_input",)}),
    ("hypothesize", {"hypothesis": "parse_input does not guard the empty string"}),
    ("plan", {"experiment_plan": "return early for empty input and re-run the test"}),
    ("change", {"proposed_change_summary": "added an empty-string guard"}),
    ("verify", {"verification_result": {"status": "pass"}}),
    ("resolve", {}),
)


def test_strict_workflow_walks_capture_to_resolved(tmp_path: Path) -> None:
    """RCA-06: bookkeeping steps must not stall a first-time strict-mode walk."""
    out_dir = str(tmp_path)
    signature = "ValueError:AAA"
    states: list[str] = []
    for action, extra in _STRICT_WALK:
        record = advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id="s",
                action=action,
                failure_signature=signature,
                gate_mode="strict",
                **extra,
            ),
        )
        assert record["gate_feedback"]["allowed"] is True, (action, record["gate_warnings"])
        assert "stalled" not in " ".join(record["gate_warnings"]).lower()
        states.append(record["state"])

    assert states == [
        "captured",
        "evidence_bound",
        "hypothesized",
        "planned",
        "changed",
        "verified",
        "resolved",
    ]


def test_repeated_unproductive_repairs_still_stall(tmp_path: Path) -> None:
    """RCA-06: the repair streak keeps counting change attempts that bring nothing new."""
    out_dir = str(tmp_path)
    signature = "ValueError:AAA"
    for action, extra in _STRICT_WALK[:4]:
        advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id="s",
                action=action,
                failure_signature=signature,
                gate_mode="strict",
                **extra,
            ),
        )

    record: dict[str, Any] | None = None
    for _ in range(3):
        record = advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id="s",
                action="change",
                failure_signature=signature,
                gate_mode="soft",
                stall_threshold=3,
            ),
        )

    assert record is not None
    assert record["repair_streak"] == 3
    assert record["state"] == "stalled"
    assert any("stalled" in warning.lower() for warning in record["gate_warnings"])


def test_denied_action_does_not_trip_stall_detection(tmp_path: Path) -> None:
    """RCA-06: a rejected request must not change the state or the allowed actions."""
    out_dir = str(tmp_path)
    signature = "ValueError:AAA"
    for _ in range(3):
        before = advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id="s",
                action="bind_evidence",
                failure_signature=signature,
                evidence_refs=("crash_frame:app.py:42:parse_input",),
                hypothesis="parse_input does not guard the empty string",
                gate_mode="soft",
                stall_threshold=3,
            ),
        )
    assert before["state"] == "evidence_bound"
    assert before["repair_streak"] == 2

    # "verify" is not an edge out of "evidence_bound", so strict mode denies it. Evaluating
    # the stall rule anyway used to persist "stalled" for this rejected request.
    denied = advance_rca_attempt(
        out_dir=out_dir,
        request=RCAAttemptRequest(
            session_id="s",
            action="verify",
            failure_signature=signature,
            gate_mode="strict",
            stall_threshold=2,
        ),
    )

    assert denied["gate_feedback"]["allowed"] is False
    assert "illegal_transition" in denied["gate_feedback"]["triggered_checks"]
    assert denied["state"] == "evidence_bound"
    assert denied["repair_streak"] == 2
    assert denied["allowed_actions"] == before["allowed_actions"]
