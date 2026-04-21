from __future__ import annotations

from pathlib import Path

from evals.adaptive_history import (
    load_persistent_adaptive_history,
    persist_adaptive_attempt_history,
)


def _snapshot() -> dict[str, object]:
    return {
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 4,
                "function": "f",
                "code": "return helper.value()",
                "locals": {"value": 1},
            }
        ],
        "pytest": {"nodeid": "test_buggy.py::test_f"},
    }


def test_persistent_adaptive_history_round_trip(tmp_path: Path) -> None:
    out_dir = tmp_path / "contexts" / "adaptive_case" / "layered_adaptive_llm"
    record = {
        "attempt": 1,
        "success": False,
        "note": "patch_verifier_rejected",
        "timed_out": False,
        "failure_signature": "sig-1",
        "attempt_metadata": {
            "retry_summary": {
                "hypothesis_tested": "The bug is confined to buggy.py",
                "new_constraint": "The patch must touch the crash file",
            }
        },
    }
    patch_meta = {
        "patch_verifier_enabled": True,
        "patch_verifier_passed": False,
        "patch_verifier_reasons": ["missed crash file"],
        "evidence_ids_new": ["e1"],
        "evidence_ids_reused": ["e2"],
    }

    session_id = persist_adaptive_attempt_history(
        out_dir=out_dir,
        snapshot=_snapshot(),
        fallback_session_id="adaptive_case:layered_adaptive_llm",
        record=record,
        patch_meta=patch_meta,
        retry_plan_text="Constrain the edit to buggy.py.",
    )
    history = load_persistent_adaptive_history(
        out_dir=out_dir,
        snapshot=_snapshot(),
        fallback_session_id="adaptive_case:layered_adaptive_llm",
    )

    assert history.error == ""
    assert history.session_id == session_id
    assert history.records_loaded == 1
    assert history.lessons
    assert history.lessons[0].source == "rca_history"
    assert "patch_verifier_rejected" in history.lessons[0].lesson
