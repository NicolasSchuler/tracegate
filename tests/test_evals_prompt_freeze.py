"""Snapshot tests ensuring eval prompt templates remain stable across changes."""

from __future__ import annotations

from evals.eval_prompts import (
    render_prompt,
    render_stage_a_request_prompt,
    render_stage_b_patch_prompt,
)
from llmdebug._hashing import xxh3_128_hexdigest_text


def _base_evidence(**overrides: object) -> dict:
    base = {
        "case_id": "case_1",
        "condition": "base_repair",
        "attempt": 2,
        "repro": ["python", "-m", "pytest", "-q"],
        "stdout": "E   AssertionError: expected 2 got 1\n",
        "stderr": "",
        "failure_signature": "AssertionError:abc123",
        "previous_failure_signature": "AssertionError:def456",
        "failure_signature_repeat": False,
        "previous_attempt_diff": {
            "attempt_prev": "attempt_01",
            "attempt_curr": "attempt_02",
            "failure_signature_same": False,
            "evidence_delta": {"new_frames": [], "dropped_frames": [], "message_changed": True},
            "change_delta": {"touched_files": ["buggy.py"], "touched_functions": ["f"]},
            "hypothesis_delta": "unknown",
            "verification_delta": {
                "old_status": "unknown",
                "new_status": "unknown",
                "status_changed": False,
            },
        },
        "file_index": "buggy.py\ntest_buggy.py",
        "files": "",
        "snapshot_available": False,
        "snapshot_error": None,
    }
    base.update(overrides)
    return base


def _xxh3_128(value: str) -> str:
    return xxh3_128_hexdigest_text(value)


def test_render_prompt_checksum_freeze() -> None:
    prompt = render_prompt(evidence=_base_evidence(), snapshot=None)
    assert _xxh3_128(prompt) == "9ebfa59b7a4c6275fab2c3be1facd69f"


def test_render_stage_a_prompt_checksum_freeze() -> None:
    prompt = render_stage_a_request_prompt(evidence=_base_evidence(), snapshot=None)
    assert _xxh3_128(prompt) == "489b3bfbd001fb44aa9df3436d591c9e"


def test_render_stage_b_prompt_checksum_freeze() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        stage_a_context="--- helper.py ---\ndef value():\n    return 2\n",
        include_project_files=False,
    )
    assert _xxh3_128(prompt) == "93b7a4429d9aa1a6a9d562cdffe3af43"
