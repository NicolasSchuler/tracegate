"""Tests for eval prompt rendering contracts and template correctness."""

from __future__ import annotations

import evals.toon_llm_encoder as toon_llm_encoder
from evals.eval_prompts import (
    _build_snapshot_prompt_text,
    _build_stage_b_prompt_sections,
    _format_snapshot_section,
    _project_snapshot_for_prompt,
    _render_retry_history_turn,
    prune_snapshot_for_prompt,
    render_prompt,
    render_retry_plan_prompt,
    render_stage_a_request_prompt,
    render_stage_b_patch_prompt,
)


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


def test_render_prompt_contains_rca_contract_sections() -> None:
    prompt = render_prompt(evidence=_base_evidence(), snapshot=None)
    assert "RCA checklist — write each section explicitly before proposing your patch" in prompt
    assert "1) Observed failure: quote the concrete evidence" in prompt
    assert "root-cause diagnosis" in prompt.lower()
    assert "root-cause hypothesis" not in prompt.lower()
    assert "Structured diff vs previous failed attempt" in prompt
    assert "hypothesis_delta" not in prompt
    assert "diagnosis_delta" in prompt


def test_render_prompt_no_format_instructions_in_user_message() -> None:
    """Format instructions were removed from user message (handled by system messages)."""
    prompt = render_prompt(evidence=_base_evidence(), snapshot=None)
    assert "Return ONLY a single unified diff" not in prompt
    assert "Do NOT wrap the diff in Markdown fences" not in prompt


def test_render_prompt_contains_loop_breaker_when_signature_repeats() -> None:
    evidence = _base_evidence(failure_signature_repeat=True)
    prompt = render_prompt(evidence=evidence, snapshot=None)
    assert "Loop breaker" in prompt
    assert "signature repeated" in prompt
    assert "Change either diagnosis or evidence basis" in prompt
    assert "Change either hypothesis or evidence basis" not in prompt


def test_render_prompt_embeds_snapshot_summary_when_present() -> None:
    snapshot = {
        "exception": {
            "type": "ValueError",
            "message": "bad input",
            "error_category": {"category": "value_error", "suggestion": "Check the input."},
        },
        "frames": [{"file_rel": "buggy.py", "line": 3, "function": "f", "code": "x = 1/0"}],
    }
    prompt = render_prompt(evidence=_base_evidence(), snapshot=snapshot)
    assert "=== llmdebug Debug Snapshot ===" in prompt
    assert "ValueError" in prompt
    assert "Category:" not in prompt
    assert "Suggestion:" not in prompt
    assert "buggy.py:3 in f()" in prompt


def test_render_prompt_snapshot_before_file_context() -> None:
    """Snapshot should appear before file index and project files."""
    snapshot = {
        "exception": {"type": "IndexError", "message": "out of range"},
        "frames": [{"file_rel": "buggy.py", "line": 5, "function": "g"}],
    }
    evidence = _base_evidence(files="--- buggy.py ---\ndef g(): pass\n")
    prompt = render_prompt(evidence=evidence, snapshot=snapshot)
    snapshot_pos = prompt.index("=== llmdebug Debug Snapshot ===")
    files_pos = prompt.index("Project files:")
    assert snapshot_pos < files_pos


def test_render_prompt_pytest_output_before_snapshot() -> None:
    """Pytest output should appear before snapshot."""
    snapshot = {
        "exception": {"type": "IndexError", "message": "out of range"},
        "frames": [{"file_rel": "buggy.py", "line": 5, "function": "g"}],
    }
    prompt = render_prompt(evidence=_base_evidence(), snapshot=snapshot)
    pytest_pos = prompt.index("Pytest output (stdout):")
    snapshot_pos = prompt.index("=== llmdebug Debug Snapshot ===")
    assert pytest_pos < snapshot_pos


def test_render_prompt_attempt_1_no_noise() -> None:
    """On attempt 1, no 'Previous failure signature' or 'Failure signature repeated'."""
    evidence = _base_evidence(
        previous_failure_signature=None,
        failure_signature_repeat=False,
        previous_attempt_diff=None,
    )
    prompt = render_prompt(evidence=evidence, snapshot=None)
    assert "Previous failure signature" not in prompt
    assert "Failure signature repeated" not in prompt
    assert "Structured diff vs previous failed attempt" not in prompt
    # RCA step 6 should be absent too.
    assert "6) Diff vs previous failed attempt" not in prompt


def test_render_prompt_attempt_2_includes_previous_context() -> None:
    """On attempt 2+, previous context is included."""
    prompt = render_prompt(evidence=_base_evidence(), snapshot=None)
    assert "Previous failure signature" in prompt
    assert "Failure signature repeated" in prompt
    assert "6) Diff vs previous failed attempt" in prompt


def test_render_stage_a_prompt_excludes_project_files() -> None:
    evidence = _base_evidence(files="--- buggy.py ---\nprint('x')\n")
    prompt = render_stage_a_request_prompt(evidence=evidence, snapshot=None)
    assert "STEP 1" in prompt
    assert "DIAGNOSE from traceback alone" in prompt
    assert "STRICT JSON only" in prompt
    assert "Allowed request types: file, search, frame, local_var" in prompt
    assert "Project files:" not in prompt


def test_render_stage_b_prompt_includes_stage_a_context() -> None:
    evidence = _base_evidence(files="--- buggy.py ---\nprint('x')\n")
    prompt = render_stage_b_patch_prompt(
        evidence=evidence,
        snapshot=None,
        stage_a_context="--- buggy.py ---\nprint('context')\n",
        stage_a_evidence_index="- e_abc123def0: file:buggy.py (32 chars)\n",
        include_project_files=False,
    )
    assert "Stage-A requested context:" in prompt
    assert "Stage-A evidence index:" in prompt
    assert "e_abc123def0" in prompt
    assert "print('context')" in prompt
    assert "Project files:" not in prompt


def test_render_prompt_instructions_target_source_fixes_not_observability_only() -> None:
    prompt = render_prompt(evidence=_base_evidence(), snapshot=None)
    assert "Prioritize non-test source files" in prompt
    assert "Ignore cosmetic CLI/TUI wording or formatting tweaks" in prompt
    assert "Do not submit observability-only patches" in prompt
    assert "CRITICAL" in prompt
    assert "ROOT CAUSE visible in the traceback" in prompt


def test_render_stage_b_prompt_includes_timeout_risk_guidance_when_timed_out() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(timed_out=True, timeout_sec=30),
        snapshot=None,
        include_project_files=False,
    )
    assert "Timeout risk signal" in prompt
    assert "timed out" in prompt
    assert "budget=30s" in prompt


def test_render_stage_b_prompt_includes_verifier_retry_guidance() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(
            previous_note="patch_verifier_rejected",
            previous_patch_verifier_rejected=True,
            previous_patch_verifier_reasons=["touches_only_test_files"],
            previous_patch_verifier_scope="adaptive_only",
        ),
        snapshot=None,
        include_project_files=False,
    )
    assert "Verifier-aware retry guidance" in prompt
    assert "touches_only_test_files" in prompt
    assert "adaptive_only" in prompt


def test_render_stage_b_prompt_includes_retry_reflection_when_enabled() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        include_project_files=False,
        prompt_retry_reflection=True,
    )
    assert "Retry reflection" in prompt
    assert "why the previous attempt failed" in prompt


def test_render_stage_b_prompt_enables_adaptive_cot_only_when_flagged() -> None:
    evidence = _base_evidence(condition="layered_adaptive_llm")
    prompt_without_flag = render_stage_b_patch_prompt(
        evidence=evidence,
        snapshot=None,
        include_project_files=False,
    )
    prompt_with_flag = render_stage_b_patch_prompt(
        evidence=evidence,
        snapshot=None,
        include_project_files=False,
        prompt_adaptive_stage_b_cot=True,
    )
    assert (
        "Before proposing a fix, think through the problem step by step" not in prompt_without_flag
    )
    assert "Before proposing a fix, think through the problem step by step" in prompt_with_flag


def test_render_stage_b_prompt_includes_repair_method_scaffold_when_enabled() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        include_project_files=False,
        prompt_repair_method_scaffold=True,
    )
    assert "Modification method" in prompt
    assert "minimum behavior-changing modification" in prompt


def test_render_stage_b_prompt_includes_retry_plan_when_provided() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        include_project_files=False,
        retry_repair_plan=(
            "Diagnosis: wrong helper return value\n"
            "Why previous attempt failed: touched the wrong branch\n"
            "New constraint: preserve helper public API\n"
            "Minimal edit plan: update helper.value\n"
            "Verification plan: rerun pytest"
        ),
    )
    assert "Structured repair plan from planning step:" in prompt
    assert "Diagnosis: wrong helper return value" in prompt


def test_stage_b_prompt_order_modes_reorder_without_changing_membership() -> None:
    snapshot = {
        "exception": {"type": "IndexError", "message": "out of range"},
        "frames": [{"file_rel": "buggy.py", "line": 5, "function": "g"}],
    }
    evidence = _base_evidence(files="--- buggy.py ---\ndef g(): pass\n")
    current_sections = _build_stage_b_prompt_sections(
        evidence=evidence,
        snapshot=snapshot,
        snapshot_prompt_text="",
        stage_a_context="Stage-A context block\n",
        stage_a_evidence_index="- e1: buggy.py\n",
        include_stage_a_evidence_index=True,
        compacted_context="Compacted context block\n",
        include_snapshot=True,
        include_pytest_output=True,
        include_stage_a_context=True,
        include_project_files=True,
        retry_packet="Retry packet\n",
        retry_history="Retry history\n",
        retry_prompt_mode="full",
        attempt=2,
        known_evidence_refs=["e0"],
        new_evidence_ids=["e1"],
        snapshot_lite_context="Snapshot lite\n",
        prompt_order_mode="current",
    )
    crash_first_sections = _build_stage_b_prompt_sections(
        evidence=evidence,
        snapshot=snapshot,
        snapshot_prompt_text="",
        stage_a_context="Stage-A context block\n",
        stage_a_evidence_index="- e1: buggy.py\n",
        include_stage_a_evidence_index=True,
        compacted_context="Compacted context block\n",
        include_snapshot=True,
        include_pytest_output=True,
        include_stage_a_context=True,
        include_project_files=True,
        retry_packet="Retry packet\n",
        retry_history="Retry history\n",
        retry_prompt_mode="full",
        attempt=2,
        known_evidence_refs=["e0"],
        new_evidence_ids=["e1"],
        snapshot_lite_context="Snapshot lite\n",
        prompt_order_mode="crash_first",
    )

    current_membership = sorted(
        (str(section["key"]), str(section["text"])) for section in current_sections
    )
    crash_first_membership = sorted(
        (str(section["key"]), str(section["text"])) for section in crash_first_sections
    )

    assert current_membership == crash_first_membership
    assert [section["key"] for section in current_sections] != [
        section["key"] for section in crash_first_sections
    ]
    assert [section["key"] for section in crash_first_sections[:4]] == [
        "instructions",
        "rca_checklist",
        "failure_signatures",
        "failure_signatures",
    ]


def test_stage_b_prompt_order_modes_keep_previous_attempt_diff_before_closing() -> None:
    snapshot = {
        "exception": {"type": "IndexError", "message": "out of range"},
        "frames": [{"file_rel": "buggy.py", "line": 5, "function": "g"}],
    }
    evidence = _base_evidence(files="--- buggy.py ---\ndef g(): pass\n")
    crash_first_sections = _build_stage_b_prompt_sections(
        evidence=evidence,
        snapshot=snapshot,
        snapshot_prompt_text="",
        stage_a_context="Stage-A context block\n",
        stage_a_evidence_index="- e1: buggy.py\n",
        include_stage_a_evidence_index=True,
        compacted_context="Compacted context block\n",
        include_snapshot=True,
        include_pytest_output=True,
        include_stage_a_context=True,
        include_project_files=True,
        retry_packet="Retry packet\n",
        retry_history="Retry history\n",
        retry_prompt_mode="full",
        attempt=2,
        known_evidence_refs=["e0"],
        new_evidence_ids=["e1"],
        snapshot_lite_context="Snapshot lite\n",
        retry_repair_plan="Diagnosis: fix helper\n",
        prompt_order_mode="crash_first",
    )
    evidence_last_sections = _build_stage_b_prompt_sections(
        evidence=evidence,
        snapshot=snapshot,
        snapshot_prompt_text="",
        stage_a_context="Stage-A context block\n",
        stage_a_evidence_index="- e1: buggy.py\n",
        include_stage_a_evidence_index=True,
        compacted_context="Compacted context block\n",
        include_snapshot=True,
        include_pytest_output=True,
        include_stage_a_context=True,
        include_project_files=True,
        retry_packet="Retry packet\n",
        retry_history="Retry history\n",
        retry_prompt_mode="full",
        attempt=2,
        known_evidence_refs=["e0"],
        new_evidence_ids=["e1"],
        snapshot_lite_context="Snapshot lite\n",
        retry_repair_plan="Diagnosis: fix helper\n",
        prompt_order_mode="evidence_last",
    )

    crash_first_keys = [str(section["key"]) for section in crash_first_sections]
    evidence_last_keys = [str(section["key"]) for section in evidence_last_sections]

    assert crash_first_keys[-1] == "closing"
    assert evidence_last_keys[-1] == "closing"
    assert "previous_attempt_diff" in crash_first_keys
    assert "previous_attempt_diff" in evidence_last_keys
    assert crash_first_keys.index("previous_attempt_diff") < crash_first_keys.index("snapshot")
    assert max(
        index for index, key in enumerate(crash_first_keys) if key == "failure_signatures"
    ) < crash_first_keys.index("previous_attempt_diff")
    assert evidence_last_keys.index("previous_attempt_diff") < evidence_last_keys.index(
        "retry_repair_plan"
    )
    assert max(
        index for index, key in enumerate(evidence_last_keys) if key == "failure_signatures"
    ) < evidence_last_keys.index("previous_attempt_diff")


def test_prune_snapshot_for_prompt_keeps_crash_and_adjacent_user_frames() -> None:
    snapshot = {
        "crash_frame_index": 2,
        "frames": [
            {"file_rel": None, "line": 1, "function": "lib0"},
            {"file_rel": "pkg/outer.py", "line": 10, "function": "outer"},
            {"file_rel": "pkg/core.py", "line": 20, "function": "crash"},
            {"file_rel": "pkg/helper.py", "line": 30, "function": "helper"},
            {"file_rel": "pkg/extra.py", "line": 40, "function": "extra"},
        ],
    }

    pruned_snapshot, meta = prune_snapshot_for_prompt(
        snapshot,
        mode="heuristic",
        pregate_score=0.9,
        failure_signature_repeat=True,
        max_user_frames=3,
    )

    assert meta["applied"] is True
    assert meta["frames_before"] == 5
    assert meta["frames_after"] == 3
    assert meta["kept_frame_indices"] == [1, 2, 3]
    assert pruned_snapshot["crash_frame_index"] == 1


def test_render_retry_plan_prompt_requests_plan_only() -> None:
    prompt = render_retry_plan_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        include_project_files=False,
        prompt_retry_reflection=True,
    )
    assert "CRITICAL — RETURN ONLY THE REPAIR PLAN:" in prompt
    assert (
        "Diagnosis, Why previous attempt failed, New constraint, Minimal edit plan, Verification plan"
        in prompt
    )
    assert "Retry reflection" in prompt
    assert "CRITICAL — YOUR PATCH MUST:" not in prompt


def test_render_stage_b_prompt_skips_retry_reflection_without_previous_attempt() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(previous_failure_signature=None, previous_attempt_diff=None),
        snapshot=None,
        include_project_files=False,
        prompt_retry_reflection=True,
    )
    assert "Retry reflection" not in prompt


def test_render_prompt_base_size_guardrail_for_token_usage() -> None:
    prompt = render_prompt(evidence=_base_evidence(), snapshot=None)
    # Guard against accidental prompt bloat from copy changes.
    assert len(prompt) <= 5000


def test_render_stage_b_prompt_delta_retry_uses_delta_sections() -> None:
    evidence = _base_evidence(files="--- buggy.py ---\nprint('x')\n")
    prompt = render_stage_b_patch_prompt(
        evidence=evidence,
        snapshot=None,
        stage_a_context="--- helper.py ---\ndef value():\n    return 2\n",
        include_project_files=True,
        retry_packet="Retry packet (delta-first, structured):\n{}",
        retry_prompt_mode="delta",
        attempt=2,
        known_evidence_refs=["ev_old"],
        new_evidence_ids=["ev_new"],
    )
    assert "Pytest output (stdout):" not in prompt
    assert "Project files:" not in prompt
    assert "Known evidence references:" in prompt
    assert "New evidence IDs in this retry:" in prompt
    assert "Stage-A requested context:" in prompt


def test_render_stage_b_prompt_delta_mode_attempt_one_is_full() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        retry_prompt_mode="delta",
        attempt=1,
    )
    assert "Pytest output (stdout):" in prompt


def test_render_stage_b_prompt_includes_retry_history_when_provided() -> None:
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        retry_history="Attempt 1\nOutcome: no_patch_proposed",
    )
    assert "Retry history from previous attempts:" in prompt
    assert "Outcome: no_patch_proposed" in prompt


def test_render_retry_history_turn_prefers_persisted_summary() -> None:
    rendered = _render_retry_history_turn(
        attempt=2,
        note="tests_still_failing",
        failure_signature="AssertionError:live",
        patch_meta={
            "retry_summary": {
                "attempt": 1,
                "hypothesis_tested": "persisted hypothesis",
                "evidence_used": "persisted evidence",
                "outcome": "persisted_outcome",
                "new_constraint": "persisted constraint",
                "failure_signature": "AssertionError:persisted",
            }
        },
    )
    assert rendered.startswith("Attempt 1")
    assert "persisted hypothesis" in rendered
    assert "Outcome: persisted_outcome" in rendered
    assert "tests_still_failing" not in rendered


# --- _format_snapshot_section tests ---


def test_format_snapshot_basic() -> None:
    snapshot = {
        "exception": {
            "type": "IndexError",
            "message": "list index out of range",
            "error_category": {
                "category": "index_error",
                "suggestion": "Check index bounds.",
            },
        },
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 6,
                "function": "last_item",
                "code": "return values[i]",
                "locals": {"values": [1, 2, 3], "i": 3},
                "locals_meta": {
                    "values": {"type": "builtins.list", "len": 3},
                    "i": {"type": "builtins.int"},
                },
            },
        ],
    }
    section = _format_snapshot_section(snapshot)
    assert "Exception: IndexError" in section
    assert "Category:" not in section
    assert "Suggestion:" not in section
    assert "Frame 0 (crash site): buggy.py:6 in last_item()" in section
    assert "Code: return values[i]" in section
    assert "values = [1, 2, 3]" in section
    assert "i = 3" in section


def test_format_snapshot_filters_framework_frames() -> None:
    snapshot = {
        "exception": {"type": "Error", "message": "boom"},
        "frames": [
            {"file_rel": "buggy.py", "line": 1, "function": "f"},
            {"file_rel": None, "line": 100, "function": "_multicall"},
            {"file_rel": "test_buggy.py", "line": 5, "function": "test_f"},
        ],
    }
    section = _format_snapshot_section(snapshot)
    assert "buggy.py" in section
    assert "test_buggy.py" in section
    assert "_multicall" not in section


def test_format_snapshot_max_frames() -> None:
    snapshot = {
        "exception": {"type": "Error", "message": "boom"},
        "frames": [{"file_rel": f"file{i}.py", "line": i, "function": f"fn{i}"} for i in range(10)],
    }
    section = _format_snapshot_section(snapshot, max_frames=2)
    assert "file0.py" in section
    assert "file1.py" in section
    assert "file2.py" not in section


def test_format_snapshot_array_locals() -> None:
    snapshot = {
        "exception": {"type": "ValueError", "message": "shape mismatch"},
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 9,
                "function": "project",
                "code": "return data @ weights",
                "locals": {
                    "data": {
                        "__array__": "numpy.ndarray",
                        "shape": [2, 4],
                        "dtype": "float64",
                    },
                    "weights": {
                        "__array__": "numpy.ndarray",
                        "shape": [3, 2],
                        "dtype": "float64",
                    },
                },
                "locals_meta": {
                    "data": {"type": "numpy.ndarray", "shape": [2, 4]},
                    "weights": {"type": "numpy.ndarray", "shape": [3, 2]},
                },
            },
        ],
    }
    section = _format_snapshot_section(snapshot)
    assert "<numpy.ndarray, shape=(2, 4), dtype=float64>" in section
    assert "<numpy.ndarray, shape=(3, 2), dtype=float64>" in section


def test_format_snapshot_skips_pytest_internal_vars() -> None:
    snapshot = {
        "exception": {"type": "Error", "message": "boom"},
        "frames": [
            {
                "file_rel": "test_buggy.py",
                "line": 7,
                "function": "test_f",
                "code": "assert f(x) == 3",
                "locals": {"@py_assert1": [1, 2, 3], "x": 5},
                "locals_meta": {
                    "@py_assert1": {"type": "builtins.list", "len": 3},
                    "x": {"type": "builtins.int"},
                },
            },
        ],
    }
    section = _format_snapshot_section(snapshot)
    assert "@py_assert1" not in section
    assert "x = 5" in section


def test_project_snapshot_for_prompt_keeps_only_stage_b_snapshot_surface() -> None:
    snapshot = {
        "schema_version": 3,
        "git": {"commit": "abc123"},
        "env": {"python": "3.12"},
        "exception": {"type": "LookupError", "message": "boom", "category": "ignored"},
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 7,
                "function": "crash",
                "code": "return values[i]",
                "locals": {"values": [1, 2, 3], "@py_assert1": [1, 2, 3], "i": 3},
                "locals_meta": {
                    "values": {"type": "builtins.list", "len": 3},
                    "@py_assert1": {"type": "builtins.list", "len": 3},
                    "i": {"type": "builtins.int"},
                },
            },
            {"file_rel": None, "line": 100, "function": "_pytest_internal"},
            {"file_rel": "helper.py", "line": 3, "function": "helper"},
            {"file_rel": "third.py", "line": 9, "function": "third"},
            {"file_rel": "ignored.py", "line": 12, "function": "ignored"},
        ],
    }

    projected = _project_snapshot_for_prompt(snapshot)

    assert projected["exception"] == {"type": "LookupError", "message": "boom"}
    assert "git" not in projected
    assert "env" not in projected
    assert [frame["file_rel"] for frame in projected["frames"]] == [
        "buggy.py",
        "helper.py",
        "third.py",
    ]
    assert "@py_assert1" not in projected["frames"][0]["locals"]
    assert "@py_assert1" not in projected["frames"][0]["locals_meta"]


def test_build_snapshot_prompt_text_skips_encoded_payload_when_not_shorter(
    monkeypatch,
) -> None:
    snapshot = {
        "exception": {"type": "IndexError", "message": "boom"},
        "frames": [{"file_rel": "buggy.py", "line": 1, "function": "f"}],
    }

    monkeypatch.setattr(
        toon_llm_encoder,
        "encode_for_llm",
        lambda *args, **kwargs: "x" * 5000,
    )

    assert _build_snapshot_prompt_text(snapshot) == ""


def test_build_snapshot_prompt_text_encodes_projected_snapshot(monkeypatch) -> None:
    snapshot = {
        "exception": {"type": "IndexError", "message": "boom"},
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 1,
                "function": "f",
                "locals": {"keep": 1, "@skip": 2},
                "locals_meta": {
                    "keep": {"type": "builtins.int"},
                    "@skip": {"type": "builtins.int"},
                },
            }
        ],
        "git": {"commit": "abc123"},
    }
    captured: dict[str, object] = {}

    def _fake_encode(payload, **kwargs):  # type: ignore[no-untyped-def]
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return "tiny"

    monkeypatch.setattr(toon_llm_encoder, "encode_for_llm", _fake_encode)

    assert _build_snapshot_prompt_text(snapshot) == "tiny"
    assert captured["payload"] == {
        "exception": {"type": "IndexError", "message": "boom"},
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 1,
                "function": "f",
                "locals": {"keep": 1},
                "locals_meta": {"keep": {"type": "builtins.int"}},
            }
        ],
    }


# --- P0 prompt intervention tests ---


def test_stage_a_diagnose_then_decide_structure() -> None:
    """P0.1: Stage-A prompt enforces STEP 1/STEP 2 diagnose-then-decide flow."""
    prompt = render_stage_a_request_prompt(evidence=_base_evidence(), snapshot=None)
    assert "STEP 1" in prompt
    assert "STEP 2" in prompt
    assert "DIAGNOSE from traceback alone" in prompt
    assert "DECIDE" in prompt
    assert '"diagnosis"' in prompt
    assert '"missing_fact"' in prompt


def test_stage_a_calibration_examples() -> None:
    """P0.2: Three calibration examples are present in the Stage-A prompt."""
    prompt = render_stage_a_request_prompt(evidence=_base_evidence(), snapshot=None)
    assert "Example 1" in prompt
    assert "Example 2" in prompt
    assert "Example 3" in prompt
    assert "IndexError" in prompt
    assert "KeyError" in prompt
    assert "AssertionError" in prompt
    assert "traceback IS sufficient" in prompt
    assert "traceback NOT sufficient" in prompt


def test_stage_a_budget_visible_when_set() -> None:
    """P0.5: Budget line appears with correct values when budget is provided."""
    prompt = render_stage_a_request_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        helper_budget_total=3,
        helper_budget_remaining=2,
    )
    assert "Budget:" in prompt
    assert "2 helper call(s) remaining" in prompt
    assert "of 3 total" in prompt


def test_stage_a_no_budget_when_zero() -> None:
    """P0.5: No budget line when defaults (0, 0)."""
    prompt = render_stage_a_request_prompt(evidence=_base_evidence(), snapshot=None)
    assert "Budget:" not in prompt


def test_stage_a_budget_remaining_decrements() -> None:
    """P0.5: Shows correct remaining/total when budget partially consumed."""
    prompt = render_stage_a_request_prompt(
        evidence=_base_evidence(),
        snapshot=None,
        helper_budget_total=5,
        helper_budget_remaining=1,
    )
    assert "1 helper call(s) remaining" in prompt
    assert "of 5 total" in prompt


def test_stage_b_distractor_mitigation_with_snapshot() -> None:
    """P0.3: Distractor mitigation present when snapshot is provided."""
    snapshot = {
        "exception": {"type": "ValueError", "message": "bad"},
        "frames": [{"file_rel": "buggy.py", "line": 1, "function": "f"}],
    }
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(), snapshot=snapshot, include_project_files=False
    )
    assert "irrelevant distractors" in prompt


def test_stage_b_no_distractor_mitigation_without_snapshot() -> None:
    """P0.3: Distractor mitigation absent when no snapshot."""
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(), snapshot=None, include_project_files=False
    )
    assert "irrelevant distractors" not in prompt


def test_stronger_closing_section() -> None:
    """P0.4: CRITICAL block present, old 'Goal:' text absent."""
    prompt = render_stage_b_patch_prompt(
        evidence=_base_evidence(), snapshot=None, include_project_files=False
    )
    assert "CRITICAL" in prompt
    assert "ROOT CAUSE visible in the traceback" in prompt
    assert "Goal: return the minimal code edit" not in prompt


# --- Lite evidence action formatters ---


def _sample_snapshot() -> dict:
    """Return a minimal snapshot dict for testing formatters."""
    return {
        "exception": {"type": "TypeError", "message": "unsupported operand"},
        "frames": [
            {
                "file_rel": "src/app.py",
                "function": "compute",
                "line": 42,
                "code": "value + 1",
                "locals": {"value": "<coroutine>", "seed": 5},
                "locals_meta": {
                    "value": {"type": "coroutine"},
                    "seed": {"type": "int"},
                },
            },
            {
                "file_rel": "src/main.py",
                "function": "run",
                "line": 10,
                "code": "result = compute(5)",
                "locals": {"x": 5},
                "locals_meta": {"x": {"type": "int"}},
            },
        ],
    }


def test_format_frame_locals_section_basic() -> None:
    """frame_locals_lite formatter includes crash frame code + locals."""
    from evals.eval_prompts import _format_frame_locals_section

    result = _format_frame_locals_section(_sample_snapshot())
    assert "=== llmdebug Crash Frame ===" in result
    assert "TypeError" in result
    assert "unsupported operand" in result
    assert "Crash site: src/app.py:42 in compute()" in result
    assert "Code: value + 1" in result
    assert "value" in result
    assert "seed" in result
    # Only crash frame — no second frame.
    assert "src/main.py" not in result


def test_format_frame_locals_section_max_locals() -> None:
    """frame_locals_lite respects max_locals cap."""
    from evals.eval_prompts import _format_frame_locals_section

    result = _format_frame_locals_section(_sample_snapshot(), max_locals=1)
    # Only one local should appear.
    lines = [
        ln
        for ln in result.splitlines()
        if ln.strip().startswith("value") or ln.strip().startswith("seed")
    ]
    assert len(lines) == 1


def test_format_frame_locals_section_empty_snapshot() -> None:
    """frame_locals_lite handles empty snapshot gracefully."""
    from evals.eval_prompts import _format_frame_locals_section

    result = _format_frame_locals_section({})
    assert "=== llmdebug Crash Frame ===" in result
    assert "===" in result


def test_format_frame_locals_section_skips_framework_frames() -> None:
    """frame_locals_lite skips frames without file_rel."""
    from evals.eval_prompts import _format_frame_locals_section

    snap = {
        "exception": {"type": "ValueError", "message": "bad"},
        "frames": [
            {"function": "internal", "line": 1},  # No file_rel → skip.
            {"file_rel": "user.py", "function": "run", "line": 5, "code": "x()"},
        ],
    }
    result = _format_frame_locals_section(snap)
    assert "user.py" in result
    assert "internal" not in result


def test_format_snapshot_full_section_basic() -> None:
    """snapshot_full_lite produces full snapshot with all frames and locals."""
    from evals.eval_prompts import _format_snapshot_full_section

    result = _format_snapshot_full_section(_sample_snapshot())
    assert "=== llmdebug Debug Snapshot ===" in result
    assert "TypeError" in result
    assert "src/app.py" in result
    assert "src/main.py" in result
    assert "value" in result
    assert "seed" in result


def test_format_snapshot_full_section_empty() -> None:
    """snapshot_full_lite handles empty snapshot."""
    from evals.eval_prompts import _format_snapshot_full_section

    result = _format_snapshot_full_section({})
    assert "=== llmdebug Debug Snapshot ===" in result


def test_lite_actions_not_in_stage_a() -> None:
    """Lite actions must NOT be in HELPER_STAGE_A_ACTIONS (they bypass Stage-A)."""
    from evals.eval_types import (
        ACTION_FRAME_LOCALS_LITE,
        ACTION_SNAPSHOT_FULL_LITE,
        HELPER_CONTEXT_MATERIALIZING_ACTIONS,
        HELPER_STAGE_A_ACTIONS,
    )

    assert ACTION_FRAME_LOCALS_LITE not in HELPER_STAGE_A_ACTIONS
    assert ACTION_SNAPSHOT_FULL_LITE not in HELPER_STAGE_A_ACTIONS
    assert ACTION_FRAME_LOCALS_LITE in HELPER_CONTEXT_MATERIALIZING_ACTIONS
    assert ACTION_SNAPSHOT_FULL_LITE in HELPER_CONTEXT_MATERIALIZING_ACTIONS


def test_lite_actions_in_online_policy() -> None:
    """Lite actions must be selectable by LinUCB."""
    from evals.adaptive_online_policy import ONLINE_POLICY_ACTIONS

    assert "frame_locals_lite" in ONLINE_POLICY_ACTIONS
    assert "snapshot_full_lite" in ONLINE_POLICY_ACTIONS
