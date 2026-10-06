"""Tests for TEP (Textual Equilibrium Propagation) components.

Covers:
- ``_tep_jaccard_similarity`` word-level Jaccard similarity
- TEP nudge escalation in ``_build_stage_b_prompt_sections``
- ``RetryPlanResult`` TEP-related dataclass fields
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.adaptive_tep import (
    TEP_NODE_CONTEXT_COMPACTION,
    TEP_NODE_PATCH_PROPOSER,
    TEP_NODE_RETRY_PLANNER,
    TEP_NODE_STAGE_A_REQUEST,
    TEP_STAGE_B_MUTABLE_KEYS,
    AdaptiveTepController,
    TepCriticResult,
    TepNodeState,
    TepNodeValidationEntry,
    _default_system_prompts_for_node,
    apply_update_payload,
    extract_mutable_stage_b_blocks,
    load_tep_state,
    next_beta,
    next_temperature,
    resolve_equilibrium_status,
    run_post_attempt_learning,
    should_accept_candidate_update,
)
from evals.eval_attempt_flow import (
    _build_full_tep_node_packets,
    _tep_jaccard_similarity,
    _update_aux_signals_from_attempt,
)
from evals.eval_prompts import _build_stage_b_prompt_sections
from evals.eval_types import PromptBudgetResult
from evals.native_stage_b_flow import RetryPlanResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_evidence(*, failure_signature_repeat: bool = False) -> dict[str, Any]:
    """Minimal evidence dict accepted by ``_build_stage_b_prompt_sections``."""
    return {
        "failure_signature_repeat": failure_signature_repeat,
        "repro": "python test.py",
        "stdout": "",
        "stderr": "AssertionError: expected 1 got 2",
        "buggy_code": "def f(): return 2",
        "failure_signature": "AssertionError",
    }


def _build_sections(
    *,
    evidence: dict[str, Any],
    enable_tep: bool = False,
    stagnation_streak: int = 0,
) -> list[dict[str, Any]]:
    """Call ``_build_stage_b_prompt_sections`` with minimal valid args."""
    return _build_stage_b_prompt_sections(
        evidence=evidence,
        snapshot=None,
        snapshot_prompt_text="",
        stage_a_context="",
        stage_a_evidence_index="",
        include_stage_a_evidence_index=False,
        compacted_context="",
        include_snapshot=False,
        include_pytest_output=False,
        include_stage_a_context=False,
        include_project_files=False,
        retry_packet="",
        retry_history="",
        retry_prompt_mode="standard",
        attempt=2,
        known_evidence_refs=None,
        new_evidence_ids=None,
        enable_tep=enable_tep,
        stagnation_streak=stagnation_streak,
    )


def _sections_by_key(
    sections: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    """Filter *sections* to those matching *key*."""
    return [s for s in sections if s.get("key") == key]


# ---------------------------------------------------------------------------
# TestTepJaccardSimilarity
# ---------------------------------------------------------------------------


class TestTepJaccardSimilarity:
    """Unit tests for ``_tep_jaccard_similarity``."""

    def test_identical_texts(self) -> None:
        assert _tep_jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different_texts(self) -> None:
        assert _tep_jaccard_similarity("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # shared: {"the", "brown"} = 2; union: {"the","quick","brown","fox","slow","dog"} = 6
        result = _tep_jaccard_similarity("the quick brown fox", "the slow brown dog")
        assert result == pytest.approx(2.0 / 6.0)

    def test_both_empty(self) -> None:
        assert _tep_jaccard_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self) -> None:
        assert _tep_jaccard_similarity("", "hello") == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        assert _tep_jaccard_similarity("Hello World", "hello world") == pytest.approx(1.0)

    def test_whitespace_only(self) -> None:
        # str.split() on whitespace-only returns [] → empty sets → both empty → 1.0
        assert _tep_jaccard_similarity("   ", "   ") == pytest.approx(1.0)

    def test_single_word_match(self) -> None:
        assert _tep_jaccard_similarity("hello", "hello") == pytest.approx(1.0)

    def test_subset(self) -> None:
        # shared: {"a","b","c"} = 3; union: {"a","b","c","d","e"} = 5
        result = _tep_jaccard_similarity("a b c", "a b c d e")
        assert result == pytest.approx(3.0 / 5.0)


# ---------------------------------------------------------------------------
# TestBuildFullTepNodePackets
# ---------------------------------------------------------------------------


class TestBuildFullTepNodePackets:
    """Unit tests for invoked-node filtering in full-mode TEP packet assembly."""

    @pytest.mark.parametrize(
        (
            "stage_a_invoked",
            "context_compaction_invoked",
            "retry_plan_invoked",
            "expected_node_ids",
        ),
        [
            (False, False, False, ["patch_proposer"]),
            (False, True, False, ["context_compaction", "patch_proposer"]),
            (False, False, True, ["retry_planner", "patch_proposer"]),
            (True, False, False, ["stage_a_request", "patch_proposer"]),
        ],
    )
    def test_only_emits_invoked_nodes(
        self,
        *,
        stage_a_invoked: bool,
        context_compaction_invoked: bool,
        retry_plan_invoked: bool,
        expected_node_ids: list[str],
    ) -> None:
        packets = _build_full_tep_node_packets(
            note="tests_still_failing",
            success=False,
            stage_a={
                "invoked": stage_a_invoked,
                "input_summary": "stage a prompt",
                "request_payload": {"need_snapshot": True},
                "text": "bundle text",
                "helper_decision_reason": "llm_requested_helper",
                "requests_count": 2,
                "request_success": True,
                "materialized_context": True,
            },
            context_compaction={
                "requested": True,
                "invoked": context_compaction_invoked,
                "success": True,
                "trigger": "auto_overflow",
                "ratio": 0.4,
                "input_summary": "compact prompt",
                "output_summary": "compacted context",
            },
            retry_plan={
                "invoked": retry_plan_invoked,
                "succeeded": True,
                "injected": True,
                "error": "",
                "input_summary": "retry prompt",
                "output_summary": "retry plan",
            },
            patch={
                "input_summary": "patch prompt",
                "output_summary": "diff --git a/buggy.py b/buggy.py",
                "patch_verifier_score": 0.7,
                "patch_verifier_passed": True,
                "syntax_preflight_cycles_used": 1,
                "postpatch_signature_repeat": False,
            },
        )

        assert [packet["node_id"] for packet in packets] == expected_node_ids

    def test_preserves_packet_payload_fields_for_invoked_nodes(self) -> None:
        packets = _build_full_tep_node_packets(
            note="tests_still_failing",
            success=True,
            stage_a={
                "invoked": True,
                "input_summary": "stage a prompt",
                "request_payload": {"need_snapshot": True},
                "text": "bundle text",
                "helper_decision_reason": "llm_requested_helper",
                "requests_count": 2,
                "request_success": True,
                "materialized_context": True,
            },
            context_compaction={
                "requested": True,
                "invoked": True,
                "success": True,
                "trigger": "auto_overflow",
                "ratio": 0.4,
                "input_summary": "compact prompt",
                "output_summary": "compacted context",
            },
            retry_plan={
                "invoked": True,
                "succeeded": False,
                "injected": False,
                "error": "timeout",
                "input_summary": "retry prompt",
                "output_summary": "timeout",
            },
            patch={
                "input_summary": "patch prompt",
                "output_summary": "diff --git a/buggy.py b/buggy.py",
                "patch_verifier_score": 0.7,
                "patch_verifier_passed": True,
                "syntax_preflight_cycles_used": 1,
                "postpatch_signature_repeat": False,
            },
        )

        assert [packet["node_id"] for packet in packets] == [
            "stage_a_request",
            "context_compaction",
            "retry_planner",
            "patch_proposer",
        ]
        assert packets[0]["output_summary"] == '{"need_snapshot": true}\nbundle text'
        assert json.loads(packets[1]["objective_packet"]) == {
            "attempt_note": "tests_still_failing",
            "success": True,
            "compaction_requested": True,
            "compaction_invoked": True,
            "compaction_success": True,
            "compaction_trigger": "auto_overflow",
            "compaction_ratio": 0.4,
        }
        assert json.loads(packets[2]["objective_packet"]) == {
            "attempt_note": "tests_still_failing",
            "success": True,
            "plan_invoked": True,
            "plan_succeeded": False,
            "plan_injected": False,
            "plan_error": "timeout",
        }
        assert json.loads(packets[3]["objective_packet"]) == {
            "attempt_note": "tests_still_failing",
            "success": True,
            "patch_verifier_score": 0.7,
            "patch_verifier_passed": True,
            "syntax_preflight_cycles_used": 1,
            "postpatch_signature_repeat": False,
        }


# ---------------------------------------------------------------------------
# TestTepNudgeEscalation
# ---------------------------------------------------------------------------


class TestTepNudgeEscalation:
    """TEP nudge escalation in ``_build_stage_b_prompt_sections``."""

    def test_no_loop_breaker_without_signature_repeat(self) -> None:
        """No loop_breaker section when failure_signature_repeat is False."""
        evidence = _base_evidence(failure_signature_repeat=False)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=3,
        )
        assert _sections_by_key(sections, "loop_breaker") == []

    def test_static_loop_breaker_when_tep_disabled(self) -> None:
        """Static loop_breaker when TEP is disabled."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=False,
            stagnation_streak=0,
        )
        lb = _sections_by_key(sections, "loop_breaker")
        assert len(lb) == 1

    def test_static_loop_breaker_when_tep_enabled_streak_zero(self) -> None:
        """Falls through to static loop_breaker when stagnation_streak=0."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=0,
        )
        lb = _sections_by_key(sections, "loop_breaker")
        assert len(lb) == 1

    def test_tep_streak_1_mild_nudge(self) -> None:
        """Streak 1: mild nudge, no diagnosis_refinement."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=1,
        )
        lb = _sections_by_key(sections, "loop_breaker")
        assert len(lb) == 1
        assert "Loop breaker" in lb[0]["text"]
        assert _sections_by_key(sections, "diagnosis_refinement") == []

    def test_tep_streak_2_warning_with_diagnosis(self) -> None:
        """Streak 2: stagnation warning, diagnosis_refinement."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=2,
        )
        lb = _sections_by_key(sections, "loop_breaker")
        assert len(lb) == 1
        assert "Stagnation warning" in lb[0]["text"]
        dr = _sections_by_key(sections, "diagnosis_refinement")
        assert len(dr) == 1

    def test_tep_streak_3_critical_with_diagnosis(self) -> None:
        """Streak 3: critical stagnation, diagnosis_refinement."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=3,
        )
        lb = _sections_by_key(sections, "loop_breaker")
        assert len(lb) == 1
        assert "CRITICAL STAGNATION" in lb[0]["text"]
        dr = _sections_by_key(sections, "diagnosis_refinement")
        assert len(dr) == 1

    def test_tep_streak_5_priority_floors_at_zero(self) -> None:
        """Priority cannot go negative: max(0, 3-5) == 0."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=5,
        )
        lb = _sections_by_key(sections, "loop_breaker")
        assert len(lb) == 1

    def test_diagnosis_refinement_not_present_at_streak_1(self) -> None:
        """Only loop_breaker, no diagnosis_refinement at streak 1."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=1,
        )
        assert len(_sections_by_key(sections, "loop_breaker")) == 1
        assert _sections_by_key(sections, "diagnosis_refinement") == []

    def test_diagnosis_refinement_present_at_streak_2(self) -> None:
        """Both loop_breaker and diagnosis_refinement at streak 2."""
        evidence = _base_evidence(failure_signature_repeat=True)
        sections = _build_sections(
            evidence=evidence,
            enable_tep=True,
            stagnation_streak=2,
        )
        lb = _sections_by_key(sections, "loop_breaker")
        dr = _sections_by_key(sections, "diagnosis_refinement")
        assert len(lb) == 1
        assert len(dr) == 1


# ---------------------------------------------------------------------------
# TestTepRetryPlanResult
# ---------------------------------------------------------------------------


def _dummy_budget_result() -> PromptBudgetResult:
    """Minimal ``PromptBudgetResult`` for constructing ``RetryPlanResult``."""
    return PromptBudgetResult(
        prompt="stub",
        used_chars=0,
        overflow_chars=0,
        dropped_sections=[],
    )


class TestTepRetryPlanResult:
    """TEP-related fields on ``RetryPlanResult``."""

    def test_default_tep_fields(self) -> None:
        result = RetryPlanResult(
            prompt="p",
            prompt_budget_result=_dummy_budget_result(),
            include_stage_a_evidence_index=False,
            stage_a_index_reason="none",
        )
        assert result.tep_equilibrium_iterations == 0
        assert result.tep_equilibrium_converged is False

    def test_tep_fields_populated(self) -> None:
        result = RetryPlanResult(
            prompt="p",
            prompt_budget_result=_dummy_budget_result(),
            include_stage_a_evidence_index=False,
            stage_a_index_reason="none",
            tep_equilibrium_iterations=3,
            tep_equilibrium_converged=True,
        )
        assert result.tep_equilibrium_iterations == 3
        assert result.tep_equilibrium_converged is True


class TestAuxiliaryRewardAlignment:
    def test_skip_helper_is_misaligned_when_pregate_wants_more_context(self) -> None:
        online_kwargs: dict[str, Any] = {}
        budget_kwargs: dict[str, Any] = {}
        patch_meta = {"adaptive_progress_component_pytest_fail_reduction": 0.25}

        _update_aux_signals_from_attempt(
            online_kwargs=online_kwargs,
            budget_kwargs=budget_kwargs,
            patch_meta=patch_meta,
            new_evidence_ids=["e1"],
            reused_evidence_ids=["e0"],
            last_context_compaction_ratio=0.4,
            pregate_should_skip=False,
            helper_action_executed="skip_helper",
        )

        assert online_kwargs["aux_pytest_fail_reduction"] == pytest.approx(0.25)
        assert online_kwargs["aux_evidence_novelty_ratio"] == pytest.approx(0.5)
        assert online_kwargs["aux_compaction_ratio"] == pytest.approx(0.4)
        assert online_kwargs["aux_informativeness_alignment"] == pytest.approx(0.0)
        assert budget_kwargs["aux_pytest_fail_reduction"] == pytest.approx(0.25)
        assert budget_kwargs["aux_evidence_novelty_ratio"] == pytest.approx(0.5)


class TestFullTepState:
    def test_extract_mutable_stage_b_blocks_locks_to_instruction_surface(self) -> None:
        sections = _build_stage_b_prompt_sections(
            evidence={
                "condition": "layered_adaptive_llm",
                "repro": "python -m pytest -q",
                "stdout": "traceback",
                "stderr": "",
                "failure_signature": "AssertionError:sig",
                "previous_failure_signature": "AssertionError:prev",
                "failure_signature_repeat": True,
                "file_index": "buggy.py",
                "files": "buggy.py",
            },
            snapshot={
                "exception": {"type": "IndexError", "message": "out of range"},
                "frames": [{"file_rel": "buggy.py", "line": 4, "function": "f"}],
            },
            snapshot_prompt_text="",
            stage_a_context="requested context",
            stage_a_evidence_index="[e1] buggy.py:4",
            include_stage_a_evidence_index=True,
            compacted_context="compacted",
            include_snapshot=True,
            include_pytest_output=True,
            include_stage_a_context=True,
            include_project_files=True,
            retry_packet="retry packet",
            retry_history="retry history",
            retry_prompt_mode="history",
            attempt=2,
            known_evidence_refs=["e0"],
            new_evidence_ids=["e1"],
            snapshot_lite_context="Frame 0: buggy.py:4",
            prompt_retry_reflection=True,
            prompt_adaptive_stage_b_cot=True,
            prompt_repair_method_scaffold=True,
            enable_tep=True,
            stagnation_streak=2,
        )

        extracted = extract_mutable_stage_b_blocks(sections)

        assert set(extracted) == set(TEP_STAGE_B_MUTABLE_KEYS)
        assert "repro" not in extracted
        assert "pytest_output" not in extracted
        assert "stage_a_context" not in extracted
        assert "project_files" not in extracted

    def test_tep_state_round_trip_preserves_prompt_surfaces(self, tmp_path) -> None:
        controller = AdaptiveTepController(
            state_path=str(tmp_path / "tep_state.json"),
            beta_init=1.0,
            beta_decay=0.9,
            validation_buffer_size=4,
        )
        controller.node("patch_proposer").actor_blocks["instructions"] = (
            "Rewrite the fix carefully."
        )
        controller.node("patch_proposer").actor_blocks["closing"] = "Return the smallest patch."
        controller.node("patch_proposer").append_validation_entry(
            case_id="case-1",
            score=7.5,
            note="seed",
            max_size=4,
        )
        controller.save()

        restored = load_tep_state(state_path=str(tmp_path / "tep_state.json"))

        assert restored.node("patch_proposer").actor_blocks["instructions"] == (
            "Rewrite the fix carefully."
        )
        assert restored.node("patch_proposer").actor_blocks["closing"] == (
            "Return the smallest patch."
        )
        assert restored.node("patch_proposer").validation_buffer[0].case_id == "case-1"

    def test_controller_primes_stage_b_defaults_and_renders_patch_prompt_template(
        self, tmp_path
    ) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep_state.json"))
        sections = [
            {"key": "instructions", "text": "Fix the failure."},
            {"key": "closing", "text": "Return only the patch."},
            {"key": "repro", "text": "python -m pytest -q"},
        ]

        controller.prime_stage_b_defaults(planning_mode=False, sections=sections)

        assert controller.patch_actor_overrides()["instructions"] == "Fix the failure."
        assert controller.patch_actor_overrides()["closing"] == "Return only the patch."
        assert (
            controller.patch_system_prompt(response_mode="full_file", target_rel="buggy.py")
            == "You fix failing tests by returning the complete repaired contents of buggy.py. "
            "Return ONLY the full contents of buggy.py. "
            "No prose. No markdown fences. Do not return a diff."
        )


class TestFullTepHeuristics:
    def test_equilibrium_stops_on_low_variance_window(self) -> None:
        converged, reason = resolve_equilibrium_status(
            scores=[7.0, 7.1, 7.0],
            improvement_kinds=["substantive", "substantive", "substantive"],
            max_iterations=20,
            window=3,
            variance_threshold=0.5,
        )

        assert converged is True
        assert reason == "score_variance"

    def test_acceptance_gate_uses_validation_floor(self) -> None:
        accepted = should_accept_candidate_update(
            free_score=6.5,
            nudged_score=6.8,
            validation_buffer=[
                TepNodeValidationEntry(case_id="c1", score=6.7),
                TepNodeValidationEntry(case_id="c2", score=6.6),
            ],
        )
        rejected = should_accept_candidate_update(
            free_score=6.5,
            nudged_score=6.4,
            validation_buffer=[
                TepNodeValidationEntry(case_id="c1", score=6.7),
            ],
        )

        assert accepted is True
        assert rejected is False

    def test_temperature_and_beta_updates_follow_accept_reject_rule(self) -> None:
        assert next_temperature(current=0.6, accepted=True) == pytest.approx(0.57)
        assert next_temperature(current=0.6, accepted=False) == pytest.approx(0.63)
        assert next_beta(current=1.0, beta_decay=0.9, accepted=True) == pytest.approx(0.9)
        assert next_beta(current=1.0, beta_decay=0.9, accepted=False) == pytest.approx(1.0)

    def test_post_attempt_learning_updates_prompt_state_when_candidate_is_accepted(
        self,
        tmp_path,
    ) -> None:
        class _FakeJsonResult:
            def __init__(self, payload: dict[str, Any]) -> None:
                self.payload = payload

        class _FakePatcher:
            # POL-17: calls are critique(current) -> update -> critique(candidate); this
            # fake used to rely on a free/nudged/update call order.
            def generate_json(self, **kwargs) -> _FakeJsonResult:  # type: ignore[no-untyped-def]
                operation = str(kwargs["operation_name"])
                if operation.endswith("_critic_current"):
                    return _FakeJsonResult(
                        {
                            "overall_score_10": 6.0,
                            "improvement_kind": "substantive",
                            "actionable_feedback": ["tighten diagnosis"],
                        }
                    )
                if operation.endswith("_critic_candidate"):
                    assert "Use the accepted updated instructions." in kwargs["prompt"]
                    return _FakeJsonResult(
                        {
                            "overall_score_10": 7.0,
                            "improvement_kind": "substantive",
                            "actionable_feedback": ["keep stronger diagnosis"],
                        }
                    )
                return _FakeJsonResult(
                    {
                        "system_prompts": {"edits": "Return only JSON edits."},
                        "blocks": {"instructions": "Use the accepted updated instructions."},
                    }
                )

        controller = AdaptiveTepController(state_path=str(tmp_path / "tep_state.json"))
        controller.node("patch_proposer").actor_blocks["instructions"] = "Old instructions."

        update_meta = run_post_attempt_learning(
            controller=controller,
            patcher=_FakePatcher(),
            context_dir=tmp_path,
            cwd=tmp_path,
            case_id="case-1",
            node_packets=[
                {
                    "node_id": "patch_proposer",
                    "input_summary": "prompt",
                    "output_summary": "diff --git a/buggy.py b/buggy.py",
                    "objective_packet": '{"success": false, "attempt_note": "tests_still_failing"}',
                    "note": "tests_still_failing",
                }
            ],
        )

        assert update_meta["enabled"] is True
        assert update_meta["node_updates"]["patch_proposer"]["accepted"] is True
        assert controller.node("patch_proposer").actor_blocks["instructions"] == (
            "Use the accepted updated instructions."
        )


# ---------------------------------------------------------------------------
# Phase 2 - new coverage tests for uncovered adaptive_tep.py paths
# ---------------------------------------------------------------------------


class TestDefaultSystemPromptsForNode:
    """_default_system_prompts_for_node returns {} for unknown node ids."""

    def test_known_nodes_return_non_empty(self) -> None:
        for node_id in (
            TEP_NODE_STAGE_A_REQUEST,
            TEP_NODE_CONTEXT_COMPACTION,
            TEP_NODE_RETRY_PLANNER,
            TEP_NODE_PATCH_PROPOSER,
        ):
            result = _default_system_prompts_for_node(node_id)
            assert isinstance(result, dict)
            assert len(result) > 0

    def test_unknown_node_returns_empty_dict(self) -> None:
        result = _default_system_prompts_for_node("totally_unknown_node_xyz")
        assert result == {}


class TestTepCriticResultFromPayload:
    """TepCriticResult.from_payload normalises unknown improvement_kind → 'stable'."""

    def test_unknown_improvement_kind_normalised_to_stable(self) -> None:
        result = TepCriticResult.from_payload(
            {
                "overall_score_10": 5.0,
                "improvement_kind": "nonsense_kind",
                "actionable_feedback": [],
            }
        )
        assert result.improvement_kind == "stable"

    def test_none_payload_is_unparsed(self) -> None:
        """POL-17: a missing payload used to read as a 'stable' critique, i.e. convergence."""
        result = TepCriticResult.from_payload(None)
        assert result.overall_score_10 == pytest.approx(0.0)
        assert result.critic_parsed is False
        assert result.improvement_kind == "unparsed"
        assert result.actionable_feedback == []

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (7, 7.0),
            ("7.5", 7.5),
            ("8/10", 8.0),
            ("seven", None),
            (float("nan"), None),
            (True, None),
        ],
    )
    def test_scores_are_parsed_defensively(self, raw: object, expected: float | None) -> None:
        result = TepCriticResult.from_payload(
            {
                "overall_score_10": raw,
                "improvement_kind": "substantive",
                "task_dependent_scores": {"a": "x", "b": "6"},
            }
        )
        assert result.critic_parsed is (expected is not None)
        if expected is not None:
            assert result.overall_score_10 == pytest.approx(expected)
        assert result.task_dependent_scores == {"b": 6.0}

    def test_known_improvement_kind_preserved(self) -> None:
        for kind in ("substantive", "stylistic", "stable", "degrading"):
            result = TepCriticResult.from_payload(
                {"overall_score_10": 7.0, "improvement_kind": kind, "actionable_feedback": []}
            )
            assert result.improvement_kind == kind


class TestTepNodeStateAppendValidationEntry:
    """append_validation_entry trims the buffer when it exceeds max_size."""

    def test_buffer_trimmed_when_over_max_size(self) -> None:
        node = TepNodeState(node_id="patch_proposer")
        for i in range(5):
            node.append_validation_entry(case_id=f"c{i}", score=float(i), note="", max_size=3)
        assert len(node.validation_buffer) == 3
        # last 3 entries: c2, c3, c4
        assert node.validation_buffer[0].case_id == "c2"
        assert node.validation_buffer[-1].case_id == "c4"

    def test_zero_max_size_never_trims(self) -> None:
        node = TepNodeState(node_id="patch_proposer")
        for i in range(10):
            node.append_validation_entry(case_id=f"c{i}", score=float(i), note="", max_size=0)
        assert len(node.validation_buffer) == 10


class TestLoadTepStateFallbacks:
    """load_tep_state falls back to defaults on OSError or bad JSON."""

    def test_fallback_when_path_does_not_exist(self) -> None:
        state = load_tep_state(state_path="/nonexistent/path/tep_state.json")
        # Should return a default state with all 4 nodes populated
        assert set(state.nodes.keys()) >= {
            TEP_NODE_STAGE_A_REQUEST,
            TEP_NODE_CONTEXT_COMPACTION,
            TEP_NODE_RETRY_PLANNER,
            TEP_NODE_PATCH_PROPOSER,
        }

    def test_fallback_when_file_contains_bad_json(self, tmp_path) -> None:
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json{{", encoding="utf-8")
        state = load_tep_state(state_path=str(bad_path))
        assert state.beta == pytest.approx(1.0)

    def test_fixes_nonpositive_validation_buffer_size(self, tmp_path) -> None:
        """When the persisted validation_buffer_size is -1 (negative), load_tep_state
        replaces it with the provided validation_buffer_size argument.
        from_dict uses ``int(value or 4)`` so 0 becomes 4 before the fix-up fires;
        use -1 to get past from_dict and trigger the fix-up branch."""
        import json as _json

        state_path = tmp_path / "tep.json"
        raw = {
            "schema_version": 1,
            "beta": 0.8,
            "beta_decay": 0.9,
            "validation_buffer_size": -1,  # negative → fix-up fires
            "nodes": {},
        }
        state_path.write_text(_json.dumps(raw), encoding="utf-8")
        state = load_tep_state(state_path=str(state_path), validation_buffer_size=6)
        assert state.validation_buffer_size == 6


class TestResolveEquilibriumStatus:
    """resolve_equilibrium_status edge cases."""

    def test_returns_not_run_for_empty_scores(self) -> None:
        converged, reason = resolve_equilibrium_status(
            scores=[],
            improvement_kinds=[],
            max_iterations=10,
            window=3,
            variance_threshold=0.5,
        )
        assert converged is False
        assert reason == "not_run"

    def test_returns_max_iterations_when_count_reached(self) -> None:
        converged, reason = resolve_equilibrium_status(
            scores=[5.0, 6.0, 7.0],
            improvement_kinds=["substantive", "substantive", "substantive"],
            max_iterations=3,
            window=5,
            variance_threshold=0.5,
        )
        assert converged is True
        assert reason == "max_iterations"

    def test_stops_on_stylistic_improvement_kind(self) -> None:
        converged, reason = resolve_equilibrium_status(
            scores=[5.0, 6.0],
            improvement_kinds=["substantive", "stylistic"],
            max_iterations=20,
            window=5,
            variance_threshold=0.5,
        )
        assert converged is True
        assert reason == "stylistic"

    def test_stops_on_stable_improvement_kind(self) -> None:
        converged, reason = resolve_equilibrium_status(
            scores=[5.0, 6.0],
            improvement_kinds=["substantive", "stable"],
            max_iterations=20,
            window=5,
            variance_threshold=0.5,
        )
        assert converged is True
        assert reason == "stable"


class TestApplyCandidateUpdateRejected:
    """apply_candidate_update increments rejected_updates when nudged < free."""

    def test_rejected_update_increments_rejected_count(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        node = controller.node(TEP_NODE_PATCH_PROPOSER)
        node.actor_blocks["instructions"] = "Original instructions."

        result = controller.apply_candidate_update(
            node_id=TEP_NODE_PATCH_PROPOSER,
            case_id="case-x",
            free_score=8.0,
            nudged_score=5.0,  # lower → rejected
            update_payload={"blocks": {"instructions": "Updated instructions."}},
            note="test",
        )

        assert result["accepted"] is False
        assert node.rejected_updates == 1
        assert node.accepted_updates == 0
        # actor_blocks must not have changed
        assert node.actor_blocks["instructions"] == "Original instructions."


class TestRunPostAttemptLearningNoPatcher:
    """run_post_attempt_learning returns disabled when patcher lacks generate_json."""

    def test_returns_disabled_when_generate_json_unavailable(self, tmp_path) -> None:
        class _NoPatcher:
            pass

        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        result = run_post_attempt_learning(
            controller=controller,
            patcher=_NoPatcher(),
            context_dir=tmp_path,
            cwd=tmp_path,
            case_id="case-1",
            node_packets=[],
        )
        assert result["enabled"] is False
        assert result["reason"] == "generate_json_unavailable"


class TestAdaptiveTepControllerAccessors:
    """Accessor methods on AdaptiveTepController return non-empty strings."""

    def test_stage_a_system_prompt_standard(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        prompt = controller.stage_a_system_prompt(strict_json=False)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_stage_a_system_prompt_strict_json(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        prompt = controller.stage_a_system_prompt(strict_json=True)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_planner_system_prompt(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        prompt = controller.planner_system_prompt()
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_compaction_system_prompt_substitutes_target_chars(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        prompt = controller.compaction_system_prompt(target_chars=500)
        assert "500" in prompt

    def test_planner_actor_overrides_returns_dict(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        overrides = controller.planner_actor_overrides()
        assert isinstance(overrides, dict)

    def test_stage_a_actor_overrides_returns_dict(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        overrides = controller.stage_a_actor_overrides()
        assert isinstance(overrides, dict)

    def test_compaction_actor_overrides_returns_dict(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        overrides = controller.compaction_actor_overrides()
        assert isinstance(overrides, dict)


class TestTepNodeTraceFinalScore:
    """TepNodeTrace.final_score returns 0.0 for empty scores list."""

    def test_final_score_empty_scores(self) -> None:
        from evals.adaptive_tep import TepNodeTrace

        trace = TepNodeTrace(node_id="patch_proposer")
        assert trace.final_score == pytest.approx(0.0)

    def test_final_score_with_scores(self) -> None:
        from evals.adaptive_tep import TepNodeTrace

        trace = TepNodeTrace(node_id="patch_proposer", scores=[5.0, 7.0, 9.0])
        assert trace.final_score == pytest.approx(9.0)


class TestApplyUpdatePayloadEdgeCases:
    """apply_update_payload skips blocks/system_prompts with empty string values."""

    def test_empty_value_in_blocks_skipped(self) -> None:
        node = TepNodeState(node_id="patch_proposer")
        apply_update_payload(
            node_state=node,
            payload={"blocks": {"instructions": "", "closing": "Keep it short."}},
        )
        # empty string → skipped; non-empty → applied
        assert "instructions" not in node.actor_blocks
        assert node.actor_blocks["closing"] == "Keep it short."

    def test_empty_value_in_system_prompts_skipped(self) -> None:
        node = TepNodeState(node_id="patch_proposer")
        apply_update_payload(
            node_state=node,
            payload={"system_prompts": {"edits": "", "diff": "Return only diff."}},
        )
        assert "edits" not in node.system_prompts
        assert node.system_prompts["diff"] == "Return only diff."

    def test_none_payload_is_no_op(self) -> None:
        node = TepNodeState(node_id="patch_proposer")
        node.actor_blocks["existing"] = "original"
        apply_update_payload(node_state=node, payload=None)
        assert node.actor_blocks["existing"] == "original"


class TestAdaptiveTepControllerRecordFreePhase:
    """record_free_phase populates latest_traces and appends to validation_buffer."""

    def test_record_free_phase_populates_trace(self, tmp_path) -> None:
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        critiques = [
            TepCriticResult(overall_score_10=score, improvement_kind="substantive")
            for score in (6.0, 7.0, 7.5)
        ]

        trace = controller.record_free_phase(
            node_id=TEP_NODE_PATCH_PROPOSER,
            case_id="case-1",
            input_summary="input",
            output_summary="output",
            critiques=critiques,
            feedback_rounds=[["feedback 1"], ["feedback 2"]],
        )

        assert trace.node_id == TEP_NODE_PATCH_PROPOSER
        assert trace.iterations == 3
        assert trace.final_score == pytest.approx(7.5)
        assert controller.latest_traces[TEP_NODE_PATCH_PROPOSER] is trace
        # validation_buffer should have one entry appended
        node = controller.node(TEP_NODE_PATCH_PROPOSER)
        assert len(node.validation_buffer) == 1
        assert node.validation_buffer[0].case_id == "case-1"

    def test_unparsed_window_does_not_converge_or_score(self, tmp_path) -> None:
        """POL-17: three malformed critic replies are not an equilibrium (codex 3-1#5)."""
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        critiques = [TepCriticResult.from_payload({"summary": "nonsense"}) for _ in range(3)]
        assert [critique.critic_parsed for critique in critiques] == [False, False, False]

        trace = controller.record_free_phase(
            node_id=TEP_NODE_PATCH_PROPOSER,
            case_id="case-1",
            input_summary="input",
            output_summary="output",
            critiques=critiques,
            feedback_rounds=[[], []],
        )

        assert trace.converged is False
        assert trace.convergence_reason == "critic_unparsed"
        assert trace.scores == []
        assert controller.node(TEP_NODE_PATCH_PROPOSER).validation_buffer == []

    def test_unparsed_critiques_are_dropped_from_a_mixed_window(self, tmp_path) -> None:
        """Only parsed critiques feed convergence and the validation buffer (POL-17)."""
        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        critiques = [
            TepCriticResult(overall_score_10=7.0, improvement_kind="substantive"),
            TepCriticResult.from_payload(None),
            TepCriticResult(overall_score_10=7.1, improvement_kind="substantive"),
        ]

        trace = controller.record_free_phase(
            node_id=TEP_NODE_PATCH_PROPOSER,
            case_id="case-1",
            input_summary="input",
            output_summary="output",
            critiques=critiques,
            feedback_rounds=[["feedback 1"], ["feedback 2"]],
        )

        assert trace.iterations == 3
        assert trace.scores == pytest.approx([7.0, 7.1])
        assert trace.improvement_kinds == ["substantive", "substantive"]
        node = controller.node(TEP_NODE_PATCH_PROPOSER)
        assert [entry.score for entry in node.validation_buffer] == pytest.approx([7.1])


# ---------------------------------------------------------------------------
# Additional coverage for 3 remaining uncovered lines
# ---------------------------------------------------------------------------


class TestLoadTepStateInitStatePath:
    """load_tep_state appends init_state_path to candidates (line 402)."""

    def test_init_state_path_used_when_state_path_absent(self, tmp_path) -> None:
        """Non-empty init_state_path → appended to candidate_paths (line 402)."""
        import json as _json

        init_path = tmp_path / "init.json"
        state = {
            "nodes": {},
            "beta": 2.5,
            "beta_decay": 0.8,
            "validation_buffer_size": 4,
            "saved_at_utc": "2024-01-01T00:00:00Z",
        }
        init_path.write_text(_json.dumps(state), encoding="utf-8")

        # state_path is empty, so only init_state_path is checked
        loaded = load_tep_state(
            state_path="",
            init_state_path=str(init_path),
        )
        assert loaded.beta == pytest.approx(2.5)
        assert loaded.beta_decay == pytest.approx(0.8)


class TestResolveEquilibriumStatusContinue:
    """resolve_equilibrium_status returns (False, 'continue') when not converged (line 464)."""

    def test_returns_continue_when_last_kind_is_substantive(self) -> None:
        """improvement_kinds present but latest is 'substantive' → (False, 'continue')."""
        converged, reason = resolve_equilibrium_status(
            scores=[5.0, 6.0, 7.0],
            improvement_kinds=["substantive", "substantive", "substantive"],
            max_iterations=100,
            window=10,
            variance_threshold=0.001,  # very tight → won't converge on variance
        )
        assert converged is False
        assert reason == "continue"


class TestRunPostAttemptLearningSkipsEmptyNodeId:
    """run_post_attempt_learning skips packets with empty node_id (line 775)."""

    def test_empty_node_id_packet_is_skipped(self, tmp_path) -> None:
        from pathlib import Path
        from types import SimpleNamespace

        from evals.adaptive_tep import AdaptiveTepController

        controller = AdaptiveTepController(state_path=str(tmp_path / "tep.json"))
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        context_dir = tmp_path / "ctx"
        context_dir.mkdir()

        critic_payload = {
            "improvement_kind": "stable",
            "overall_score_10": 7.5,
            "feedback": "looks good",
            "suggested_changes": [],
        }
        update_payload = {
            "blocks": {},
            "system_prompts": {},
        }

        call_count = {"n": 0}

        def _generate_json(**kw):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return SimpleNamespace(payload=critic_payload, meta={})
            return SimpleNamespace(payload=update_payload, meta={})

        fake_patcher = SimpleNamespace(generate_json=_generate_json)

        result = run_post_attempt_learning(
            controller=controller,
            patcher=fake_patcher,
            context_dir=context_dir,
            cwd=cwd,
            case_id="case-skip",
            node_packets=[
                {
                    "node_id": "",
                    "input_summary": "",
                    "output_summary": "",
                    "objective_packet": "",
                },  # empty node_id → skip
                {
                    "node_id": TEP_NODE_PATCH_PROPOSER,
                    "input_summary": "x",
                    "output_summary": "y",
                    "objective_packet": "z",
                },
            ],
        )
        # Should succeed (not raise) even though one packet had empty node_id
        assert isinstance(result, dict)
        assert result.get("enabled") is True


# ---------------------------------------------------------------------------
# POL-17: paired critic comparison, ties rejected, unparsed critics skip the update
# ---------------------------------------------------------------------------


class _ScriptedJsonResult:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self.payload = payload


class _ScriptedPatcher:
    def __init__(self, current: dict[str, Any] | None, candidate: dict[str, Any] | None) -> None:
        self.current = current
        self.candidate = candidate
        self.operations: list[str] = []

    def generate_json(self, **kwargs: Any) -> _ScriptedJsonResult:
        operation = str(kwargs["operation_name"])
        self.operations.append(operation)
        if operation.endswith("_critic_current"):
            return _ScriptedJsonResult(self.current)
        if operation.endswith("_critic_candidate"):
            return _ScriptedJsonResult(self.candidate)
        return _ScriptedJsonResult({"blocks": {"instructions": "New instructions."}})


def _learn(
    tmp_path: Path, patcher: _ScriptedPatcher
) -> tuple[dict[str, Any], AdaptiveTepController]:
    controller = AdaptiveTepController(state_path=str(tmp_path / "tep_state.json"))
    controller.node("patch_proposer").actor_blocks["instructions"] = "Old instructions."
    meta = run_post_attempt_learning(
        controller=controller,
        patcher=patcher,
        context_dir=tmp_path,
        cwd=tmp_path,
        case_id="case-1",
        node_packets=[{"node_id": "patch_proposer", "note": "tests_still_failing"}],
    )
    return meta["node_updates"]["patch_proposer"], controller


def _critique(score: float) -> dict[str, Any]:
    return {"overall_score_10": score, "improvement_kind": "substantive", "actionable_feedback": []}


def test_tep_rejects_a_candidate_that_only_ties(tmp_path: Path) -> None:
    update, controller = _learn(tmp_path, _ScriptedPatcher(_critique(6.0), _critique(6.0)))
    assert update["accepted"] is False
    assert controller.node("patch_proposer").actor_blocks["instructions"] == "Old instructions."


def test_tep_skips_the_update_when_the_current_critique_is_unparsed(tmp_path: Path) -> None:
    patcher = _ScriptedPatcher(None, _critique(9.0))
    update, controller = _learn(tmp_path, patcher)
    assert update == {"accepted": False, "reason": "critic_unparsed", "critic_parsed": False}
    assert patcher.operations == ["tep_patch_proposer_critic_current"]
    assert controller.node("patch_proposer").update_count == 0


def test_tep_rejects_when_the_candidate_critique_is_unparsed(tmp_path: Path) -> None:
    update, controller = _learn(tmp_path, _ScriptedPatcher(_critique(6.0), {"summary": "x"}))
    assert update["accepted"] is False
    assert update["reason"] == "critic_unparsed"
    assert controller.node("patch_proposer").actor_blocks["instructions"] == "Old instructions."


# ---------------------------------------------------------------------------
# ATT-17: TEP state-save failures are reported, not swallowed
# ---------------------------------------------------------------------------


def test_tep_save_returns_the_error_instead_of_raising(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file", encoding="utf-8")
    controller = AdaptiveTepController(state_path=str(blocker / "tep_state.json"))
    error = controller.save()
    assert error.startswith(("FileExistsError", "NotADirectoryError"))

    ok = AdaptiveTepController(state_path=str(tmp_path / "state" / "tep.json"))
    assert ok.save() == ""
    assert (tmp_path / "state" / "tep.json").exists()
    assert AdaptiveTepController(state_path="").save() == ""
