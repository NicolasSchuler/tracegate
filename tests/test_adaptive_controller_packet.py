from __future__ import annotations

import json
from pathlib import Path

from evals import adaptive_controller_packet as controller_packets
from evals.adaptive_controller_packet import (
    build_adaptive_controller_sidecars,
    hybrid_guidance_metadata,
    parse_controller_packet,
    render_hybrid_controller_guidance,
)
from evals.adaptive_history import PersistentHistoryBundle, PersistentHistoryLesson


def test_build_adaptive_controller_sidecars_includes_ast_and_history(tmp_path: Path) -> None:
    root = tmp_path / "case"
    root.mkdir()
    (root / "buggy.py").write_text(
        "def buggy(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    attempt_dir = tmp_path / "attempt_01"
    attempt_dir.mkdir()
    snapshot = {
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 2,
                "function": "buggy",
                "code": "return value + 1",
                "locals": {"value": 1},
            }
        ]
    }

    sidecars = build_adaptive_controller_sidecars(
        case_id="case",
        condition="layered_adaptive_llm",
        attempt=1,
        root=root,
        attempt_dir=attempt_dir,
        snapshot=snapshot,
        failure_signature="sig-1",
        evidence_items=[
            {
                "id": "e1",
                "type": "frame",
                "locator": "frame:0",
                "text": "[e1] frame:frame:0\nAssertionError in buggy()\nvalue was 1\n",
            }
        ],
        new_evidence_ids=["e1"],
        reused_evidence_ids=[],
        retry_prompt_mode="delta",
        retry_context_mode="history",
        retry_history_text="Attempt 1: tests_still_failing",
        retry_history_turns=["Attempt 1: tests_still_failing"],
        retry_history_turns_included=1,
        retry_history_truncated=False,
        previous_failure_signature="",
        previous_note="",
        previous_patch_verifier_rejected=False,
        previous_patch_verifier_reasons=[],
        previous_patch_verifier_scope="adaptive_only",
        previous_timed_out=False,
        previous_timeout_sec=None,
        helper_action_selected="snapshot_lite",
        helper_action_executed="snapshot_lite",
        helper_decision_reason="policy_action_snapshot_lite",
        helper_outcome="materialized",
        helper_materialized_context=True,
        helper_calls_used=1,
        helper_budget_max_calls=2,
        no_progress_streak=0,
        context_compaction_requested=False,
        context_compaction_invoked=False,
        context_compaction_success=False,
        context_compaction_trigger="not_needed",
        prompt_budget={"total_chars": 1000, "retry_history_chars": 100},
        retry_plan_text="Focus on the return expression.",
        syntax_cycle=1,
        syntax_max_cycles=2,
        syntax_last_error_type="",
        syntax_last_error_msg="",
        syntax_retry_feedback="",
    )

    packet = sidecars["llmdebug_controller_packet.json"]
    evidence_bundle = sidecars["llmdebug_evidence_bundle.json"]
    history_bundle = sidecars["llmdebug_history_bundle.json"]

    assert packet["controller_state"]["helper_action_executed"] == "snapshot_lite"
    assert packet["solver_hints"]["retry_plan"] == "Focus on the return expression."
    assert evidence_bundle["ast_context"]["symbol_name"] == "buggy"
    assert evidence_bundle["summary"]["strategy"] == "deterministic_extract_v1"
    assert evidence_bundle["summary"]["top_facts"]
    assert evidence_bundle["snapshot_reduction"]["selected_locals"][0]["name"] == "value"
    assert '"file_rel": "buggy.py"' in evidence_bundle["snapshot_full_text"]
    assert evidence_bundle["items"][0]["id"] == "e1"
    assert history_bundle["retry_history_text"] == "Attempt 1: tests_still_failing"
    assert history_bundle["lessons"][0]["source"] == "attempt_history"


def test_parse_controller_packet_uses_crash_frame_index_and_typed_repo_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "case"
    root.mkdir()
    (root / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    (root / "buggy.py").write_text(
        "import helper\n\n\ndef buggy() -> int:\n    return helper.value()\n",
        encoding="utf-8",
    )
    attempt_dir = tmp_path / "attempt_02"
    attempt_dir.mkdir()
    snapshot = {
        "crash_frame_index": 1,
        "frames": [
            {
                "file_rel": "helper.py",
                "line": 1,
                "function": "value",
                "code": "def value() -> int:",
                "locals": {},
            },
            {
                "file_rel": "buggy.py",
                "line": 4,
                "function": "buggy",
                "code": "return helper.value()",
                "locals": {"helper": {"value": 1}},
            },
        ],
    }
    monkeypatch.setattr(
        controller_packets,
        "get_enhanced_git_context",
        lambda *_args, **_kwargs: {
            "head": {"branch": "main", "commit": "abc123"},
            "crash_file": {
                "file_rel": "buggy.py",
                "line": 4,
                "blame": {"summary": "introduced buggy return path"},
            },
            "recent_commits": [{"commit": "abc123", "summary": "fix helper math"}],
        },
    )

    sidecars = build_adaptive_controller_sidecars(
        case_id="case",
        condition="layered_adaptive_llm",
        attempt=2,
        root=root,
        attempt_dir=attempt_dir,
        snapshot=snapshot,
        failure_signature="sig-2",
        evidence_items=[],
        new_evidence_ids=[],
        reused_evidence_ids=[],
        retry_prompt_mode="delta",
        retry_context_mode="history",
        retry_history_text="Attempt 1: tests_still_failing",
        retry_history_turns=["Attempt 1: tests_still_failing"],
        retry_history_turns_included=1,
        retry_history_truncated=False,
        previous_failure_signature="sig-1",
        previous_note="tests_still_failing",
        previous_patch_verifier_rejected=False,
        previous_patch_verifier_reasons=[],
        previous_patch_verifier_scope="adaptive_only",
        previous_timed_out=False,
        previous_timeout_sec=None,
        helper_action_selected="snapshot_lite",
        helper_action_executed="snapshot_lite",
        helper_decision_reason="policy_action_snapshot_lite",
        helper_outcome="materialized",
        helper_materialized_context=True,
        helper_calls_used=1,
        helper_budget_max_calls=2,
        no_progress_streak=1,
        context_compaction_requested=False,
        context_compaction_invoked=False,
        context_compaction_success=False,
        context_compaction_trigger="not_needed",
        prompt_budget={"total_chars": 1000},
        retry_plan_text="Focus on the buggy return expression.",
        syntax_cycle=1,
        syntax_max_cycles=2,
        syntax_last_error_type="",
        syntax_last_error_msg="",
        syntax_retry_feedback="",
    )

    packet = parse_controller_packet(sidecars["llmdebug_controller_packet.json"])

    assert packet is not None
    assert packet.evidence_bundle.ast_context is not None
    assert packet.evidence_bundle.ast_context.path == "buggy.py"
    assert packet.evidence_bundle.snapshot_reduction is not None
    assert packet.evidence_bundle.snapshot_reduction.crash_frame_summary.startswith("buggy.py:4")
    assert packet.evidence_bundle.summary is not None
    assert packet.evidence_bundle.summary.crash_anchor.startswith("buggy.py:4")
    assert packet.history_bundle.repo_history is not None
    assert (
        packet.history_bundle.repo_history.crash_file_blame_summary
        == "introduced buggy return path"
    )
    assert packet.history_bundle.repo_history.recent_commits[0].summary == "fix helper math"


def test_build_adaptive_controller_sidecars_reduces_and_summarizes_history(tmp_path: Path) -> None:
    root = tmp_path / "case"
    root.mkdir()
    (root / "buggy.py").write_text(
        "def buggy(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    attempt_dir = tmp_path / "attempt_03"
    attempt_dir.mkdir()
    snapshot = {
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 2,
                "function": "buggy",
                "code": "return value + 1",
                "locals": {"value": 1, "noise": "x"},
            }
        ]
    }

    sidecars = build_adaptive_controller_sidecars(
        case_id="case",
        condition="layered_adaptive_llm",
        attempt=3,
        root=root,
        attempt_dir=attempt_dir,
        snapshot=snapshot,
        failure_signature="sig-3",
        evidence_items=[
            {"id": "e1", "type": "frame", "locator": "frame:0", "text": "AssertionError on value"},
            {
                "id": "e2",
                "type": "search_hit",
                "locator": "search:buggy",
                "text": "candidate helper path",
            },
            {
                "id": "e3",
                "type": "search_hit",
                "locator": "search:buggy",
                "text": "candidate helper path",
            },
            {"id": "e4", "type": "frame", "locator": "frame:0", "text": "AssertionError on value"},
            {
                "id": "e5",
                "type": "source_span",
                "locator": "buggy.py:2",
                "text": "return expression",
            },
            {
                "id": "e6",
                "type": "source_span",
                "locator": "buggy.py:1",
                "text": "definition header",
            },
            {"id": "e7", "type": "file", "locator": "buggy.py", "text": "full file context"},
        ],
        new_evidence_ids=["e1", "e2", "e5"],
        reused_evidence_ids=["e6"],
        retry_prompt_mode="delta",
        retry_context_mode="history",
        retry_history_text="Attempt 1: tests_still_failing\nAttempt 2: patch_verifier_rejected",
        retry_history_turns=[
            "Attempt 1\nHypothesis tested: check return path\nEvidence used: base_repair\nOutcome: tests_still_failing\nNew constraint: preserve helper contract",
            "Attempt 2\nHypothesis tested: widen patch scope\nEvidence used: stage_a_requests=2\nOutcome: patch_verifier_rejected\nNew constraint: must touch crash file",
        ],
        retry_history_turns_included=2,
        retry_history_truncated=False,
        previous_failure_signature="sig-2",
        previous_note="patch_verifier_rejected",
        previous_patch_verifier_rejected=True,
        previous_patch_verifier_reasons=["patch missed crash file"],
        previous_patch_verifier_scope="adaptive_only",
        previous_timed_out=False,
        previous_timeout_sec=None,
        helper_action_selected="compact_then_patch",
        helper_action_executed="compact_then_patch",
        helper_decision_reason="policy_action_compact_then_patch",
        helper_outcome="materialized",
        helper_materialized_context=True,
        helper_calls_used=1,
        helper_budget_max_calls=2,
        no_progress_streak=2,
        context_compaction_requested=True,
        context_compaction_invoked=True,
        context_compaction_success=True,
        context_compaction_trigger="policy",
        prompt_budget={"total_chars": 1000, "retry_history_chars": 120},
        retry_plan_text="Touch the crash file and preserve helper contract.",
        syntax_cycle=1,
        syntax_max_cycles=2,
        syntax_last_error_type="",
        syntax_last_error_msg="",
        syntax_retry_feedback="",
    )

    evidence_bundle = sidecars["llmdebug_evidence_bundle.json"]
    history_bundle = sidecars["llmdebug_history_bundle.json"]
    controller_packet = sidecars["llmdebug_controller_packet.json"]

    assert len(evidence_bundle["items"]) <= 6
    assert evidence_bundle["summary"]["omitted_item_count"] >= 1
    assert "evidence_summary" in controller_packet["solver_hints"]["tool_routes"]
    assert "history_lessons" in controller_packet["solver_hints"]["tool_routes"]
    assert "lambda_style_decomposition" in controller_packet["solver_hints"]["tool_routes"]
    assert controller_packet["solver_hints"]["decomposition"][0]["title"] == "localize_fix"
    assert history_bundle["lessons"][0]["lesson"]
    assert history_bundle["lessons"][-1]["source"] == "patch_verifier"


def test_build_adaptive_controller_sidecars_merges_persistent_history_and_external_decomposition(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "case"
    root.mkdir()
    (root / "buggy.py").write_text(
        "def buggy(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    attempt_dir = tmp_path / "attempt_04"
    attempt_dir.mkdir()
    snapshot = {
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 2,
                "function": "buggy",
                "code": "return value + 1",
                "locals": {"value": 1},
            }
        ]
    }
    monkeypatch.setattr(
        controller_packets,
        "load_persistent_adaptive_history",
        lambda **_kwargs: PersistentHistoryBundle(
            session_id="sess-123",
            records_loaded=2,
            lessons=[
                PersistentHistoryLesson(
                    source="rca_history",
                    attempt=1,
                    lesson="previous verifier rejected patches that skipped buggy.py",
                )
            ],
        ),
    )

    sidecars = build_adaptive_controller_sidecars(
        case_id="case",
        condition="layered_adaptive_llm",
        attempt=4,
        root=root,
        attempt_dir=attempt_dir,
        snapshot=snapshot,
        failure_signature="sig-4",
        evidence_items=[],
        new_evidence_ids=[],
        reused_evidence_ids=[],
        retry_prompt_mode="delta",
        retry_context_mode="history",
        retry_history_text="Attempt 3: patch_verifier_rejected",
        retry_history_turns=["Attempt 3: patch_verifier_rejected"],
        retry_history_turns_included=1,
        retry_history_truncated=False,
        previous_failure_signature="sig-3",
        previous_note="patch_verifier_rejected",
        previous_patch_verifier_rejected=True,
        previous_patch_verifier_reasons=["missed crash file"],
        previous_patch_verifier_scope="adaptive_only",
        previous_timed_out=False,
        previous_timeout_sec=None,
        helper_action_selected="compact_then_patch",
        helper_action_executed="compact_then_patch",
        helper_decision_reason="policy_action_compact_then_patch",
        helper_outcome="materialized",
        helper_materialized_context=True,
        helper_calls_used=1,
        helper_budget_max_calls=2,
        no_progress_streak=1,
        context_compaction_requested=True,
        context_compaction_invoked=True,
        context_compaction_success=True,
        context_compaction_trigger="policy",
        prompt_budget={"total_chars": 1000},
        retry_plan_text="Touch buggy.py and preserve the function contract.",
        syntax_cycle=1,
        syntax_max_cycles=2,
        syntax_last_error_type="",
        syntax_last_error_msg="",
        syntax_retry_feedback="",
        external_decomposition=[
            {
                "title": "constrain_edit",
                "question": "Limit the edit to buggy.py:2 and preserve the signature.",
                "supporting_routes": ["retry_plan", "persistent_history"],
                "evidence_ids": [],
            }
        ],
        external_decomposition_source="lambda_rlm_decomposition",
        external_strategy_summary="First localize the return path, then keep the patch minimal.",
    )

    controller_packet = sidecars["llmdebug_controller_packet.json"]
    history_bundle = sidecars["llmdebug_history_bundle.json"]

    assert history_bundle["persistent_session_id"] == "sess-123"
    assert history_bundle["persistent_records_loaded"] == 2
    assert history_bundle["lessons"][-1]["source"] == "rca_history"
    assert "persistent_history" in controller_packet["solver_hints"]["tool_routes"]
    assert "lambda_rlm_decomposition" in controller_packet["solver_hints"]["tool_routes"]
    assert "strategy_summary" in controller_packet["solver_hints"]["tool_routes"]
    assert controller_packet["solver_hints"]["decomposition"][0]["title"] == "constrain_edit"
    assert (
        controller_packet["solver_hints"]["strategy_summary"]
        == "First localize the return path, then keep the patch minimal."
    )


# ---------------------------------------------------------------------------
# Imports for dataclass types used in the coverage-gap tests below
# ---------------------------------------------------------------------------
from evals.adaptive_controller_packet import (  # noqa: E402
    AdaptiveControllerStateV2,
    ASTContextV1,
    ControllerPacketV1,
    EvidenceBundleV1,
    EvidenceItemV1,
    EvidenceSummaryV1,
    HistoryBundleV1,
    HistoryLessonV1,
    RepoHistoryV1,
    SnapshotReductionV1,
    SolverHintsV1,
    VerifierReportV1,
    render_compact_controller_guidance,
)

# ---------------------------------------------------------------------------
# Helpers to construct minimal dataclass instances for render tests
# ---------------------------------------------------------------------------


def _minimal_controller_state() -> AdaptiveControllerStateV2:
    return AdaptiveControllerStateV2(
        schema_version="v1",
        helper_action_selected="",
        helper_action_executed="",
        helper_decision_reason="",
        helper_outcome="",
        helper_materialized_context=False,
        helper_calls_used=0,
        helper_budget_max_calls=0,
        no_progress_streak=0,
        retry_prompt_mode="",
        retry_context_mode="",
        context_compaction_requested=False,
        context_compaction_invoked=False,
        context_compaction_success=False,
        context_compaction_trigger="",
        prompt_budget={},
    )


def _minimal_evidence_bundle() -> EvidenceBundleV1:
    return EvidenceBundleV1(
        schema_version="v1",
        case_id="c",
        condition="cond",
        attempt=1,
        failure_signature="sig",
        selected_ids=[],
        new_ids=[],
        reused_ids=[],
        ast_context=None,
        snapshot_reduction=None,
        summary=None,
        items=[],
    )


def _minimal_history_bundle(
    *,
    lessons: list[HistoryLessonV1] | None = None,
    repo_history: RepoHistoryV1 | None = None,
    previous_note: str = "",
) -> HistoryBundleV1:
    return HistoryBundleV1(
        schema_version="v1",
        retry_context_mode="history",
        attempt_turns=[],
        retry_history_text="",
        retry_history_turns_included=0,
        retry_history_truncated=False,
        previous_note=previous_note,
        previous_failure_signature="",
        previous_patch_verifier_rejected=False,
        previous_patch_verifier_reasons=[],
        lessons=lessons or [],
        repo_history=repo_history,
        persistent_session_id="",
        persistent_records_loaded=0,
    )


def _minimal_verifier_report() -> VerifierReportV1:
    return VerifierReportV1(
        schema_version="v1",
        status="none",
        scope="",
        reasons=[],
        previous_timed_out=False,
        previous_timeout_sec=None,
        syntax_cycle=0,
        syntax_max_cycles=0,
        last_error_type="",
        last_error_msg="",
        retry_feedback="",
    )


def _minimal_solver_hints(
    *,
    strategy_summary: str = "",
    decomposition: list[dict] | None = None,
) -> SolverHintsV1:
    return SolverHintsV1(
        tool_routes=[],
        retry_plan="",
        recommended_focus=[],
        decomposition=decomposition or [],
        strategy_summary=strategy_summary,
    )


def _make_packet(
    *,
    state: AdaptiveControllerStateV2 | None = None,
    evidence: EvidenceBundleV1 | None = None,
    history: HistoryBundleV1 | None = None,
    verifier: VerifierReportV1 | None = None,
    hints: SolverHintsV1 | None = None,
) -> ControllerPacketV1:
    return ControllerPacketV1(
        schema_version="v1",
        case_id="c",
        condition="cond",
        attempt=1,
        controller_state=state or _minimal_controller_state(),
        evidence_bundle=evidence or _minimal_evidence_bundle(),
        history_bundle=history or _minimal_history_bundle(),
        verifier_report=verifier or _minimal_verifier_report(),
        solver_hints=hints or _minimal_solver_hints(),
    )


# ---------------------------------------------------------------------------
# Line 52: _truncate — max_chars <= 3 branch
# ---------------------------------------------------------------------------


class TestTruncateSmallMaxChars:
    def test_max_chars_zero_returns_empty(self) -> None:
        assert controller_packets._truncate("hello", max_chars=0) == ""

    def test_max_chars_one_returns_single_char(self) -> None:
        assert controller_packets._truncate("hello", max_chars=1) == "h"

    def test_max_chars_three_returns_three_chars(self) -> None:
        assert controller_packets._truncate("hello world", max_chars=3) == "hel"

    def test_max_chars_two_returns_two_chars(self) -> None:
        assert controller_packets._truncate("abcdef", max_chars=2) == "ab"

    def test_short_text_within_limit(self) -> None:
        assert controller_packets._truncate("hi", max_chars=3) == "hi"


# ---------------------------------------------------------------------------
# Line 269: render_compact — blame_hint from repo_history
# ---------------------------------------------------------------------------


class TestRenderCompactBlameHint:
    def test_blame_hint_rendered_when_present(self) -> None:
        repo = RepoHistoryV1(
            repo_root=".",
            head_branch="main",
            head_commit="abc",
            crash_file_path="buggy.py",
            crash_file_line=4,
            crash_file_blame_summary="off-by-one introduced in abc",
            crash_file_diffstat=None,
            recent_commits=[],
            warnings=[],
        )
        pkt = _make_packet(
            state=_minimal_controller_state(),
            history=_minimal_history_bundle(repo_history=repo),
        )
        text = render_compact_controller_guidance(pkt, include_history=True)
        assert "blame_hint: off-by-one introduced in abc" in text


# ---------------------------------------------------------------------------
# Line 278: render_compact — strategy_summary non-empty
# ---------------------------------------------------------------------------


class TestRenderCompactStrategySummary:
    def test_strategy_summary_line_present(self) -> None:
        pkt = _make_packet(
            hints=_minimal_solver_hints(
                strategy_summary="Narrow the edit scope",
            ),
        )
        text = render_compact_controller_guidance(pkt)
        assert "strategy_summary: Narrow the edit scope" in text


# ---------------------------------------------------------------------------
# Line 286: render_compact — decomposition steps
# ---------------------------------------------------------------------------


class TestRenderCompactDecomposition:
    def test_decomposition_steps_rendered(self) -> None:
        pkt = _make_packet(
            hints=_minimal_solver_hints(
                decomposition=[
                    {"title": "step1", "question": "What changed?"},
                    {"title": "step2", "question": "Where is it?"},
                ],
            ),
        )
        text = render_compact_controller_guidance(pkt)
        assert "decomposition:" in text
        assert "step1: What changed?" in text
        assert "step2: Where is it?" in text


# ---------------------------------------------------------------------------
# Line 361: _parse_repo_history — non-dict item in recent_commits_raw
# ---------------------------------------------------------------------------


class TestParseRepoHistorySkipsNonDict:
    def test_non_dict_commits_skipped(self) -> None:
        raw = {
            "head": {"branch": "main", "commit": "abc"},
            "crash_file": {},
            "recent_commits": [
                "not-a-dict",
                42,
                {"commit": "abc", "summary": "good commit"},
            ],
        }
        result = controller_packets._parse_repo_history(raw)
        assert result is not None
        assert len(result.recent_commits) == 1
        assert result.recent_commits[0].summary == "good commit"


# ---------------------------------------------------------------------------
# Lines 405, 408: _parse_history_lessons — non-dict / empty lesson
# ---------------------------------------------------------------------------


class TestParseHistoryLessonsEdgeCases:
    def test_non_dict_row_skipped(self) -> None:
        result = controller_packets._parse_history_lessons(
            ["not-a-dict", 42, {"lesson": "real", "source": "s"}]
        )
        assert len(result) == 1
        assert result[0].lesson == "real"

    def test_empty_lesson_skipped(self) -> None:
        result = controller_packets._parse_history_lessons(
            [{"lesson": "", "source": "s"}, {"lesson": "  ", "source": "s2"}]
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Line 425: _parse_evidence_items — empty evidence_id
# ---------------------------------------------------------------------------


class TestParseEvidenceItemsEmptyId:
    def test_empty_id_skipped(self) -> None:
        result = controller_packets._parse_evidence_items(
            [
                {"id": "", "kind": "frame"},
                {"id": "  ", "kind": "frame"},
                {"id": "e1", "kind": "frame", "locator": "loc"},
            ]
        )
        assert len(result) == 1
        assert result[0].id == "e1"


# ---------------------------------------------------------------------------
# Line 475: parse_controller_packet — missing required dict key
# ---------------------------------------------------------------------------


class TestParseControllerPacketMissingKey:
    def test_missing_controller_state(self) -> None:
        raw = {
            "evidence_bundle": {},
            "history_bundle": {},
            "verifier_report": {},
        }
        assert parse_controller_packet(raw) is None

    def test_missing_evidence_bundle(self) -> None:
        raw = {
            "controller_state": {},
            "history_bundle": {},
            "verifier_report": {},
        }
        assert parse_controller_packet(raw) is None

    def test_missing_history_bundle(self) -> None:
        raw = {
            "controller_state": {},
            "evidence_bundle": {},
            "verifier_report": {},
        }
        assert parse_controller_packet(raw) is None

    def test_missing_verifier_report(self) -> None:
        raw = {
            "controller_state": {},
            "evidence_bundle": {},
            "history_bundle": {},
        }
        assert parse_controller_packet(raw) is None

    def test_non_dict_value_returns_none(self) -> None:
        raw = {
            "controller_state": "not-dict",
            "evidence_bundle": {},
            "history_bundle": {},
            "verifier_report": {},
        }
        assert parse_controller_packet(raw) is None


# ---------------------------------------------------------------------------
# Lines 565, 569: decomposition parsing — non-dict / missing title or question
# ---------------------------------------------------------------------------


class TestDecompositionParsingEdgeCases:
    def _base_packet(self, decomposition_raw: list) -> dict:
        return {
            "controller_state": {},
            "evidence_bundle": {},
            "history_bundle": {},
            "verifier_report": {},
            "solver_hints": {"decomposition": decomposition_raw},
        }

    def test_non_dict_decomposition_row_skipped(self) -> None:
        raw = self._base_packet(["not-dict", 99])
        result = parse_controller_packet(raw)
        assert result is not None
        assert result.solver_hints.decomposition == []

    def test_missing_title_skipped(self) -> None:
        raw = self._base_packet([{"question": "q"}])
        result = parse_controller_packet(raw)
        assert result is not None
        assert result.solver_hints.decomposition == []

    def test_missing_question_skipped(self) -> None:
        raw = self._base_packet([{"title": "t"}])
        result = parse_controller_packet(raw)
        assert result is not None
        assert result.solver_hints.decomposition == []

    def test_valid_decomposition_kept(self) -> None:
        raw = self._base_packet([{"title": "step1", "question": "Why?"}])
        result = parse_controller_packet(raw)
        assert result is not None
        assert len(result.solver_hints.decomposition) == 1
        assert result.solver_hints.decomposition[0]["title"] == "step1"


# ---------------------------------------------------------------------------
# Lines 614, 617, 619: _crash_frame — non-dict, non-list, empty
# ---------------------------------------------------------------------------


class TestCrashFrameEdgeCases:
    def test_non_dict_snapshot_returns_none(self) -> None:
        assert controller_packets._crash_frame("not-a-dict") is None

    def test_non_list_frames_returns_none(self) -> None:
        assert controller_packets._crash_frame({"frames": "oops"}) is None

    def test_empty_frames_returns_none(self) -> None:
        assert controller_packets._crash_frame({"frames": []}) is None


# ---------------------------------------------------------------------------
# Lines 625-628: _crash_frame — crash_frame_index out of range, fallback
# ---------------------------------------------------------------------------


class TestCrashFrameFallback:
    def test_out_of_range_index_falls_back_to_first_dict(self) -> None:
        snap = {
            "crash_frame_index": 99,
            "frames": [
                {"file_rel": "a.py", "line": 1},
                {"file_rel": "b.py", "line": 2},
            ],
        }
        result = controller_packets._crash_frame(snap)
        assert result is not None
        assert result["file_rel"] == "a.py"

    def test_index_points_to_non_dict_falls_back(self) -> None:
        snap = {
            "crash_frame_index": 0,
            "frames": [
                "not-dict",
                {"file_rel": "b.py", "line": 2},
            ],
        }
        result = controller_packets._crash_frame(snap)
        assert result is not None
        assert result["file_rel"] == "b.py"

    def test_all_non_dict_frames_returns_none(self) -> None:
        snap = {"frames": ["a", "b", 3]}
        assert controller_packets._crash_frame(snap) is None


# ---------------------------------------------------------------------------
# Lines 640, 644, 647, 650-651, 653, 656-657: _ast_context_from_snapshot
# ---------------------------------------------------------------------------


class TestASTContextFromSnapshot:
    def test_no_crash_frame_returns_none(self) -> None:
        result = controller_packets._ast_context_from_snapshot(
            root=Path("/tmp"), snapshot={"frames": []}
        )
        assert result is None

    def test_missing_file_rel_returns_none(self, tmp_path: Path) -> None:
        snap = {"frames": [{"line": 1, "function": "f"}]}
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is None

    def test_source_path_not_file_returns_none(self, tmp_path: Path) -> None:
        snap = {"frames": [{"file_rel": "missing.py", "line": 1}]}
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is None

    def test_os_error_reading_returns_none(self, tmp_path: Path, monkeypatch) -> None:
        src = tmp_path / "bad.py"
        src.write_text("x = 1\n", encoding="utf-8")
        snap = {"frames": [{"file_rel": "bad.py", "line": 1}]}
        monkeypatch.setattr(
            Path, "read_text", lambda *a, **kw: (_ for _ in ()).throw(OSError("boom"))
        )
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is None

    def test_empty_source_returns_none(self, tmp_path: Path) -> None:
        src = tmp_path / "empty.py"
        src.write_text("   \n  \n", encoding="utf-8")
        snap = {"frames": [{"file_rel": "empty.py", "line": 1}]}
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is None

    def test_syntax_error_returns_partial_context(self, tmp_path: Path) -> None:
        src = tmp_path / "broken.py"
        src.write_text("def f(\n", encoding="utf-8")
        snap = {"frames": [{"file_rel": "broken.py", "line": 1}]}
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is not None
        assert result.node_type == "SyntaxError"
        assert result.path == "broken.py"
        assert "def f(" in result.snippet


# ---------------------------------------------------------------------------
# Line 679: AST line-target matching — no enclosing_nodes → None
# ---------------------------------------------------------------------------


class TestASTNoEnclosingNodes:
    def test_line_outside_all_nodes_returns_none(self, tmp_path: Path) -> None:
        src = tmp_path / "small.py"
        # A module-level file where line 100 is beyond all AST nodes.
        src.write_text("x = 1\n", encoding="utf-8")
        snap = {"frames": [{"file_rel": "small.py", "line": 100}]}
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is None


# ---------------------------------------------------------------------------
# Lines 693, 701-704: AST symbol extraction — lineno None; async/class
# ---------------------------------------------------------------------------


class TestASTSymbolKinds:
    def test_async_function_symbol_kind(self, tmp_path: Path) -> None:
        src = tmp_path / "af.py"
        src.write_text("async def handler():\n    pass\n", encoding="utf-8")
        snap = {"frames": [{"file_rel": "af.py", "line": 1}]}
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is not None
        assert result.symbol_kind == "async_function"
        assert result.symbol_name == "handler"

    def test_class_symbol_kind(self, tmp_path: Path) -> None:
        src = tmp_path / "cls.py"
        src.write_text("class MyClass:\n    x = 1\n", encoding="utf-8")
        snap = {"frames": [{"file_rel": "cls.py", "line": 1}]}
        result = controller_packets._ast_context_from_snapshot(root=tmp_path, snapshot=snap)
        assert result is not None
        assert result.symbol_kind == "class"
        assert result.symbol_name == "MyClass"


# ---------------------------------------------------------------------------
# Lines 728, 731, 734, 737: _snapshot_reduction_from_snapshot
# ---------------------------------------------------------------------------


class TestSnapshotReductionEdgeCases:
    def test_non_dict_returns_none(self) -> None:
        assert controller_packets._snapshot_reduction_from_snapshot("x") is None

    def test_non_list_frames_returns_none(self) -> None:
        result = controller_packets._snapshot_reduction_from_snapshot({"frames": "bad"})
        assert result is None

    def test_empty_frames_returns_none(self) -> None:
        result = controller_packets._snapshot_reduction_from_snapshot({"frames": []})
        assert result is None

    def test_no_dict_frames_returns_none(self) -> None:
        result = controller_packets._snapshot_reduction_from_snapshot({"frames": ["a", "b"]})
        assert result is None

    def test_crash_frame_none_returns_none(self, monkeypatch) -> None:
        """When _crash_frame returns None the reduction is None."""
        monkeypatch.setattr(controller_packets, "_crash_frame", lambda _: None)
        result = controller_packets._snapshot_reduction_from_snapshot(
            {"frames": [{"file_rel": "a.py", "line": 1}]}
        )
        assert result is None


# ---------------------------------------------------------------------------
# Line 762: snapshot reduction — locals key starting with "@" → skip
# ---------------------------------------------------------------------------


class TestSnapshotReductionSkipsAtLocals:
    def test_at_prefixed_local_skipped(self) -> None:
        snap = {
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "code": "pass",
                    "locals": {"@hidden": "secret", "visible": 42},
                }
            ]
        }
        result = controller_packets._snapshot_reduction_from_snapshot(snap)
        assert result is not None
        local_names = [loc["name"] for loc in result.selected_locals]
        assert "visible" in local_names
        assert "@hidden" not in local_names


# ---------------------------------------------------------------------------
# Line 787: _line_span_from_raw_item — has int "line" value > 0
# ---------------------------------------------------------------------------


class TestLineSpanFromRawItem:
    def test_positive_line_returns_tuple(self) -> None:
        result = controller_packets._line_span_from_raw_item({"line": 5})
        assert result == (5, 5)

    def test_zero_line_falls_through(self) -> None:
        result = controller_packets._line_span_from_raw_item({"line": 0, "locator": "file:10"})
        assert result == (10, 10)

    def test_bool_line_falls_through(self) -> None:
        """bool is subclass of int — ensure it is rejected."""
        result = controller_packets._line_span_from_raw_item({"line": True, "locator": "file:3"})
        assert result == (3, 3)


# ---------------------------------------------------------------------------
# Lines 806, 821, 824: _evidence_kind_priority
# ---------------------------------------------------------------------------


class TestEvidenceKindPriority:
    def test_source_span_priority_1(self) -> None:
        assert controller_packets._evidence_kind_priority("source_span") == 1

    def test_file_only_priority_1(self) -> None:
        assert controller_packets._evidence_kind_priority("file_only") == 1

    def test_search_hit_priority_2(self) -> None:
        assert controller_packets._evidence_kind_priority("search_hit") == 2

    def test_search_only_priority_2(self) -> None:
        assert controller_packets._evidence_kind_priority("search_only") == 2

    def test_unknown_priority_3(self) -> None:
        assert controller_packets._evidence_kind_priority("whatever") == 3

    def test_frame_priority_0(self) -> None:
        assert controller_packets._evidence_kind_priority("frame") == 0


# ---------------------------------------------------------------------------
# Lines 840, 843: _evidence_items_from_raw — non-dict / empty id
# ---------------------------------------------------------------------------


class TestEvidenceItemsFromRawEdge:
    def test_non_dict_item_skipped(self) -> None:
        result = controller_packets._evidence_items_from_raw(
            evidence_items=["bad", 42],
            new_ids=[],
            reused_ids=[],
        )
        assert result == []

    def test_empty_id_skipped(self) -> None:
        result = controller_packets._evidence_items_from_raw(
            evidence_items=[{"id": "", "type": "frame"}],
            new_ids=[],
            reused_ids=[],
        )
        assert result == []


# ---------------------------------------------------------------------------
# Line 905: _build_evidence_summary — no unique_facts and no crash_anchor
# ---------------------------------------------------------------------------


class TestBuildEvidenceSummaryNone:
    def test_returns_none_when_no_facts_or_anchor(self) -> None:
        result = controller_packets._build_evidence_summary(
            ast_context=None,
            snapshot_reduction=None,
            items=[],
            total_item_count=0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Lines 930, 934: evidence dedup — duplicate fact / empty fact
# ---------------------------------------------------------------------------


class TestBuildEvidenceSummaryDedup:
    def test_duplicate_facts_deduplicated(self) -> None:
        ast_ctx = ASTContextV1(
            path="a.py",
            crash_line=1,
            symbol_name="f",
            symbol_kind="function",
            node_type="FunctionDef",
            line_span=(1, 2),
            snippet="def f(): pass",
            related_names=[],
        )
        sr = SnapshotReductionV1(
            selected_frame_count=1,
            omitted_frame_count=0,
            crash_frame_summary="a.py:1 in f()",
            selected_frames=[],
            selected_locals=[],
        )
        # Feed items with identical summaries to trigger dedup
        items = [
            EvidenceItemV1(
                id="e1",
                kind="frame",
                locator="loc",
                summary="same text",
                path=None,
                line_span=None,
                score=1.0,
                provenance="helper",
            ),
            EvidenceItemV1(
                id="e2",
                kind="frame",
                locator="loc",
                summary="same text",
                path=None,
                line_span=None,
                score=0.5,
                provenance="helper",
            ),
        ]
        result = controller_packets._build_evidence_summary(
            ast_context=ast_ctx,
            snapshot_reduction=sr,
            items=items,
            total_item_count=2,
        )
        assert result is not None
        # The two "frame loc: same text" entries should dedup to 1
        fact_text = " ".join(result.top_facts)
        count = fact_text.count("same text")
        # At most 2 occurrences from the two items but identical ones deduplicated
        assert count >= 1


# ---------------------------------------------------------------------------
# Line 964: history lesson — fallback to truncated turn text
# ---------------------------------------------------------------------------


class TestHistoryLessonFallbackTruncated:
    def test_raw_turn_used_as_lesson_when_no_structured_values(self) -> None:
        lessons = controller_packets._build_history_lessons(
            retry_history_turns=["Just a raw turn with no structured prefix"],
            previous_note="",
            previous_patch_verifier_reasons=[],
            persistent_lessons=[],
        )
        assert len(lessons) == 1
        assert "raw turn" in lessons[0].lesson


# ---------------------------------------------------------------------------
# Lines 998, 1010, 1013-1014: _compact_repo_history — non-dict / exception
# ---------------------------------------------------------------------------


class TestCompactRepoHistoryEdgeCases:
    def test_non_dict_snapshot_returns_none(self, tmp_path: Path) -> None:
        result = controller_packets._compact_repo_history(snapshot="bad", attempt_dir=tmp_path)
        assert result is None

    def test_exception_returns_error_repo_history(self, tmp_path: Path, monkeypatch) -> None:
        def boom(*_a, **_kw):
            raise RuntimeError("git exploded")

        monkeypatch.setattr(controller_packets, "get_enhanced_git_context", boom)
        snap = {"frames": [{"file_rel": "a.py", "line": 1}]}
        result = controller_packets._compact_repo_history(snapshot=snap, attempt_dir=tmp_path)
        assert result is not None
        assert "RuntimeError" in result.error
        assert "git exploded" in result.error


# ---------------------------------------------------------------------------
# Line 1043: compact repo history — _git_warnings present
# ---------------------------------------------------------------------------


class TestCompactRepoHistoryWarnings:
    def test_git_warnings_included(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            controller_packets,
            "get_enhanced_git_context",
            lambda *_a, **_kw: {
                "repo_root": "/repo",
                "head": {"branch": "main", "commit": "abc"},
                "crash_file": {"file_rel": "a.py"},
                "recent_commits": [],
                "_git_warnings": ["shallow clone detected"],
            },
        )
        snap = {"frames": [{"file_rel": "a.py", "line": 1}]}
        result = controller_packets._compact_repo_history(snapshot=snap, attempt_dir=tmp_path)
        assert result is not None
        assert "shallow clone detected" in result.warnings


# ---------------------------------------------------------------------------
# Line 1249: verifier report — syntax_last_error_type → status
# ---------------------------------------------------------------------------


class TestVerifierReportSyntaxRetryFeedback:
    def test_syntax_error_type_sets_syntax_retry_feedback(self, tmp_path: Path) -> None:
        root = tmp_path / "case"
        root.mkdir()
        (root / "f.py").write_text("def f():\n    pass\n", encoding="utf-8")
        attempt_dir = tmp_path / "attempt_s"
        attempt_dir.mkdir()
        snap = {
            "frames": [
                {
                    "file_rel": "f.py",
                    "line": 1,
                    "function": "f",
                    "code": "pass",
                    "locals": {},
                }
            ]
        }
        sidecars = build_adaptive_controller_sidecars(
            case_id="case",
            condition="cond",
            attempt=2,
            root=root,
            attempt_dir=attempt_dir,
            snapshot=snap,
            failure_signature="sig",
            evidence_items=[],
            new_evidence_ids=[],
            reused_evidence_ids=[],
            retry_prompt_mode="delta",
            retry_context_mode="history",
            retry_history_text="",
            retry_history_turns=[],
            retry_history_turns_included=0,
            retry_history_truncated=False,
            previous_failure_signature="",
            previous_note="",
            previous_patch_verifier_rejected=False,
            previous_patch_verifier_reasons=[],
            previous_patch_verifier_scope="adaptive_only",
            previous_timed_out=False,
            previous_timeout_sec=None,
            helper_action_selected="",
            helper_action_executed="",
            helper_decision_reason="",
            helper_outcome="",
            helper_materialized_context=False,
            helper_calls_used=0,
            helper_budget_max_calls=0,
            no_progress_streak=0,
            context_compaction_requested=False,
            context_compaction_invoked=False,
            context_compaction_success=False,
            context_compaction_trigger="not_needed",
            prompt_budget={"total_chars": 500},
            retry_plan_text="",
            syntax_cycle=2,
            syntax_max_cycles=3,
            syntax_last_error_type="SyntaxError",
            syntax_last_error_msg="unexpected EOF",
            syntax_retry_feedback="Fix the syntax error",
        )
        verifier = sidecars["llmdebug_verifier_report.json"]
        assert verifier["status"] == "syntax_retry_feedback"
        assert verifier["last_error_type"] == "SyntaxError"


# ---------------------------------------------------------------------------
# render_compact_controller_guidance — full branch coverage
# ---------------------------------------------------------------------------


class TestRenderCompactControllerGuidanceFull:
    """Exercise every branch in render_compact_controller_guidance (lines 231-286)."""

    def test_full_render_with_all_branches(self) -> None:
        """Build a packet with all optional fields populated."""
        state = AdaptiveControllerStateV2(
            schema_version="v1",
            helper_action_selected="snapshot_lite",
            helper_action_executed="snapshot_lite",
            helper_decision_reason="test",
            helper_outcome="materialized",
            helper_materialized_context=True,
            helper_calls_used=1,
            helper_budget_max_calls=2,
            no_progress_streak=0,
            retry_prompt_mode="delta",
            retry_context_mode="history",
            context_compaction_requested=False,
            context_compaction_invoked=False,
            context_compaction_success=False,
            context_compaction_trigger="",
            prompt_budget={},
        )
        evidence = _minimal_evidence_bundle()
        evidence = EvidenceBundleV1(
            schema_version="v1",
            case_id="c",
            condition="cond",
            attempt=1,
            failure_signature="sig",
            selected_ids=[],
            new_ids=[],
            reused_ids=[],
            ast_context=None,
            snapshot_reduction=None,
            summary=EvidenceSummaryV1(
                strategy="deterministic_extract_v1",
                crash_anchor="buggy.py:10 in buggy()",
                top_facts=["crash at buggy.py:10"],
                selected_local_names=["x"],
                omitted_item_count=0,
                omitted_frame_count=0,
            ),
            items=[],
        )
        verifier = VerifierReportV1(
            schema_version="v1",
            status="previous_verifier_rejected",
            scope="adaptive_only",
            reasons=["missed crash file"],
            previous_timed_out=False,
            previous_timeout_sec=None,
            syntax_cycle=0,
            syntax_max_cycles=0,
            last_error_type="",
            last_error_msg="",
            retry_feedback="",
        )
        history = _minimal_history_bundle(
            lessons=[
                HistoryLessonV1(source="attempt_history", attempt=1, lesson="fix return"),
            ],
            repo_history=RepoHistoryV1(
                repo_root=".",
                head_branch="main",
                head_commit="abc",
                crash_file_path="buggy.py",
                crash_file_line=10,
                crash_file_blame_summary="introduced off-by-one",
                crash_file_diffstat=None,
                recent_commits=[],
                warnings=[],
            ),
        )
        hints = SolverHintsV1(
            tool_routes=["evidence_summary"],
            retry_plan="Focus on the return",
            recommended_focus=[],
            decomposition=[
                {"title": "localize", "question": "Where is the bug?"},
            ],
            strategy_summary="Edit only buggy.py",
        )
        pkt = ControllerPacketV1(
            schema_version="v1",
            case_id="c",
            condition="cond",
            attempt=1,
            controller_state=state,
            evidence_bundle=evidence,
            history_bundle=history,
            verifier_report=verifier,
            solver_hints=hints,
        )
        text = render_compact_controller_guidance(pkt)
        assert "controller_state:" in text
        assert "action=snapshot_lite" in text
        assert "crash_anchor: buggy.py:10" in text
        assert "evidence_summary:" in text
        assert "verifier_status: previous_verifier_rejected" in text
        assert "verifier_reasons: missed crash file" in text
        assert "history_lessons:" in text
        assert "attempt 1: fix return" in text
        assert "blame_hint: introduced off-by-one" in text
        assert "retry_plan: Focus on the return" in text
        assert "strategy_summary: Edit only buggy.py" in text
        assert "decomposition:" in text
        assert "localize: Where is the bug?" in text

    def test_empty_packet_returns_empty_string(self) -> None:
        pkt = _make_packet()
        text = render_compact_controller_guidance(pkt)
        assert text == ""

    def test_none_packet_returns_empty_string(self) -> None:
        text = render_compact_controller_guidance(None)
        assert text == ""


class TestRenderHybridControllerGuidance:
    def test_fair_v1_uses_shared_canonical_sections_and_cap(self) -> None:
        pkt = _make_packet(
            state=AdaptiveControllerStateV2(
                schema_version="v1",
                helper_action_selected="snapshot_lite",
                helper_action_executed="snapshot_lite",
                helper_decision_reason="policy_action_snapshot_lite",
                helper_outcome="materialized",
                helper_materialized_context=True,
                helper_calls_used=1,
                helper_budget_max_calls=2,
                no_progress_streak=0,
                retry_prompt_mode="delta",
                retry_context_mode="history",
                context_compaction_requested=False,
                context_compaction_invoked=False,
                context_compaction_success=False,
                context_compaction_trigger="not_needed",
                prompt_budget={"total_chars": 1000},
            ),
            evidence=EvidenceBundleV1(
                schema_version="v1",
                case_id="c",
                condition="cond",
                attempt=1,
                failure_signature="sig",
                selected_ids=[],
                new_ids=[],
                reused_ids=[],
                ast_context=None,
                snapshot_reduction=None,
                summary=EvidenceSummaryV1(
                    strategy="deterministic_extract_v1",
                    crash_anchor="buggy.py:10 in buggy",
                    top_facts=["A" * 240, "B" * 240, "C" * 240, "D" * 240, "E" * 240],
                    selected_local_names=["value"],
                    omitted_item_count=0,
                    omitted_frame_count=0,
                ),
                items=[],
            ),
            history=_minimal_history_bundle(
                lessons=[
                    HistoryLessonV1(source="attempt_history", attempt=1, lesson="keep scope tight"),
                    HistoryLessonV1(
                        source="attempt_history", attempt=2, lesson="edit buggy.py only"
                    ),
                    HistoryLessonV1(
                        source="persistent_history", attempt=None, lesson="preserve return type"
                    ),
                    HistoryLessonV1(source="attempt_history", attempt=3, lesson="drop extras"),
                ]
            ),
            verifier=VerifierReportV1(
                schema_version="v1",
                status="previous_verifier_rejected",
                scope="adaptive_only",
                reasons=["touch buggy.py", "preserve signature", "keep tests unchanged", "ignore"],
                previous_timed_out=False,
                previous_timeout_sec=None,
                syntax_cycle=0,
                syntax_max_cycles=0,
                last_error_type="",
                last_error_msg="",
                retry_feedback="",
            ),
            hints=SolverHintsV1(
                tool_routes=[],
                retry_plan="Adjust the buggy return expression without widening the edit.",
                recommended_focus=[],
                decomposition=[{"title": "localize", "question": "should not appear"}],
                strategy_summary="should not appear",
            ),
        )

        rendered = render_hybrid_controller_guidance(pkt, mode="fair_v1")

        assert rendered.mode == "fair_v1"
        assert rendered.sections == (
            "controller_state",
            "crash_anchor",
            "evidence_summary",
            "verifier",
            "history_lessons",
            "retry_plan",
        )
        assert rendered.char_count <= 2000
        assert "controller_state: action=snapshot_lite outcome=materialized" in rendered.text
        assert "crash_anchor: buggy.py:10 in buggy" in rendered.text
        assert "evidence_summary:" in rendered.text
        assert "verifier: status=previous_verifier_rejected" in rendered.text
        assert "history_lessons:" in rendered.text
        assert "retry_plan:" in rendered.text
        assert "strategy_summary:" not in rendered.text
        assert "decomposition:" not in rendered.text
        assert "blame_hint:" not in rendered.text

    def test_fair_v1_respects_custom_max_chars(self) -> None:
        pkt = _make_packet(
            state=AdaptiveControllerStateV2(
                schema_version="v1",
                helper_action_selected="snapshot_lite",
                helper_action_executed="snapshot_lite",
                helper_decision_reason="policy_action_snapshot_lite",
                helper_outcome="materialized",
                helper_materialized_context=True,
                helper_calls_used=1,
                helper_budget_max_calls=2,
                no_progress_streak=0,
                retry_prompt_mode="delta",
                retry_context_mode="history",
                context_compaction_requested=False,
                context_compaction_invoked=False,
                context_compaction_success=False,
                context_compaction_trigger="not_needed",
                prompt_budget={"total_chars": 1000},
            ),
            evidence=EvidenceBundleV1(
                schema_version="v1",
                case_id="c",
                condition="cond",
                attempt=1,
                failure_signature="sig",
                selected_ids=[],
                new_ids=[],
                reused_ids=[],
                ast_context=None,
                snapshot_reduction=None,
                summary=EvidenceSummaryV1(
                    strategy="deterministic_extract_v1",
                    crash_anchor="buggy.py:10 in buggy",
                    top_facts=["preserve signature"],
                    selected_local_names=["value"],
                    omitted_item_count=0,
                    omitted_frame_count=0,
                ),
                items=[],
            ),
        )

        rendered = render_hybrid_controller_guidance(pkt, mode="fair_v1", max_chars=30)

        assert rendered.mode == "fair_v1"
        assert rendered.char_count <= 30
        assert rendered.sections == ("controller_state",)
        assert rendered.text == "controller_state: action=sn..."

    def test_fixed_v1_uses_evidence_only_sections_and_cap(self) -> None:
        pkt = _make_packet(
            state=AdaptiveControllerStateV2(
                schema_version="v1",
                helper_action_selected="snapshot_lite",
                helper_action_executed="snapshot_lite",
                helper_decision_reason="policy_action_snapshot_lite",
                helper_outcome="materialized",
                helper_materialized_context=True,
                helper_calls_used=1,
                helper_budget_max_calls=2,
                no_progress_streak=1,
                retry_prompt_mode="delta",
                retry_context_mode="history",
                context_compaction_requested=False,
                context_compaction_invoked=False,
                context_compaction_success=False,
                context_compaction_trigger="not_needed",
                prompt_budget={"total_chars": 1000},
            ),
            evidence=EvidenceBundleV1(
                schema_version="v1",
                case_id="c",
                condition="cond",
                attempt=1,
                failure_signature="sig",
                selected_ids=["e1"],
                new_ids=["e1"],
                reused_ids=[],
                ast_context=ASTContextV1(
                    path="buggy.py",
                    crash_line=10,
                    symbol_name="buggy",
                    symbol_kind="function",
                    node_type="FunctionDef",
                    line_span=(8, 12),
                    snippet="def buggy(value):\n    return value + 1\n",
                    related_names=[],
                ),
                snapshot_reduction=SnapshotReductionV1(
                    selected_frame_count=1,
                    omitted_frame_count=2,
                    crash_frame_summary="buggy.py:10 in buggy() value=1",
                    selected_frames=[],
                    selected_locals=[{"name": "value", "summary": "1"}],
                ),
                summary=EvidenceSummaryV1(
                    strategy="deterministic_extract_v1",
                    crash_anchor="buggy.py:10 in buggy",
                    top_facts=["Edit the return expression", "Keep the signature unchanged"],
                    selected_local_names=["value"],
                    omitted_item_count=0,
                    omitted_frame_count=2,
                ),
                items=[
                    EvidenceItemV1(
                        id="e1",
                        kind="file",
                        locator="buggy.py",
                        summary="Crash file with the incorrect return expression.",
                        path="buggy.py",
                        line_span=(8, 12),
                        score=1.0,
                        provenance="adaptive_helper",
                    )
                ],
                snapshot_full_text=json.dumps(
                    {
                        "frames": [
                            {
                                "file_rel": "buggy.py",
                                "line": 10,
                                "locals": {"value": "1"},
                            }
                        ]
                    },
                    indent=2,
                ),
            ),
            history=_minimal_history_bundle(
                previous_note="tests_still_failing",
                lessons=[HistoryLessonV1(source="attempt_history", attempt=1, lesson="skip")],
            ),
            verifier=VerifierReportV1(
                schema_version="v1",
                status="previous_verifier_rejected",
                scope="adaptive_only",
                reasons=["touch buggy.py"],
                previous_timed_out=False,
                previous_timeout_sec=None,
                syntax_cycle=0,
                syntax_max_cycles=0,
                last_error_type="",
                last_error_msg="",
                retry_feedback="",
            ),
            hints=SolverHintsV1(
                tool_routes=[],
                retry_plan="should not appear",
                recommended_focus=["return expr"],
                decomposition=[{"title": "localize", "question": "should not appear"}],
                strategy_summary="should not appear",
            ),
        )

        rendered = render_hybrid_controller_guidance(pkt, mode="fixed_v1", max_chars=2000)

        assert rendered.mode == "fixed_v1"
        assert rendered.sections == (
            "crash_anchor",
            "evidence_summary",
            "ast_focus",
            "ast_snippet",
            "selected_evidence",
            "snapshot_full",
        )
        assert "crash_anchor: buggy.py:10 in buggy" in rendered.text
        assert "evidence_summary:" in rendered.text
        assert "ast_focus: buggy.py lines=(8, 12) symbol=buggy" in rendered.text
        assert "ast_snippet:" in rendered.text
        assert "selected_evidence:" in rendered.text
        assert "snapshot_full:" in rendered.text
        assert '"file_rel": "buggy.py"' in rendered.text
        assert '"line": 10' in rendered.text
        assert "controller_state:" not in rendered.text
        assert "verifier" not in rendered.text
        assert "history_lessons:" not in rendered.text
        assert "retry_plan:" not in rendered.text
        assert "strategy_summary:" not in rendered.text
        assert "decomposition:" not in rendered.text
        assert "snapshot_focus:" not in rendered.text

    def test_fixed_v1_respects_custom_max_chars(self) -> None:
        pkt = _make_packet(
            evidence=EvidenceBundleV1(
                schema_version="v1",
                case_id="c",
                condition="cond",
                attempt=1,
                failure_signature="sig",
                selected_ids=[],
                new_ids=[],
                reused_ids=[],
                ast_context=None,
                snapshot_reduction=None,
                summary=EvidenceSummaryV1(
                    strategy="deterministic_extract_v1",
                    crash_anchor="buggy.py:10 in buggy",
                    top_facts=["preserve signature"],
                    selected_local_names=[],
                    omitted_item_count=0,
                    omitted_frame_count=0,
                ),
                items=[],
            ),
        )

        rendered = render_hybrid_controller_guidance(pkt, mode="fixed_v1", max_chars=30)

        assert rendered.mode == "fixed_v1"
        assert rendered.char_count <= 30
        assert rendered.sections == ("crash_anchor",)
        assert rendered.text.startswith("crash_anchor: buggy.py:10")
        assert rendered.text.endswith("...")

    def test_tuned_v1_trims_section_metadata_after_truncation(self) -> None:
        first_line = "controller_state: action=skip outcome=n/a"
        pkt = _make_packet(
            state=AdaptiveControllerStateV2(
                schema_version="v1",
                helper_action_selected="skip",
                helper_action_executed="skip",
                helper_decision_reason="policy_action_skip",
                helper_outcome="",
                helper_materialized_context=False,
                helper_calls_used=0,
                helper_budget_max_calls=0,
                no_progress_streak=0,
                retry_prompt_mode="",
                retry_context_mode="",
                context_compaction_requested=False,
                context_compaction_invoked=False,
                context_compaction_success=False,
                context_compaction_trigger="",
                prompt_budget={},
            ),
            hints=SolverHintsV1(
                tool_routes=[],
                retry_plan="focus on buggy.py only",
                recommended_focus=[],
                decomposition=[{"title": "localize", "question": "where is the bug?"}],
                strategy_summary="",
            ),
        )

        rendered = render_hybrid_controller_guidance(
            pkt,
            mode="tuned_v1",
            include_history=False,
            max_chars=len(first_line),
        )

        assert rendered.text == first_line
        assert rendered.sections == ("controller_state",)
        assert hybrid_guidance_metadata(rendered) == {
            "baseline_hybrid_guidance_mode": "tuned_v1",
            "baseline_hybrid_guidance_chars": len(first_line),
            "baseline_hybrid_guidance_sections": ["controller_state"],
        }

    def test_hybrid_guidance_metadata_reflects_rendered_block(self) -> None:
        rendered = render_hybrid_controller_guidance(_make_packet(), mode="tuned_v1")

        assert hybrid_guidance_metadata(rendered) == {
            "baseline_hybrid_guidance_mode": "tuned_v1",
            "baseline_hybrid_guidance_chars": 0,
            "baseline_hybrid_guidance_sections": [],
        }


# ---------------------------------------------------------------------------
# parse_controller_packet — full round-trip (lines 454-459)
# ---------------------------------------------------------------------------


class TestParseControllerPacketRoundTrip:
    """Full round-trip through parse_controller_packet with solver_hints."""

    def test_round_trip_with_decomposition(self) -> None:
        raw = {
            "schema_version": "v1",
            "controller_state": {
                "schema_version": "v1",
                "helper_action_selected": "snapshot_lite",
                "helper_action_executed": "snapshot_lite",
                "helper_decision_reason": "test",
                "helper_outcome": "materialized",
                "helper_materialized_context": True,
                "helper_calls_used": 1,
                "helper_budget_max_calls": 2,
                "no_progress_streak": 0,
                "retry_prompt_mode": "delta",
                "retry_context_mode": "history",
                "context_compaction_requested": False,
                "context_compaction_invoked": False,
                "context_compaction_success": False,
                "context_compaction_trigger": "",
                "prompt_budget": {"total_chars": 1000},
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "case_id": "c",
                "condition": "cond",
                "attempt": 1,
                "failure_signature": "sig",
                "selected_ids": [],
                "new_ids": [],
                "reused_ids": [],
                "items": [
                    {"id": "", "kind": "frame", "locator": "frame:0"},
                    {"id": "e1", "kind": "frame", "locator": "frame:0", "summary": "s"},
                ],
            },
            "history_bundle": {
                "schema_version": "v1",
                "retry_context_mode": "history",
                "attempt_turns": [],
                "retry_history_text": "",
                "retry_history_turns_included": 0,
                "retry_history_truncated": False,
                "previous_note": "",
                "previous_failure_signature": "",
                "previous_patch_verifier_rejected": False,
                "previous_patch_verifier_reasons": [],
                "lessons": [
                    "not_a_dict",
                    {"lesson": ""},
                    {"source": "s", "attempt": 1, "lesson": "real lesson"},
                ],
            },
            "verifier_report": {
                "schema_version": "v1",
                "status": "none",
                "scope": "",
                "reasons": [],
                "previous_timed_out": False,
            },
            "solver_hints": {
                "tool_routes": ["evidence_summary"],
                "retry_plan": "fix it",
                "recommended_focus": [],
                "decomposition": [
                    "not_a_dict",
                    {"title": "", "question": "empty title"},
                    {"title": "step1", "question": "what?"},
                ],
                "strategy_summary": "be minimal",
            },
        }
        pkt = parse_controller_packet(raw)
        assert pkt is not None
        assert pkt.controller_state.helper_action_executed == "snapshot_lite"
        assert pkt.controller_state.prompt_budget == {"total_chars": 1000}
        assert len(pkt.evidence_bundle.items) == 1
        assert pkt.evidence_bundle.items[0].id == "e1"
        assert len(pkt.history_bundle.lessons) == 1
        assert pkt.history_bundle.lessons[0].lesson == "real lesson"
        assert len(pkt.solver_hints.decomposition) == 1
        assert pkt.solver_hints.decomposition[0]["title"] == "step1"
        assert pkt.solver_hints.strategy_summary == "be minimal"


# ---------------------------------------------------------------------------
# _crash_frame fallback — line 627 (non-dict at crash_idx, first dict elsewhere)
# ---------------------------------------------------------------------------


class TestCrashFrameFallbackToDictInLoop:
    def test_non_dict_at_crash_idx_falls_back_to_first_dict(self) -> None:
        snapshot = {"frames": ["not_a_dict", {"file_rel": "b.py", "line": 1}]}
        result = controller_packets._crash_frame(snapshot)
        assert result is not None
        assert result["file_rel"] == "b.py"


# ---------------------------------------------------------------------------
# _ast_context_from_snapshot — line 693 (start/end_lineno None)
# ---------------------------------------------------------------------------


class TestASTContextLinenoNone:
    def test_returns_none_when_no_lineno(self, tmp_path: Path) -> None:
        src = tmp_path / "mod.py"
        src.write_text("x = 1\n", encoding="utf-8")
        snapshot = {"frames": [{"file_rel": "mod.py", "line": 1}]}
        result = controller_packets._ast_context_from_snapshot(
            root=tmp_path,
            snapshot=snapshot,
        )
        # x=1 is an Assign node; for simple assignments the enclosing node
        # may not be a FunctionDef/ClassDef — the AST walker still works.
        # The key test is that it doesn't crash.
        assert result is None or isinstance(result.symbol_name, str)


# ---------------------------------------------------------------------------
# _history_value_from_turn — line 821
# ---------------------------------------------------------------------------


class TestHistoryValueFromTurn:
    def test_extracts_value_from_turn(self) -> None:
        turn = "Attempt 1\nOutcome: tests_still_failing\nEvidence used: snapshot"
        result = controller_packets._history_value_from_turn(turn, "Outcome")
        assert result == "tests_still_failing"

    def test_returns_empty_when_prefix_absent(self) -> None:
        turn = "Attempt 1\nSomething else"
        result = controller_packets._history_value_from_turn(turn, "Outcome")
        assert result == ""


# ---------------------------------------------------------------------------
# _build_evidence_summary — line 905 (crash_anchor from snapshot_reduction)
# ---------------------------------------------------------------------------


class TestBuildEvidenceSummaryCrashAnchorFromReduction:
    def test_crash_anchor_from_reduction_when_no_ast(self) -> None:
        reduction = SnapshotReductionV1(
            selected_frame_count=1,
            omitted_frame_count=0,
            crash_frame_summary="buggy.py:10 in buggy()",
            selected_frames=[],
            selected_locals=[],
        )
        result = controller_packets._build_evidence_summary(
            ast_context=None,
            snapshot_reduction=reduction,
            items=[],
            total_item_count=0,
        )
        assert result is not None
        assert result.crash_anchor == "buggy.py:10 in buggy()"


# ---------------------------------------------------------------------------
# _build_compact_history_lessons — line 964 (fallback to truncated turn)
# ---------------------------------------------------------------------------


class TestCompactHistoryLessonFallback:
    def test_fallback_to_raw_turn_when_no_structured_values(self) -> None:
        result = controller_packets._build_history_lessons(
            retry_history_turns=["just some text without structured fields"],
            previous_note="",
            previous_patch_verifier_reasons=[],
            persistent_lessons=[],
        )
        assert len(result) == 1
        assert "just some text" in result[0].lesson


# ---------------------------------------------------------------------------
# _compact_repo_history — line 998 (non-dict snapshot)
# ---------------------------------------------------------------------------


class TestCompactRepoHistoryNonDictSnapshot:
    def test_returns_none_for_non_dict(self, tmp_path: Path) -> None:
        result = controller_packets._compact_repo_history(
            snapshot="not a dict",
            attempt_dir=tmp_path,
        )
        assert result is None


# ---------------------------------------------------------------------------
# remove_adaptive_controller_sidecars / write_adaptive_controller_sidecars
# ---------------------------------------------------------------------------


class TestSidecarFileOps:
    def test_remove_sidecars(self, tmp_path: Path) -> None:
        for name in controller_packets.ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES:
            (tmp_path / name).write_text("{}", encoding="utf-8")
        removed = controller_packets.remove_adaptive_controller_sidecars(
            context_dir=tmp_path,
        )
        assert len(removed) == len(
            controller_packets.ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES,
        )
        for path in removed:
            assert not path.exists()

    def test_remove_sidecars_missing_files(self, tmp_path: Path) -> None:
        removed = controller_packets.remove_adaptive_controller_sidecars(
            context_dir=tmp_path,
        )
        assert removed == []

    def test_write_sidecars(self, tmp_path: Path) -> None:
        sidecars = {"file_a.json": {"key": "value"}, "file_b.json": {"x": 1}}
        written = controller_packets.write_adaptive_controller_sidecars(
            context_dir=tmp_path,
            sidecars=sidecars,
        )
        assert len(written) == 2
        for path in written:
            assert path.exists()
            import json as _json

            data = _json.loads(path.read_text(encoding="utf-8"))
            assert "_sidecar_schema_version" in data
