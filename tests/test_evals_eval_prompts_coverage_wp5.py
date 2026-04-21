"""Coverage tests for evals.eval_prompts - wp5 batch.

Targets uncovered lines and branches: _format_snapshot_index_section frame
limit, _truncate_text_for_section marker-too-large and pytest-output negative
head, _build_context_compaction_bundle pytest/files scope, truncation of
combined bundle, _build_retry_history_section clipping and empty, _render_
retry_history_turn with failure signature, _build_retry_summary._compact
truncation and _new_constraint verifier-rejected fallback, and
_prompt_safe_previous_attempt_diff rewrite.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from evals import eval_prompts as mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_with_frames(n: int) -> dict[str, Any]:
    """Build a snapshot with *n* frames each having file_rel."""
    return {
        "frames": [
            {"file_rel": f"src/mod{i}.py", "line": i + 1, "function": f"fn{i}"} for i in range(n)
        ],
        "exception": {"type": "ValueError", "message": "test"},
    }


# ---------------------------------------------------------------------------
# _format_snapshot_index_section - frame limit exceeded
# ---------------------------------------------------------------------------


class TestFormatSnapshotIndexSectionFrameLimit:
    """When frame count exceeds max_frames the loop breaks early."""

    def test_max_frames_limit(self) -> None:
        """Only max_frames frames are shown."""
        snap = _snapshot_with_frames(10)
        result = mod._format_snapshot_index_section(snap, max_frames=3)
        assert result.count("Frame") == 3

    def test_default_limit_is_six(self) -> None:
        """Default max_frames=6 limits output."""
        snap = _snapshot_with_frames(10)
        result = mod._format_snapshot_index_section(snap)
        assert result.count("Frame") == 6


# ---------------------------------------------------------------------------
# _truncate_text_for_section - edge cases
# ---------------------------------------------------------------------------


class TestTruncateTextForSection:
    """Truncation behaviour for various key types and limits."""

    def test_marker_too_large(self) -> None:
        """When marker >= max_chars, raw truncation without marker."""
        long_text = "x" * 100
        result = mod._truncate_text_for_section(key="default", text=long_text, max_chars=18)
        # marker is "\n...[truncated]...\n" (20 chars), so with max_chars=18 the
        # function returns raw truncation
        assert len(result) <= 18
        assert "[truncated]" not in result

    def test_pytest_output_negative_head(self) -> None:
        """For pytest_output with tight limit, returns tail of text."""
        long_text = "A" * 50 + "B" * 50
        # head_chars = max_chars - max_chars//2 - len(marker)
        # With max_chars=30, marker=20, tail=15, head = 30 - 15 - 20 = -5 → negative
        result = mod._truncate_text_for_section(key="pytest_output", text=long_text, max_chars=30)
        assert len(result) <= 30

    def test_zero_max_chars(self) -> None:
        """Zero max_chars returns empty string."""
        assert mod._truncate_text_for_section(key="k", text="hello", max_chars=0) == ""

    def test_text_within_limit(self) -> None:
        """Text shorter than limit is returned as-is."""
        text = "short"
        assert mod._truncate_text_for_section(key="k", text=text, max_chars=100) == text


# ---------------------------------------------------------------------------
# _build_context_compaction_bundle - scope variants
# ---------------------------------------------------------------------------


class TestBuildContextCompactionBundle:
    """Bundle construction with different scopes."""

    def test_pytest_scope_with_stdout(self) -> None:
        """'pytest' scope collects stdout into the bundle."""
        result = mod._build_context_compaction_bundle(
            scope={"pytest"},
            snapshot=None,
            stdout="FAIL test_a",
            stderr="",
            stage_a_context="",
            project_files_context="",
            max_input_chars=0,
        )
        assert "pytest_output" in result.source_sections
        assert result.replace_pytest_output is True
        assert "FAIL test_a" in result.text

    def test_pytest_scope_with_stderr(self) -> None:
        """'pytest' scope collects stderr too."""
        result = mod._build_context_compaction_bundle(
            scope={"pytest"},
            snapshot=None,
            stdout="",
            stderr="E: x != y",
            stage_a_context="",
            project_files_context="",
            max_input_chars=0,
        )
        assert "pytest_output" in result.source_sections
        assert "E: x != y" in result.text

    def test_files_scope_with_stage_a(self) -> None:
        """'files' scope adds stage_a_context."""
        result = mod._build_context_compaction_bundle(
            scope={"files"},
            snapshot=None,
            stdout="",
            stderr="",
            stage_a_context="file content here",
            project_files_context="",
            max_input_chars=0,
        )
        assert "stage_a_context" in result.source_sections
        assert result.replace_stage_a_context is True

    def test_files_scope_with_project_files(self) -> None:
        """'files' scope adds project_files_context."""
        result = mod._build_context_compaction_bundle(
            scope={"files"},
            snapshot=None,
            stdout="",
            stderr="",
            stage_a_context="",
            project_files_context="project content",
            max_input_chars=0,
        )
        assert "project_files" in result.source_sections
        assert result.replace_project_files is True

    def test_bundle_truncated_when_exceeds_limit(self) -> None:
        """Combined bundle exceeding max_input_chars is truncated."""
        result = mod._build_context_compaction_bundle(
            scope={"pytest", "files"},
            snapshot=None,
            stdout="x" * 500,
            stderr="y" * 500,
            stage_a_context="z" * 500,
            project_files_context="w" * 500,
            max_input_chars=200,
        )
        assert len(result.text) <= 200 + 50  # allow marker overhead

    def test_empty_scope_produces_empty_text(self) -> None:
        """Empty scope with no matching sections yields empty text."""
        result = mod._build_context_compaction_bundle(
            scope=set(),
            snapshot=None,
            stdout="",
            stderr="",
            stage_a_context="",
            project_files_context="",
            max_input_chars=0,
        )
        assert result.text == ""


# ---------------------------------------------------------------------------
# _build_retry_history_section - clipping and empty
# ---------------------------------------------------------------------------


class TestBuildRetryHistorySection:
    """Retry history section construction edge cases."""

    def test_clipping_single_long_turn(self) -> None:
        """When no turns fit completely, the most recent is clipped."""
        turns = ["A" * 200]
        text, _used, count, truncated = mod._build_retry_history_section(
            prior_turns=turns,
            max_chars=50,
        )
        assert len(text) <= 50
        assert truncated is True
        assert count == 1

    def test_empty_when_all_whitespace(self) -> None:
        """All-whitespace turns yield empty result."""
        text, _used, count, _truncated = mod._build_retry_history_section(
            prior_turns=["  ", "\n", "\t"],
            max_chars=100,
        )
        assert text == ""
        assert count == 0

    def test_tiny_max_chars_returns_empty(self) -> None:
        """With max_chars=0, returns empty even with turns."""
        text, _used, _count, truncated = mod._build_retry_history_section(
            prior_turns=["hello"],
            max_chars=0,
        )
        assert text == ""
        assert truncated is True

    def test_no_turns_at_all(self) -> None:
        """Empty list of turns yields empty result."""
        text, _used, _count, _truncated = mod._build_retry_history_section(
            prior_turns=[],
            max_chars=100,
        )
        assert text == ""


# ---------------------------------------------------------------------------
# _render_retry_history_turn - failure signature
# ---------------------------------------------------------------------------


class TestRenderRetryHistoryTurn:
    """Retry history turn rendering with failure signature."""

    def test_with_failure_signature(self) -> None:
        """When summary has failure_signature, it appears in the output."""
        result = mod._render_retry_history_turn(
            attempt=1,
            note="failed",
            failure_signature="AssertionError: x != y",
            patch_meta=None,
        )
        assert "Failure signature:" in result
        assert "AssertionError" in result

    def test_without_failure_signature(self) -> None:
        """When failure_signature is empty, line is omitted."""
        result = mod._render_retry_history_turn(
            attempt=1,
            note="failed",
            failure_signature="",
            patch_meta=None,
        )
        assert "Failure signature:" not in result


# ---------------------------------------------------------------------------
# _build_retry_summary._compact - truncation
# ---------------------------------------------------------------------------


class TestBuildRetrySummaryCompact:
    """Compact truncation and _new_constraint edge cases."""

    def test_long_note_truncated(self) -> None:
        """A note longer than 220 chars is truncated in the summary."""
        summary = mod._build_retry_summary(
            attempt=1,
            note="x" * 300,
            failure_signature="",
            patch_meta=None,
        )
        new_constraint = summary["new_constraint"]
        assert "...[truncated]" in new_constraint or len(new_constraint) <= 140

    def test_verifier_rejected_no_reasons(self) -> None:
        """patch_verifier_rejected without reasons gives fallback text."""
        summary = mod._build_retry_summary(
            attempt=1,
            note="patch_verifier_rejected",
            failure_signature="",
            patch_meta={},
        )
        assert summary["new_constraint"] == "must satisfy patch verifier before apply"

    def test_verifier_rejected_with_reasons(self) -> None:
        """patch_verifier_rejected with reasons renders them."""
        summary = mod._build_retry_summary(
            attempt=1,
            note="patch_verifier_rejected",
            failure_signature="",
            patch_meta={"patch_verifier_reasons": ["score_too_low", "misses_crash"]},
        )
        assert "must satisfy verifier" in summary["new_constraint"]
        assert "score_too_low" in summary["new_constraint"]

    def test_long_failure_signature_truncated(self) -> None:
        """Failure signature > 120 chars is truncated."""
        long_sig = "Z" * 200
        summary = mod._build_retry_summary(
            attempt=1,
            note="failed",
            failure_signature=long_sig,
            patch_meta=None,
        )
        assert len(summary["failure_signature"]) <= 135  # 120 + suffix


# ---------------------------------------------------------------------------
# _prompt_safe_previous_attempt_diff - rewrite dict
# ---------------------------------------------------------------------------


class TestPromptSafePreviousAttemptDiff:
    """Key rewriting in previous attempt diffs."""

    def test_none_returns_none(self) -> None:
        assert mod._prompt_safe_previous_attempt_diff(None) is None

    def test_non_dict_returns_none(self) -> None:
        assert mod._prompt_safe_previous_attempt_diff("not a dict") is None  # type: ignore[arg-type]

    def test_hypothesis_delta_renamed(self) -> None:
        """hypothesis_delta key is renamed to diagnosis_delta."""
        diff: dict[str, Any] = {"hypothesis_delta": "some change", "other": 42}
        result = mod._prompt_safe_previous_attempt_diff(diff)
        assert result is not None
        assert "diagnosis_delta" in result
        assert "hypothesis_delta" not in result
        assert result["other"] == 42

    def test_nested_rewrite(self) -> None:
        """Nested dicts also get hypothesis_delta renamed."""
        diff: dict[str, Any] = {"inner": {"hypothesis_delta": "nested"}}
        result = mod._prompt_safe_previous_attempt_diff(diff)
        assert result is not None
        assert result["inner"]["diagnosis_delta"] == "nested"

    def test_list_values_rewritten(self) -> None:
        """Lists containing dicts are recursively rewritten."""
        diff: dict[str, Any] = {"items": [{"hypothesis_delta": "x"}]}
        result = mod._prompt_safe_previous_attempt_diff(diff)
        assert result is not None
        assert result["items"][0]["diagnosis_delta"] == "x"
