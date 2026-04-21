"""Tests for evals.progress_output covering uncovered branches and edge cases."""

from __future__ import annotations

import io

import pytest

from evals.progress_output import (
    ProgressPalette,
    RunProgressSummary,
    _humanize_reason,
    _looks_like_context_limit,
    _looks_like_timeout,
    _normalize_reason_token,
    _stdout_is_tty,
    build_progress_palette,
    classify_failure_reason,
    humanize_failure_reason,
    normalize_failure_reason,
)

# ---------------------------------------------------------------------------
# _stdout_is_tty edge cases (lines 35-40)
# ---------------------------------------------------------------------------


class TestStdoutIsTty:
    """Cover _stdout_is_tty when isatty is not callable or raises."""

    def test_non_callable_isatty_returns_false(self) -> None:
        """Line 36: isatty is not callable → False."""
        stream = io.StringIO()
        stream.isatty = "not-a-callable"  # type: ignore[assignment]
        assert _stdout_is_tty(stream) is False  # type: ignore[arg-type]

    def test_isatty_raises_returns_false(self) -> None:
        """Lines 39-40: isatty() raises → False."""

        class _BrokenStream:
            def isatty(self) -> bool:
                raise OSError("broken")

        assert _stdout_is_tty(_BrokenStream()) is False  # type: ignore[arg-type]

    def test_true_when_tty(self) -> None:
        class _TtyStream:
            def isatty(self) -> bool:
                return True

        assert _stdout_is_tty(_TtyStream()) is True  # type: ignore[arg-type]

    def test_false_when_not_tty(self) -> None:
        stream = io.StringIO()
        assert _stdout_is_tty(stream) is False  # type: ignore[arg-type]

    def test_defaults_to_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout", io.StringIO())
        assert _stdout_is_tty() is False


# ---------------------------------------------------------------------------
# build_progress_palette (line 71)
# ---------------------------------------------------------------------------


class TestBuildProgressPalette:
    def test_non_tty_stream_disables_palette(self) -> None:
        palette = build_progress_palette(io.StringIO())  # type: ignore[arg-type]
        assert palette.enabled is False

    def test_no_color_env_disables_palette(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _TtyStream:
            def isatty(self) -> bool:
                return True

        monkeypatch.setenv("NO_COLOR", "1")
        palette = build_progress_palette(_TtyStream())  # type: ignore[arg-type]
        assert palette.enabled is False


# ---------------------------------------------------------------------------
# _humanize_reason edge cases (lines 82-88)
# ---------------------------------------------------------------------------


class TestHumanizeReason:
    """Cover all branches of _humanize_reason."""

    def test_transport_runtime_error(self) -> None:
        """Line 83: transport_runtime_error token."""
        assert _humanize_reason("transport_runtime_error") == "transport/runtime error"

    def test_runtime_error(self) -> None:
        """Line 85: runtime_error token."""
        assert _humanize_reason("runtime_error") == "runtime error"

    def test_patch_verifier_prefix(self) -> None:
        """Line 87: patch_verifier_ prefix."""
        result = _humanize_reason("patch_verifier_syntax_error")
        assert result == "patch verifier: syntax error"

    def test_generic_token(self) -> None:
        """Line 88: generic fallback."""
        assert _humanize_reason("some_other_failure") == "some other failure"


# ---------------------------------------------------------------------------
# _looks_like_timeout (lines 92-95)
# ---------------------------------------------------------------------------


class TestLooksLikeTimeout:
    def test_timed_out_key_true(self) -> None:
        """Line 93: timed_out=True → True."""
        assert _looks_like_timeout({"timed_out": True}) is True

    def test_timed_out_key_false(self) -> None:
        assert _looks_like_timeout({"timed_out": False}) is False

    def test_error_timeout_string(self) -> None:
        assert _looks_like_timeout({"error": "timeout"}) is True

    def test_error_timed_out_string(self) -> None:
        assert _looks_like_timeout({"error": "timed_out"}) is True

    def test_no_timeout_signals(self) -> None:
        assert _looks_like_timeout({"error": "syntax_error"}) is False

    def test_empty_meta(self) -> None:
        assert _looks_like_timeout({}) is False


# ---------------------------------------------------------------------------
# _looks_like_context_limit (lines 98-106)
# ---------------------------------------------------------------------------


class TestLooksLikeContextLimit:
    @pytest.mark.parametrize(
        "key",
        ["body", "message", "error_message", "stdout", "stderr"],
    )
    def test_context_limit_markers_in_various_keys(self, key: str) -> None:
        """Lines 103-105: marker match triggers True."""
        meta = {key: "The prompt exceeds the available context size limit"}
        assert _looks_like_context_limit(meta) is True

    def test_maximum_context_length_marker(self) -> None:
        meta = {"body": "Error: maximum context length is 4096 tokens"}
        assert _looks_like_context_limit(meta) is True

    def test_prompt_too_long_marker(self) -> None:
        meta = {"error_message": "prompt is too long"}
        assert _looks_like_context_limit(meta) is True

    def test_no_match_returns_false(self) -> None:
        meta = {"body": "something unrelated", "message": "no issue"}
        assert _looks_like_context_limit(meta) is False

    def test_non_string_values_skipped(self) -> None:
        meta = {"body": 42, "message": None}
        assert _looks_like_context_limit(meta) is False

    def test_empty_meta(self) -> None:
        assert _looks_like_context_limit({}) is False


# ---------------------------------------------------------------------------
# normalize_failure_reason — all branches (lines 127-187)
# ---------------------------------------------------------------------------


class TestNormalizeFailureReason:
    def test_transport_runtime_error_detected(self) -> None:
        """Line 131: transport marker → transport_runtime_error."""
        result = normalize_failure_reason(runtime_error="Connection refused by server")
        assert result == "transport_runtime_error"

    def test_generic_runtime_error(self) -> None:
        """Line 132: non-transport runtime error."""
        result = normalize_failure_reason(runtime_error="out of memory")
        assert result == "runtime_error"

    def test_no_records_returns_failed(self) -> None:
        """Line 135: empty case_records → 'failed'."""
        assert normalize_failure_reason() == "failed"
        assert normalize_failure_reason(case_records=[]) == "failed"

    def test_timeout_from_timed_out_key(self) -> None:
        """Line 146: _looks_like_timeout from terminal record."""
        records = [{"timed_out": True}]
        assert normalize_failure_reason(case_records=records) == "timeout"

    def test_timeout_from_patch_meta(self) -> None:
        records = [{"patch_meta": {"timed_out": True}}]
        assert normalize_failure_reason(case_records=records) == "timeout"

    def test_context_limit_from_patch_meta(self) -> None:
        """Line 149: context limit detected."""
        records = [{"patch_meta": {"body": "maximum context length exceeded"}}]
        assert normalize_failure_reason(case_records=records) == "context_limit"

    def test_context_limit_from_attempt_metadata(self) -> None:
        records = [{"attempt_metadata": {"body": "prompt is too long"}}]
        assert normalize_failure_reason(case_records=records) == "context_limit"

    def test_context_limit_from_baseline_meta(self) -> None:
        records = [
            {
                "patch_meta": {"baseline_meta": {"stderr": "exceeds the available context size"}},
            }
        ]
        assert normalize_failure_reason(case_records=records) == "context_limit"

    def test_case_fail_fast_reason(self) -> None:
        records = [{"patch_meta": {"case_fail_fast_reason": "Bad Input"}}]
        result = normalize_failure_reason(case_records=records)
        assert result == "bad_input"

    def test_adaptive_stop_reason(self) -> None:
        records = [
            {
                "patch_meta": {"adaptive_progress_stop_note": "Budget Exhausted"},
            }
        ]
        result = normalize_failure_reason(case_records=records)
        assert result == "budget_exhausted"

    def test_patch_verifier_reasons(self) -> None:
        """Lines 173-176: verifier reasons list."""
        records = [{"patch_meta": {"patch_verifier_reasons": ["Syntax Error"]}}]
        result = normalize_failure_reason(case_records=records)
        assert result == "patch_verifier_syntax_error"

    def test_patch_verifier_reasons_skips_empty_items(self) -> None:
        records = [{"patch_meta": {"patch_verifier_reasons": ["", "  ", "Real Issue"]}}]
        result = normalize_failure_reason(case_records=records)
        assert result == "patch_verifier_real_issue"

    def test_error_text_from_meta(self) -> None:
        """Line 181: error text in meta dicts."""
        records = [{"patch_meta": {"error": "import failed"}}]
        result = normalize_failure_reason(case_records=records)
        assert result == "import_failed"

    def test_note_as_failure_reason(self) -> None:
        """Line 185: note field used as last-resort reason."""
        records = [{"note": "unexpected_crash"}]
        result = normalize_failure_reason(case_records=records)
        assert result == "unexpected_crash"

    def test_note_in_success_set_ignored(self) -> None:
        """Line 184: note 'success' is not treated as a failure reason."""
        records = [{"note": "success"}]
        assert normalize_failure_reason(case_records=records) == "failed"

    def test_fallback_to_failed(self) -> None:
        """Line 187: nothing matched → 'failed'."""
        records = [{"some_other_key": "value"}]
        assert normalize_failure_reason(case_records=records) == "failed"


# ---------------------------------------------------------------------------
# humanize_failure_reason (line 191)
# ---------------------------------------------------------------------------


class TestHumanizeFailureReason:
    def test_delegates_to_humanize_reason(self) -> None:
        """Line 191: public wrapper."""
        assert humanize_failure_reason("timeout") == "timeout"
        assert humanize_failure_reason("runtime_error") == "runtime error"
        assert humanize_failure_reason("transport_runtime_error") == "transport/runtime error"
        assert humanize_failure_reason("patch_verifier_bad_diff") == "patch verifier: bad diff"


# ---------------------------------------------------------------------------
# classify_failure_reason (lines 194-203)
# ---------------------------------------------------------------------------


class TestClassifyFailureReason:
    def test_returns_normalized_and_humanized(self) -> None:
        norm, human = classify_failure_reason(
            runtime_error="Connection refused",
        )
        assert norm == "transport_runtime_error"
        assert human == "transport/runtime error"

    def test_no_args_returns_failed(self) -> None:
        norm, human = classify_failure_reason()
        assert norm == "failed"
        assert human == "failed"


# ---------------------------------------------------------------------------
# RunProgressSummary (lines 206-248)
# ---------------------------------------------------------------------------


class TestRunProgressSummary:
    def test_record_runtime_error(self) -> None:
        summary = RunProgressSummary()
        summary.record_runtime_error("connection refused")
        assert summary.runtime_error_units == 1
        assert summary.failure_reasons["transport_runtime_error"] == 1

    def test_record_case_records_success(self) -> None:
        summary = RunProgressSummary()
        summary.record_case_records([{"success": True}])
        assert summary.success_units == 1
        assert summary.failed_units == 0

    def test_record_case_records_failure(self) -> None:
        summary = RunProgressSummary()
        summary.record_case_records([{"success": False, "timed_out": True}])
        assert summary.failed_units == 1
        assert summary.failure_reasons["timeout"] == 1

    def test_record_case_records_interrupted_infra(self) -> None:
        summary = RunProgressSummary()
        summary.record_case_records(
            [
                {
                    "record_type": "infra_event",
                    "success": False,
                    "patch_meta": {"infra_resume_eligible": True},
                }
            ]
        )
        assert summary.interrupted_units == 1
        assert summary.failed_units == 0

    def test_record_case_records_interrupted_resume_eligible_attempt(self) -> None:
        summary = RunProgressSummary()
        summary.record_case_records(
            [
                {
                    "record_type": "attempt",
                    "success": False,
                    "patch_meta": {"infra_resume_eligible": True},
                }
            ]
        )
        assert summary.interrupted_units == 1
        assert summary.failed_units == 0

    def test_render_lines_no_failures(self) -> None:
        summary = RunProgressSummary()
        palette = ProgressPalette(enabled=False)
        lines = summary.render_lines(skipped_units=0, palette=palette)
        assert len(lines) == 2
        assert "success=0" in lines[0]
        assert "Top failure reasons: none" in lines[1]

    def test_render_lines_with_failures(self) -> None:
        summary = RunProgressSummary()
        summary.record_case_records([{"success": False, "timed_out": True}])
        summary.record_case_records([{"success": False, "timed_out": True}])
        summary.record_case_records([{"success": False, "note": "unexpected_crash"}])
        palette = ProgressPalette(enabled=False)
        lines = summary.render_lines(skipped_units=1, palette=palette)
        assert "failed=3" in lines[0]
        assert "skipped=1" in lines[0]
        assert "Top failure reasons:" in lines[1]
        assert "timeout=2" in lines[1]

    def test_render_lines_with_palette_enabled(self) -> None:
        summary = RunProgressSummary()
        summary.success_units = 5
        summary.failed_units = 2
        summary.interrupted_units = 1
        summary.runtime_error_units = 1
        palette = ProgressPalette(enabled=True)
        lines = summary.render_lines(skipped_units=0, palette=palette)
        # ANSI codes should be present in the success count
        assert "\x1b[32m5\x1b[0m" in lines[0]


# ---------------------------------------------------------------------------
# _normalize_reason_token edge cases
# ---------------------------------------------------------------------------


class TestNormalizeReasonToken:
    def test_empty_string_returns_failed(self) -> None:
        assert _normalize_reason_token("") == "failed"

    def test_whitespace_only_returns_failed(self) -> None:
        assert _normalize_reason_token("   ") == "failed"

    def test_special_characters_normalized(self) -> None:
        assert _normalize_reason_token("foo--bar!!baz") == "foo_bar_baz"


# ---------------------------------------------------------------------------
# ProgressPalette edge cases
# ---------------------------------------------------------------------------


class TestProgressPalette:
    def test_disabled_palette_returns_plain_text(self) -> None:
        palette = ProgressPalette(enabled=False)
        assert palette.success_count(5) == "5"
        assert palette.failure_count(0) == "0"
        assert palette.warning_count(3) == "3"
        assert palette.info_count(0) == "0"

    def test_enabled_palette_zero_values_no_color(self) -> None:
        palette = ProgressPalette(enabled=True)
        assert palette.success_count(0) == "0"
        assert palette.failure_count(0) == "0"
        assert palette.warning_count(0) == "0"
        assert palette.info_count(0) == "0"

    def test_enabled_palette_positive_values_colored(self) -> None:
        palette = ProgressPalette(enabled=True)
        assert "\x1b[32m" in palette.success_count(1)
        assert "\x1b[31m" in palette.failure_count(1)
        assert "\x1b[33m" in palette.warning_count(1)
        assert "\x1b[36m" in palette.info_count(1)

    def test_wrap_empty_string_returns_empty(self) -> None:
        palette = ProgressPalette(enabled=True)
        assert palette._wrap("", "\x1b[32m") == ""
