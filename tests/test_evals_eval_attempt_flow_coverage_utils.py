"""Coverage-boost tests for evals/eval_attempt_flow.py — part 1: result/meta utilities.

Tests _result_meta_dict, _result_with_meta, _summarize_recovery_usage_meta,
_is_forced_noop_marker_diff, _build_runner_forced_attempt_diff,
_format_snapshot_index_section, _classify_patcher_request_failure,
_is_context_limit_http_error_message, _is_retryable_transport_http_status,
_is_transport_runtime_error.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals import eval_attempt_flow

# ---------------------------------------------------------------------------
# _result_meta_dict / _result_with_meta
# ---------------------------------------------------------------------------


class TestResultMetaDict:
    def test_meta_dict_returns_copy(self) -> None:
        obj = SimpleNamespace(meta={"key": "val"})
        result = eval_attempt_flow._result_meta_dict(obj)
        assert result == {"key": "val"}
        result["key"] = "changed"
        assert obj.meta["key"] == "val"

    def test_meta_non_dict_returns_empty(self) -> None:
        assert eval_attempt_flow._result_meta_dict(SimpleNamespace(meta="not_a_dict")) == {}

    def test_meta_missing_returns_empty(self) -> None:
        assert eval_attempt_flow._result_meta_dict(SimpleNamespace()) == {}


class TestResultWithMeta:
    def test_patch_result_returns_new_instance(self) -> None:
        from evals.patchers.base import PatchResult

        original = PatchResult(diff="--- a/f.py\n+++ b/f.py\n", meta={"old": True})
        updated = eval_attempt_flow._result_with_meta(original, {"new": True})
        assert isinstance(updated, PatchResult)
        assert updated.meta == {"new": True}
        assert updated.diff == original.diff

    def test_context_request_result(self) -> None:
        from evals.patchers.base import ContextRequestResult

        original = ContextRequestResult(request_json={"r": 1}, raw_response="raw", meta={})
        updated = eval_attempt_flow._result_with_meta(original, {"transport": "ok"})
        assert isinstance(updated, ContextRequestResult)
        assert updated.meta == {"transport": "ok"}

    def test_context_compaction_result(self) -> None:
        from evals.patchers.base import ContextCompactionResult

        original = ContextCompactionResult(compacted_text="short", raw_response="raw", meta={})
        updated = eval_attempt_flow._result_with_meta(original, {"compacted": True})
        assert isinstance(updated, ContextCompactionResult)

    def test_patch_plan_result(self) -> None:
        from evals.patchers.base import PatchPlanResult

        original = PatchPlanResult(plan_text="plan", raw_response="raw", meta={})
        updated = eval_attempt_flow._result_with_meta(original, {"planned": True})
        assert isinstance(updated, PatchPlanResult)

    def test_unknown_type_returns_original(self) -> None:
        original = SimpleNamespace(meta={"old": True})
        result = eval_attempt_flow._result_with_meta(original, {"new": True})
        assert result is original


# ---------------------------------------------------------------------------
# _summarize_recovery_usage_meta
# ---------------------------------------------------------------------------


class TestSummarizeRecoveryUsageMeta:
    def test_minimal_meta(self) -> None:
        assert eval_attempt_flow._summarize_recovery_usage_meta({}) == {}

    def test_with_llm_call_count_and_status(self) -> None:
        result = eval_attempt_flow._summarize_recovery_usage_meta(
            {"llm_call_count": 2, "status": 200}
        )
        assert result["llm_call_count"] == 2
        assert result["status"] == 200

    def test_with_tokens(self) -> None:
        result = eval_attempt_flow._summarize_recovery_usage_meta(
            {"tokens_prompt": 100, "tokens_completion": 50, "tokens_total": 150}
        )
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] == 100

    def test_with_error_and_body(self) -> None:
        result = eval_attempt_flow._summarize_recovery_usage_meta(
            {"error": "timeout", "body": "response body"}
        )
        assert result["error"] == "timeout"
        assert result["body"] == "response body"

    def test_status_non_int_excluded(self) -> None:
        assert "status" not in eval_attempt_flow._summarize_recovery_usage_meta({"status": "nope"})

    def test_seed_keys(self) -> None:
        result = eval_attempt_flow._summarize_recovery_usage_meta(
            {
                "api_compat_seed_requested": True,
                "api_compat_seed_acknowledged": True,
                "api_compat_seed_response": 42,
            }
        )
        assert result["api_compat_seed_requested"] is True


# ---------------------------------------------------------------------------
# _is_forced_noop_marker_diff
# ---------------------------------------------------------------------------


class TestIsForcedNoopMarkerDiff:
    def test_valid_marker_diff(self) -> None:
        diff = (
            "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
            "@@ -1 +1,2 @@\n pass\n+# llmdebug-force-attempt-1\n"
        )
        assert eval_attempt_flow._is_forced_noop_marker_diff(diff) is True

    def test_real_change_not_noop(self) -> None:
        diff = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+return 1\n"
        assert eval_attempt_flow._is_forced_noop_marker_diff(diff) is False

    def test_empty_diff(self) -> None:
        assert eval_attempt_flow._is_forced_noop_marker_diff("") is False

    def test_only_headers(self) -> None:
        diff = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n pass\n"
        assert eval_attempt_flow._is_forced_noop_marker_diff(diff) is False


# ---------------------------------------------------------------------------
# _build_runner_forced_attempt_diff
# ---------------------------------------------------------------------------


class TestBuildRunnerForcedAttemptDiff:
    def test_builds_diff(self, tmp_path: Path) -> None:
        (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        result = eval_attempt_flow._build_runner_forced_attempt_diff(
            cwd=tmp_path, file_index="buggy.py", attempt=1
        )
        assert result is not None
        assert "llmdebug-force-attempt-1" in result[1]

    def test_empty_file_index(self, tmp_path: Path) -> None:
        assert (
            eval_attempt_flow._build_runner_forced_attempt_diff(
                cwd=tmp_path, file_index="", attempt=1
            )
            is None
        )

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert (
            eval_attempt_flow._build_runner_forced_attempt_diff(
                cwd=tmp_path, file_index="nonexistent.py", attempt=1
            )
            is None
        )

    def test_marker_collision(self, tmp_path: Path) -> None:
        (tmp_path / "buggy.py").write_text(
            "def f():\n    return 1\n# llmdebug-force-attempt-1\n", encoding="utf-8"
        )
        result = eval_attempt_flow._build_runner_forced_attempt_diff(
            cwd=tmp_path, file_index="buggy.py", attempt=1
        )
        assert result is not None
        assert "llmdebug-force-attempt-1-2" in result[1]


# ---------------------------------------------------------------------------
# _format_snapshot_index_section
# ---------------------------------------------------------------------------


class TestFormatSnapshotIndexSection:
    def test_basic_snapshot(self) -> None:
        snapshot = {
            "exception": {"type": "AssertionError", "message": "assert 1 == 2"},
            "frames": [
                {
                    "file_rel": "test.py",
                    "line": 10,
                    "function": "test_add",
                    "code": "assert add(1, 1) == 3",
                }
            ],
        }
        result = eval_attempt_flow._format_snapshot_index_section(snapshot)
        assert "=== llmdebug Snapshot Index ===" in result
        assert "AssertionError" in result

    def test_empty_snapshot(self) -> None:
        result = eval_attempt_flow._format_snapshot_index_section({})
        assert "=== llmdebug Snapshot Index ===" in result


# ---------------------------------------------------------------------------
# _classify_patcher_request_failure
# ---------------------------------------------------------------------------


class TestClassifyPatcherRequestFailure:
    def test_no_error(self) -> None:
        assert eval_attempt_flow._classify_patcher_request_failure({})[0] is False

    def test_transport_error(self) -> None:
        f, k = eval_attempt_flow._classify_patcher_request_failure({"error": "request_failed"})
        assert f is True and k == "transport"

    def test_context_limit(self) -> None:
        f, k = eval_attempt_flow._classify_patcher_request_failure(
            {"error": "http_error", "body": "maximum context length"}
        )
        assert f is True and k == "context_limit"

    def test_retryable_status(self) -> None:
        f, k = eval_attempt_flow._classify_patcher_request_failure(
            {"error": "http_error", "status": 429}
        )
        assert f is True and k == "transport"

    def test_unrecognized(self) -> None:
        assert eval_attempt_flow._classify_patcher_request_failure({"error": "unknown"})[0] is False


# ---------------------------------------------------------------------------
# _is_context_limit_http_error_message / _is_retryable_transport_http_status
# ---------------------------------------------------------------------------


class TestTransportHelpers:
    def test_context_limit_message(self) -> None:
        assert (
            eval_attempt_flow._is_context_limit_http_error_message("maximum context length") is True
        )
        assert eval_attempt_flow._is_context_limit_http_error_message("") is False
        assert eval_attempt_flow._is_context_limit_http_error_message(None) is False

    def test_retryable_status(self) -> None:
        assert eval_attempt_flow._is_retryable_transport_http_status(429) is True
        assert eval_attempt_flow._is_retryable_transport_http_status(500) is True
        assert eval_attempt_flow._is_retryable_transport_http_status(200) is False
        assert eval_attempt_flow._is_retryable_transport_http_status("429") is False

    def test_transport_runtime_error(self) -> None:
        assert eval_attempt_flow._is_transport_runtime_error("Connection refused") is True
        assert eval_attempt_flow._is_transport_runtime_error("Segmentation fault") is True
        assert eval_attempt_flow._is_transport_runtime_error("") is False
        assert eval_attempt_flow._is_transport_runtime_error("KeyError") is False
