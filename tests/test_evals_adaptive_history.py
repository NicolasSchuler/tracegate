"""Tests for evals/adaptive_history.py — persistent adaptive-history helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals.adaptive_history import (
    PersistentHistoryBundle,
    PersistentHistoryLesson,
    _coerce_int,
    _dedupe_evidence_refs,
    _lesson_from_record,
    _truncate,
    load_persistent_adaptive_history,
    persist_adaptive_attempt_history,
)

# ── _truncate ──────────────────────────────────────────────────────────


class TestTruncate:
    """Exercise truncation boundaries and the small-budget fast path."""

    def test_short_string_unchanged(self) -> None:
        """Strings within the limit are returned as-is."""
        assert _truncate("hi", max_chars=10) == "hi"

    def test_max_chars_zero_returns_empty(self) -> None:
        """A zero budget drops the string entirely."""
        assert _truncate("hello world", max_chars=0) == ""

    def test_max_chars_one(self) -> None:
        """max_chars=1 → first character only."""
        assert _truncate("hello", max_chars=1) == "h"

    def test_max_chars_two(self) -> None:
        """Budgets below the ellipsis threshold keep a raw prefix."""
        assert _truncate("hello world", max_chars=2) == "he"

    def test_max_chars_three_boundary(self) -> None:
        """max_chars=3 is the exact boundary for the <=3 branch."""
        assert _truncate("hello world", max_chars=3) == "hel"

    def test_max_chars_four_uses_ellipsis(self) -> None:
        """Once the budget allows it, truncation reserves room for an ellipsis."""
        assert _truncate("hello world", max_chars=4) == "h..."

    def test_normal_truncation_ellipsis(self) -> None:
        """Normal truncation appends '...'."""
        result = _truncate("hello world", max_chars=8)
        assert result.endswith("...")
        assert len(result) <= 8

    def test_empty_input(self) -> None:
        """Empty string returns empty."""
        assert _truncate("", max_chars=5) == ""

    def test_none_coerced(self) -> None:
        """None-ish input is coerced to ''."""
        assert _truncate(None, max_chars=5) == ""  # type: ignore[arg-type]

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace is stripped before measuring."""
        assert _truncate("  hi  ", max_chars=10) == "hi"


# ── _coerce_int ────────────────────────────────────────────────────────


class TestCoerceInt:
    """Exercise integer coercion, including explicit bool rejection."""

    def test_valid_int(self) -> None:
        assert _coerce_int(42) == 42

    def test_zero(self) -> None:
        assert _coerce_int(0) == 0

    def test_negative(self) -> None:
        assert _coerce_int(-7) == -7

    @pytest.mark.parametrize("val", [True, False])
    def test_bool_returns_none(self, val: bool) -> None:
        """Bool values are rejected even though ``bool`` subclasses ``int``."""
        assert _coerce_int(val) is None

    @pytest.mark.parametrize("val", [None, "abc", 3.14, [], {}])
    def test_non_int_returns_none(self, val: Any) -> None:
        assert _coerce_int(val) is None


# ── _dedupe_evidence_refs ──────────────────────────────────────────────


class TestDedupeEvidenceRefs:
    """Exercise deduplication and truncation of evidence refs."""

    def test_basic_dedup(self) -> None:
        """Duplicate items are removed while keeping first-seen order."""
        result = _dedupe_evidence_refs(["a", "b", "a"], ["c", "b"])
        assert result == ["a", "b", "c"]

    def test_empty_strings_filtered(self) -> None:
        """Empty and whitespace-only items are skipped."""
        result = _dedupe_evidence_refs(["", " ", "x"], ["  ", "y"])
        assert result == ["x", "y"]

    def test_preserves_order(self) -> None:
        """new_ids appear before reused_ids."""
        result = _dedupe_evidence_refs(["n1", "n2"], ["r1", "r2"])
        assert result == ["n1", "n2", "r1", "r2"]

    def test_truncation_at_max(self) -> None:
        """Evidence refs are capped at the configured maximum."""
        ids = [f"id{i}" for i in range(20)]
        result = _dedupe_evidence_refs(ids, [])
        assert len(result) == 12
        assert result == ids[:12]

    def test_empty_inputs(self) -> None:
        assert _dedupe_evidence_refs([], []) == []


# ── _lesson_from_record ───────────────────────────────────────────────


class TestLessonFromRecord:
    """Exercise lesson rendering precedence across record shapes."""

    def test_patch_verifier_rejected_with_reasons(self) -> None:
        """Verifier rejection reasons outrank generic status-only output."""
        rec: dict[str, Any] = {
            "verification_result": {
                "status": "fail",
                "patch_verifier_rejected": True,
                "patch_verifier_reasons": ["reason1", "reason2"],
            },
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson.startswith("fail: reason1")

    def test_patch_verifier_rejected_no_reasons(self) -> None:
        """status + patch_verifier_rejected, no reasons → fallback."""
        rec: dict[str, Any] = {
            "verification_result": {
                "status": "fail",
                "patch_verifier_rejected": True,
            },
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert "prior verifier rejection" in lesson.lesson

    def test_status_and_hypothesis(self) -> None:
        """Status is prefixed when a hypothesis is available."""
        rec: dict[str, Any] = {
            "verification_result": {"status": "fail"},
            "hypothesis": "bad import",
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "fail: bad import"

    def test_status_and_experiment_plan(self) -> None:
        """The experiment plan is used when no hypothesis was recorded."""
        rec: dict[str, Any] = {
            "verification_result": {"status": "retry"},
            "experiment_plan": "try plan B",
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "retry: try plan B"

    def test_hypothesis_and_experiment_plan(self) -> None:
        """Without status, hypothesis and plan are composed into one lesson."""
        rec: dict[str, Any] = {
            "hypothesis": "off by one",
            "experiment_plan": "add bounds check",
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert "hypothesis=off by one" in lesson.lesson
        assert "plan=add bounds check" in lesson.lesson

    def test_hypothesis_only(self) -> None:
        """Hypothesis-only records still produce a lesson."""
        rec: dict[str, Any] = {"hypothesis": "stale cache"}
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "hypothesis=stale cache"

    def test_experiment_plan_only(self) -> None:
        """Plan-only records still produce a lesson."""
        rec: dict[str, Any] = {"experiment_plan": "increase timeout"}
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "plan=increase timeout"

    def test_change_summary_only(self) -> None:
        """Change summaries are used when richer RCA fields are absent."""
        rec: dict[str, Any] = {
            "proposed_change_summary": "Touched files: a.py",
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "Touched files: a.py"

    def test_gate_warnings_only(self) -> None:
        """Gate warnings provide a final fallback before status-only output."""
        rec: dict[str, Any] = {
            "gate_warnings": ["stall detected", "budget low"],
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "stall detected"

    def test_status_only(self) -> None:
        """Status-only records fall back to the raw status text."""
        rec: dict[str, Any] = {
            "verification_result": {"status": "unknown"},
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "unknown"

    def test_empty_record_returns_none(self) -> None:
        """Completely empty records produce no lesson."""
        assert _lesson_from_record({}, max_chars=300) is None

    def test_all_empty_strings_returns_none(self) -> None:
        """All fields present but empty → None."""
        rec: dict[str, Any] = {
            "verification_result": {"status": ""},
            "hypothesis": "",
            "experiment_plan": "",
            "proposed_change_summary": "",
            "gate_warnings": [],
        }
        assert _lesson_from_record(rec, max_chars=300) is None

    def test_attempt_coerced(self) -> None:
        """attempt field from verification_result is coerced to int."""
        rec: dict[str, Any] = {
            "verification_result": {"status": "ok", "attempt": 3},
            "hypothesis": "test",
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.attempt == 3

    def test_non_dict_verification_result(self) -> None:
        """Non-dict verification_result is treated as empty dict."""
        rec: dict[str, Any] = {
            "verification_result": "not a dict",
            "hypothesis": "fallback",
        }
        lesson = _lesson_from_record(rec, max_chars=300)
        assert lesson is not None
        assert lesson.lesson == "hypothesis=fallback"


# ── load_persistent_adaptive_history ──────────────────────────────────


class TestLoadPersistentAdaptiveHistory:
    """Exercise load_persistent_adaptive_history branches."""

    def test_max_records_zero_early_return(self, tmp_path: Path) -> None:
        """Non-positive record budgets return an empty bundle immediately."""
        bundle = load_persistent_adaptive_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="test",
            max_records=0,
        )
        assert bundle.session_id == ""
        assert bundle.records_loaded == 0
        assert bundle.lessons == []

    def test_max_lessons_zero_early_return(self, tmp_path: Path) -> None:
        """Non-positive lesson budgets return an empty bundle immediately."""
        bundle = load_persistent_adaptive_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="test",
            max_lessons=0,
        )
        assert bundle.session_id == ""
        assert bundle.lessons == []

    def test_import_failure_returns_error_bundle(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Import failures are reported through the bundle error field."""
        import builtins

        real_import = builtins.__import__

        def _failing_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
            if name == "llmdebug.rca_state":
                raise ImportError("rca_state unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _failing_import)

        # Also clear from sys.modules so it is re-imported.
        saved = sys.modules.pop("llmdebug.rca_state", None)
        try:
            bundle = load_persistent_adaptive_history(
                out_dir=tmp_path,
                snapshot=None,
                fallback_session_id="test",
            )
            assert bundle.error
            assert "ImportError" in bundle.error
        finally:
            if saved is not None:
                sys.modules["llmdebug.rca_state"] = saved

    def test_derive_session_id_raises_returns_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """History-load exceptions are surfaced as bundle errors."""
        monkeypatch.setattr(
            "evals.adaptive_history._derive_session_id",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        bundle = load_persistent_adaptive_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="test",
        )
        assert "RuntimeError" in bundle.error
        assert "boom" in bundle.error

    def test_non_dict_records_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-dict entries are ignored rather than aborting the load."""
        monkeypatch.setattr(
            "evals.adaptive_history._derive_session_id",
            lambda **kw: "sid",
        )
        fake_rca = types.ModuleType("llmdebug.rca_state")
        fake_rca.get_rca_history = lambda **kw: [  # type: ignore[attr-defined]
            "not-a-dict",
            42,
            None,
            {"hypothesis": "good one"},
        ]
        fake_rca.derive_session_id = lambda *a, **kw: "sid"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug.rca_state", fake_rca)

        bundle = load_persistent_adaptive_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="test",
        )
        # 4 raw records loaded (len of the returned list)
        assert bundle.records_loaded == 4
        # Only the dict produced a lesson
        assert len(bundle.lessons) == 1
        assert bundle.lessons[0].lesson == "hypothesis=good one"

    def test_duplicate_lessons_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Duplicate rendered lessons are deduplicated."""
        monkeypatch.setattr(
            "evals.adaptive_history._derive_session_id",
            lambda **kw: "sid",
        )
        fake_rca = types.ModuleType("llmdebug.rca_state")
        fake_rca.get_rca_history = lambda **kw: [  # type: ignore[attr-defined]
            {"hypothesis": "same"},
            {"hypothesis": "same"},
            {"hypothesis": "different"},
        ]
        fake_rca.derive_session_id = lambda *a, **kw: "sid"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug.rca_state", fake_rca)

        bundle = load_persistent_adaptive_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="test",
        )
        lessons_text = [lesson.lesson for lesson in bundle.lessons]
        assert lessons_text == ["hypothesis=same", "hypothesis=different"]

    def test_max_lessons_respected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lesson collection stops once the configured cap is reached."""
        monkeypatch.setattr(
            "evals.adaptive_history._derive_session_id",
            lambda **kw: "sid",
        )
        fake_rca = types.ModuleType("llmdebug.rca_state")
        fake_rca.get_rca_history = lambda **kw: [  # type: ignore[attr-defined]
            {"hypothesis": f"h{i}"} for i in range(10)
        ]
        fake_rca.derive_session_id = lambda *a, **kw: "sid"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug.rca_state", fake_rca)

        bundle = load_persistent_adaptive_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="test",
            max_lessons=2,
        )
        assert len(bundle.lessons) == 2

    def test_happy_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full happy path with real-looking records."""
        monkeypatch.setattr(
            "evals.adaptive_history._derive_session_id",
            lambda **kw: "sess-1",
        )
        fake_rca = types.ModuleType("llmdebug.rca_state")
        fake_rca.get_rca_history = lambda **kw: [  # type: ignore[attr-defined]
            {
                "verification_result": {"status": "fail", "attempt": 1},
                "hypothesis": "wrong import",
            },
        ]
        fake_rca.derive_session_id = lambda *a, **kw: "sess-1"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug.rca_state", fake_rca)

        bundle = load_persistent_adaptive_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="test",
        )
        assert bundle.session_id == "sess-1"
        assert bundle.records_loaded == 1
        assert len(bundle.lessons) == 1
        assert bundle.lessons[0].attempt == 1


# ── persist_adaptive_attempt_history ──────────────────────────────────


class TestPersistAdaptiveAttemptHistory:
    """Exercise persist_adaptive_attempt_history branches."""

    @pytest.fixture()
    def _mock_rca(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[MagicMock, MagicMock]:
        """Patch rca_state.advance_rca_attempt and derive_session_id."""
        advance_mock = MagicMock(return_value={})
        derive_mock = MagicMock(return_value="mock-session")
        monkeypatch.setattr(
            "evals.adaptive_history._derive_session_id",
            derive_mock,
        )
        # Patch the module-level imports used inside the function.
        fake_rca = types.ModuleType("llmdebug.rca_state")
        fake_rca.advance_rca_attempt = advance_mock  # type: ignore[attr-defined]
        fake_rca.derive_session_id = derive_mock  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug.rca_state", fake_rca)

        fake_types = types.ModuleType("llmdebug.rca_types")

        from llmdebug.rca_types import RCAAttemptRequest

        fake_types.RCAAttemptRequest = RCAAttemptRequest  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llmdebug.rca_types", fake_types)
        return advance_mock, derive_mock

    def test_gate_reason_string_sets_adaptive_stop_reason(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """String gate reasons are forwarded into verification metadata."""
        advance_mock, _ = _mock_rca
        persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={"attempt": 1, "note": "fail"},
            patch_meta={
                "adaptive_stop_trigger_reason": "budget_exceeded",
            },
            retry_plan_text="",
        )
        call_kwargs = advance_mock.call_args[1]
        vr = call_kwargs["request"].verification_result
        assert vr["adaptive_stop_reason"] == "budget_exceeded"

    def test_empty_failure_signature_defaults(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """Empty failure signatures fall back to ``unknown_failure``."""
        advance_mock, _ = _mock_rca
        persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={"attempt": 1},
            patch_meta={},
            retry_plan_text="",
        )
        call_kwargs = advance_mock.call_args[1]
        assert call_kwargs["request"].failure_signature == "unknown_failure"

    def test_success_record_action_resolve(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """success=True → action='resolve'."""
        advance_mock, _ = _mock_rca
        persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={"attempt": 2, "success": True},
            patch_meta={},
            retry_plan_text="",
        )
        call_kwargs = advance_mock.call_args[1]
        assert call_kwargs["request"].action == "resolve"

    def test_patch_verifier_reasons_forwarded(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """patch_verifier_reasons list is forwarded to verification_result."""
        advance_mock, _ = _mock_rca
        persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={"attempt": 1},
            patch_meta={
                "patch_verifier_reasons": ["lint fail", "test fail"],
            },
            retry_plan_text="",
        )
        call_kwargs = advance_mock.call_args[1]
        vr = call_kwargs["request"].verification_result
        assert vr["patch_verifier_reasons"] == ["lint fail", "test fail"]

    def test_patch_files_in_change_summary(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """patch_file_paths → proposed_change_summary."""
        advance_mock, _ = _mock_rca
        persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={"attempt": 1},
            patch_meta={"patch_file_paths": ["a.py", "b.py"]},
            retry_plan_text="",
        )
        call_kwargs = advance_mock.call_args[1]
        summary = call_kwargs["request"].proposed_change_summary
        assert "a.py" in summary
        assert "b.py" in summary

    def test_returns_session_id(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """Function returns the derived session id."""
        _, _ = _mock_rca
        sid = persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={"attempt": 1},
            patch_meta=None,
            retry_plan_text="plan",
        )
        assert sid == "mock-session"

    def test_retry_summary_hypothesis_forwarded(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """hypothesis_tested from retry_summary is forwarded."""
        advance_mock, _ = _mock_rca
        persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={
                "attempt": 1,
                "attempt_metadata": {
                    "retry_summary": {
                        "hypothesis_tested": "cache miss",
                        "failure_signature": "sig1",
                    },
                },
            },
            patch_meta={},
            retry_plan_text="",
        )
        call_kwargs = advance_mock.call_args[1]
        assert call_kwargs["request"].hypothesis == "cache miss"
        assert call_kwargs["request"].failure_signature == "sig1"

    def test_evidence_ids_deduped(
        self,
        tmp_path: Path,
        _mock_rca: tuple[MagicMock, MagicMock],
    ) -> None:
        """Evidence IDs from patch_meta are deduped."""
        advance_mock, _ = _mock_rca
        persist_adaptive_attempt_history(
            out_dir=tmp_path,
            snapshot=None,
            fallback_session_id="t",
            record={"attempt": 1},
            patch_meta={
                "evidence_ids_new": ["e1", "e2", "e1"],
                "evidence_ids_reused": ["e2", "e3"],
            },
            retry_plan_text="",
        )
        call_kwargs = advance_mock.call_args[1]
        refs = list(call_kwargs["request"].evidence_refs)
        assert refs == ["e1", "e2", "e3"]


# ── Dataclass smoke tests ────────────────────────────────────────────


class TestDataclasses:
    """Smoke-test the frozen dataclasses."""

    def test_persistent_history_lesson_frozen(self) -> None:
        lesson = PersistentHistoryLesson(source="rca", attempt=1, lesson="test")
        with pytest.raises(AttributeError):
            lesson.lesson = "changed"  # type: ignore[misc]

    def test_persistent_history_bundle_defaults(self) -> None:
        bundle = PersistentHistoryBundle(session_id="s", records_loaded=0, lessons=[])
        assert bundle.error == ""
