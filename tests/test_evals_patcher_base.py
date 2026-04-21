"""Tests for the base Patcher interface (evals/patchers/base.py).

Covers the _emit_progress callback machinery, default sync/async delegation
for every optional pipeline step, and the apropose_patch delegation path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals.patchers.base import (
    ContextCompactionResult,
    ContextRequestResult,
    Patcher,
    PatchResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_kwargs() -> dict[str, Any]:
    """Return the minimal keyword arguments accepted by most Patcher methods."""
    return {
        "prompt": "Fix the bug.",
        "context_dir": Path("/tmp/ctx"),
        "cwd": Path("/tmp/repo"),
    }


class _ConcretePatcher(Patcher):
    """Minimal subclass that overrides ``propose_patch`` with a fixed diff."""

    name = "concrete-stub"

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        return PatchResult(diff="diff --git stub")


# ---------------------------------------------------------------------------
# _emit_progress tests (lines 144-151, branches [145→146], [145→147])
# ---------------------------------------------------------------------------


class TestEmitProgressNoCallback:
    """When no callback is registered, _emit_progress returns immediately."""

    def test_no_callback_set(self) -> None:
        """Line 145→146: callback is None ⇒ early return, no error."""
        p = Patcher()
        # Must not raise
        p._emit_progress("test_event")
        # No callback is registered — verify no error attribute was set
        assert p._current_progress_callback() is None

    def test_no_callback_with_payload(self) -> None:
        """Same branch with a payload dict — still safe with no callback."""
        p = Patcher()
        p._emit_progress("evt", {"key": "val"})
        # No callback is registered — verify no error attribute was set
        assert p._current_progress_callback() is None


class TestEmitProgressWithCallback:
    """When a valid callback is set, it should receive event + payload."""

    def test_callback_receives_event_and_dict_payload(self) -> None:
        """Lines 144-149, branch 145→147: callback fires with copied dict."""
        received: list[tuple[str, dict[str, Any] | None]] = []

        def _cb(event: str, payload: dict[str, Any] | None) -> None:
            received.append((event, payload))

        p = Patcher()
        p.set_progress_callback(_cb)
        p._emit_progress("step_done", {"key": "val"})

        assert len(received) == 1
        assert received[0][0] == "step_done"
        assert received[0][1] == {"key": "val"}

    def test_payload_is_shallow_copied(self) -> None:
        """Line 148: dict(payload) creates a fresh copy, not the original."""
        captured: list[dict[str, Any] | None] = []

        def _cb(event: str, payload: dict[str, Any] | None) -> None:
            captured.append(payload)

        original: dict[str, Any] = {"a": 1}
        p = Patcher()
        p.set_progress_callback(_cb)
        p._emit_progress("e", original)

        assert captured[0] is not original
        assert captured[0] == original

    def test_non_dict_payload_forwarded_as_none(self) -> None:
        """Line 148: non-dict payload ⇒ callback receives None."""
        captured: list[tuple[str, dict[str, Any] | None]] = []

        def _cb(event: str, payload: dict[str, Any] | None) -> None:
            captured.append((event, payload))

        p = Patcher()
        p.set_progress_callback(_cb)
        p._emit_progress("e", None)

        assert captured[0] == ("e", None)


class TestEmitProgressCallbackRaises:
    """When the callback raises, the exception must be silently swallowed."""

    def test_exception_swallowed(self) -> None:
        """Lines 150-151: callback raises ⇒ no propagation."""

        def _bad_cb(event: str, payload: dict[str, Any] | None) -> None:
            raise RuntimeError("boom")

        p = Patcher()
        p.set_progress_callback(_bad_cb)
        # Must NOT raise
        p._emit_progress("evt", {"x": 1})

    def test_keyboard_interrupt_not_swallowed(self) -> None:
        """BaseException subclasses other than Exception still propagate."""

        def _fatal_cb(event: str, payload: dict[str, Any] | None) -> None:
            raise KeyboardInterrupt

        p = Patcher()
        p.set_progress_callback(_fatal_cb)
        with pytest.raises(KeyboardInterrupt):
            p._emit_progress("evt")


# ---------------------------------------------------------------------------
# Sync default return values
# ---------------------------------------------------------------------------


class TestSyncDefaults:
    """Default sync methods return None (compact_context, plan_patch)."""

    def test_compact_context_returns_none(self) -> None:
        """Line 229: compact_context default returns None."""
        p = Patcher()
        result = p.compact_context(
            **_dummy_kwargs(),
            context_text="some text",
            target_chars=500,
        )
        assert result is None

    def test_plan_patch_returns_none(self) -> None:
        """Line 267: plan_patch default returns None."""
        p = Patcher()
        result = p.plan_patch(**_dummy_kwargs())
        assert result is None


class TestCapabilityDefaults:
    """Default capability hooks preserve legacy method-override detection."""

    def test_base_patcher_capabilities_default_to_unsupported(self) -> None:
        p = Patcher()

        assert p.supports_request_context() is False
        assert p.supports_context_compaction() is False
        assert p.supports_strict_json_first() is False

    def test_request_context_override_implies_support(self) -> None:
        class _LegacyRequestContextPatcher(_ConcretePatcher):
            def request_context(
                self,
                *,
                prompt: str,
                context_dir: Path,
                cwd: Path,
                strict_json_first: bool = False,
            ) -> ContextRequestResult:
                del prompt, context_dir, cwd, strict_json_first
                return ContextRequestResult(request_json=None, raw_response="{}", meta={})

        p = _LegacyRequestContextPatcher()

        assert p.supports_request_context() is True
        assert p.supports_strict_json_first() is True
        assert p.supports_context_compaction() is False

    def test_compact_context_override_implies_support(self) -> None:
        class _LegacyCompactionPatcher(_ConcretePatcher):
            def compact_context(
                self,
                *,
                prompt: str,
                context_text: str,
                context_dir: Path,
                cwd: Path,
                target_chars: int,
            ) -> ContextCompactionResult:
                del prompt, context_text, context_dir, cwd, target_chars
                return ContextCompactionResult(compacted_text="summary", raw_response="summary")

        p = _LegacyCompactionPatcher()

        assert p.supports_context_compaction() is True
        assert p.supports_request_context() is False
        assert p.supports_strict_json_first() is False


# ---------------------------------------------------------------------------
# Async delegation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arequest_context_delegates_to_sync() -> None:
    """Line 194: arequest_context delegates to request_context (→ None)."""
    p = Patcher()
    result = await p.arequest_context(**_dummy_kwargs())
    assert result is None


@pytest.mark.asyncio
async def test_acompact_context_delegates_to_sync() -> None:
    """Line 241: acompact_context delegates to compact_context (→ None)."""
    p = Patcher()
    result = await p.acompact_context(
        **_dummy_kwargs(),
        context_text="large blob",
        target_chars=200,
    )
    assert result is None


@pytest.mark.asyncio
async def test_aplan_patch_delegates_to_sync() -> None:
    """Line 273: aplan_patch delegates to plan_patch (→ None)."""
    p = Patcher()
    result = await p.aplan_patch(**_dummy_kwargs())
    assert result is None


@pytest.mark.asyncio
async def test_apropose_patch_delegates_to_sync() -> None:
    """Line 301: apropose_patch delegates to propose_patch on subclass."""
    p = _ConcretePatcher()
    result = await p.apropose_patch(**_dummy_kwargs())
    assert isinstance(result, PatchResult)
    assert result.diff == "diff --git stub"


# ---------------------------------------------------------------------------
# Additional coverage for previously missing lines
# ---------------------------------------------------------------------------


class TestProbeServerReady:
    """Cover line 165: probe_server_ready default implementation."""

    def test_probe_server_ready_returns_false_and_error_meta(self) -> None:
        p = Patcher()
        ready, meta = p.probe_server_ready()
        assert ready is False
        assert "error" in meta
        assert meta["error"] == "server_ready_probe_unsupported"


class TestSupportsRequestContextWithBaseMethod:
    """Cover lines 180-184: supports_request_context when callable resolves to base method."""

    def test_non_callable_request_context_attr_returns_false(self) -> None:
        """Line 180: callable check guards against non-callable attrs."""

        class _BrokenPatcher(_ConcretePatcher):
            request_context = "not_a_callable"  # type: ignore[assignment]

        p = _BrokenPatcher()
        assert p.supports_request_context() is False

    def test_base_class_request_context_not_overridden_returns_false(self) -> None:
        """Lines 180-184: _method_is_inherited_noop catches unoverridden base method."""
        # Patcher.request_context is the base method, so support is False
        p = _ConcretePatcher()
        # _ConcretePatcher does NOT override request_context
        assert p.supports_request_context() is False


class TestSupportsContextCompactionWithBaseMethod:
    """Cover lines 196-200: supports_context_compaction when callable resolves to base method."""

    def test_non_callable_compact_context_attr_returns_false(self) -> None:
        """Line 196: callable check guards against non-callable attrs."""

        class _BrokenPatcher(_ConcretePatcher):
            compact_context = 42  # type: ignore[assignment]

        p = _BrokenPatcher()
        assert p.supports_context_compaction() is False

    def test_base_class_compact_context_not_overridden_returns_false(self) -> None:
        """Lines 196-200: _method_is_inherited_noop catches unoverridden base method."""
        p = _ConcretePatcher()
        assert p.supports_context_compaction() is False


class TestSupportsStrictJsonFirstSignatureError:
    """Cover lines 217-218: inspect.signature raises TypeError/ValueError."""

    def test_inspect_signature_raises_returns_false(self) -> None:
        """Lines 217-218: TypeError or ValueError during signature inspection → False."""
        import inspect
        import unittest.mock as mock

        class _RequestCtxPatcher(_ConcretePatcher):
            def request_context(  # type: ignore[override]
                self,
                *,
                prompt: str,
                context_dir: Path,
                cwd: Path,
                strict_json_first: bool = False,
            ) -> None:
                return None

        p = _RequestCtxPatcher()
        assert p.supports_request_context() is True

        # Simulate inspect.signature raising TypeError
        with mock.patch.object(inspect, "signature", side_effect=TypeError("cannot introspect")):
            result = p.supports_strict_json_first()
        assert result is False

    def test_inspect_signature_raises_value_error_returns_false(self) -> None:
        """Line 218: ValueError from inspect.signature → False."""
        import inspect
        import unittest.mock as mock

        class _RequestCtxPatcher2(_ConcretePatcher):
            def request_context(  # type: ignore[override]
                self,
                *,
                prompt: str,
                context_dir: Path,
                cwd: Path,
                strict_json_first: bool = False,
            ) -> None:
                return None

        p = _RequestCtxPatcher2()
        with mock.patch.object(inspect, "signature", side_effect=ValueError("no sig")):
            result = p.supports_strict_json_first()
        assert result is False


class TestSupportsStrictJsonFirstNonCallableAfterContextCheck:
    """Line 214: request_context_fn is not callable when supports_request_context() is True."""

    def test_non_callable_request_context_returns_false(self) -> None:
        """Line 214: when request_context is non-callable but supports_request_context() is True."""

        class _OverridePatcher(_ConcretePatcher):
            def supports_request_context(self) -> bool:
                return True

        p = _OverridePatcher()
        p.request_context = 42  # type: ignore[assignment]  — not callable
        assert p.supports_strict_json_first() is False
