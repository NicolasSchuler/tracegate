"""Tests for DebugSession schema helpers."""

from __future__ import annotations

from typing import Any

import pytest

from llmdebug._debug_session import (
    DEBUG_SESSION_KIND,
    DEBUG_SESSION_SCHEMA_VERSION,
    build_debug_session,
    coerce_crash_frame_index,
    is_debug_session_payload,
    normalize_snapshot_view,
    split_extra_context,
)


def test_split_extra_context_none() -> None:
    context, trace_payload, eval_payload = split_extra_context(None)
    assert context == {}
    assert trace_payload is None
    assert eval_payload is None


def test_split_extra_context_extracts_trace_and_eval() -> None:
    extra: dict[str, Any] = {
        "hook": "sys.excepthook",
        "trace": {"spans": [1]},
        "eval_outcome": {"status": "failed"},
    }
    context, trace_payload, eval_payload = split_extra_context(extra)
    assert context == {"hook": "sys.excepthook"}
    assert trace_payload == {"spans": [1]}
    assert eval_payload == {"status": "failed"}
    # Ensure source dict is not mutated
    assert "trace" in extra
    assert "eval_outcome" in extra


def test_split_extra_context_keeps_eval_outcome_when_eval_is_present() -> None:
    """CAP-19: "eval" wins, and "eval_outcome" stays in the context instead of vanishing."""
    extra: dict[str, Any] = {"eval": {"status": "passed"}, "eval_outcome": {"status": "failed"}}

    context, _, eval_payload = split_extra_context(extra)

    assert eval_payload == {"status": "passed"}
    assert context == {"eval_outcome": {"status": "failed"}}


def test_split_extra_context_keeps_explicit_none_eval() -> None:
    context, _, eval_payload = split_extra_context({"eval": None, "eval_outcome": {"x": 1}})

    assert eval_payload is None
    assert context == {"eval_outcome": {"x": 1}}


def test_build_debug_session_minimal() -> None:
    payload = build_debug_session(
        name="unit_test",
        timestamp_utc="2026-02-10T12:00:00Z",
        llmdebug_version="2.1.0",
        snapshot={"exception": {"type": "ValueError"}, "frames": []},
    )
    assert payload["schema_version"] == "2.1"
    assert payload["kind"] == "llmdebug.debug_session"
    assert payload["session"]["name"] == "unit_test"
    assert payload["snapshot"]["exception"]["type"] == "ValueError"
    assert "context" not in payload
    assert "trace" not in payload
    assert "eval" not in payload


def test_is_debug_session_payload() -> None:
    assert not is_debug_session_payload({})
    assert not is_debug_session_payload({"session": {}, "snapshot": {}})
    assert not is_debug_session_payload(
        {"kind": "other.kind", "session": {"name": "x"}, "snapshot": {"frames": []}}
    )
    assert not is_debug_session_payload({"session": {"name": "x"}, "snapshot": {"frames": []}})
    assert is_debug_session_payload(
        {"kind": "llmdebug.debug_session", "session": {"name": "x"}, "snapshot": {"frames": []}}
    )
    # Kind is optional for backward compatibility
    assert is_debug_session_payload(
        {"schema_version": "2.0", "session": {"name": "x"}, "snapshot": {"frames": []}}
    )


def test_normalize_snapshot_view_passthrough_for_legacy() -> None:
    legacy = {"schema_version": "1.0", "name": "legacy", "exception": {"type": "E"}, "frames": []}
    out = normalize_snapshot_view(legacy)
    assert out is legacy


def test_normalize_snapshot_view_envelope_context_overrides_snapshot() -> None:
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "session": {"name": "case", "timestamp_utc": "2026-02-10T12:00:00Z"},
        "snapshot": {"exception": {"type": "E"}, "frames": [], "hook": "from_snapshot"},
        "context": {"hook": "from_context"},
    }
    out = normalize_snapshot_view(payload)
    assert out["name"] == "case"
    assert out["timestamp_utc"] == "2026-02-10T12:00:00Z"
    assert out["hook"] == "from_context"


def test_normalize_snapshot_view_fallbacks_and_extras() -> None:
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "name": "top_name",
        "timestamp_utc": "2026-02-10T12:34:00Z",
        "llmdebug_version": "2.1.0",
        "session": {},  # Intentionally partial
        "snapshot": {"exception": {"type": "TypeError"}, "frames": []},
        "trace": {"spans": [1, 2]},
        "eval": {"status": "failed"},
        "custom_key": {"x": 1},
    }
    out = normalize_snapshot_view(payload)
    assert out["name"] == "top_name"
    assert out["timestamp_utc"] == "2026-02-10T12:34:00Z"
    assert out["llmdebug_version"] == "2.1.0"
    assert out["trace"] == {"spans": [1, 2]}
    assert out["eval"] == {"status": "failed"}
    assert out["custom_key"] == {"x": 1}


def test_coerce_crash_frame_index_handles_invalid_values() -> None:
    snapshot = {
        "frames": [
            {"function": "crash"},
            {"function": "caller"},
        ],
        "crash_frame_index": "abc",
    }
    assert coerce_crash_frame_index(snapshot) == 0


def test_coerce_crash_frame_index_falls_back_to_first_dict_frame() -> None:
    snapshot = {
        "frames": [
            "bad-frame",
            {"function": "good-frame"},
            {"function": "other-frame"},
        ],
        "crash_frame_index": 0,
    }
    assert coerce_crash_frame_index(snapshot) == 1


def test_coerce_crash_frame_index_defaults_when_frames_missing() -> None:
    snapshot = {
        "frames": None,
        "crash_frame_index": 3,
    }
    assert coerce_crash_frame_index(snapshot) == 0


def test_coerce_crash_frame_index_defaults_when_no_dict_frame_exists() -> None:
    snapshot = {
        "frames": ["bad-frame", 3, None],
        "crash_frame_index": 1,
    }
    assert coerce_crash_frame_index(snapshot) == 0


# ---------------------------------------------------------------------------
# build_debug_session — optional fields
# ---------------------------------------------------------------------------


def test_build_debug_session_with_context() -> None:
    payload = build_debug_session(
        name="ctx_test",
        timestamp_utc="2026-01-01T00:00:00Z",
        llmdebug_version="2.0.0",
        snapshot={"exception": {"type": "E"}, "frames": []},
        context={"hook": "sys.excepthook"},
    )
    assert payload["context"] == {"hook": "sys.excepthook"}


def test_build_debug_session_with_trace_and_eval() -> None:
    payload = build_debug_session(
        name="full_test",
        timestamp_utc="2026-01-01T00:00:00Z",
        llmdebug_version="2.0.0",
        snapshot={"exception": {"type": "E"}, "frames": []},
        trace_payload={"spans": []},
        eval_payload={"status": "pass"},
    )
    assert payload["trace"] == {"spans": []}
    assert payload["eval"] == {"status": "pass"}


def test_build_debug_session_empty_context_omitted() -> None:
    payload = build_debug_session(
        name="no_ctx",
        timestamp_utc="2026-01-01T00:00:00Z",
        llmdebug_version="2.0.0",
        snapshot={"exception": {"type": "E"}, "frames": []},
        context={},
    )
    assert "context" not in payload


# ---------------------------------------------------------------------------
# is_debug_session_payload — non-dict input (line 68)
# ---------------------------------------------------------------------------


def test_is_debug_session_payload_rejects_non_dict() -> None:
    assert not is_debug_session_payload("not-a-dict")  # type: ignore[arg-type]
    assert not is_debug_session_payload(None)  # type: ignore[arg-type]
    assert not is_debug_session_payload(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalize_snapshot_view — branch coverage for session/data fallbacks
# ---------------------------------------------------------------------------


def test_normalize_snapshot_view_no_schema_version() -> None:
    """schema_version absent → not added to out."""
    payload = {
        "kind": "llmdebug.debug_session",
        "session": {"name": "x"},
        "snapshot": {"frames": []},
    }
    out = normalize_snapshot_view(payload)
    assert "schema_version" not in out
    assert out["name"] == "x"


def test_normalize_snapshot_view_session_name_fallback_to_top_level() -> None:
    """Session dict has no 'name' but top-level 'name' exists → use it."""
    payload = {
        "kind": "llmdebug.debug_session",
        "name": "top_level_name",
        "session": {},
        "snapshot": {"frames": []},
    }
    out = normalize_snapshot_view(payload)
    assert out["name"] == "top_level_name"


def test_normalize_snapshot_view_session_timestamp_fallback_to_top_level() -> None:
    """Session dict has no 'timestamp_utc' but top-level key exists → use it."""
    payload = {
        "kind": "llmdebug.debug_session",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "session": {"name": "y"},
        "snapshot": {"frames": []},
    }
    out = normalize_snapshot_view(payload)
    assert out["timestamp_utc"] == "2026-01-01T00:00:00Z"


def test_normalize_snapshot_view_non_dict_session_fallbacks() -> None:
    """Empty session dict with no name/timestamp → absent in output."""
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "session": {},
        "snapshot": {"frames": []},
    }
    out = normalize_snapshot_view(payload)
    assert "name" not in out
    assert "timestamp_utc" not in out


def test_normalize_snapshot_view_llmdebug_version_from_session() -> None:
    """llmdebug_version in session dict → propagated to out."""
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "session": {"name": "v", "llmdebug_version": "3.0.0"},
        "snapshot": {"frames": []},
    }
    out = normalize_snapshot_view(payload)
    assert out["llmdebug_version"] == "3.0.0"


def test_normalize_snapshot_view_llmdebug_version_fallback_from_data() -> None:
    """llmdebug_version absent from session dict → fall back to top-level data key."""
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "llmdebug_version": "2.5.0",
        "session": {"name": "v"},
        "snapshot": {"frames": []},
    }
    out = normalize_snapshot_view(payload)
    assert out["llmdebug_version"] == "2.5.0"


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 65, 83, 115-118, 132, 149
# ---------------------------------------------------------------------------


def test_is_debug_session_payload_backward_compat_no_kind_schema_2() -> None:
    """Line 65/78-80: backward-compatible detection with schema_version='2.0' and no kind."""
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "session": {"name": "compat"},
        "snapshot": {"frames": []},
    }
    assert is_debug_session_payload(payload) is True


def test_is_debug_session_payload_schema_version_not_2() -> None:
    """Backward-compat path: schema_version doesn't start with '2' → False."""
    payload: dict[str, Any] = {
        "schema_version": "1.5",
        "session": {"name": "old"},
        "snapshot": {"frames": []},
    }
    assert is_debug_session_payload(payload) is False


def test_normalize_snapshot_view_session_not_dict_fallback() -> None:
    """Lines 114-118: the name/timestamp else branch for a session that is not a dict.

    The branch is hit when session passes is_debug_session_payload but then the
    code's else branch (session not isinstance dict) is taken for name/timestamp.

    Note: is_debug_session_payload requires session to be a dict, so the only way
    to reach lines 114-118 is if session is a dict (passes the guard) but then
    becomes something else. Since that's structurally impossible via normal flow,
    we test a close approximation: session is a dict with no name/timestamp,
    and top-level data has name/timestamp.
    """
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "name": "fallback_name",
        "timestamp_utc": "2026-06-01T00:00:00Z",
        "session": {},
        "snapshot": {"frames": []},
    }
    out = normalize_snapshot_view(payload)
    assert out["name"] == "fallback_name"
    assert out["timestamp_utc"] == "2026-06-01T00:00:00Z"


def test_coerce_crash_frame_index_none_crash_frame_index() -> None:
    """Line 132/162: crash_frame_index that raises TypeError on int()."""
    snapshot: dict[str, Any] = {
        "frames": [{"function": "f"}],
        "crash_frame_index": None,
    }
    # None → int(None) raises TypeError → fallback to 0
    assert coerce_crash_frame_index(snapshot) == 0


def test_coerce_crash_frame_index_non_dict_frame_at_stored_idx() -> None:
    """Line 165-170: stored index is valid int but points to non-dict frame → scan."""
    snapshot: dict[str, Any] = {
        "frames": ["bad-string", "also-bad", {"function": "good"}],
        "crash_frame_index": 0,
    }
    assert coerce_crash_frame_index(snapshot) == 2


def test_coerce_crash_frame_index_out_of_range_scans_for_dict() -> None:
    """Index out of range → scan for first dict frame."""
    snapshot: dict[str, Any] = {
        "frames": [42, {"function": "first_dict"}, {"function": "second"}],
        "crash_frame_index": 99,
    }
    assert coerce_crash_frame_index(snapshot) == 1


# ---------------------------------------------------------------------------
# Legacy schema handling — coverage for lines 115-118, 120->124, 132
# ---------------------------------------------------------------------------


class TestLegacySchemaHandling:
    """Exercise normalize_snapshot_view branches that handle legacy/malformed payloads.

    Lines 115-118 and 132 are the ``else`` branches when ``session`` is not a dict
    inside ``normalize_snapshot_view``.  Since ``is_debug_session_payload`` requires
    ``isinstance(session, dict)``, these branches are unreachable via the public API
    unless we monkeypatch the guard.
    """

    def test_no_session_dict_name_and_timestamp_from_top_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lines 115-118: session is not a dict → name/timestamp taken from data."""
        monkeypatch.setattr("llmdebug._debug_session.is_debug_session_payload", lambda _data: True)
        data: dict[str, Any] = {
            "schema_version": "2.0",
            "name": "legacy_name",
            "timestamp_utc": "2025-01-01T00:00:00Z",
            "session": "not-a-dict",
            "snapshot": {"frames": []},
        }
        out = normalize_snapshot_view(data)
        assert out["name"] == "legacy_name"
        assert out["timestamp_utc"] == "2025-01-01T00:00:00Z"

    def test_no_session_dict_missing_name_and_timestamp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lines 115-118: session is not a dict AND no top-level name/timestamp."""
        monkeypatch.setattr("llmdebug._debug_session.is_debug_session_payload", lambda _data: True)
        data: dict[str, Any] = {
            "schema_version": "2.0",
            "session": 42,
            "snapshot": {"frames": []},
        }
        out = normalize_snapshot_view(data)
        assert "name" not in out
        assert "timestamp_utc" not in out

    def test_no_session_dict_llmdebug_version_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Line 132: session not a dict → llmdebug_version from top-level data."""
        monkeypatch.setattr("llmdebug._debug_session.is_debug_session_payload", lambda _data: True)
        data: dict[str, Any] = {
            "schema_version": "2.0",
            "llmdebug_version": "1.0.0",
            "session": None,
            "snapshot": {"frames": []},
        }
        out = normalize_snapshot_view(data)
        assert out["llmdebug_version"] == "1.0.0"

    def test_no_session_dict_no_llmdebug_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Line 132: session not a dict AND no llmdebug_version anywhere."""
        monkeypatch.setattr("llmdebug._debug_session.is_debug_session_payload", lambda _data: True)
        data: dict[str, Any] = {
            "schema_version": "2.0",
            "session": [],
            "snapshot": {"frames": []},
        }
        out = normalize_snapshot_view(data)
        assert "llmdebug_version" not in out

    def test_snapshot_not_dict_skips_update(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch 120->124: snapshot is not a dict → out.update(snapshot) is skipped."""
        monkeypatch.setattr("llmdebug._debug_session.is_debug_session_payload", lambda _data: True)
        data: dict[str, Any] = {
            "schema_version": "2.0",
            "session": {"name": "s", "timestamp_utc": "t", "llmdebug_version": "v"},
            "snapshot": "not-a-dict",
        }
        out = normalize_snapshot_view(data)
        assert out["name"] == "s"
        # snapshot content was NOT merged since it's not a dict
        assert "frames" not in out

    def test_context_not_dict_skips_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch around line 124: context present but not a dict → no update."""
        monkeypatch.setattr("llmdebug._debug_session.is_debug_session_payload", lambda _data: True)
        data: dict[str, Any] = {
            "schema_version": "2.0",
            "session": {"name": "s"},
            "snapshot": {"frames": []},
            "context": "not-a-dict",
        }
        out = normalize_snapshot_view(data)
        assert out["name"] == "s"
        assert "context" not in out  # not merged since not a dict

    def test_extra_keys_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 141-144: non-reserved keys propagated to output."""
        monkeypatch.setattr("llmdebug._debug_session.is_debug_session_payload", lambda _data: True)
        data: dict[str, Any] = {
            "session": {"name": "x"},
            "snapshot": {"frames": []},
            "custom_field": "value",
            "another": 42,
        }
        out = normalize_snapshot_view(data)
        assert out["custom_field"] == "value"
        assert out["another"] == 42


# ---------------------------------------------------------------------------
# Module-level coverage — force reload so coverage traces import-time lines
# ---------------------------------------------------------------------------


def test_module_level_lines_covered_via_reload() -> None:
    """Force-reload the module so coverage sees import-time statements.

    Lines 7-15, 31, 65, 83, 149 are module-level imports, constants, and
    ``def`` statements that execute at import time.  When the conftest chain
    imports the package before ``pytest-cov`` starts tracing, those lines are
    missed.  A ``reload`` re-executes them under the active tracer.
    """
    import importlib

    import llmdebug._debug_session as mod

    importlib.reload(mod)

    # Sanity-check the reloaded module exposes the expected public API.
    assert mod.DEBUG_SESSION_SCHEMA_VERSION == "2.1"
    assert mod.DEBUG_SESSION_KIND == "llmdebug.debug_session"
    assert callable(mod.split_extra_context)
    assert callable(mod.build_debug_session)
    assert callable(mod.is_debug_session_payload)
    assert callable(mod.normalize_snapshot_view)
    assert callable(mod.coerce_crash_frame_index)
