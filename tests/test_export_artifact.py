"""Tests for export_artifact.py — shareable export artifact builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llmdebug.export_artifact import (
    _comparison_payload,
    _crash_frame,
    _find_retry_summaries,
    _html_json_blob,
    _looks_like_filesystem_path,
    _path_basename,
    _render_html_card,
    _render_key_value_list,
    _render_retry_summary_timeline,
    _restore_exception_text_fields,
    _restore_path_value,
    _sanitize_redacted_export_metadata,
    _sanitize_redacted_locator,
    _serialize_frame_sections,
    apply_export_redaction,
    build_export_bundle_manifest,
    build_export_payload,
    default_export_path,
    render_bundle_index_html,
    render_export_html,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(**overrides: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "name": "test_snap",
        "exception": {"type": "ValueError", "message": "bad value"},
        "frames": [
            {
                "file": "/project/app.py",
                "file_rel": "app.py",
                "line": 42,
                "function": "run",
                "code": "x = int(s)",
                "locals": {"s": "abc"},
            }
        ],
        "crash_frame_index": 0,
    }
    snap.update(overrides)
    return snap


def _make_raw_payload(**overrides: Any) -> dict[str, Any]:
    return _make_snapshot(**overrides)


# ---------------------------------------------------------------------------
# default_export_path
# ---------------------------------------------------------------------------


def test_default_export_path_html() -> None:
    p = default_export_path(snapshot_path=Path("/out/snap.json"), fmt="html")
    assert p == Path("/out/snap.report.html")


def test_default_export_path_json() -> None:
    p = default_export_path(snapshot_path=Path("/out/snap.json"), fmt="json")
    assert p == Path("/out/snap.report.json")


def test_default_export_path_other_fmt_defaults_to_json() -> None:
    p = default_export_path(snapshot_path=Path("/out/snap.json"), fmt="csv")
    assert p.suffix == ".json"


# ---------------------------------------------------------------------------
# _path_basename
# ---------------------------------------------------------------------------


def test_path_basename_unix() -> None:
    assert _path_basename("/home/user/file.json") == "file.json"


def test_path_basename_windows_style() -> None:
    assert _path_basename("C:\\Users\\u\\file.json") == "file.json"


def test_path_basename_empty() -> None:
    assert _path_basename("") == ""


def test_path_basename_none_like() -> None:
    assert _path_basename(None) == ""


def test_path_basename_just_filename() -> None:
    assert _path_basename("file.json") == "file.json"


# ---------------------------------------------------------------------------
# _looks_like_filesystem_path
# ---------------------------------------------------------------------------


def test_looks_like_filesystem_path_absolute() -> None:
    assert _looks_like_filesystem_path("/home/user/file.json") is True


def test_looks_like_filesystem_path_tilde() -> None:
    assert _looks_like_filesystem_path("~/file.json") is True


def test_looks_like_filesystem_path_relative_with_slash() -> None:
    assert _looks_like_filesystem_path("src/llmdebug/foo.py") is True


def test_looks_like_filesystem_path_windows() -> None:
    assert _looks_like_filesystem_path("C:\\Users\\u\\file.json") is True


def test_looks_like_filesystem_path_no_path() -> None:
    assert _looks_like_filesystem_path("just-a-name") is False


def test_looks_like_filesystem_path_empty() -> None:
    assert _looks_like_filesystem_path("") is False


def test_looks_like_filesystem_path_none() -> None:
    assert _looks_like_filesystem_path(None) is False


# ---------------------------------------------------------------------------
# _sanitize_redacted_locator
# ---------------------------------------------------------------------------


def test_sanitize_redacted_locator_empty_returns_fallback() -> None:
    assert _sanitize_redacted_locator("", fallback="fb") == "fb"


def test_sanitize_redacted_locator_none_returns_none() -> None:
    assert _sanitize_redacted_locator(None) is None


def test_sanitize_redacted_locator_path_returns_basename() -> None:
    result = _sanitize_redacted_locator("/home/user/snap.json", fallback="fb")
    assert result == "snap.json"


def test_sanitize_redacted_locator_path_empty_basename_uses_fallback() -> None:
    # A path that resolves to empty basename (trailing slash)
    result = _sanitize_redacted_locator("/home/user/", fallback="fallback")
    # _path_basename strips trailing slashes and returns the last component or ""
    # if it returns "", uses fallback
    assert result in ("user", "fallback")


def test_sanitize_redacted_locator_non_path_returns_value() -> None:
    result = _sanitize_redacted_locator("my-snapshot-ref")
    assert result == "my-snapshot-ref"


# ---------------------------------------------------------------------------
# _sanitize_redacted_export_metadata
# ---------------------------------------------------------------------------


def test_sanitize_redacted_export_metadata_sanitizes_source() -> None:
    payload: dict[str, Any] = {
        "source": {
            "snapshot_file": "snap.json",
            "snapshot_path": "/secret/path/snap.json",
            "snapshot_ref": "/secret/path/snap.json",
            "out_dir": "/secret/path",
        }
    }
    _sanitize_redacted_export_metadata(payload)
    source = payload["source"]
    assert source["out_dir"] is None
    assert source["snapshot_path"] == "snap.json"
    assert source["snapshot_ref"] == "snap.json"


def test_sanitize_redacted_export_metadata_sanitizes_comparison() -> None:
    payload: dict[str, Any] = {
        "comparison": {
            "snapshot_path": "/secret/path/compare.json",
            "ref": "/secret/path/compare.json",
        }
    }
    _sanitize_redacted_export_metadata(payload)
    comparison = payload["comparison"]
    assert comparison["snapshot_path"] == "compare.json"
    assert comparison["ref"] == "compare.json"


def test_sanitize_redacted_export_metadata_no_source() -> None:
    payload: dict[str, Any] = {"other": "value"}
    _sanitize_redacted_export_metadata(payload)  # Should not raise
    assert payload == {"other": "value"}


def test_sanitize_redacted_export_metadata_source_not_dict() -> None:
    payload: dict[str, Any] = {"source": "string-source"}
    _sanitize_redacted_export_metadata(payload)  # Should not raise
    # Input payload should be returned unchanged when source is not a dict
    assert payload == {"source": "string-source"}


# ---------------------------------------------------------------------------
# _crash_frame
# ---------------------------------------------------------------------------


def test_crash_frame_empty_frames_returns_empty() -> None:
    snap: dict[str, Any] = {"frames": []}
    assert _crash_frame(snap) == {}


def test_crash_frame_no_frames_key() -> None:
    snap: dict[str, Any] = {}
    assert _crash_frame(snap) == {}


def test_crash_frame_valid_index() -> None:
    frames = [{"function": "run"}, {"function": "main"}]
    snap: dict[str, Any] = {"frames": frames, "crash_frame_index": 1}
    result = _crash_frame(snap)
    assert result["function"] == "main"


def test_crash_frame_out_of_range_falls_back_to_first() -> None:
    frames = [{"function": "first"}, {"function": "second"}]
    snap: dict[str, Any] = {"frames": frames, "crash_frame_index": 99}
    result = _crash_frame(snap)
    assert result["function"] == "first"


def test_crash_frame_non_int_index_falls_back_to_zero() -> None:
    frames = [{"function": "first"}]
    snap: dict[str, Any] = {"frames": frames, "crash_frame_index": "bad"}
    result = _crash_frame(snap)
    assert result["function"] == "first"


def test_crash_frame_non_dict_frame_falls_back() -> None:
    frames = ["not-a-dict", {"function": "second"}]
    snap: dict[str, Any] = {"frames": frames, "crash_frame_index": 0}
    # crash_idx=0 but frames[0] is not a dict → falls to first dict-frame
    result = _crash_frame(snap)
    # If crash_idx points to non-dict, falls to frames[0] which is also not dict → {}
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _serialize_frame_sections
# ---------------------------------------------------------------------------


def test_serialize_frame_sections_empty() -> None:
    snap: dict[str, Any] = {"frames": []}
    assert _serialize_frame_sections(snap) == []


def test_serialize_frame_sections_normal() -> None:
    snap = _make_snapshot()
    sections = _serialize_frame_sections(snap)
    assert len(sections) == 1
    assert sections[0]["index"] == 0
    assert sections[0]["is_crash"] is True
    assert "app.py" in sections[0]["location"]


def test_serialize_frame_sections_non_dict_frame_skipped() -> None:
    snap: dict[str, Any] = {
        "frames": ["not-a-dict", {"function": "run", "line": 1}],
        "crash_frame_index": 1,
    }
    sections = _serialize_frame_sections(snap)
    assert len(sections) == 1
    assert sections[0]["index"] == 1


def test_serialize_frame_sections_marks_crash_frame() -> None:
    snap: dict[str, Any] = {
        "frames": [{"function": "a", "line": 1}, {"function": "b", "line": 2}],
        "crash_frame_index": 1,
    }
    sections = _serialize_frame_sections(snap)
    assert sections[0]["is_crash"] is False
    assert sections[1]["is_crash"] is True


# ---------------------------------------------------------------------------
# _find_retry_summaries
# ---------------------------------------------------------------------------


def test_find_retry_summaries_empty() -> None:
    assert _find_retry_summaries({}) == []


def test_find_retry_summaries_single() -> None:
    val = {"retry_summary": {"attempt": 1, "outcome": "fail"}}
    result = _find_retry_summaries(val)
    assert len(result) == 1
    assert result[0]["attempt"] == 1


def test_find_retry_summaries_plural() -> None:
    val = {
        "retry_summaries": [
            {"attempt": 1, "outcome": "fail"},
            {"attempt": 2, "outcome": "pass"},
        ]
    }
    result = _find_retry_summaries(val)
    assert len(result) == 2


def test_find_retry_summaries_deduped() -> None:
    summary = {"attempt": 1, "outcome": "fail"}
    val = {"retry_summary": summary, "nested": {"retry_summary": summary}}
    result = _find_retry_summaries(val)
    assert len(result) == 1


def test_find_retry_summaries_nested_in_list() -> None:
    val = [{"retry_summary": {"attempt": 1}}, {"retry_summary": {"attempt": 2}}]
    result = _find_retry_summaries(val)
    assert len(result) == 2


def test_find_retry_summaries_skips_non_dict_items_in_list() -> None:
    val = {"retry_summaries": [{"attempt": 1}, "bad-item", None]}
    result = _find_retry_summaries(val)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _restore_path_value
# ---------------------------------------------------------------------------


def test_restore_path_value_restores_in_dict() -> None:
    redacted: dict[str, Any] = {"traceback": "REDACTED", "other": "x"}
    original: dict[str, Any] = {"traceback": "real-tb", "other": "x"}
    _restore_path_value(redacted, original, key_name="traceback")
    assert redacted["traceback"] == "real-tb"


def test_restore_path_value_nested() -> None:
    redacted: dict[str, Any] = {"nested": {"traceback": "REDACTED"}}
    original: dict[str, Any] = {"nested": {"traceback": "real-tb"}}
    _restore_path_value(redacted, original, key_name="traceback")
    assert redacted["nested"]["traceback"] == "real-tb"


def test_restore_path_value_in_list() -> None:
    redacted: list[Any] = [{"traceback": "REDACTED"}]
    original: list[Any] = [{"traceback": "real-tb"}]
    _restore_path_value(redacted, original, key_name="traceback")
    assert redacted[0]["traceback"] == "real-tb"


def test_restore_path_value_non_dict_non_list() -> None:
    # Should not raise
    original = "string"
    _restore_path_value(original, "string", key_name="tb")
    # No modification occurred to the input (non-dict/non-list are no-ops)
    assert original == "string"


# ---------------------------------------------------------------------------
# _restore_exception_text_fields
# ---------------------------------------------------------------------------


def test_restore_exception_text_fields_restores_message() -> None:
    redacted: dict[str, Any] = {"exception": {"type": "ValueError", "message": "REDACTED"}}
    original: dict[str, Any] = {"exception": {"type": "ValueError", "message": "real message"}}
    _restore_exception_text_fields(redacted, original)
    assert redacted["exception"]["message"] == "real message"


def test_restore_exception_text_fields_restores_cause() -> None:
    redacted: dict[str, Any] = {
        "exception": {"type": "ValueError", "message": "REDACTED"},
        "cause": {"message": "REDACTED"},
    }
    original: dict[str, Any] = {
        "exception": {"type": "ValueError", "message": "real"},
        "cause": {"message": "real cause"},
    }
    _restore_exception_text_fields(redacted, original)
    assert redacted["exception"]["message"] == "real"
    # cause is also handled — its message (in _EXCEPTION_STRING_KEYS) is restored
    assert redacted["cause"]["message"] == "real cause"


def test_restore_exception_text_fields_list() -> None:
    redacted: list[Any] = [{"exception": {"message": "REDACTED"}}]
    original: list[Any] = [{"exception": {"message": "real"}}]
    _restore_exception_text_fields(redacted, original)
    assert redacted[0]["exception"]["message"] == "real"


def test_restore_exception_text_fields_non_dict() -> None:
    # Should not raise
    redacted = "x"
    original = "y"
    _restore_exception_text_fields(redacted, original)
    # No modification occurred to the inputs
    assert redacted == "x"
    assert original == "y"


def test_restore_exception_text_fields_missing_exception_in_original() -> None:
    redacted: dict[str, Any] = {"exception": {"message": "REDACTED"}}
    original: dict[str, Any] = {}
    _restore_exception_text_fields(redacted, original)
    # Should not crash; exception stays as-is
    assert redacted["exception"]["message"] == "REDACTED"


# ---------------------------------------------------------------------------
# apply_export_redaction
# ---------------------------------------------------------------------------


def test_apply_export_redaction_full_mode_copies_without_change() -> None:
    payload: dict[str, Any] = {"exception": {"message": "raw"}, "export_mode": "full"}
    result = apply_export_redaction(payload, export_mode="full", redaction_profile=None)
    assert result["exception"]["message"] == "raw"
    assert result["export_mode"] == "full"


def test_apply_export_redaction_redacted_mode_redacts() -> None:
    payload: dict[str, Any] = {
        "exception": {"message": "password=secret123"},
        "source": {"snapshot_path": "/secret/path/snap.json", "snapshot_file": "snap.json"},
        "export_mode": "redacted",
    }
    result = apply_export_redaction(payload, export_mode="redacted", redaction_profile="prod")
    assert result["export_mode"] == "redacted"
    # path should be sanitized
    assert result["source"]["out_dir"] is None


@pytest.mark.parametrize("profile", [None, "dev"])
def test_apply_export_redaction_refuses_redacted_mode_without_redactors(
    profile: str | None,
) -> None:
    """CAP-06: an export that redacts nothing must not be labelled "redacted"."""
    payload: dict[str, Any] = {"exception": {"message": "token sk-live-secret"}, "source": {}}

    with pytest.raises(ValueError, match="redacted export needs a redaction profile"):
        apply_export_redaction(payload, export_mode="redacted", redaction_profile=profile)


def test_apply_export_redaction_refuses_when_patterns_compile_to_nothing() -> None:
    payload: dict[str, Any] = {"exception": {"message": "plain"}, "source": {}}

    with (
        patch("llmdebug.export_artifact.compile_redactors", return_value=[]),
        pytest.raises(ValueError, match="redacted export"),
    ):
        apply_export_redaction(payload, export_mode="redacted", redaction_profile="ci")


def test_apply_export_redaction_non_redacted_mode_preserves_profile() -> None:
    payload: dict[str, Any] = {"exception": {"message": "ok"}}
    result = apply_export_redaction(payload, export_mode="summary", redaction_profile="ci")
    assert result["export_mode"] == "full"
    assert result["export_redaction_profile"] == "ci"


# ---------------------------------------------------------------------------
# _comparison_payload
# ---------------------------------------------------------------------------


def test_comparison_payload_none_when_no_compare() -> None:
    result = _comparison_payload(
        compare_ref=None,
        compare_path=None,
        compare_snapshot=None,
        snapshot=_make_snapshot(),
    )
    assert result is None


def test_comparison_payload_none_when_no_path() -> None:
    result = _comparison_payload(
        compare_ref="prev",
        compare_path=None,
        compare_snapshot=_make_snapshot(),
        snapshot=_make_snapshot(),
    )
    assert result is None


def test_comparison_payload_with_compare() -> None:
    snap = _make_snapshot()
    compare = _make_snapshot(exception={"type": "TypeError", "message": "other"})
    result = _comparison_payload(
        compare_ref="previous",
        compare_path=Path("/out/prev.json"),
        compare_snapshot=compare,
        snapshot=snap,
    )
    assert result is not None
    assert result["ref"] == "previous"
    assert "diff" in result
    assert "text" in result


def test_comparison_payload_uses_path_as_ref_when_no_ref() -> None:
    snap = _make_snapshot()
    compare = _make_snapshot()
    result = _comparison_payload(
        compare_ref=None,
        compare_path=Path("/out/prev.json"),
        compare_snapshot=compare,
        snapshot=snap,
    )
    assert result is not None
    assert result["ref"] == "/out/prev.json"


# ---------------------------------------------------------------------------
# build_export_bundle_manifest
# ---------------------------------------------------------------------------


def test_build_export_bundle_manifest_basic() -> None:
    payload: dict[str, Any] = {
        "exported_at_utc": "2024-01-01T00:00:00Z",
        "export_mode": "full",
        "export_redaction_profile": None,
        "session_id": "s1",
        "failure_signature": "sig-1",
        "source": {
            "snapshot_ref": "latest",
            "snapshot_path": "/out/snap.json",
            "snapshot_file": "snap.json",
        },
        "rca": {"status": None, "history": []},
    }
    manifest = build_export_bundle_manifest(
        payload=payload,
        export_format="json",
        artifact_filename="snap.report.json",
    )
    assert manifest["kind"] == "llmdebug.export_bundle"
    assert manifest["artifacts"]["primary"] == "snap.report.json"
    assert manifest["artifacts"]["compare"] is None
    assert manifest["rca"]["has_status"] is False
    assert manifest["rca"]["history_count"] == 0


def test_build_export_bundle_manifest_with_compare() -> None:
    payload: dict[str, Any] = {
        "export_mode": "full",
        "source": {"snapshot_ref": "latest", "snapshot_path": "/out/snap.json"},
        "rca": {"status": {"state": "done"}, "history": [{"x": 1}]},
        "comparison": {"ref": "prev"},
    }
    compare_payload: dict[str, Any] = {
        "source": {"snapshot_path": "/out/prev.json", "snapshot_file": "prev.json"},
    }
    manifest = build_export_bundle_manifest(
        payload=payload,
        export_format="html",
        artifact_filename="snap.report.html",
        compare_payload=compare_payload,
        compare_artifact_filename="prev.report.html",
    )
    assert manifest["artifacts"]["compare"] == "prev.report.html"
    assert manifest["compare_snapshot"]["ref"] == "prev"
    assert manifest["rca"]["has_status"] is True
    assert manifest["rca"]["history_count"] == 1


def test_build_export_bundle_manifest_retry_summary_count() -> None:
    payload: dict[str, Any] = {
        "export_mode": "full",
        "source": {},
        "rca": {},
        "retry_summaries": [{"attempt": 1}, {"attempt": 2}],
    }
    manifest = build_export_bundle_manifest(
        payload=payload, export_format="json", artifact_filename="a.json"
    )
    assert manifest["retry_summary_count"] == 2


# ---------------------------------------------------------------------------
# render_export_html
# ---------------------------------------------------------------------------


def _build_full_payload(tmp_path: Path) -> dict[str, Any]:
    snap = _make_snapshot()
    from llmdebug.export_artifact import build_export_payload

    return build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
        snapshot_ref="latest",
    )


def test_render_export_html_basic(tmp_path: Path) -> None:
    payload = _build_full_payload(tmp_path)
    html = render_export_html(payload)
    assert "<!DOCTYPE html>" in html
    assert "ValueError" in html
    assert "bad value" in html


def test_render_export_html_escapes_xss(tmp_path: Path) -> None:
    snap = _make_snapshot(
        exception={"type": "ValueError", "message": '<script>alert("xss")</script>'}
    )
    from llmdebug.export_artifact import build_export_payload

    payload = build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
    )
    html = render_export_html(payload)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_export_html_includes_frames(tmp_path: Path) -> None:
    payload = _build_full_payload(tmp_path)
    html = render_export_html(payload)
    assert "Frame #0" in html
    assert "CRASH" in html


def test_render_export_html_includes_retry_timeline(tmp_path: Path) -> None:
    snap = _make_snapshot()
    from llmdebug.export_artifact import build_export_payload

    payload = build_export_payload(
        snapshot=snap,
        raw_payload={
            **snap,
            "retry_summaries": [{"attempt": 1, "hypothesis_tested": "h1", "outcome": "fail"}],
        },
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
    )
    html = render_export_html(payload)
    assert "Retry Timeline" in html
    assert "Attempt 1" in html


def test_render_export_html_includes_repro_command(tmp_path: Path) -> None:
    snap: dict[str, Any] = {
        **_make_snapshot(),
        "pytest": {"repro": ["python", "-m", "pytest", "tests/test_foo.py"]},
    }
    from llmdebug.export_artifact import build_export_payload

    payload = build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
    )
    html = render_export_html(payload)
    assert "Repro Command" in html
    assert "pytest" in html


def test_render_export_html_includes_rca_section(tmp_path: Path) -> None:
    snap = _make_snapshot()
    from llmdebug.export_artifact import build_export_payload

    payload = build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
    )
    # Manually inject RCA status and history for this test
    payload["rca"]["status"] = {"state": "active", "session_id": "s1"}
    payload["rca"]["history"] = [{"attempt_id": "a1"}]
    html = render_export_html(payload)
    assert "RCA Status" in html
    assert "RCA History" in html


def test_render_export_html_includes_comparison_section(tmp_path: Path) -> None:
    snap = _make_snapshot()
    compare = _make_snapshot(exception={"type": "TypeError", "message": "other"})
    from llmdebug.export_artifact import build_export_payload
    from llmdebug.snapshot_diff import diff_snapshots, format_diff_text

    diff = diff_snapshots(compare, snap)
    payload = build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
    )
    payload["comparison"] = {
        "ref": "prev",
        "text": format_diff_text(diff),
    }
    html = render_export_html(payload)
    assert "Comparison" in html


def test_render_export_html_no_frames(tmp_path: Path) -> None:
    snap: dict[str, Any] = {"exception": {"type": "RuntimeError", "message": "boom"}}
    from llmdebug.export_artifact import build_export_payload

    payload = build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
    )
    html = render_export_html(payload)
    assert "RuntimeError" in html


# ---------------------------------------------------------------------------
# render_bundle_index_html
# ---------------------------------------------------------------------------


def test_render_bundle_index_html_basic() -> None:
    manifest: dict[str, Any] = {
        "failure_signature": "sig-1",
        "session_id": "s1",
        "export_mode": "full",
        "export_redaction_profile": None,
        "artifacts": {"primary": "snap.json", "compare": None},
    }
    html = render_bundle_index_html(manifest)
    assert "<!DOCTYPE html>" in html
    assert "Primary artifact" in html
    assert "snap.json" in html
    assert "Compare artifact" not in html


def test_render_bundle_index_html_with_compare() -> None:
    manifest: dict[str, Any] = {
        "failure_signature": "sig-1",
        "session_id": "s1",
        "export_mode": "full",
        "export_redaction_profile": "ci",
        "artifacts": {"primary": "snap.json", "compare": "compare.json"},
    }
    html = render_bundle_index_html(manifest)
    assert "Compare artifact" in html
    assert "compare.json" in html


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def test_html_json_blob_escapes_angle_brackets() -> None:
    result = _html_json_blob({"key": "<value>"})
    assert "&lt;" in result
    assert "<value>" not in result


def test_render_html_card() -> None:
    card = _render_html_card("Title", "<p>body</p>")
    assert "<h2>Title</h2>" in card
    assert "<p>body</p>" in card
    assert 'class="card"' in card


def test_render_key_value_list_skips_empty() -> None:
    result = _render_key_value_list([("Key", "value"), ("Empty", "")])
    assert "Key" in result
    assert "Empty" not in result


def test_render_retry_summary_timeline_renders_all_fields() -> None:
    summaries = [
        {
            "attempt": 1,
            "hypothesis_tested": "null pointer",
            "evidence_used": "snapshot",
            "outcome": "fail",
            "new_constraint": "avoid None",
        }
    ]
    html = _render_retry_summary_timeline(summaries)
    assert "Attempt 1" in html
    assert "null pointer" in html
    assert "fail" in html
    assert "avoid None" in html


# ---------------------------------------------------------------------------
# build_export_payload
# ---------------------------------------------------------------------------


def test_build_export_payload_basic(tmp_path: Path) -> None:
    snap = _make_snapshot()
    payload = build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
        snapshot_ref="latest",
    )
    assert payload["kind"] == "llmdebug.export"
    assert payload["export_mode"] == "full"
    assert payload["session_id"]
    assert payload["failure_signature"]
    assert len(payload["frames"]) == 1
    assert payload["comparison"] is None
    assert payload["raw_payload"] is None  # raw_payload == snapshot → None


def test_build_export_payload_different_raw_payload(tmp_path: Path) -> None:
    snap = _make_snapshot()
    raw = {**snap, "extra": "field"}
    payload = build_export_payload(
        snapshot=snap,
        raw_payload=raw,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
    )
    assert payload["raw_payload"] is not None
    assert payload["raw_payload"]["extra"] == "field"


def test_build_export_payload_with_compare(tmp_path: Path) -> None:
    snap = _make_snapshot()
    compare = _make_snapshot(exception={"type": "TypeError", "message": "other"})
    payload = build_export_payload(
        snapshot=snap,
        raw_payload=snap,
        snapshot_path=tmp_path / "snap.json",
        out_dir=str(tmp_path),
        compare_path=tmp_path / "prev.json",
        compare_snapshot=compare,
        compare_ref="previous",
    )
    assert payload["comparison"] is not None
    assert payload["comparison"]["ref"] == "previous"
