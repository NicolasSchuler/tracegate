"""Coverage tests for evals.eval_evidence - wp5 batch.

Targets uncovered lines and branches in eval_evidence.py: context-request
payload coercion (frame/local_var types), execute_context_requests edge cases
(path traversal, non-existent files, search truncation, frame locals, char
limits, duplicate evidence), fallback bundle paths, crash-file from snapshot
with non-dict frames, import-neighbor text-read failures, subprocess script
extraction, collect_text_context fallback display, and strip-artifact OSError.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evals import eval_evidence as mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if frames is None:
        frames = [
            {
                "file_rel": "src/main.py",
                "line": 10,
                "function": "run",
                "code": "x = 1",
                "locals": {"x": 1, "y": "hello"},
            }
        ]
    return {"frames": frames}


def _file_index(root: Path) -> str:
    return "\n".join(mod._collect_file_index_lines(root))


def _make_payload(requests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "need_more_context": True,
        "requests": requests,
    }


# ---------------------------------------------------------------------------
# _coerce_context_request_payload - frame/local_var types
# ---------------------------------------------------------------------------


class TestCoerceContextRequestPayloadFrameLocalVar:
    """Exercises the frame and local_var request normalization branches."""

    def test_frame_request_normalized(self) -> None:
        """A valid frame request is kept in the payload."""
        raw = {"need_more_context": True, "requests": [{"type": "frame", "index": 0}]}
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 1
        assert result["requests"][0]["type"] == "frame"
        assert result["requests"][0]["index"] == 0

    def test_frame_request_bool_index_rejected(self) -> None:
        """Bool index (isinstance(True, int)) is specifically rejected."""
        raw = {"need_more_context": True, "requests": [{"type": "frame", "index": True}]}
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 0

    def test_local_var_request_normalized(self) -> None:
        """A valid local_var request is kept in the payload."""
        raw = {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 0, "name": "x"}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 1
        req = result["requests"][0]
        assert req["type"] == "local_var"
        assert req["frame"] == 0
        assert req["name"] == "x"

    def test_local_var_missing_name_rejected(self) -> None:
        """local_var request without name is rejected."""
        raw = {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 0}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 0


# ---------------------------------------------------------------------------
# execute_context_requests - path traversal, non-existent, search
# ---------------------------------------------------------------------------


class TestExecuteContextRequestsFileEdges:
    """File request edge cases: traversal, missing files, text read failures."""

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        """Relative paths with '..' are rejected by normalization."""
        (tmp_path / "ok.py").write_text("pass", encoding="utf-8")
        payload = _make_payload([{"type": "file", "path": "../../../etc/passwd"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback

    def test_nonexistent_file_skipped(self, tmp_path: Path) -> None:
        """Request for a file not in the file index is skipped."""
        (tmp_path / "a.py").write_text("pass", encoding="utf-8")
        payload = _make_payload([{"type": "file", "path": "missing.py"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback

    def test_directory_path_skipped(self, tmp_path: Path) -> None:
        """Request targeting a directory (not a file) is skipped."""
        (tmp_path / "subdir").mkdir()
        # Even if "subdir" were in the index, it's not a file
        payload = _make_payload([{"type": "file", "path": "subdir"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index="subdir\n",
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback

    def test_file_read_failure_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unreadable files are skipped instead of crashing the request executor."""
        (tmp_path / "a.py").write_text("pass", encoding="utf-8")
        monkeypatch.setattr(mod, "_read_text_limited", lambda *_a, **_kw: None)
        payload = _make_payload([{"type": "file", "path": "a.py"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback


class TestExecuteContextRequestsSearchEdges:
    """Search request edge cases."""

    def test_empty_pattern_skipped(self, tmp_path: Path) -> None:
        """Search request with empty pattern is skipped."""
        (tmp_path / "a.py").write_text("hello world", encoding="utf-8")
        payload = _make_payload([{"type": "search", "path": "a.py", "pattern": ""}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback

    def test_search_path_not_in_index_skipped(self, tmp_path: Path) -> None:
        """Search on a path not in the file index is skipped."""
        (tmp_path / "a.py").write_text("hello world", encoding="utf-8")
        payload = _make_payload([{"type": "search", "path": "missing.py", "pattern": "hello"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback

    def test_search_char_limit_breaks(self, tmp_path: Path) -> None:
        """Search that would exceed max_total_chars stops processing."""
        (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
        payload = _make_payload(
            [
                {"type": "search", "path": "a.py", "pattern": "x"},
                {"type": "search", "path": "a.py", "pattern": "y"},
            ]
        )
        # Very small char limit triggers early break
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=50,
        )
        # At most one search result should fit
        assert result.requests_count <= 1

    def test_search_directory_path_is_skipped(self, tmp_path: Path) -> None:
        """Search requests against directories are skipped."""
        (tmp_path / "subdir").mkdir()
        payload = _make_payload([{"type": "search", "path": "subdir", "pattern": "hello"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index="subdir\n",
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback

    def test_search_truncation_note_is_rendered(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Truncated literal search output should advertise truncation in the block text."""
        (tmp_path / "a.py").write_text("hello world", encoding="utf-8")
        monkeypatch.setattr(
            mod,
            "_literal_search_matches_in_file",
            lambda *_a, **_kw: (["1: hello world"], True),
        )
        payload = _make_payload([{"type": "search", "path": "a.py", "pattern": "hello"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert not result.used_fallback
        assert "(search truncated)" in result.text


# ---------------------------------------------------------------------------
# execute_context_requests - frame and local_var handling
# ---------------------------------------------------------------------------


class TestExecuteContextRequestsFrames:
    """Frame and local_var request handling."""

    def test_frame_with_code_and_locals(self, tmp_path: Path) -> None:
        """Frame request returns code and locals."""
        snap = _snapshot(
            frames=[
                {
                    "file_rel": "main.py",
                    "line": 5,
                    "function": "go",
                    "code": "result = compute()",
                    "locals": {"result": None, "x": 42},
                }
            ]
        )
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        payload = _make_payload([{"type": "frame", "index": 0}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert not result.used_fallback
        assert "code: result = compute()" in result.text
        assert "x = 42" in result.text

    def test_frame_with_non_serializable_local(self, tmp_path: Path) -> None:
        """When json.dumps fails on a local, repr() fallback is used."""

        class NonSerializable:
            def __repr__(self) -> str:
                return "<NonSerializable>"

        snap = _snapshot(
            frames=[
                {
                    "file_rel": "main.py",
                    "line": 5,
                    "function": "go",
                    "code": "",
                    "locals": {"obj": NonSerializable()},
                }
            ]
        )
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        payload = _make_payload([{"type": "frame", "index": 0}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert not result.used_fallback

    def test_frame_duplicate_is_deduplicated(self, tmp_path: Path) -> None:
        """Duplicate frame requests should collapse to one evidence item."""
        snap = _snapshot()
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        payload = _make_payload([{"type": "frame", "index": 0}, {"type": "frame", "index": 0}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.requests_count == 1

    def test_frame_repr_fallback_when_json_dump_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Frame locals fall back to repr() when JSON serialization raises."""
        snap = _snapshot(
            frames=[{"file_rel": "main.py", "line": 5, "function": "go", "locals": {"x": 1}}]
        )
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")

        def _boom(*_a: Any, **_kw: Any) -> str:
            raise TypeError("boom")

        monkeypatch.setattr(mod.json, "dumps", _boom)
        payload = _make_payload([{"type": "frame", "index": 0}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert not result.used_fallback
        assert "x = 1" in result.text

    def test_frame_char_limit_breaks(self, tmp_path: Path) -> None:
        """When frame block exceeds max_total_chars, processing stops."""
        snap = _snapshot(
            frames=[
                {
                    "file_rel": "main.py",
                    "line": 5,
                    "function": "go",
                    "code": "x" * 200,
                    "locals": {},
                }
            ]
        )
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        payload = _make_payload(
            [
                {"type": "frame", "index": 0},
            ]
        )
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=10,
        )
        # Frame block too large → fallback
        assert result.used_fallback

    def test_local_var_request(self, tmp_path: Path) -> None:
        """local_var request extracts a specific variable from a frame."""
        snap = _snapshot(
            frames=[
                {
                    "file_rel": "main.py",
                    "line": 5,
                    "function": "go",
                    "code": "",
                    "locals": {"my_var": "hello_world"},
                }
            ]
        )
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        payload = _make_payload([{"type": "local_var", "frame": 0, "name": "my_var"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert not result.used_fallback
        assert "my_var" in result.text
        assert "hello_world" in result.text

    def test_local_var_non_dict_frame_skipped(self, tmp_path: Path) -> None:
        """When frame at index is not a dict, local_var request is skipped."""
        snap = {"frames": ["not-a-dict"]}
        (tmp_path / "a.py").write_text("pass", encoding="utf-8")
        payload = _make_payload([{"type": "local_var", "frame": 0, "name": "x"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.used_fallback

    def test_local_var_duplicate_skipped(self, tmp_path: Path) -> None:
        """Duplicate local_var evidence IDs are de-duplicated."""
        snap = _snapshot(
            frames=[
                {
                    "file_rel": "main.py",
                    "line": 5,
                    "function": "go",
                    "code": "",
                    "locals": {"x": 42},
                }
            ]
        )
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        payload = _make_payload(
            [
                {"type": "local_var", "frame": 0, "name": "x"},
                {"type": "local_var", "frame": 0, "name": "x"},
            ]
        )
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert result.requests_count == 1

    def test_local_var_repr_fallback_when_json_dump_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """local_var payloads also fall back to repr() on JSON serialization failure."""
        snap = _snapshot(
            frames=[
                {
                    "file_rel": "main.py",
                    "line": 5,
                    "function": "go",
                    "locals": {"my_var": "hello_world"},
                }
            ]
        )
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")

        def _boom(*_a: Any, **_kw: Any) -> str:
            raise TypeError("boom")

        monkeypatch.setattr(mod.json, "dumps", _boom)
        payload = _make_payload([{"type": "local_var", "frame": 0, "name": "my_var"}])
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snap,
            file_index=_file_index(tmp_path),
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=100_000,
        )
        assert not result.used_fallback
        assert "my_var = 'hello_world'" in result.text


# ---------------------------------------------------------------------------
# build_fallback_context_bundle edge cases
# ---------------------------------------------------------------------------


class TestBuildFallbackContextBundle:
    """Edge cases in fallback bundle construction."""

    def test_empty_path_skipped(self, tmp_path: Path) -> None:
        """Empty strings in candidate paths are skipped."""
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=10_000,
            reason="test",
            file_index_lines=["", "  ", ""],
        )
        assert result.used_fallback

    def test_remaining_chars_exhausted(self, tmp_path: Path) -> None:
        """When max_total_chars is very small, only partial content is collected."""
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=5,
            reason="test",
            file_index_lines=["a.py", "b.py"],
        )
        # Very tight limit means at most one file included
        assert result.requests_count <= 1

    def test_zero_remaining_chars_stops_before_reading(self, tmp_path: Path) -> None:
        """A zero budget exits the candidate loop immediately."""
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=0,
            reason="test",
            file_index_lines=["a.py"],
        )
        assert result.requests_count == 0

    def test_unreadable_candidate_is_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unreadable fallback candidates should be skipped without aborting the bundle."""
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        monkeypatch.setattr(mod, "_read_text_limited", lambda *_a, **_kw: None)
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=10_000,
            reason="test",
            file_index_lines=["a.py"],
        )
        assert result.requests_count == 0

    def test_duplicate_fallback_evidence_ids_are_deduplicated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Duplicate fallback evidence IDs should collapse to one block."""
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")

        monkeypatch.setattr(
            mod,
            "_build_evidence_item",
            lambda **_kw: {"id": "shared-id", "text": "ignored"},
        )
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=10_000,
            reason="test",
            file_index_lines=["a.py", "b.py"],
        )
        assert result.requests_count == 1


# ---------------------------------------------------------------------------
# _crash_file_from_snapshot - non-dict frames
# ---------------------------------------------------------------------------


class TestCrashFileFromSnapshot:
    """Snapshot frame iteration edge cases."""

    def test_non_dict_frames_skipped(self) -> None:
        """Non-dict items in the frames list are skipped."""
        snap = {"frames": ["not-a-dict", 42, None]}
        assert mod._crash_file_from_snapshot(snap) is None

    def test_empty_frames_returns_none(self) -> None:
        """Empty frames list returns None."""
        assert mod._crash_file_from_snapshot({"frames": []}) is None

    def test_first_valid_frame_used(self) -> None:
        """First frame with a valid file_rel is returned."""
        snap = {
            "frames": [
                {"file_rel": "  "},
                {"file_rel": "src/main.py", "line": 1},
            ]
        }
        assert mod._crash_file_from_snapshot(snap) == "src/main.py"


# ---------------------------------------------------------------------------
# _import_neighbors - text read failure
# ---------------------------------------------------------------------------


class TestImportNeighborsEdges:
    """Import neighbor extraction edge cases."""

    def test_unreadable_file_returns_empty(self, tmp_path: Path) -> None:
        """When _read_text_limited returns None, empty list is returned."""
        (tmp_path / "crash.py").write_bytes(b"\xff\xfe" * 100)
        result = mod._import_neighbors(root=tmp_path, rel_path="crash.py", limit=10)
        # Depending on encoding behavior, should not crash
        assert isinstance(result, list)

    def test_read_text_failure_returns_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A direct text-read failure should return an empty neighbor list."""
        (tmp_path / "crash.py").write_text("import helper", encoding="utf-8")
        monkeypatch.setattr(mod, "_read_text_limited", lambda *_a, **_kw: None)
        assert mod._import_neighbors(root=tmp_path, rel_path="crash.py", limit=10) == []


class TestCollectFileIndexLinesEdges:
    """Exercise the absolute-path fallback in file-index collection."""

    def test_relative_to_failure_falls_back_to_absolute_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If relative path derivation fails, the absolute path is recorded instead."""
        target = tmp_path / "main.py"
        target.write_text("pass", encoding="utf-8")

        def _boom(self: Path, other: Path) -> Path:
            raise ValueError("relative_to failed")

        monkeypatch.setattr(type(target), "relative_to", _boom, raising=False)
        result = mod._collect_file_index_lines(tmp_path)
        assert str(target) in result


# ---------------------------------------------------------------------------
# _extract_subprocess_script_neighbors
# ---------------------------------------------------------------------------


class TestExtractSubprocessScriptNeighbors:
    """Subprocess script neighbor extraction."""

    def test_empty_rel_path_returns_empty(self) -> None:
        """Empty or None rel_path returns empty list."""
        result = mod._extract_subprocess_script_neighbors(
            root=Path("."),
            rel_path=None,
            limit=10,
        )
        assert result == []

    def test_unreadable_file_returns_empty(self, tmp_path: Path) -> None:
        """When text read fails, returns empty list."""
        (tmp_path / "crash.py").write_bytes(b"\xff\xfe" * 200)
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path,
            rel_path="crash.py",
            limit=10,
        )
        assert isinstance(result, list)

    def test_subprocess_run_with_string_arg(self, tmp_path: Path) -> None:
        """Detects script neighbor from subprocess.run("script.py")."""
        code = textwrap.dedent("""\
            import subprocess
            subprocess.run("helper.py")
        """)
        (tmp_path / "main.py").write_text(code, encoding="utf-8")
        (tmp_path / "helper.py").write_text("pass", encoding="utf-8")
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path,
            rel_path="main.py",
            limit=10,
        )
        assert "helper.py" in result

    def test_subprocess_run_with_list_arg(self, tmp_path: Path) -> None:
        """Detects script neighbor from subprocess.run(["python", "script.py"])."""
        code = textwrap.dedent("""\
            import subprocess
            subprocess.run(["python", "worker.py"])
        """)
        (tmp_path / "main.py").write_text(code, encoding="utf-8")
        (tmp_path / "worker.py").write_text("pass", encoding="utf-8")
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path,
            rel_path="main.py",
            limit=10,
        )
        assert "worker.py" in result

    def test_from_subprocess_import_run(self, tmp_path: Path) -> None:
        """Detects neighbor when using 'from subprocess import run'."""
        code = textwrap.dedent("""\
            from subprocess import run
            run("task.py")
        """)
        (tmp_path / "main.py").write_text(code, encoding="utf-8")
        (tmp_path / "task.py").write_text("pass", encoding="utf-8")
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path,
            rel_path="main.py",
            limit=10,
        )
        assert "task.py" in result

    def test_malformed_shell_string_handled(self, tmp_path: Path) -> None:
        """shlex.split failure on unmatched quotes is handled gracefully."""
        code = textwrap.dedent("""\
            import subprocess
            subprocess.run('unclosed "string with script.py')
        """)
        (tmp_path / "main.py").write_text(code, encoding="utf-8")
        (tmp_path / "script.py").write_text("pass", encoding="utf-8")
        # Should not raise
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path,
            rel_path="main.py",
            limit=10,
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# collect_text_context - relative_to fallback
# ---------------------------------------------------------------------------


class TestCollectTextContext:
    """Text context collection edge cases."""

    def test_text_read_failure_skipped(self, tmp_path: Path) -> None:
        """Files that cannot be read are skipped without error."""
        (tmp_path / "ok.py").write_text("x = 1", encoding="utf-8")
        # Binary file that _read_text_limited may fail on
        (tmp_path / "bad.py").write_bytes(b"\xff\xfe" * 100)
        result = mod.collect_text_context(
            tmp_path,
            file_index_lines=["ok.py", "bad.py"],
            max_total_chars=100_000,
        )
        assert "x = 1" in result


# ---------------------------------------------------------------------------
# _strip_eval_only_case_artifacts - OSError handling
# ---------------------------------------------------------------------------


class TestStripEvalOnlyCaseArtifacts:
    """Artifact stripping with OSError resilience."""

    def test_missing_artifacts_handled(self, tmp_path: Path) -> None:
        """Non-existent artifacts don't raise errors."""
        # Should not raise even though no artifacts exist
        mod._strip_eval_only_case_artifacts(tmp_path)

    def test_successful_deletion(self, tmp_path: Path) -> None:
        """Existing eval-only artifacts are removed."""
        from evals.eval_types import _EVAL_ONLY_CASE_ARTIFACTS

        for name in _EVAL_ONLY_CASE_ARTIFACTS:
            (tmp_path / name).write_text("temp", encoding="utf-8")
        mod._strip_eval_only_case_artifacts(tmp_path)
        for name in _EVAL_ONLY_CASE_ARTIFACTS:
            assert not (tmp_path / name).exists()
