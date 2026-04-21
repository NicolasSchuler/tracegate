"""Tests for evals.eval_evidence - evidence items, context requests, and search utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals import eval_evidence as mod
from evals.stage_a_contract import normalize_stage_a_request_payload

# ---------------------------------------------------------------------------
# _annotate_evidence_text
# ---------------------------------------------------------------------------


class TestAnnotateEvidenceText:
    """Tests for the _annotate_evidence_text helper."""

    def test_empty_body_returns_header_only(self) -> None:
        """When text body is empty, only the header line is returned."""
        result = mod._annotate_evidence_text(
            evidence_id="e_abc", kind="file", locator="foo.py", text=""
        )
        assert result == "[e_abc] file:foo.py\n"

    def test_nonempty_body_appended(self) -> None:
        """When text is present, header + body are returned."""
        result = mod._annotate_evidence_text(
            evidence_id="e_abc", kind="file", locator="foo.py", text="hello"
        )
        assert result == "[e_abc] file:foo.py\nhello\n"


# ---------------------------------------------------------------------------
# _render_evidence_items
# ---------------------------------------------------------------------------


class TestRenderEvidenceItems:
    """Tests for _render_evidence_items."""

    def test_empty_list(self) -> None:
        """Empty input yields empty string."""
        assert mod._render_evidence_items([]) == ""

    def test_items_with_empty_text_skipped(self) -> None:
        """Items with falsy text are skipped."""
        items: list[dict[str, Any]] = [{"text": ""}, {"text": None}]
        assert mod._render_evidence_items(items) == ""

    def test_single_item(self) -> None:
        """Single item renders correctly."""
        items = [{"text": "hello world"}]
        assert mod._render_evidence_items(items) == "hello world\n"

    def test_multiple_items_joined(self) -> None:
        """Multiple items are joined with double newlines."""
        items = [{"text": "alpha"}, {"text": "beta"}]
        result = mod._render_evidence_items(items)
        assert result == "alpha\n\nbeta\n"


# ---------------------------------------------------------------------------
# _render_evidence_index
# ---------------------------------------------------------------------------


class TestRenderEvidenceIndex:
    """Tests for _render_evidence_index."""

    def test_empty_items(self) -> None:
        """Empty list yields empty string."""
        assert mod._render_evidence_index([]) == ""

    def test_non_dict_items_skipped(self) -> None:
        """Non-dict items are silently skipped."""
        items: list[Any] = ["not-a-dict", 42]
        assert mod._render_evidence_index(items) == ""

    def test_missing_id_skipped(self) -> None:
        """Items without an id are skipped."""
        items: list[dict[str, Any]] = [{"type": "file", "locator": "x.py"}]
        assert mod._render_evidence_index(items) == ""

    def test_item_with_locator_and_chars(self) -> None:
        """Item with locator and positive chars renders correctly."""
        items: list[dict[str, Any]] = [
            {"id": "e_1", "type": "file", "locator": "foo.py", "chars": 100}
        ]
        result = mod._render_evidence_index(items)
        assert result == "- e_1: file:foo.py (100 chars)\n"

    def test_item_without_locator(self) -> None:
        """Item without locator omits the locator portion."""
        items: list[dict[str, Any]] = [{"id": "e_2", "type": "search"}]
        result = mod._render_evidence_index(items)
        assert result == "- e_2: search\n"

    def test_item_with_zero_chars_no_suffix(self) -> None:
        """Zero chars should not emit a chars suffix."""
        items: list[dict[str, Any]] = [{"id": "e_3", "type": "file", "locator": "a.py", "chars": 0}]
        result = mod._render_evidence_index(items)
        assert "chars" not in result

    def test_fallback_path_as_locator(self) -> None:
        """When locator is absent, path is used as fallback."""
        items: list[dict[str, Any]] = [{"id": "e_4", "type": "file", "path": "bar.py"}]
        result = mod._render_evidence_index(items)
        assert "bar.py" in result


# ---------------------------------------------------------------------------
# _merge_evidence_items
# ---------------------------------------------------------------------------


class TestMergeEvidenceItems:
    """Tests for _merge_evidence_items."""

    def test_deduplicates_by_id(self) -> None:
        """Duplicate evidence IDs across groups are dropped."""
        g1: list[dict[str, Any]] = [{"id": "e_1", "text": "a"}]
        g2: list[dict[str, Any]] = [{"id": "e_1", "text": "a"}, {"id": "e_2", "text": "b"}]
        result = mod._merge_evidence_items(g1, g2)
        assert len(result) == 2
        assert [r["id"] for r in result] == ["e_1", "e_2"]

    def test_non_dict_skipped(self) -> None:
        """Non-dict entries in groups are skipped."""
        g: list[Any] = ["not-a-dict", {"id": "e_1"}]
        result = mod._merge_evidence_items(g)
        assert len(result) == 1

    def test_missing_id_skipped(self) -> None:
        """Items without id are skipped."""
        g: list[dict[str, Any]] = [{"text": "no id"}, {"id": "", "text": "empty id"}]
        result = mod._merge_evidence_items(g)
        assert result == []


# ---------------------------------------------------------------------------
# _coerce_payload_bool
# ---------------------------------------------------------------------------


class TestCoercePayloadBool:
    """Tests for _coerce_payload_bool."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            (None, False),
            (1, True),
            (0, False),
            (0.0, False),
            (1.0, True),
            (1.5, False),
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
            ("random", False),
            ([], False),
            ([1], False),
        ],
    )
    def test_coerce_values(self, value: Any, expected: bool) -> None:
        """Various types are correctly coerced to bool."""
        assert mod._coerce_payload_bool(value) is expected


# ---------------------------------------------------------------------------
# _coerce_context_request_payload
# ---------------------------------------------------------------------------


class TestCoerceContextRequestPayload:
    """Tests for _coerce_context_request_payload."""

    def test_non_dict_returns_none(self) -> None:
        """Non-dict input returns None."""
        assert mod._coerce_context_request_payload("string") is None
        assert mod._coerce_context_request_payload(42) is None

    def test_requests_not_list_returns_none(self) -> None:
        """When requests is present but not a list, returns None."""
        assert mod._coerce_context_request_payload({"requests": "bad"}) is None

    def test_minimal_valid_payload(self) -> None:
        """Minimal valid dict produces a well-formed payload."""
        result = mod._coerce_context_request_payload({"need_more_context": True})
        assert result is not None
        assert result["need_more_context"] is True
        assert result["requests"] == []

    def test_file_request_entry(self) -> None:
        """File request entries are normalized."""
        raw: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "file", "path": "a.py"}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 1
        assert result["requests"][0]["type"] == "file"
        assert result["requests"][0]["path"] == "a.py"

    def test_entry_without_type_skipped(self) -> None:
        """Entries with missing/empty type are skipped."""
        raw: dict[str, Any] = {
            "requests": [{"path": "a.py"}, {"type": "", "path": "b.py"}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 0

    def test_non_dict_entry_skipped(self) -> None:
        """Non-dict entries in the requests list are skipped."""
        raw: dict[str, Any] = {"requests": ["not-a-dict", {"type": "file"}]}
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 1

    def test_frame_request_valid(self) -> None:
        """Valid frame request is normalized."""
        raw: dict[str, Any] = {
            "requests": [{"type": "frame", "index": 0}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 1
        assert result["requests"][0]["index"] == 0

    def test_frame_request_bool_index_rejected(self) -> None:
        """Boolean index on frame request is rejected."""
        raw: dict[str, Any] = {
            "requests": [{"type": "frame", "index": True}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 0

    def test_frame_request_no_index_rejected(self) -> None:
        """Frame request without index is rejected."""
        raw: dict[str, Any] = {
            "requests": [{"type": "frame"}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 0

    def test_local_var_request_valid(self) -> None:
        """Valid local_var request is normalized."""
        raw: dict[str, Any] = {
            "requests": [{"type": "local_var", "frame": 0, "name": "x"}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        entries = result["requests"]
        assert len(entries) == 1
        assert entries[0]["frame"] == 0
        assert entries[0]["name"] == "x"

    def test_local_var_bool_frame_rejected(self) -> None:
        """Boolean frame on local_var request is rejected."""
        raw: dict[str, Any] = {
            "requests": [{"type": "local_var", "frame": True, "name": "x"}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 0

    def test_local_var_empty_name_rejected(self) -> None:
        """Empty name on local_var request is rejected."""
        raw: dict[str, Any] = {
            "requests": [{"type": "local_var", "frame": 0, "name": ""}],
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert len(result["requests"]) == 0

    def test_optional_fields_set_when_present(self) -> None:
        """Optional string fields are included when present and non-empty."""
        raw: dict[str, Any] = {
            "need_more_context": False,
            "snapshot_justification": "need stack",
            "compact_context_reason": "large",
            "sufficiency_rationale": "enough",
            "diagnosis": " root cause ",
            "missing_fact": " some fact ",
        }
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert result["snapshot_justification"] == "need stack"
        assert result["compact_context_reason"] == "large"
        assert result["sufficiency_rationale"] == "enough"
        assert result["diagnosis"] == "root cause"
        assert result["missing_fact"] == "some fact"

    def test_missing_fact_explicit_none(self) -> None:
        """Explicitly null missing_fact is preserved."""
        raw: dict[str, Any] = {"missing_fact": None}
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert "missing_fact" in result
        assert result["missing_fact"] is None

    def test_missing_fact_empty_string_omitted(self) -> None:
        """Empty missing_fact string is not included."""
        raw: dict[str, Any] = {"missing_fact": "  "}
        result = mod._coerce_context_request_payload(raw)
        assert result is not None
        assert "missing_fact" not in result

    def test_shared_normalizer_parity(self) -> None:
        raw: dict[str, Any] = {
            "diagnosis": " off by one ",
            "missing_fact": None,
            "need_more_context": "maybe",
            "need_snapshot": "on",
            "need_compact_context": "off",
            "requests": [
                {"type": "frame", "index": "bad"},
                {"type": "local_var", "frame": 0, "name": " value "},
                {"type": "file", "path": " buggy.py "},
            ],
        }
        assert mod._coerce_context_request_payload(raw) == normalize_stage_a_request_payload(raw)


# ---------------------------------------------------------------------------
# _normalize_context_rel_path
# ---------------------------------------------------------------------------


class TestNormalizeContextRelPath:
    """Tests for _normalize_context_rel_path."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("foo.py", "foo.py"),
            ("./foo.py", "foo.py"),
            ("dir/./foo.py", "dir/foo.py"),
            ("", None),
            ("  ", None),
            ("/absolute/path.py", None),
            ("C:/windows/path.py", None),
            ("dir/../escape.py", None),
            ("dir//foo.py", "dir/foo.py"),
            ("dir\\foo.py", "dir/foo.py"),
        ],
    )
    def test_various_paths(self, raw: str, expected: str | None) -> None:
        """Normalize various path strings."""
        assert mod._normalize_context_rel_path(raw) == expected


# ---------------------------------------------------------------------------
# _read_text_limited
# ---------------------------------------------------------------------------


class TestReadTextLimited:
    """Tests for _read_text_limited."""

    def test_reads_small_file(self, tmp_path: Path) -> None:
        """Small file content is returned verbatim."""
        p = tmp_path / "small.txt"
        p.write_text("hello", encoding="utf-8")
        assert mod._read_text_limited(p, max_chars=100) == "hello"

    def test_truncates_large_file(self, tmp_path: Path) -> None:
        """Files exceeding max_chars are truncated with marker."""
        p = tmp_path / "big.txt"
        p.write_text("x" * 200, encoding="utf-8")
        result = mod._read_text_limited(p, max_chars=50)
        assert result is not None
        assert result.endswith("...[TRUNC]\n")
        assert len(result) < 200

    def test_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Missing file returns None."""
        p = tmp_path / "missing.txt"
        assert mod._read_text_limited(p, max_chars=100) is None


# ---------------------------------------------------------------------------
# _literal_search_matches_in_file
# ---------------------------------------------------------------------------


class TestLiteralSearchMatchesInFile:
    """Tests for _literal_search_matches_in_file."""

    def test_empty_pattern_returns_empty(self, tmp_path: Path) -> None:
        """Empty pattern always returns empty list."""
        p = tmp_path / "f.py"
        p.write_text("hello", encoding="utf-8")
        matches, trunc = mod._literal_search_matches_in_file(
            p, "", max_matches=5, max_scan_chars=1000
        )
        assert matches == []
        assert trunc is False

    def test_finds_matches(self, tmp_path: Path) -> None:
        """Matching lines are returned with line numbers."""
        p = tmp_path / "f.py"
        p.write_text("foo\nbar\nfoo baz\n", encoding="utf-8")
        matches, _ = mod._literal_search_matches_in_file(
            p, "foo", max_matches=10, max_scan_chars=10000
        )
        assert len(matches) == 2
        assert matches[0].startswith("1:")
        assert matches[1].startswith("3:")

    def test_truncation_at_max_matches(self, tmp_path: Path) -> None:
        """Stops collecting after max_matches."""
        p = tmp_path / "f.py"
        p.write_text("\n".join(["match"] * 20), encoding="utf-8")
        matches, _ = mod._literal_search_matches_in_file(
            p, "match", max_matches=3, max_scan_chars=100000
        )
        assert len(matches) == 3

    def test_scan_cap_truncation(self, tmp_path: Path) -> None:
        """Returns truncated=True when scan cap is reached."""
        p = tmp_path / "f.py"
        p.write_text("x" * 100 + "\nmatch\n", encoding="utf-8")
        _matches, trunc = mod._literal_search_matches_in_file(
            p, "match", max_matches=10, max_scan_chars=50
        )
        assert trunc is True

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing file returns empty list, no exception."""
        p = tmp_path / "nope.py"
        matches, trunc = mod._literal_search_matches_in_file(
            p, "x", max_matches=5, max_scan_chars=1000
        )
        assert matches == []
        assert trunc is False

    def test_long_line_truncated(self, tmp_path: Path) -> None:
        """Very long matching lines are truncated to 220 chars."""
        p = tmp_path / "f.py"
        p.write_text("match" + "x" * 300, encoding="utf-8")
        matches, _ = mod._literal_search_matches_in_file(
            p, "match", max_matches=5, max_scan_chars=100000
        )
        assert len(matches) == 1
        assert matches[0].endswith("...")


# ---------------------------------------------------------------------------
# _collect_file_index_lines
# ---------------------------------------------------------------------------


class TestCollectFileIndexLines:
    """Tests for _collect_file_index_lines."""

    def test_collects_eligible_files(self, tmp_path: Path) -> None:
        """Collects .py, .txt, .json, .md files and excludes artifacts."""
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        (tmp_path / "readme.md").write_text("hi", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        result = mod._collect_file_index_lines(tmp_path)
        assert "main.py" in result
        assert "readme.md" in result
        assert "data.json" in result
        assert "notes.txt" in result
        assert "image.png" not in result

    def test_excludes_eval_only_artifacts(self, tmp_path: Path) -> None:
        """Files in _EVAL_ONLY_CASE_ARTIFACTS are excluded."""
        (tmp_path / "case.json").write_text("{}", encoding="utf-8")
        (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
        (tmp_path / "real.py").write_text("pass", encoding="utf-8")
        result = mod._collect_file_index_lines(tmp_path)
        names = [Path(r).name for r in result]
        assert "case.json" not in names
        assert "meta.json" not in names
        assert "real.py" in names

    def test_excludes_hidden_dirs(self, tmp_path: Path) -> None:
        """Hidden and excluded directories are skipped."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.py").write_text("pass", encoding="utf-8")
        (tmp_path / "ok.py").write_text("pass", encoding="utf-8")
        result = mod._collect_file_index_lines(tmp_path)
        assert any("ok.py" in r for r in result)
        assert not any("__pycache__" in r for r in result)


# ---------------------------------------------------------------------------
# _crash_file_from_snapshot
# ---------------------------------------------------------------------------


class TestCrashFileFromSnapshot:
    """Tests for _crash_file_from_snapshot."""

    def test_none_snapshot(self) -> None:
        """None snapshot returns None."""
        assert mod._crash_file_from_snapshot(None) is None

    def test_no_frames(self) -> None:
        """Snapshot without frames returns None."""
        assert mod._crash_file_from_snapshot({"frames": []}) is None

    def test_extracts_first_file_rel(self) -> None:
        """Returns file_rel from the first valid frame."""
        snap: dict[str, Any] = {
            "frames": [
                {"file_rel": "  buggy.py  ", "line": 1},
            ]
        }
        assert mod._crash_file_from_snapshot(snap) == "buggy.py"

    def test_skips_non_dict_frames(self) -> None:
        """Non-dict frame entries are skipped."""
        snap: dict[str, Any] = {"frames": ["bad", {"file_rel": "ok.py"}]}
        assert mod._crash_file_from_snapshot(snap) == "ok.py"

    def test_frames_not_list(self) -> None:
        """If frames is not a list, returns None."""
        snap: dict[str, Any] = {"frames": "not-a-list"}
        assert mod._crash_file_from_snapshot(snap) is None


# ---------------------------------------------------------------------------
# _detect_failing_test_file
# ---------------------------------------------------------------------------


class TestDetectFailingTestFile:
    """Tests for _detect_failing_test_file."""

    def test_detects_from_stdout(self, tmp_path: Path) -> None:
        """Extracts test file path from stdout."""
        (tmp_path / "test_app.py").write_text("pass", encoding="utf-8")
        result = mod._detect_failing_test_file(
            stdout="FAILED test_app.py::test_foo",
            stderr="",
            root=tmp_path,
        )
        assert result == "test_app.py"

    def test_fallback_to_file_index(self, tmp_path: Path) -> None:
        """Falls back to first test_*.py in file index when no match in output."""
        result = mod._detect_failing_test_file(
            stdout="no test info here",
            stderr="",
            root=tmp_path,
            file_index_lines=["main.py", "test_main.py"],
        )
        assert result == "test_main.py"

    def test_returns_none_when_no_test(self, tmp_path: Path) -> None:
        """Returns None when no test file is found anywhere."""
        result = mod._detect_failing_test_file(
            stdout="no info",
            stderr="",
            root=tmp_path,
            file_index_lines=["main.py", "utils.py"],
        )
        assert result is None


# ---------------------------------------------------------------------------
# _import_neighbors
# ---------------------------------------------------------------------------


class TestImportNeighbors:
    """Tests for _import_neighbors."""

    def test_none_rel_path(self, tmp_path: Path) -> None:
        """None rel_path returns empty list."""
        assert mod._import_neighbors(root=tmp_path, rel_path=None, limit=5) == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent file returns empty list."""
        assert mod._import_neighbors(root=tmp_path, rel_path="nope.py", limit=5) == []

    def test_finds_import_neighbors(self, tmp_path: Path) -> None:
        """Discovers .py files referenced by import statements."""
        (tmp_path / "helper.py").write_text("def go(): pass", encoding="utf-8")
        (tmp_path / "main.py").write_text(
            "import helper\nfrom utils import foo\n", encoding="utf-8"
        )
        (tmp_path / "utils.py").write_text("def foo(): pass", encoding="utf-8")
        result = mod._import_neighbors(root=tmp_path, rel_path="main.py", limit=10)
        assert "helper.py" in result
        assert "utils.py" in result

    def test_respects_limit(self, tmp_path: Path) -> None:
        """Stops at the given limit."""
        for i in range(5):
            (tmp_path / f"mod{i}.py").write_text("pass", encoding="utf-8")
        imports = "\n".join(f"import mod{i}" for i in range(5))
        (tmp_path / "main.py").write_text(imports, encoding="utf-8")
        result = mod._import_neighbors(root=tmp_path, rel_path="main.py", limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _resolve_subprocess_script_path
# ---------------------------------------------------------------------------


class TestResolveSubprocessScriptPath:
    """Tests for _resolve_subprocess_script_path."""

    def test_empty_returns_none(self, tmp_path: Path) -> None:
        """Empty string returns None."""
        assert mod._resolve_subprocess_script_path(root=tmp_path, raw_path="") is None

    def test_non_py_returns_none(self, tmp_path: Path) -> None:
        """Non-.py path returns None."""
        assert mod._resolve_subprocess_script_path(root=tmp_path, raw_path="run.sh") is None

    def test_dash_prefix_returns_none(self, tmp_path: Path) -> None:
        """Paths starting with dash are rejected."""
        assert mod._resolve_subprocess_script_path(root=tmp_path, raw_path="-m.py") is None

    def test_resolves_relative_path(self, tmp_path: Path) -> None:
        """Resolves a valid relative .py path under root."""
        (tmp_path / "script.py").write_text("pass", encoding="utf-8")
        result = mod._resolve_subprocess_script_path(root=tmp_path, raw_path="./script.py")
        assert result == "script.py"

    def test_basename_fallback(self, tmp_path: Path) -> None:
        """Basename-only resolves via file index lines when single match."""
        result = mod._resolve_subprocess_script_path(
            root=tmp_path,
            raw_path="worker.py",
            file_index_lines=["src/worker.py"],
        )
        assert result == "src/worker.py"

    def test_basename_ambiguous_returns_none(self, tmp_path: Path) -> None:
        """Multiple matches for basename returns None."""
        result = mod._resolve_subprocess_script_path(
            root=tmp_path,
            raw_path="worker.py",
            file_index_lines=["src/worker.py", "lib/worker.py"],
        )
        assert result is None

    def test_path_with_slash_no_fallback(self, tmp_path: Path) -> None:
        """Paths with slashes that don't exist won't use basename fallback."""
        result = mod._resolve_subprocess_script_path(
            root=tmp_path,
            raw_path="sub/missing.py",
            file_index_lines=["sub/missing.py"],
        )
        # File doesn't exist on disk, so resolution fails
        assert result is None

    def test_unsafe_path_returns_none(self, tmp_path: Path) -> None:
        """Unsafe relative path (with ..) returns None."""
        assert mod._resolve_subprocess_script_path(root=tmp_path, raw_path="../escape.py") is None


# ---------------------------------------------------------------------------
# _extract_subprocess_script_neighbors
# ---------------------------------------------------------------------------


class TestExtractSubprocessScriptNeighbors:
    """Tests for _extract_subprocess_script_neighbors."""

    def test_empty_rel_path(self, tmp_path: Path) -> None:
        """None rel_path returns empty."""
        result = mod._extract_subprocess_script_neighbors(root=tmp_path, rel_path=None, limit=5)
        assert result == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent file returns empty."""
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path, rel_path="nope.py", limit=5
        )
        assert result == []

    def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        """File with syntax errors returns empty list."""
        bad = tmp_path / "bad.py"
        bad.write_text("def (broken syntax", encoding="utf-8")
        result = mod._extract_subprocess_script_neighbors(root=tmp_path, rel_path="bad.py", limit=5)
        assert result == []

    def test_finds_subprocess_run_scripts(self, tmp_path: Path) -> None:
        """Discovers scripts passed to subprocess.run()."""
        worker = tmp_path / "worker.py"
        worker.write_text("print('hi')", encoding="utf-8")
        caller = tmp_path / "caller.py"
        caller.write_text(
            "import subprocess\nsubprocess.run(['python', 'worker.py'])\n",
            encoding="utf-8",
        )
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path, rel_path="caller.py", limit=5
        )
        assert "worker.py" in result

    def test_finds_from_import_member(self, tmp_path: Path) -> None:
        """Discovers scripts when subprocess member is directly imported."""
        worker = tmp_path / "worker.py"
        worker.write_text("print('hi')", encoding="utf-8")
        caller = tmp_path / "caller.py"
        caller.write_text(
            "from subprocess import run\nrun(['python', 'worker.py'])\n",
            encoding="utf-8",
        )
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path, rel_path="caller.py", limit=5
        )
        assert "worker.py" in result

    def test_subprocess_string_arg(self, tmp_path: Path) -> None:
        """Discovers scripts from string arguments to subprocess."""
        worker = tmp_path / "worker.py"
        worker.write_text("print('hi')", encoding="utf-8")
        caller = tmp_path / "caller.py"
        caller.write_text(
            "import subprocess\nsubprocess.run('python worker.py', shell=True)\n",
            encoding="utf-8",
        )
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path, rel_path="caller.py", limit=5
        )
        assert "worker.py" in result

    def test_subprocess_keyword_args(self, tmp_path: Path) -> None:
        """Discovers scripts from keyword 'args' argument."""
        worker = tmp_path / "worker.py"
        worker.write_text("print('hi')", encoding="utf-8")
        caller = tmp_path / "caller.py"
        caller.write_text(
            "import subprocess\nsubprocess.run(args=['python', 'worker.py'])\n",
            encoding="utf-8",
        )
        result = mod._extract_subprocess_script_neighbors(
            root=tmp_path, rel_path="caller.py", limit=5
        )
        assert "worker.py" in result

    def test_limit_respected(self, tmp_path: Path) -> None:
        """Stops collecting after reaching the limit."""
        for i in range(5):
            (tmp_path / f"s{i}.py").write_text("pass", encoding="utf-8")
        lines = [f"subprocess.run(['python', 's{i}.py'])" for i in range(5)]
        caller = tmp_path / "c.py"
        caller.write_text(
            "import subprocess\n" + "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        result = mod._extract_subprocess_script_neighbors(root=tmp_path, rel_path="c.py", limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# collect_text_context
# ---------------------------------------------------------------------------


class TestCollectTextContext:
    """Tests for collect_text_context."""

    def test_collects_files(self, tmp_path: Path) -> None:
        """Collects text from files in the directory."""
        (tmp_path / "a.py").write_text("print('a')", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('b')", encoding="utf-8")
        result = mod.collect_text_context(tmp_path)
        assert "a.py" in result
        assert "b.py" in result

    def test_respects_max_total_chars(self, tmp_path: Path) -> None:
        """Stops when max_total_chars is exceeded."""
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text("x" * 1000, encoding="utf-8")
        result = mod.collect_text_context(tmp_path, max_total_chars=500)
        # Should have stopped well before all 10 files
        assert len(result) < 5000

    def test_uses_provided_file_index_lines(self, tmp_path: Path) -> None:
        """Uses provided file_index_lines instead of scanning."""
        (tmp_path / "a.py").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.py").write_text("beta", encoding="utf-8")
        result = mod.collect_text_context(tmp_path, file_index_lines=["a.py"])
        assert "alpha" in result
        assert "beta" not in result

    def test_nonexistent_file_in_index_skipped(self, tmp_path: Path) -> None:
        """Missing files in the index are silently skipped."""
        result = mod.collect_text_context(tmp_path, file_index_lines=["missing.py"])
        assert result == ""


# ---------------------------------------------------------------------------
# collect_file_index
# ---------------------------------------------------------------------------


class TestCollectFileIndex:
    """Tests for collect_file_index."""

    def test_returns_newline_delimited(self, tmp_path: Path) -> None:
        """Returns newline-separated file list."""
        (tmp_path / "a.py").write_text("pass", encoding="utf-8")
        (tmp_path / "b.py").write_text("pass", encoding="utf-8")
        result = mod.collect_file_index(tmp_path)
        assert "a.py" in result
        assert "b.py" in result
        assert "\n" in result


# ---------------------------------------------------------------------------
# _strip_eval_only_case_artifacts
# ---------------------------------------------------------------------------


class TestStripEvalOnlyCaseArtifacts:
    """Tests for _strip_eval_only_case_artifacts."""

    def test_removes_case_artifacts(self, tmp_path: Path) -> None:
        """Removes case.json, meta.json, oracle.patch if present."""
        for name in ("case.json", "meta.json", "oracle.patch"):
            (tmp_path / name).write_text("{}", encoding="utf-8")
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        mod._strip_eval_only_case_artifacts(tmp_path)
        assert not (tmp_path / "case.json").exists()
        assert not (tmp_path / "meta.json").exists()
        assert not (tmp_path / "oracle.patch").exists()
        assert (tmp_path / "main.py").exists()

    def test_ignores_missing_artifacts(self, tmp_path: Path) -> None:
        """Does not raise when artifact files are already absent."""
        mod._strip_eval_only_case_artifacts(tmp_path)  # should not raise
        # Directory should be unchanged (still empty, no files created)
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# execute_context_requests - frame and local_var handling
# ---------------------------------------------------------------------------


def _make_file_index(tmp_path: Path, files: dict[str, str]) -> str:
    """Helper: write files under tmp_path and return a newline-delimited file index."""
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return "\n".join(files)


class TestExecuteContextRequestsFrame:
    """Tests for frame request handling in execute_context_requests."""

    def test_frame_request_returns_frame_block(self, tmp_path: Path) -> None:
        """Valid frame request produces a frame block."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {
            "frames": [
                {
                    "file_rel": "main.py",
                    "line": 5,
                    "function": "foo",
                    "code": "return 42",
                    "locals": {"x": 1, "y": "hello"},
                }
            ]
        }
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "frame", "index": 0}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is False
        assert result.requests_count == 1
        assert any(b["type"] == "frame" for b in result.blocks)
        assert "frame[0]" in result.text
        assert "x = 1" in result.text

    def test_frame_request_no_snapshot_falls_through(self, tmp_path: Path) -> None:
        """Frame request with no snapshot skips that request."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "frame", "index": 0}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        # Falls back because no valid requests succeeded
        assert result.used_fallback is True

    def test_frame_out_of_range(self, tmp_path: Path) -> None:
        """Frame index beyond available frames is skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {"frames": [{"file_rel": "main.py", "line": 1}]}
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "frame", "index": 99}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is True

    def test_frame_non_dict_frame_skipped(self, tmp_path: Path) -> None:
        """Non-dict frame entries in snapshot are skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {"frames": ["not-a-dict"]}
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "frame", "index": 0}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is True

    def test_frame_locals_at_prefix_skipped(self, tmp_path: Path) -> None:
        """Locals with '@' prefix keys are excluded."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {
            "frames": [
                {
                    "file_rel": "main.py",
                    "line": 1,
                    "function": "f",
                    "code": "x",
                    "locals": {"@return": None, "real": 42},
                }
            ]
        }
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "frame", "index": 0}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert "@return" not in result.text
        assert "real = 42" in result.text


class TestExecuteContextRequestsLocalVar:
    """Tests for local_var request handling."""

    def test_local_var_valid(self, tmp_path: Path) -> None:
        """Valid local_var request returns the variable value."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {
            "frames": [
                {
                    "file_rel": "main.py",
                    "line": 10,
                    "function": "bar",
                    "locals": {"x": 42},
                }
            ]
        }
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 0, "name": "x"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is False
        assert "x = 42" in result.text

    def test_local_var_no_snapshot(self, tmp_path: Path) -> None:
        """local_var with no snapshot is skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 0, "name": "x"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is True

    def test_local_var_missing_name(self, tmp_path: Path) -> None:
        """local_var with missing variable name is skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {
            "frames": [{"file_rel": "main.py", "line": 1, "locals": {"y": 1}}]
        }
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 0, "name": "x"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        # 'x' not in locals, so skipped -> fallback
        assert result.used_fallback is True

    def test_local_var_frame_out_of_range(self, tmp_path: Path) -> None:
        """local_var with out-of-range frame index is skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {
            "frames": [{"file_rel": "main.py", "line": 1, "locals": {"x": 1}}]
        }
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 99, "name": "x"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is True

    def test_local_var_large_value_truncated(self, tmp_path: Path) -> None:
        """Large local variable values are truncated."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        snapshot: dict[str, Any] = {
            "frames": [
                {
                    "file_rel": "main.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"big": "x" * 5000},
                }
            ]
        }
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 0, "name": "big"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=snapshot,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is False
        assert "..." in result.text


# ---------------------------------------------------------------------------
# execute_context_requests - search handling
# ---------------------------------------------------------------------------


class TestExecuteContextRequestsSearch:
    """Tests for search request handling in execute_context_requests."""

    def test_search_returns_matches(self, tmp_path: Path) -> None:
        """Search request finds matching lines in files."""
        fi = _make_file_index(tmp_path, {"main.py": "foo = 1\nbar = 2\nfoo = 3\n"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "search", "path": "main.py", "pattern": "foo"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is False
        assert "foo" in result.text

    def test_search_no_matches(self, tmp_path: Path) -> None:
        """Search with no matches shows '(no literal matches)'."""
        fi = _make_file_index(tmp_path, {"main.py": "hello world\n"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [
                {"type": "search", "path": "main.py", "pattern": "zzz_not_found"},
            ],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is False
        assert "(no literal matches)" in result.text

    def test_search_empty_pattern_skipped(self, tmp_path: Path) -> None:
        """Search with empty pattern is skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "hello\n"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "search", "path": "main.py", "pattern": ""}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is True


# ---------------------------------------------------------------------------
# execute_context_requests - edge cases
# ---------------------------------------------------------------------------


class TestExecuteContextRequestsEdge:
    """Edge-case tests for execute_context_requests."""

    def test_malformed_requests_triggers_fallback(self, tmp_path: Path) -> None:
        """Payload with non-list requests triggers fallback."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": "not-a-list",
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is True

    def test_need_more_context_false_returns_empty(self, tmp_path: Path) -> None:
        """When need_more_context is false, returns empty with no fallback."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        payload: dict[str, Any] = {
            "need_more_context": False,
            "requests": [{"type": "file", "path": "main.py"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is False
        assert result.requests_count == 0
        assert result.text == ""

    def test_file_not_in_index_skipped(self, tmp_path: Path) -> None:
        """File request for path not in the index is skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [{"type": "file", "path": "unknown.py"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is True

    def test_total_chars_limit_stops_collection(self, tmp_path: Path) -> None:
        """Requests stop when max_total_chars would be exceeded."""
        fi = _make_file_index(tmp_path, {"a.py": "x" * 100, "b.py": "y" * 100})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": [
                {"type": "file", "path": "a.py"},
                {"type": "file", "path": "b.py"},
            ],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=10,
            max_total_chars=120,
        )
        assert result.used_fallback is False
        assert result.requests_count == 1

    def test_non_dict_request_entry_skipped(self, tmp_path: Path) -> None:
        """Non-dict entries in requests list are skipped."""
        fi = _make_file_index(tmp_path, {"main.py": "pass"})
        payload: dict[str, Any] = {
            "need_more_context": True,
            "requests": ["not-a-dict", {"type": "file", "path": "main.py"}],
        }
        result = mod.execute_context_requests(
            request_payload=payload,
            root=tmp_path,
            snapshot=None,
            file_index=fi,
            stdout="",
            stderr="",
            max_requests=5,
            max_total_chars=60000,
        )
        assert result.used_fallback is False
        assert result.requests_count == 1


# ---------------------------------------------------------------------------
# build_fallback_context_bundle
# ---------------------------------------------------------------------------


class TestBuildFallbackContextBundle:
    """Tests for build_fallback_context_bundle."""

    def test_uses_crash_file_and_test_file(self, tmp_path: Path) -> None:
        """Includes crash file and detected test file."""
        (tmp_path / "buggy.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "test_buggy.py").write_text("assert False", encoding="utf-8")
        snapshot: dict[str, Any] = {"frames": [{"file_rel": "buggy.py", "line": 1}]}
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=snapshot,
            stdout="FAILED test_buggy.py::test_x",
            stderr="",
            max_total_chars=60000,
            reason="test",
        )
        assert result.used_fallback is True
        assert any("buggy.py" in b.get("path", "") for b in result.blocks)

    def test_no_candidates_falls_back_to_file_index(self, tmp_path: Path) -> None:
        """When no crash/test file, uses first files from index."""
        (tmp_path / "a.py").write_text("pass", encoding="utf-8")
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=60000,
            reason="test",
            file_index_lines=["a.py"],
        )
        assert result.used_fallback is True
        assert len(result.blocks) >= 1

    def test_empty_rel_path_skipped(self, tmp_path: Path) -> None:
        """Empty strings in candidate paths are skipped."""
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=60000,
            reason="test",
            file_index_lines=["", "  "],
        )
        assert result.blocks == []

    def test_nonexistent_candidate_skipped(self, tmp_path: Path) -> None:
        """Non-existent candidate files are skipped."""
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=60000,
            reason="test",
            file_index_lines=["missing.py"],
        )
        assert result.blocks == []

    def test_max_total_chars_respected(self, tmp_path: Path) -> None:
        """Stops when max_total_chars is reached."""
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("x" * 500, encoding="utf-8")
        result = mod.build_fallback_context_bundle(
            root=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            max_total_chars=600,
            reason="test",
            file_index_lines=[f"f{i}.py" for i in range(5)],
        )
        assert result.chars_returned <= 600


# ---------------------------------------------------------------------------
# _iter_file_index_lines
# ---------------------------------------------------------------------------


class TestIterFileIndexLines:
    """Tests for _iter_file_index_lines."""

    def test_string_input(self) -> None:
        """String input is split by newlines."""
        result = list(mod._iter_file_index_lines("a.py\nb.py"))
        assert result == ["a.py", "b.py"]

    def test_sequence_input(self) -> None:
        """Sequence input yields each element."""
        result = list(mod._iter_file_index_lines(["a.py", "b.py"]))
        assert result == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# _build_evidence_item and _build_evidence_item_id
# ---------------------------------------------------------------------------


class TestBuildEvidenceItem:
    """Tests for _build_evidence_item."""

    def test_returns_well_formed_dict(self) -> None:
        """Returned dict has all expected keys."""
        item = mod._build_evidence_item(
            kind="file", locator="main.py", text="hello", meta={"path": "main.py"}
        )
        assert "id" in item
        assert item["type"] == "file"
        assert item["locator"] == "main.py"
        assert item["path"] == "main.py"
        assert item["chars"] > 0
        assert item["id"].startswith("e_")


class TestBuildEvidenceItemId:
    """Tests for _build_evidence_item_id."""

    def test_deterministic(self) -> None:
        """Same inputs produce the same id."""
        a = mod._build_evidence_item_id(kind="file", locator="x.py", text="hello")
        b = mod._build_evidence_item_id(kind="file", locator="x.py", text="hello")
        assert a == b
        assert a.startswith("e_")

    def test_different_text_different_id(self) -> None:
        """Different text produces different ids."""
        a = mod._build_evidence_item_id(kind="file", locator="x.py", text="hello")
        b = mod._build_evidence_item_id(kind="file", locator="x.py", text="world")
        assert a != b


# ---------------------------------------------------------------------------
# _compute_evidence_fingerprint
# ---------------------------------------------------------------------------


class TestComputeEvidenceFingerprint:
    """Tests for _compute_evidence_fingerprint."""

    def test_deterministic(self) -> None:
        """Same inputs produce same fingerprint."""
        kwargs: dict[str, Any] = {
            "failure_signature": "sig",
            "stage_a_blocks": [{"id": "e_1"}],
            "stage_a_context": "ctx",
            "structured_seed_blocks": [],
        }
        a = mod._compute_evidence_fingerprint(**kwargs)
        b = mod._compute_evidence_fingerprint(**kwargs)
        assert a == b

    def test_different_inputs_differ(self) -> None:
        """Different inputs produce different fingerprints."""
        base: dict[str, Any] = {
            "failure_signature": "sig",
            "stage_a_blocks": [],
            "stage_a_context": "ctx",
            "structured_seed_blocks": [],
        }
        a = mod._compute_evidence_fingerprint(**base)
        b = mod._compute_evidence_fingerprint(**{**base, "failure_signature": "other"})
        assert a != b


# ---------------------------------------------------------------------------
# _evidence_ids_from_items
# ---------------------------------------------------------------------------


class TestEvidenceIdsFromItems:
    """Tests for _evidence_ids_from_items."""

    def test_extracts_ids(self) -> None:
        """Extracts IDs from items with valid id keys."""
        items: list[dict[str, Any]] = [
            {"id": "e_1"},
            {"id": ""},
            {"no_id": True},
            {"id": "e_2"},
        ]
        result = mod._evidence_ids_from_items(items)
        assert result == ["e_1", "e_2"]

    def test_non_dict_skipped(self) -> None:
        """Non-dict entries are skipped."""
        items: list[Any] = ["bad", {"id": "e_1"}]
        result = mod._evidence_ids_from_items(items)
        assert result == ["e_1"]
