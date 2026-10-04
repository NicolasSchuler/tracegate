"""Comprehensive tests for BugsInPy, PyBugHive, and TrickyBugs importers.

Targets ≥85% coverage for each module by exercising:
- Metadata parsing (valid, malformed, edge cases)
- Patch/file extraction and filtering
- Case iteration with various skip reasons
- Full import pipeline with mocked externals
- Error handling (ValueError, validation failures)
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest

from evals.importers.bugsinpy_importer import (
    BugsInPyBugSpec,
    _build_case_metadata,
    _extract_python_files_from_patch,
    _iter_bugs,
    _parse_key_value_file,
    import_bugsinpy,
)
from evals.importers.native_repro_helpers import NativePytestCommand
from evals.importers.pybughive_importer import (
    PyBugHiveIssueSpec,
    _build_archive_member_index,
    _choose_target_source_file,
    _load_issue_specs,
    _looks_like_test_path,
    _normalize_parent_commit,
    _read_archive_file,
    _resolve_archive_member,
    import_pybughive,
)
from evals.importers.pybughive_importer import (
    _build_case_metadata as pbh_build_case_metadata,
)
from evals.importers.trickybugs_importer import (
    TrickyBugsCandidate,
    _candidate_from_record,
    _first_string,
    _iter_dict_nodes,
    _iter_python_candidates,
    _read_relative_file,
    _safe_slug,
    import_trickybugs,
)
from llmdebug import _json as json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NATIVE_CMD = NativePytestCommand(
    source_test_command="pytest -q tests/test_demo.py::test_case",
    python_args=["-m", "pytest", "-q", "tests/test_demo.py::test_case"],
    source_test_selector="tests/test_demo.py::test_case",
    test_paths=["tests/test_demo.py"],
)

BUGGY_CODE = "def add(a, b):\n    return a - b\n"
FIXED_CODE = "def add(a, b):\n    return a + b\n"
TEST_CODE = "def test_add():\n    assert add(1, 2) == 3\n"


def _make_bugsinpy_repo(
    tmp_path: Path,
    *,
    github_url: str = "https://github.com/example/demo",
    buggy_commit: str = "abc123",
    fixed_commit: str = "def456",
    pythonpath: str = "",
    run_test: str = "pytest -q tests/test_demo.py::test_case\n",
    patch_text: str | None = None,
    test_file: str = "tests/test_demo.py",
    project_name: str = "demo_project",
    bug_id: str = "1",
    include_project_info: bool = True,
    include_bug_info: bool = True,
    include_run_test: bool = True,
    include_patch: bool = True,
) -> Path:
    """Create a minimal BugsInPy repository structure."""
    repo = tmp_path / "BugsInPy"
    bug_dir = repo / "projects" / project_name / "bugs" / bug_id
    bug_dir.mkdir(parents=True, exist_ok=True)

    if include_project_info:
        (repo / "projects" / project_name / "project.info").write_text(
            f'github_url="{github_url}"\n', encoding="utf-8"
        )

    if include_bug_info:
        (bug_dir / "bug.info").write_text(
            f'buggy_commit_id="{buggy_commit}"\n'
            f'fixed_commit_id="{fixed_commit}"\n'
            f'test_file="{test_file}"\n'
            f'pythonpath="{pythonpath}"\n',
            encoding="utf-8",
        )

    if include_run_test:
        (bug_dir / "run_test.sh").write_text(run_test, encoding="utf-8")

    if include_patch:
        default_patch = (
            "diff --git a/src/demo.py b/src/demo.py\n"
            "--- a/src/demo.py\n"
            "+++ b/src/demo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-return 1\n"
            "+return 2\n"
        )
        (bug_dir / "bug_patch.txt").write_text(
            patch_text if patch_text is not None else default_patch,
            encoding="utf-8",
        )

    return repo


def _fake_git_show(repo_dir: Path, request: Any) -> str | None:
    """Simulate git_show_file for BugsInPy tests."""
    del repo_dir
    if request.rel_path == "src/demo.py":
        if request.commitish == "abc123":
            return BUGGY_CODE
        if request.commitish == "def456":
            return FIXED_CODE
    if request.rel_path == "tests/test_demo.py":
        return TEST_CODE
    return None


def _make_pybughive_repo(
    tmp_path: Path,
    *,
    payload: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a minimal PyBugHive repository with dataset JSON."""
    repo = tmp_path / "pybughive"
    dataset_dir = repo / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if payload is None:
        payload = [
            {
                "username": "example",
                "repository": "demo",
                "issues": [
                    {
                        "id": 7,
                        "title": "Fix demo",
                        "testSteps": "pytest -q tests/test_demo.py",
                        "commits": [
                            {
                                "hash": "fix123",
                                "parents": "bug123",
                                "stat": {"files": [{"filename": "src/demo.py"}]},
                            }
                        ],
                    }
                ],
            }
        ]
    (dataset_dir / "pybughive_current.json").write_text(json.dumps(payload), encoding="utf-8")
    return repo


def _make_pybughive_archive(
    tmp_path: Path,
    name: str = "demo.zip",
    members: dict[str, str] | None = None,
) -> Path:
    """Create a zip archive with given members."""
    if members is None:
        members = {
            "dump/bug123/src/demo.py": BUGGY_CODE,
            "dump/fix123/src/demo.py": FIXED_CODE,
            "dump/fix123/tests/test_demo.py": TEST_CODE,
        }
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w") as zf:
        for member_path, content in members.items():
            zf.writestr(member_path, content)
    return archive


# ===================================================================
# BugsInPy Importer Tests
# ===================================================================


class TestBugsInPyParseKeyValueFile:
    """Tests for _parse_key_value_file."""

    def test_parses_valid_entries(self, tmp_path: Path) -> None:
        """Valid key=value lines are parsed correctly."""
        f = tmp_path / "test.info"
        f.write_text(
            'github_url="https://example.com"\nbuggy_commit_id="abc"\n',
            encoding="utf-8",
        )
        result = _parse_key_value_file(f)
        assert result["github_url"] == "https://example.com"
        assert result["buggy_commit_id"] == "abc"

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        """Empty lines and comment lines are skipped (covers L51)."""
        f = tmp_path / "test.info"
        f.write_text(
            '# this is a comment\n\n  mykey="value1"  \n  # another comment\n\n',
            encoding="utf-8",
        )
        result = _parse_key_value_file(f)
        assert result == {"mykey": "value1"}

    def test_skips_non_matching_lines(self, tmp_path: Path) -> None:
        """Lines that don't match key=value pattern are skipped (covers L54)."""
        f = tmp_path / "test.info"
        f.write_text(
            'this is not a key value line\n=== header ===\nvalid_key="yes"\n',
            encoding="utf-8",
        )
        result = _parse_key_value_file(f)
        assert result == {"valid_key": "yes"}

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file returns empty dict."""
        f = tmp_path / "test.info"
        f.write_text("", encoding="utf-8")
        assert _parse_key_value_file(f) == {}


class TestBugsInPyExtractPythonFiles:
    """Tests for _extract_python_files_from_patch."""

    def test_single_python_file(self) -> None:
        """Extracts a single .py file from patch."""
        diff_patch = "diff --git a/src/demo.py b/src/demo.py\n--- a/src/demo.py\n"
        assert _extract_python_files_from_patch(diff_patch) == ["src/demo.py"]

    def test_non_python_file_excluded(self) -> None:
        """Non-.py files after normalization are excluded (covers L71)."""
        diff_patch = "diff --git a/readme.md b/readme.md\n"
        assert _extract_python_files_from_patch(diff_patch) == []

    def test_deduplicates_files(self) -> None:
        """Duplicate file entries are deduplicated."""
        diff_patch = (
            "diff --git a/src/demo.py b/src/demo.py\ndiff --git a/src/demo.py b/src/demo.py\n"
        )
        assert _extract_python_files_from_patch(diff_patch) == ["src/demo.py"]

    def test_multiple_python_files(self) -> None:
        """Multiple distinct .py files are all extracted."""
        diff_patch = "diff --git a/src/a.py b/src/a.py\ndiff --git a/src/b.py b/src/b.py\n"
        result = _extract_python_files_from_patch(diff_patch)
        assert result == ["src/a.py", "src/b.py"]

    def test_non_diff_lines_ignored(self) -> None:
        """Lines not matching diff --git pattern are ignored."""
        diff_patch = (
            "index 1234..5678\n"
            "--- a/src/demo.py\n"
            "+++ b/src/demo.py\n"
            "diff --git a/src/demo.py b/src/demo.py\n"
        )
        assert _extract_python_files_from_patch(diff_patch) == ["src/demo.py"]

    def test_unsafe_path_skipped(self) -> None:
        """Paths with .. components are excluded (unsafe path → L71)."""
        diff_patch = "diff --git a/../evil.py b/../evil.py\n"
        assert _extract_python_files_from_patch(diff_patch) == []


class TestBugsInPyIterBugs:
    """Tests for _iter_bugs with various skip conditions."""

    def test_no_projects_dir(self, tmp_path: Path) -> None:
        """Returns empty when projects/ dir doesn't exist (covers L80)."""
        repo = tmp_path / "BugsInPy"
        repo.mkdir()
        assert _iter_bugs(repo) == []

    def test_non_dir_project_entry_skipped(self, tmp_path: Path) -> None:
        """Non-directory entries in projects/ are skipped (covers L85)."""
        repo = tmp_path / "BugsInPy"
        projects = repo / "projects"
        projects.mkdir(parents=True)
        (projects / "not_a_dir.txt").write_text("file", encoding="utf-8")
        assert _iter_bugs(repo) == []

    def test_missing_project_info_skipped(self, tmp_path: Path) -> None:
        """Project dir without project.info is skipped (covers L89)."""
        repo = _make_bugsinpy_repo(tmp_path, include_project_info=False)
        assert _iter_bugs(repo) == []

    def test_empty_github_url_skipped(self, tmp_path: Path) -> None:
        """Project with empty github_url is skipped (covers L93)."""
        repo = _make_bugsinpy_repo(tmp_path, github_url="")
        assert _iter_bugs(repo) == []

    def test_no_bugs_dir_skipped(self, tmp_path: Path) -> None:
        """Project without bugs/ dir is skipped (covers L97)."""
        repo = tmp_path / "BugsInPy"
        proj = repo / "projects" / "proj"
        proj.mkdir(parents=True)
        (proj / "project.info").write_text('github_url="https://example.com"\n', encoding="utf-8")
        # No bugs/ directory
        assert _iter_bugs(repo) == []

    def test_missing_bug_files_skipped(self, tmp_path: Path) -> None:
        """Bug dir missing required files is skipped (covers L104)."""
        repo = _make_bugsinpy_repo(tmp_path, include_bug_info=False)
        assert _iter_bugs(repo) == []

    def test_missing_commit_ids_skipped(self, tmp_path: Path) -> None:
        """Bug with empty commit IDs is skipped (covers L110)."""
        repo = _make_bugsinpy_repo(tmp_path, buggy_commit="", fixed_commit="")
        assert _iter_bugs(repo) == []


class TestBugsInPyBuildCaseMetadata:
    """Tests for _build_case_metadata."""

    def test_easy_difficulty(self) -> None:
        """Fix lines ≤3 → easy difficulty."""
        spec = BugsInPyBugSpec(
            project_name="proj",
            bug_id="1",
            github_url="https://example.com",
            buggy_commit_id="abc",
            fixed_commit_id="def",
            test_file="tests/test.py",
            run_test_cmd="pytest tests/test.py",
            patch_text="diff",
            pythonpath="",
        )
        meta = _build_case_metadata(spec=spec, native_command=_NATIVE_CMD, fix_lines=2)
        assert meta["difficulty"] == "easy"
        assert meta["bug_source"] == "bugsinpy"
        assert meta["repro_mode"] == "native_dataset_test"

    def test_medium_difficulty(self) -> None:
        """Fix lines 4-10 → medium difficulty."""
        spec = BugsInPyBugSpec(
            project_name="proj",
            bug_id="2",
            github_url="u",
            buggy_commit_id="a",
            fixed_commit_id="b",
            test_file="t.py",
            run_test_cmd="pytest t.py",
            patch_text="",
            pythonpath="",
        )
        meta = _build_case_metadata(spec=spec, native_command=_NATIVE_CMD, fix_lines=7)
        assert meta["difficulty"] == "medium"

    def test_hard_difficulty(self) -> None:
        """Fix lines >10 → hard difficulty."""
        spec = BugsInPyBugSpec(
            project_name="proj",
            bug_id="3",
            github_url="u",
            buggy_commit_id="a",
            fixed_commit_id="b",
            test_file="t.py",
            run_test_cmd="pytest t.py",
            patch_text="",
            pythonpath="",
        )
        meta = _build_case_metadata(spec=spec, native_command=_NATIVE_CMD, fix_lines=20)
        assert meta["difficulty"] == "hard"


class TestBugsInPyImport:
    """Integration tests for import_bugsinpy."""

    def test_max_cases_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stops importing after max_cases reached (covers L184)."""
        repo = tmp_path / "BugsInPy"
        for i in range(1, 4):
            _make_bugsinpy_repo(tmp_path, project_name="proj", bug_id=str(i))
        # Just reuse the same repo structure
        repo = _make_bugsinpy_repo(tmp_path, project_name="proj", bug_id="1")
        # Make 3 bugs
        for i in range(2, 4):
            bug_dir = repo / "projects" / "proj" / "bugs" / str(i)
            bug_dir.mkdir(parents=True, exist_ok=True)
            (bug_dir / "bug.info").write_text(
                f'buggy_commit_id="abc{i}"\nfixed_commit_id="def{i}"\n'
                f'test_file="tests/test_demo.py"\npythonpath=""\n',
                encoding="utf-8",
            )
            (bug_dir / "run_test.sh").write_text(
                "pytest -q tests/test_demo.py::test_case\n", encoding="utf-8"
            )
            (bug_dir / "bug_patch.txt").write_text(
                "diff --git a/src/demo.py b/src/demo.py\n", encoding="utf-8"
            )

        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )

        call_count = 0

        def _counted_git_show(repo_dir: Path, request: Any) -> str | None:
            nonlocal call_count
            call_count += 1
            return _fake_git_show(repo_dir, request)

        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _counted_git_show)

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            max_cases=1,
            validate=False,
            dry_run=True,
        )
        assert summary.imported == 1

    def test_case_already_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing case directory is skipped (covers L190-191)."""
        repo = _make_bugsinpy_repo(tmp_path)
        out_dir = tmp_path / "cases"
        case_dir = out_dir / "bip_demo_project_1"
        case_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)

        summary = import_bugsinpy(
            out_dir,
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert summary.skip_reasons.get("case_already_exists") == 1

    def test_unsupported_pythonpath(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-trivial pythonpath is skipped (covers L194-195)."""
        repo = _make_bugsinpy_repo(tmp_path, pythonpath="/usr/lib")
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert summary.skip_reasons.get("unsupported_pythonpath") == 1

    def test_unsupported_native_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unsupported test command is skipped (covers L199-200)."""
        repo = _make_bugsinpy_repo(tmp_path, run_test="pipenv run pytest tests/test.py\n")
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "unsupported_native_test_command" in summary.skip_reasons

    def test_multiple_patch_files_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Patch touching multiple .py files is skipped (covers L204-205)."""
        multi_patch = "diff --git a/src/a.py b/src/a.py\ndiff --git a/src/b.py b/src/b.py\n"
        repo = _make_bugsinpy_repo(tmp_path, patch_text=multi_patch)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "unsupported_patch_file_count" in summary.skip_reasons

    def test_git_clone_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failed git clone is skipped (covers L211-212)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (False, "git_clone_failed"),
        )

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "git_clone_failed" in summary.skip_reasons

    def test_git_show_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failed git show returns None → skipped (covers L223-224)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.git_show_file",
            lambda *a, **kw: None,
        )

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "git_show_failed" in summary.skip_reasons

    def test_unsafe_test_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unsafe test path skipped (covers L231-233)."""
        repo = _make_bugsinpy_repo(
            tmp_path,
            run_test="pytest -q ../evil/test.py\n",
            patch_text="diff --git a/src/demo.py b/src/demo.py\n",
        )
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        # The unsafe path makes normalize_native_pytest_command reject it, or
        # the path is caught as unsupported command / unsafe.
        assert summary.imported == 0

    def test_test_file_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test file not found in git (covers L245-247, 250)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )

        def _show_no_tests(repo_dir: Path, request: Any) -> str | None:
            if request.rel_path == "src/demo.py":
                if request.commitish == "abc123":
                    return BUGGY_CODE
                if request.commitish == "def456":
                    return FIXED_CODE
            # Return None for test files → both fixed and buggy fail
            return None

        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _show_no_tests)

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "native_test_file_not_found" in summary.skip_reasons

    def test_test_file_fallback_to_buggy_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test file fallback to buggy commit (covers L240)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )

        def _show_buggy_only_test(repo_dir: Path, request: Any) -> str | None:
            if request.rel_path == "src/demo.py":
                if request.commitish == "abc123":
                    return BUGGY_CODE
                if request.commitish == "def456":
                    return FIXED_CODE
            if request.rel_path == "tests/test_demo.py":
                # Fixed commit returns None, buggy commit has the file
                if request.commitish == "abc123":
                    return TEST_CODE
                return None
            return None

        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.git_show_file",
            _show_buggy_only_test,
        )

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert summary.imported == 1

    def test_empty_oracle_patch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Identical buggy/fixed code → empty patch → skipped (covers L254-255)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )

        def _show_same_code(repo_dir: Path, request: Any) -> str | None:
            if request.rel_path == "src/demo.py":
                return BUGGY_CODE  # Same for both commits
            if request.rel_path == "tests/test_demo.py":
                return TEST_CODE
            return None

        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _show_same_code)

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "empty_oracle_patch" in summary.skip_reasons

    def test_dry_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry run doesn't create files (covers L263-264)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)

        out_dir = tmp_path / "cases"
        summary = import_bugsinpy(
            out_dir,
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
            dry_run=True,
        )
        assert summary.imported == 1
        assert not (out_dir / "bip_demo_project_1").exists()

    def test_write_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ValueError from write_native_case_files → skipped (covers L274-277)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.write_native_case_files",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("unsafe")),
        )

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "unsafe_dataset_path" in summary.skip_reasons

    def test_validation_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Schema validation failure → skipped (covers L280-284)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (True, None),
        )
        monkeypatch.setattr("evals.importers.bugsinpy_importer.git_show_file", _fake_git_show)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.validate_generated_case",
            lambda *a, **kw: (False, "bad schema"),
        )

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=True,
        )
        assert "schema_validation_failed" in summary.skip_reasons

    def test_git_clone_failed_no_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failed git clone with None reason uses fallback (covers L211)."""
        repo = _make_bugsinpy_repo(tmp_path)
        monkeypatch.setattr(
            "evals.importers.bugsinpy_importer.ensure_git_repo",
            lambda *a, **kw: (False, None),
        )

        summary = import_bugsinpy(
            tmp_path / "cases",
            bugsinpy_repo=repo,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "git_clone_failed" in summary.skip_reasons


# ===================================================================
# PyBugHive Importer Tests
# ===================================================================


class TestPyBugHiveHelpers:
    """Tests for PyBugHive helper functions."""

    def test_normalize_parent_commit_basic(self) -> None:
        """Extracts first token from parent commit string."""
        assert _normalize_parent_commit("abc123 def456") == "abc123"

    def test_normalize_parent_commit_empty(self) -> None:
        """Empty string returns empty."""
        assert _normalize_parent_commit("") == ""
        assert _normalize_parent_commit("  ") == ""

    def test_looks_like_test_path_in_tests_dir(self) -> None:
        """Path containing /tests/ is detected."""
        assert _looks_like_test_path("src/tests/test_parser.py") is True

    def test_looks_like_test_path_test_prefix(self) -> None:
        """File starting with test_ is detected."""
        assert _looks_like_test_path("test_parser.py") is True

    def test_looks_like_test_path_source(self) -> None:
        """Non-test file is not detected."""
        assert _looks_like_test_path("src/parser.py") is False


class TestPyBugHiveLoadIssueSpecs:
    """Tests for _load_issue_specs with malformed data."""

    def test_payload_not_list(self, tmp_path: Path) -> None:
        """Non-list JSON returns empty (covers L64)."""
        repo = _make_pybughive_repo(tmp_path, payload={"not": "a list"})  # type: ignore[arg-type]
        assert _load_issue_specs(repo) == []

    def test_project_not_dict(self, tmp_path: Path) -> None:
        """Non-dict project entry is skipped (covers L68)."""
        repo = _make_pybughive_repo(tmp_path, payload=["not_a_dict"])  # type: ignore[list-item]
        assert _load_issue_specs(repo) == []

    def test_missing_username_or_repo(self, tmp_path: Path) -> None:
        """Missing username/repo → skipped (covers L73)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[{"username": "", "repository": "demo", "issues": []}],
        )
        assert _load_issue_specs(repo) == []

    def test_issues_not_list(self, tmp_path: Path) -> None:
        """issues not a list → skipped (covers L73)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[{"username": "u", "repository": "r", "issues": "not_list"}],
        )
        assert _load_issue_specs(repo) == []

    def test_issue_not_dict(self, tmp_path: Path) -> None:
        """Non-dict issue entry is skipped (covers L77)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[{"username": "u", "repository": "r", "issues": ["bad"]}],
        )
        assert _load_issue_specs(repo) == []

    def test_issue_id_not_int(self, tmp_path: Path) -> None:
        """Non-int issue id → skipped (covers L82)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [{"id": "notint", "commits": []}],
                }
            ],
        )
        assert _load_issue_specs(repo) == []

    def test_commits_not_list(self, tmp_path: Path) -> None:
        """commits not a list → skipped (covers L82)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [{"id": 1, "commits": "bad"}],
                }
            ],
        )
        assert _load_issue_specs(repo) == []

    def test_commits_empty(self, tmp_path: Path) -> None:
        """Empty commits list → skipped (covers L82)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [{"id": 1, "commits": []}],
                }
            ],
        )
        assert _load_issue_specs(repo) == []

    def test_commit_not_dict(self, tmp_path: Path) -> None:
        """Non-dict commit → skipped (covers L86)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [{"id": 1, "commits": ["bad"]}],
                }
            ],
        )
        assert _load_issue_specs(repo) == []

    def test_missing_fix_or_parent_commit(self, tmp_path: Path) -> None:
        """Empty hash/parents → skipped (covers L91)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [
                        {
                            "id": 1,
                            "commits": [{"hash": "", "parents": "abc"}],
                        }
                    ],
                }
            ],
        )
        assert _load_issue_specs(repo) == []

    def test_file_item_not_dict(self, tmp_path: Path) -> None:
        """Non-dict file entry is skipped (covers L98)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [
                        {
                            "id": 1,
                            "title": "t",
                            "commits": [
                                {
                                    "hash": "fix",
                                    "parents": "bug",
                                    "stat": {"files": ["not_a_dict"]},
                                }
                            ],
                        }
                    ],
                }
            ],
        )
        specs = _load_issue_specs(repo)
        assert len(specs) == 1
        assert specs[0].changed_python_files == []


class TestPyBugHiveChooseTarget:
    """Tests for _choose_target_source_file."""

    def test_single_non_test_file(self) -> None:
        """One non-test file → selected (covers L124)."""
        spec = PyBugHiveIssueSpec(
            username="u",
            repository="r",
            issue_id=1,
            title="t",
            fix_commit="f",
            parent_commit="p",
            changed_python_files=["src/mod.py", "tests/test_mod.py"],
            test_steps="",
            test_steps_full="",
        )
        assert _choose_target_source_file(spec) == "src/mod.py"

    def test_single_total_file(self) -> None:
        """One total file → selected even if test (covers L125-126)."""
        spec = PyBugHiveIssueSpec(
            username="u",
            repository="r",
            issue_id=1,
            title="t",
            fix_commit="f",
            parent_commit="p",
            changed_python_files=["tests/test_mod.py"],
            test_steps="",
            test_steps_full="",
        )
        assert _choose_target_source_file(spec) == "tests/test_mod.py"

    def test_multiple_non_test_files_returns_none(self) -> None:
        """Multiple non-test files → None."""
        spec = PyBugHiveIssueSpec(
            username="u",
            repository="r",
            issue_id=1,
            title="t",
            fix_commit="f",
            parent_commit="p",
            changed_python_files=["src/a.py", "src/b.py"],
            test_steps="",
            test_steps_full="",
        )
        assert _choose_target_source_file(spec) is None


class TestPyBugHiveArchive:
    """Tests for archive member resolution."""

    def test_build_index_bad_zip(self, tmp_path: Path) -> None:
        """Corrupt zip files are counted as scan failures."""
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        index, failures = _build_archive_member_index({"key": bad})
        assert index == {}
        assert failures == 1

    def test_build_index_skips_directories(self, tmp_path: Path) -> None:
        """Directory entries in zip are excluded from index."""
        archive = tmp_path / "test.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("dir/", "")
            zf.writestr("dir/file.py", "code")
        index, _ = _build_archive_member_index({"k": archive})
        members = index[archive]
        assert "dir/" not in members
        assert "dir/file.py" in members

    def test_resolve_unsafe_path(self) -> None:
        """Unsafe path returns skip reason."""
        result, reason = _resolve_archive_member({}, commitish="c", rel_path="../evil.py")
        assert result is None
        assert reason == "unsafe_dataset_path"

    def test_resolve_not_found(self) -> None:
        """No matching member → not found."""
        result, reason = _resolve_archive_member(
            {Path("a.zip"): ("x/y.py",)},
            commitish="c",
            rel_path="missing.py",
        )
        assert result is None
        assert reason == "offline_dump_path_not_found"

    def test_resolve_ambiguous(self) -> None:
        """Multiple commit matches → ambiguous."""
        idx: dict[Path, tuple[str, ...]] = {
            Path("a.zip"): ("c1/src/demo.py", "c1_extra/src/demo.py"),
        }
        result, reason = _resolve_archive_member(idx, commitish="c1", rel_path="src/demo.py")
        assert result is None
        assert reason == "offline_dump_path_ambiguous"

    def test_resolve_commit_match_single(self) -> None:
        """Single commit match → success."""
        p = Path("a.zip")
        idx: dict[Path, tuple[str, ...]] = {
            p: ("c1/src/demo.py", "c2/src/demo.py"),
        }
        result, reason = _resolve_archive_member(idx, commitish="c1", rel_path="src/demo.py")
        assert result == (p, "c1/src/demo.py")
        assert reason is None

    def test_resolve_no_commit_match_single_all(self) -> None:
        """No commit match but single total match → success."""
        p = Path("a.zip")
        idx: dict[Path, tuple[str, ...]] = {p: ("other/src/demo.py",)}
        result, reason = _resolve_archive_member(idx, commitish="c1", rel_path="src/demo.py")
        assert result == (p, "other/src/demo.py")
        assert reason is None

    def test_resolve_no_commit_match_multiple_all(self) -> None:
        """No commit match + multiple total matches → ambiguous."""
        idx: dict[Path, tuple[str, ...]] = {
            Path("a.zip"): ("x/src/demo.py",),
            Path("b.zip"): ("y/src/demo.py",),
        }
        result, reason = _resolve_archive_member(idx, commitish="nomatch", rel_path="src/demo.py")
        assert result is None
        assert reason == "offline_dump_path_ambiguous"

    def test_read_archive_file_success(self, tmp_path: Path) -> None:
        """Successful read from archive."""
        archive = _make_pybughive_archive(tmp_path)
        index, _ = _build_archive_member_index({"k": archive})
        content, reason = _read_archive_file(index, commitish="bug123", rel_path="src/demo.py")
        assert content is not None
        assert reason is None
        assert "return a - b" in content

    def test_read_archive_file_reason_propagated(self) -> None:
        """When resolve fails, reason is propagated."""
        content, reason = _read_archive_file({}, commitish="c", rel_path="../bad.py")
        assert content is None
        assert reason == "unsafe_dataset_path"

    def test_read_archive_file_bad_read(self, tmp_path: Path) -> None:
        """Corrupt archive data returns scan failure reason."""
        archive = tmp_path / "bad_content.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("src/demo.py", "content")
        # Build index, then corrupt
        index, _ = _build_archive_member_index({"k": archive})
        # Overwrite archive with garbage to make reading fail
        archive.write_bytes(b"corrupted data here")
        content, reason = _read_archive_file(index, commitish="x", rel_path="src/demo.py")
        assert content is None
        assert reason == "offline_dump_archive_scan_failed"


class TestPyBugHiveBuildMetadata:
    """Tests for _build_case_metadata."""

    def test_metadata_fields(self) -> None:
        """Metadata has all expected fields."""
        spec = PyBugHiveIssueSpec(
            username="u",
            repository="repo",
            issue_id=42,
            title="title",
            fix_commit="f",
            parent_commit="p",
            changed_python_files=["src/mod.py"],
            test_steps="pytest tests/test.py",
            test_steps_full="",
        )
        meta = pbh_build_case_metadata(spec=spec, native_command=_NATIVE_CMD, fix_lines=5)
        assert meta["bug_source"] == "pybughive"
        assert meta["difficulty"] == "medium"
        assert meta["pybughive_project"] == "u/repo"
        assert meta["pybughive_issue_id"] == 42


class TestPyBugHiveImport:
    """Integration tests for import_pybughive."""

    def test_empty_archives(self, tmp_path: Path) -> None:
        """No offline archives → all skipped (covers L258-269)."""
        repo = _make_pybughive_repo(tmp_path)
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={},
            data_root=tmp_path / "work",
        )
        assert summary.imported == 0
        assert "offline_dumps_missing" in summary.skip_reasons

    def test_all_archives_unreadable(self, tmp_path: Path) -> None:
        """All archives corrupt → empty index (covers L272-285)."""
        repo = _make_pybughive_repo(tmp_path)
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"bad": bad},
            data_root=tmp_path / "work",
        )
        assert summary.imported == 0
        assert "offline_dump_archive_scan_failed" in summary.skip_reasons

    def test_case_already_exists(self, tmp_path: Path) -> None:
        """Existing case dir → skipped (covers L295-296)."""
        repo = _make_pybughive_repo(tmp_path)
        archive = _make_pybughive_archive(tmp_path)
        out_dir = tmp_path / "cases"
        (out_dir / "pbh_demo_7").mkdir(parents=True)

        summary = import_pybughive(
            out_dir,
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "case_already_exists" in summary.skip_reasons

    def test_unsupported_native_command(self, tmp_path: Path) -> None:
        """Unsupported test command → skipped (covers L301-302)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [
                        {
                            "id": 1,
                            "title": "t",
                            "testSteps": "tox -e py39",
                            "commits": [
                                {
                                    "hash": "f",
                                    "parents": "p",
                                    "stat": {"files": [{"filename": "src/mod.py"}]},
                                }
                            ],
                        }
                    ],
                }
            ],
        )
        archive = _make_pybughive_archive(tmp_path)
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
        )
        assert "unsupported_native_test_command" in summary.skip_reasons

    def test_unsupported_patch_file_count(self, tmp_path: Path) -> None:
        """Multiple non-test changed files → skipped (covers L306-307)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [
                        {
                            "id": 1,
                            "title": "t",
                            "testSteps": "pytest -q tests/test.py",
                            "commits": [
                                {
                                    "hash": "f",
                                    "parents": "p",
                                    "stat": {
                                        "files": [
                                            {"filename": "src/a.py"},
                                            {"filename": "src/b.py"},
                                        ]
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        )
        archive = _make_pybughive_archive(tmp_path)
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
        )
        assert "unsupported_patch_file_count" in summary.skip_reasons

    def test_empty_oracle_patch(self, tmp_path: Path) -> None:
        """Identical buggy/fixed code → empty patch (covers L352-353)."""
        repo = _make_pybughive_repo(tmp_path)
        # Same content in both commits
        archive = _make_pybughive_archive(
            tmp_path,
            members={
                "dump/bug123/src/demo.py": BUGGY_CODE,
                "dump/fix123/src/demo.py": BUGGY_CODE,  # Same as buggy
                "dump/fix123/tests/test_demo.py": TEST_CODE,
            },
        )
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
        )
        assert "empty_oracle_patch" in summary.skip_reasons

    def test_test_file_fallback_to_parent(self, tmp_path: Path) -> None:
        """Test file not in fix commit → fallback to parent (covers L332, 338)."""
        repo = _make_pybughive_repo(tmp_path)
        archive = _make_pybughive_archive(
            tmp_path,
            members={
                "dump/bug123/src/demo.py": BUGGY_CODE,
                "dump/fix123/src/demo.py": FIXED_CODE,
                # Test only in parent commit
                "dump/bug123/tests/test_demo.py": TEST_CODE,
            },
        )
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
            validate=False,
        )
        assert summary.imported == 1

    def test_test_file_not_found(self, tmp_path: Path) -> None:
        """Test file missing in both commits (covers L338-340)."""
        repo = _make_pybughive_repo(tmp_path)
        archive = _make_pybughive_archive(
            tmp_path,
            members={
                "dump/bug123/src/demo.py": BUGGY_CODE,
                "dump/fix123/src/demo.py": FIXED_CODE,
                # No test files at all
            },
        )
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
        )
        assert summary.imported == 0
        # Test file wasn't found in either commit
        total_skips = sum(summary.skip_reasons.values())
        assert total_skips >= 1

    def test_unsafe_test_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unsafe test path after normalization (covers L343-345, 348).

        We need _read_archive_file to succeed for the test file, but then
        normalize_relative_path to return None when called directly on the
        test path string at L341 inside the import loop.
        """
        repo = _make_pybughive_repo(tmp_path)
        archive = _make_pybughive_archive(tmp_path)

        original_normalize = __import__(
            "evals.importers.native_repro_helpers",
            fromlist=["normalize_relative_path"],
        ).normalize_relative_path

        # Track calls for the specific test path.
        # Call #1: inside _read_archive_file → _resolve_archive_member
        # Call #2: direct call at pybughive_importer.py L341
        test_path_calls = [0]

        def _normalize_that_fails_on_direct_test_call(raw: str) -> str | None:
            result = original_normalize(raw)
            if result == "tests/test_demo.py":
                test_path_calls[0] += 1
                # Fail on the 2nd call (the direct L341 call)
                if test_path_calls[0] >= 2:
                    return None
            return result

        monkeypatch.setattr(
            "evals.importers.pybughive_importer.normalize_relative_path",
            _normalize_that_fails_on_direct_test_call,
        )

        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
            validate=False,
        )
        assert summary.imported == 0
        assert "unsafe_dataset_path" in summary.skip_reasons

    def test_write_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ValueError from write_native_case_files (covers L372-375)."""
        repo = _make_pybughive_repo(tmp_path)
        archive = _make_pybughive_archive(tmp_path)
        monkeypatch.setattr(
            "evals.importers.pybughive_importer.write_native_case_files",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("unsafe")),
        )

        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "unsafe_dataset_path" in summary.skip_reasons

    def test_validation_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Schema validation failure (covers L378-382)."""
        repo = _make_pybughive_repo(tmp_path)
        archive = _make_pybughive_archive(tmp_path)
        monkeypatch.setattr(
            "evals.importers.pybughive_importer.validate_generated_case",
            lambda *a, **kw: (False, "bad schema"),
        )

        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
            validate=True,
        )
        assert "schema_validation_failed" in summary.skip_reasons

    def test_max_cases_limit(self, tmp_path: Path) -> None:
        """Stops importing after max_cases (covers L289)."""
        repo = _make_pybughive_repo(
            tmp_path,
            payload=[
                {
                    "username": "u",
                    "repository": "r",
                    "issues": [
                        {
                            "id": i,
                            "title": f"issue {i}",
                            "testSteps": "pytest -q tests/test_demo.py",
                            "commits": [
                                {
                                    "hash": f"fix{i}",
                                    "parents": f"bug{i}",
                                    "stat": {"files": [{"filename": "src/demo.py"}]},
                                }
                            ],
                        }
                        for i in range(1, 4)
                    ],
                }
            ],
        )
        archive = _make_pybughive_archive(
            tmp_path,
            members={
                **{f"dump/bug{i}/src/demo.py": BUGGY_CODE for i in range(1, 4)},
                **{f"dump/fix{i}/src/demo.py": FIXED_CODE for i in range(1, 4)},
                **{f"dump/fix{i}/tests/test_demo.py": TEST_CODE for i in range(1, 4)},
            },
        )
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
            max_cases=1,
            validate=False,
            dry_run=True,
        )
        assert summary.imported == 1

    def test_scan_failures_note(self, tmp_path: Path) -> None:
        """Scan failures produce a note in the summary (covers L393)."""
        repo = _make_pybughive_repo(tmp_path)
        good_archive = _make_pybughive_archive(tmp_path)
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")

        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"good": good_archive, "bad": bad},
            data_root=tmp_path / "work",
            validate=False,
        )
        assert any("unreadable" in n for n in summary.notes)

    def test_dry_run_import(self, tmp_path: Path) -> None:
        """Dry run records case IDs without writing files."""
        repo = _make_pybughive_repo(tmp_path)
        archive = _make_pybughive_archive(tmp_path)
        out = tmp_path / "cases"
        summary = import_pybughive(
            out,
            pybughive_repo=repo,
            offline_archives={"k": archive},
            data_root=tmp_path / "work",
            dry_run=True,
        )
        assert summary.imported == 1
        assert not (out / "pbh_demo_7" / "case.json").exists()

    def test_empty_archives_no_specs(self, tmp_path: Path) -> None:
        """No specs + empty archives → zero attempted."""
        repo = _make_pybughive_repo(tmp_path, payload=[])
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={},
            data_root=tmp_path / "work",
        )
        assert summary.attempted == 0
        assert summary.skip_reasons == {}

    def test_all_archives_corrupt_no_specs(self, tmp_path: Path) -> None:
        """No specs + all archives corrupt → zero attempted."""
        repo = _make_pybughive_repo(tmp_path, payload=[])
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        summary = import_pybughive(
            tmp_path / "cases",
            pybughive_repo=repo,
            offline_archives={"k": bad},
            data_root=tmp_path / "work",
        )
        assert summary.attempted == 0
        assert summary.skip_reasons == {}


# ===================================================================
# TrickyBugs Importer Tests
# ===================================================================


class TestTrickyBugsHelpers:
    """Tests for TrickyBugs helper functions."""

    def test_safe_slug_basic(self) -> None:
        """Basic slug generation."""
        assert _safe_slug("Hello World") == "hello_world"

    def test_safe_slug_empty(self) -> None:
        """Empty input → 'unknown'."""
        assert _safe_slug("") == "unknown"
        assert _safe_slug("   ") == "unknown"

    def test_safe_slug_special_chars(self) -> None:
        """Special characters are replaced."""
        assert _safe_slug("a/b@c") == "a_b_c"

    def test_first_string_found(self) -> None:
        """Returns first matching string value."""
        record = {"a": "hello", "b": "world"}
        assert _first_string(record, ["a", "b"]) == "hello"

    def test_first_string_skips_empty(self) -> None:
        """Skips empty/whitespace values."""
        record: dict[str, Any] = {"a": "", "b": "  ", "c": "found"}
        assert _first_string(record, ["a", "b", "c"]) == "found"

    def test_first_string_skips_non_string(self) -> None:
        """Skips non-string values."""
        record: dict[str, Any] = {"a": 123, "b": "ok"}
        assert _first_string(record, ["a", "b"]) == "ok"

    def test_first_string_none(self) -> None:
        """Returns None when no match."""
        assert _first_string({}, ["a"]) is None

    def test_iter_dict_nodes_dict(self) -> None:
        """Yields dict and nested dicts."""
        data: dict[str, Any] = {"a": {"b": 1}}
        nodes = list(_iter_dict_nodes(data))
        assert len(nodes) == 2

    def test_iter_dict_nodes_list(self) -> None:
        """Yields dicts from nested lists."""
        data = [{"a": 1}, {"b": 2}]
        nodes = list(_iter_dict_nodes(data))
        assert len(nodes) == 2

    def test_iter_dict_nodes_scalar(self) -> None:
        """Scalar yields nothing."""
        assert list(_iter_dict_nodes(42)) == []
        assert list(_iter_dict_nodes("string")) == []


class TestTrickyBugsCandidateFromRecord:
    """Tests for _candidate_from_record."""

    def test_valid_record(self) -> None:
        """Valid record produces candidate."""
        record: dict[str, Any] = {
            "pid": "p1",
            "program_id": "prog",
            "language": "python",
            "buggy_file": "src/buggy.py",
            "fixed_file": "src/fixed.py",
            "test_command": "pytest tests/test.py",
        }
        c = _candidate_from_record(record=record, origin="test")
        assert c is not None
        assert c.pid == "p1"
        assert c.buggy_rel_path == "src/buggy.py"

    def test_missing_buggy_or_fixed(self) -> None:
        """Missing buggy/fixed file → None (covers L103)."""
        record: dict[str, Any] = {
            "pid": "p1",
            "buggy_file": "src/buggy.py",
            # No fixed_file
        }
        assert _candidate_from_record(record=record, origin="test") is None

    def test_wrong_language(self) -> None:
        """Non-Python language → None (covers L107)."""
        record: dict[str, Any] = {
            "language": "java",
            "buggy_file": "src/buggy.py",
            "fixed_file": "src/fixed.py",
        }
        assert _candidate_from_record(record=record, origin="test") is None

    def test_unsafe_paths(self) -> None:
        """Unsafe paths → None (covers L112)."""
        record: dict[str, Any] = {
            "buggy_file": "../evil/buggy.py",
            "fixed_file": "src/fixed.py",
        }
        assert _candidate_from_record(record=record, origin="test") is None

    def test_non_py_extension(self) -> None:
        """Non-.py extension → None (covers L114)."""
        record: dict[str, Any] = {
            "buggy_file": "src/buggy.txt",
            "fixed_file": "src/fixed.txt",
        }
        assert _candidate_from_record(record=record, origin="test") is None

    def test_no_language_key_accepted(self) -> None:
        """No language key → accepted (Python assumed)."""
        record: dict[str, Any] = {
            "buggy_file": "src/buggy.py",
            "fixed_file": "src/fixed.py",
        }
        c = _candidate_from_record(record=record, origin="test")
        assert c is not None

    def test_python3_language_accepted(self) -> None:
        """'python3' language → accepted."""
        record: dict[str, Any] = {
            "language": "Python3",
            "buggy_file": "src/buggy.py",
            "fixed_file": "src/fixed.py",
        }
        c = _candidate_from_record(record=record, origin="test")
        assert c is not None


class TestTrickyBugsIterCandidates:
    """Tests for _iter_python_candidates."""

    def test_large_json_skipped(self, tmp_path: Path) -> None:
        """JSON files exceeding size limit are skipped (covers L135)."""
        root = tmp_path / "tb"
        root.mkdir()
        big = root / "big.json"
        big.write_text("x" * 11_000_000, encoding="utf-8")
        assert _iter_python_candidates(root) == []

    def test_os_error_handled(self, tmp_path: Path) -> None:
        """OSError during JSON read is handled (covers L137-138)."""
        root = tmp_path / "tb"
        root.mkdir()
        # Create a directory named .json which will fail on read_text
        bad = root / "bad.json"
        bad.mkdir()
        assert _iter_python_candidates(root) == []

    def test_invalid_json_handled(self, tmp_path: Path) -> None:
        """Invalid JSON is handled (covers L139-140)."""
        root = tmp_path / "tb"
        root.mkdir()
        (root / "bad.json").write_text("{invalid json", encoding="utf-8")
        assert _iter_python_candidates(root) == []

    def test_deduplication(self, tmp_path: Path) -> None:
        """Duplicate candidates are deduplicated."""
        root = tmp_path / "tb"
        root.mkdir()
        record: dict[str, Any] = {
            "pid": "p1",
            "program_id": "prog",
            "buggy_file": "src/buggy.py",
            "fixed_file": "src/fixed.py",
        }
        # Write same record in two files
        (root / "a.json").write_text(json.dumps([record]), encoding="utf-8")
        (root / "b.json").write_text(json.dumps([record]), encoding="utf-8")
        candidates = _iter_python_candidates(root)
        assert len(candidates) == 1


class TestTrickyBugsReadRelativeFile:
    """Tests for _read_relative_file."""

    def test_reads_existing_file(self, tmp_path: Path) -> None:
        """Successfully reads a file."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "mod.py").write_text("code", encoding="utf-8")
        assert _read_relative_file(tmp_path, "src/mod.py") == "code"

    def test_unsafe_path_returns_none(self, tmp_path: Path) -> None:
        """Unsafe path → None (covers L174)."""
        assert _read_relative_file(tmp_path, "../evil.py") is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Nonexistent file → None (covers L177)."""
        assert _read_relative_file(tmp_path, "missing.py") is None

    def test_directory_returns_none(self, tmp_path: Path) -> None:
        """Directory path → None (not a file)."""
        (tmp_path / "src").mkdir()
        assert _read_relative_file(tmp_path, "src") is None


class TestTrickyBugsImport:
    """Integration tests for import_trickybugs."""

    def _make_trickybugs_root(
        self,
        tmp_path: Path,
        *,
        records: list[dict[str, Any]] | None = None,
        create_source_files: bool = True,
    ) -> Path:
        """Create a TrickyBugs dataset root."""
        root = tmp_path / "TrickyBugs"
        src = root / "dataset"
        src.mkdir(parents=True, exist_ok=True)

        if create_source_files:
            (src / "buggy.py").write_text(BUGGY_CODE, encoding="utf-8")
            (src / "fixed.py").write_text(FIXED_CODE, encoding="utf-8")
            (src / "test_it.py").write_text(TEST_CODE, encoding="utf-8")

        if records is None:
            records = [
                {
                    "pid": "p1",
                    "program_id": "prog",
                    "language": "python",
                    "buggy_file": "dataset/buggy.py",
                    "fixed_file": "dataset/fixed.py",
                    "test_command": "pytest -q dataset/test_it.py",
                }
            ]
        (root / "metadata.json").write_text(json.dumps(records), encoding="utf-8")
        return root

    def test_max_cases_limit(self, tmp_path: Path) -> None:
        """Stops after max_cases (covers L203)."""
        root = self._make_trickybugs_root(tmp_path)
        # Add second record
        src = root / "dataset"
        (src / "buggy2.py").write_text("def f():\n    return 0\n", encoding="utf-8")
        (src / "fixed2.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        payload = [
            {
                "pid": "p1",
                "program_id": "prog1",
                "buggy_file": "dataset/buggy.py",
                "fixed_file": "dataset/fixed.py",
                "test_command": "pytest -q dataset/test_it.py",
            },
            {
                "pid": "p2",
                "program_id": "prog2",
                "buggy_file": "dataset/buggy2.py",
                "fixed_file": "dataset/fixed2.py",
                "test_command": "pytest -q dataset/test_it.py",
            },
        ]
        (root / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
            max_cases=1,
            validate=False,
            dry_run=True,
        )
        assert summary.imported == 1

    def test_case_already_exists(self, tmp_path: Path) -> None:
        """Existing case dir → skipped (covers L209-210)."""
        root = self._make_trickybugs_root(tmp_path)
        out_dir = tmp_path / "cases"
        (out_dir / "tb_p1_prog").mkdir(parents=True)

        summary = import_trickybugs(
            out_dir,
            trickybugs_root=root,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "case_already_exists" in summary.skip_reasons

    def test_missing_test_command(self, tmp_path: Path) -> None:
        """No test_command → skipped (covers L212-213 in import)."""
        root = self._make_trickybugs_root(
            tmp_path,
            records=[
                {
                    "pid": "p1",
                    "program_id": "prog",
                    "buggy_file": "dataset/buggy.py",
                    "fixed_file": "dataset/fixed.py",
                    # No test_command
                }
            ],
        )
        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        assert "missing_native_test_oracle_mapping" in summary.skip_reasons

    def test_unsupported_native_command(self, tmp_path: Path) -> None:
        """Unsupported test command → skipped (covers L218-219)."""
        root = self._make_trickybugs_root(
            tmp_path,
            records=[
                {
                    "pid": "p1",
                    "program_id": "prog",
                    "buggy_file": "dataset/buggy.py",
                    "fixed_file": "dataset/fixed.py",
                    "test_command": "tox -e py39",
                }
            ],
        )
        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        assert "unsupported_native_test_command" in summary.skip_reasons

    def test_missing_buggy_source(self, tmp_path: Path) -> None:
        """Buggy file doesn't exist → skipped (covers L224-225)."""
        root = self._make_trickybugs_root(
            tmp_path,
            records=[
                {
                    "pid": "p1",
                    "program_id": "prog",
                    "buggy_file": "dataset/nonexistent.py",
                    "fixed_file": "dataset/fixed.py",
                    "test_command": "pytest -q dataset/test_it.py",
                }
            ],
        )
        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        assert "missing_buggy_source" in summary.skip_reasons

    def test_missing_fixed_source(self, tmp_path: Path) -> None:
        """Fixed file doesn't exist → skipped (covers L227-228)."""
        root = self._make_trickybugs_root(
            tmp_path,
            records=[
                {
                    "pid": "p1",
                    "program_id": "prog",
                    "buggy_file": "dataset/buggy.py",
                    "fixed_file": "dataset/nonexistent.py",
                    "test_command": "pytest -q dataset/test_it.py",
                }
            ],
        )
        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        assert "missing_fixed_source" in summary.skip_reasons

    def test_empty_source_file(self, tmp_path: Path) -> None:
        """Empty source file → skipped (covers L230-231)."""
        root = self._make_trickybugs_root(tmp_path)
        (root / "dataset" / "buggy.py").write_text("   \n", encoding="utf-8")

        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        assert "empty_source_file" in summary.skip_reasons

    def test_unsafe_test_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unsafe test path → skipped (covers L238-240, 248)."""
        root = self._make_trickybugs_root(
            tmp_path,
            records=[
                {
                    "pid": "p1",
                    "program_id": "prog",
                    "buggy_file": "dataset/buggy.py",
                    "fixed_file": "dataset/fixed.py",
                    "test_command": "pytest -q dataset/test_it.py",
                }
            ],
        )

        # Make normalize_relative_path return None for test paths
        original_normalize = __import__(
            "evals.importers.native_repro_helpers",
            fromlist=["normalize_relative_path"],
        ).normalize_relative_path

        def _normalize_unsafe_for_test(raw: str) -> str | None:
            result = original_normalize(raw)
            # Source paths stay valid; only the native test path is unsafe.
            if result and "test_it" in result:
                return None
            return result

        monkeypatch.setattr(
            "evals.importers.trickybugs_importer.normalize_relative_path",
            _normalize_unsafe_for_test,
        )

        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        # This used to pass only because the schema check rejected every native case.
        assert summary.imported == 0
        assert summary.skip_reasons == {"unsafe_dataset_path": 1}

    def test_test_file_not_found(self, tmp_path: Path) -> None:
        """Test file missing on disk (covers L243-245)."""
        root = self._make_trickybugs_root(
            tmp_path,
            records=[
                {
                    "pid": "p1",
                    "program_id": "prog",
                    "buggy_file": "dataset/buggy.py",
                    "fixed_file": "dataset/fixed.py",
                    "test_command": "pytest -q dataset/missing_test.py",
                }
            ],
        )
        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        assert "native_test_file_not_found" in summary.skip_reasons

    def test_empty_oracle_patch(self, tmp_path: Path) -> None:
        """Same buggy/fixed code → empty patch (covers L256-257)."""
        root = self._make_trickybugs_root(tmp_path)
        # Make both files identical
        (root / "dataset" / "fixed.py").write_text(BUGGY_CODE, encoding="utf-8")

        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
        )
        assert "empty_oracle_patch" in summary.skip_reasons

    def test_dry_run(self, tmp_path: Path) -> None:
        """Dry run records IDs without writing files (covers L288-290)."""
        root = self._make_trickybugs_root(tmp_path)
        out = tmp_path / "cases"
        summary = import_trickybugs(
            out,
            trickybugs_root=root,
            data_root=tmp_path / "work",
            validate=False,
            dry_run=True,
        )
        assert summary.imported == 1
        assert not (out / "tb_p1_prog").exists()

    def test_write_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ValueError from write → skipped (covers L292-293, 300-303)."""
        root = self._make_trickybugs_root(tmp_path)
        monkeypatch.setattr(
            "evals.importers.trickybugs_importer.write_native_case_files",
            lambda *a, **kw: (_ for _ in ()).throw(ValueError("unsafe")),
        )

        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
            validate=False,
        )
        assert "unsafe_dataset_path" in summary.skip_reasons

    def test_validation_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Schema validation failure (covers L305-310)."""
        root = self._make_trickybugs_root(tmp_path)
        monkeypatch.setattr(
            "evals.importers.trickybugs_importer.validate_generated_case",
            lambda *a, **kw: (False, "bad schema"),
        )

        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
            validate=True,
        )
        assert "schema_validation_failed" in summary.skip_reasons

    def test_successful_import_with_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full successful import with validation (covers L312)."""
        root = self._make_trickybugs_root(tmp_path)
        monkeypatch.setattr(
            "evals.importers.trickybugs_importer.validate_generated_case",
            lambda *a, **kw: (True, "ok"),
        )

        summary = import_trickybugs(
            tmp_path / "cases",
            trickybugs_root=root,
            data_root=tmp_path / "work",
            validate=True,
        )
        assert summary.imported == 1
        assert summary.skipped == 0
