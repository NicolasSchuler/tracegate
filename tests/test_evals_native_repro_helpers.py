"""Tests for native reproduction helpers (path normalization, pytest commands,
git operations, case file writing, and internal parsing functions).

Covers missing lines: 47, 69, 75-76, 78, 80, 88, 102, 126, 128-129,
136-140, 147, 157-158, 160, 199, 201, 205, 208, 214, 256, 267, 269,
271, 283-285, 291.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from evals.importers.native_repro_helpers import (
    SKIP_GIT_CLONE_FAILED,
    GitShowFileRequest,
    NativePytestCommand,
    _collect_selector_tokens,
    _extract_single_command_line,
    _extract_test_paths,
    _is_supported_selector_path,
    _option_arity,
    _starts_with_env_assignment,
    _write_relative_file,
    ensure_git_repo,
    git_show_file,
    normalize_native_pytest_command,
    normalize_relative_path,
    write_native_case_files,
)
from llmdebug import _json as json

# ---------------------------------------------------------------------------
# normalize_relative_path
# ---------------------------------------------------------------------------


class TestNormalizeRelativePath:
    """Tests for ``normalize_relative_path``."""

    def test_valid_path(self) -> None:
        assert normalize_relative_path("src/demo.py") == "src/demo.py"

    def test_dotslash_prefix_stripped(self) -> None:
        """``./`` prefix is removed via parts filter."""
        assert normalize_relative_path("./src/demo.py") == "src/demo.py"

    def test_rejects_parent_traversal(self) -> None:
        assert normalize_relative_path("../demo.py") is None

    def test_rejects_absolute(self) -> None:
        assert normalize_relative_path("/abs/path.py") is None

    def test_empty_string_returns_none(self) -> None:
        """Line 47: empty string after strip → None."""
        assert normalize_relative_path("") is None

    def test_whitespace_only_returns_none(self) -> None:
        """Line 47: whitespace-only string → None."""
        assert normalize_relative_path("   ") is None

    def test_backslash_normalized(self) -> None:
        assert normalize_relative_path("src\\demo.py") == "src/demo.py"


# ---------------------------------------------------------------------------
# normalize_native_pytest_command
# ---------------------------------------------------------------------------


class TestNormalizeNativePytestCommand:
    """Tests for ``normalize_native_pytest_command``."""

    def test_pytest_with_selector(self) -> None:
        cmd = normalize_native_pytest_command("pytest -q tests/test_a.py::test_x")
        assert cmd is not None
        assert cmd.python_args == [
            "-m",
            "pytest",
            "-q",
            "tests/test_a.py::test_x",
        ]
        assert cmd.test_paths == ["tests/test_a.py"]

    def test_python_m_pytest(self) -> None:
        cmd = normalize_native_pytest_command("python -m pytest tests/test_b.py")
        assert cmd is not None
        assert cmd.source_test_selector == "tests/test_b.py"

    def test_directory_selector(self) -> None:
        cmd = normalize_native_pytest_command("pytest -q tests/")
        assert cmd is not None
        assert cmd.test_paths == ["tests"]

    def test_rejects_shell_composition(self) -> None:
        assert normalize_native_pytest_command("pytest tests/test_a.py && echo done") is None

    def test_rejects_unsupported_executable(self) -> None:
        assert normalize_native_pytest_command("pipenv run pytest -- tests/test_a.py") is None

    def test_rejects_bare_pytest_no_selector(self) -> None:
        assert normalize_native_pytest_command("pytest") is None

    def test_rejects_k_only_no_paths(self) -> None:
        """No file-path selectors → None."""
        assert normalize_native_pytest_command("pytest -k expr") is None

    def test_multiline_script_rejected(self) -> None:
        """Line 69/208: multi-line scripts rejected by _extract_single_command_line."""
        script = "#!/bin/bash\npytest tests/test_a.py\npytest tests/test_b.py"
        assert normalize_native_pytest_command(script) is None

    def test_shlex_error_returns_none(self) -> None:
        """Lines 75-76: un-parseable shell quoting → None."""
        assert normalize_native_pytest_command("pytest 'unclosed") is None

    def test_empty_tokens_returns_none(self) -> None:
        """Line 78: empty token list after shlex → None."""
        # A command text that becomes a blank single line.
        assert normalize_native_pytest_command("   ") is None

    def test_env_assignment_prefix_rejected(self) -> None:
        """Line 80: leading env var assignment → None."""
        assert normalize_native_pytest_command("FOO=bar pytest tests/test_a.py") is None

    def test_python_without_m_pytest_rejected(self) -> None:
        """Line 88: python + wrong subcommand → None."""
        assert normalize_native_pytest_command("python tests/test_a.py") is None

    def test_python_m_without_pytest_rejected(self) -> None:
        """Line 88: python -m <not-pytest> → None."""
        assert normalize_native_pytest_command("python -m unittest tests/test_a.py") is None

    def test_python_only_m_rejected(self) -> None:
        """Line 88: ``python -m`` with no following token → None."""
        assert normalize_native_pytest_command("python -m") is None

    def test_empty_selector_fallback(self) -> None:
        """Line 102: selector is empty after strip → falls back to
        normalized_tail join."""
        cmd = normalize_native_pytest_command("pytest --rootdir /tmp tests/test_a.py")
        assert cmd is not None
        assert "tests/test_a.py" in cmd.source_test_selector

    def test_pytest_dot_test_form(self) -> None:
        """``py.test`` is accepted as an alias for pytest."""
        cmd = normalize_native_pytest_command("py.test tests/test_a.py")
        assert cmd is not None
        assert cmd.test_paths == ["tests/test_a.py"]

    def test_python3_m_pytest(self) -> None:
        """``python3 -m pytest`` accepted."""
        cmd = normalize_native_pytest_command("python3 -m pytest tests/test_x.py")
        assert cmd is not None
        assert cmd.test_paths == ["tests/test_x.py"]

    def test_pipe_rejected(self) -> None:
        assert normalize_native_pytest_command("pytest tests/t.py | tee log") is None

    def test_backtick_rejected(self) -> None:
        assert normalize_native_pytest_command("pytest `echo tests/t.py`") is None

    def test_dollar_paren_rejected(self) -> None:
        assert normalize_native_pytest_command("pytest $(echo tests/t.py)") is None


# ---------------------------------------------------------------------------
# _extract_single_command_line (private helper)
# ---------------------------------------------------------------------------


class TestExtractSingleCommandLine:
    """Tests for ``_extract_single_command_line``."""

    def test_simple_line(self) -> None:
        assert _extract_single_command_line("pytest tests/t.py") == ("pytest tests/t.py")

    def test_blank_lines_skipped(self) -> None:
        """Line 199: blank lines are skipped."""
        assert _extract_single_command_line("\n\npytest tests/t.py\n\n") == "pytest tests/t.py"

    def test_shebang_skipped(self) -> None:
        """Line 201: shebang line is ignored."""
        text = "#!/bin/bash\npytest tests/t.py"
        assert _extract_single_command_line(text) == "pytest tests/t.py"

    def test_comment_skipped(self) -> None:
        """Comments are ignored."""
        text = "# a comment\npytest tests/t.py"
        assert _extract_single_command_line(text) == "pytest tests/t.py"

    def test_set_directive_skipped(self) -> None:
        """Line 205: ``set -e`` style lines are skipped."""
        text = "set -e\npytest tests/t.py"
        assert _extract_single_command_line(text) == "pytest tests/t.py"

    def test_multiline_returns_none(self) -> None:
        """Line 208: more than one real command line → None."""
        text = "pytest tests/a.py\npytest tests/b.py"
        assert _extract_single_command_line(text) is None

    def test_no_command_returns_none(self) -> None:
        """Line 208: zero real lines → None."""
        assert _extract_single_command_line("# only comments\n") is None

    def test_empty_input(self) -> None:
        assert _extract_single_command_line("") is None


# ---------------------------------------------------------------------------
# _starts_with_env_assignment
# ---------------------------------------------------------------------------


class TestStartsWithEnvAssignment:
    """Tests for ``_starts_with_env_assignment``."""

    def test_empty_tokens(self) -> None:
        """Line 214: empty list → False."""
        assert _starts_with_env_assignment([]) is False

    def test_env_assignment(self) -> None:
        assert _starts_with_env_assignment(["FOO=bar", "pytest"]) is True

    def test_no_env_assignment(self) -> None:
        assert _starts_with_env_assignment(["pytest", "tests/"]) is False


# ---------------------------------------------------------------------------
# _option_arity
# ---------------------------------------------------------------------------


class TestOptionArity:
    """Tests for ``_option_arity``."""

    def test_value_option_returns_two(self) -> None:
        assert _option_arity("-k") == 2

    def test_long_option_with_eq_returns_one(self) -> None:
        """Line 256: ``--opt=val`` → arity 1."""
        assert _option_arity("--rootdir=/tmp") == 1

    def test_unknown_flag_returns_one(self) -> None:
        assert _option_arity("-v") == 1


# ---------------------------------------------------------------------------
# _collect_selector_tokens
# ---------------------------------------------------------------------------


class TestCollectSelectorTokens:
    """Tests for ``_collect_selector_tokens``."""

    def test_simple_path(self) -> None:
        assert _collect_selector_tokens(["tests/t.py"]) == ["tests/t.py"]

    def test_double_dash_skipped(self) -> None:
        result = _collect_selector_tokens(["--", "tests/t.py"])
        assert result == ["tests/t.py"]

    def test_option_with_value_skipped(self) -> None:
        result = _collect_selector_tokens(["-k", "expr", "tests/t.py"])
        assert result == ["tests/t.py"]


# ---------------------------------------------------------------------------
# _extract_test_paths
# ---------------------------------------------------------------------------


class TestExtractTestPaths:
    """Tests for ``_extract_test_paths``."""

    def test_basic_path(self) -> None:
        assert _extract_test_paths(["tests/test_a.py"]) == [
            "tests/test_a.py",
        ]

    def test_strips_node_id(self) -> None:
        assert _extract_test_paths(["tests/test_a.py::TestClass::test_m"]) == ["tests/test_a.py"]

    def test_unsafe_path_skipped(self) -> None:
        """Line 267: normalize returns None for absolute path → skip."""
        assert _extract_test_paths(["/abs/test_a.py"]) == []

    def test_unsupported_selector_skipped(self) -> None:
        """Line 269: bare word that's not test/tests → skip."""
        assert _extract_test_paths(["randomword"]) == []

    def test_duplicate_path_deduplicated(self) -> None:
        """Line 271: seen set prevents duplicates."""
        result = _extract_test_paths(["tests/test_a.py::test_x", "tests/test_a.py::test_y"])
        assert result == ["tests/test_a.py"]


# ---------------------------------------------------------------------------
# _is_supported_selector_path
# ---------------------------------------------------------------------------


class TestIsSupportedSelectorPath:
    """Tests for ``_is_supported_selector_path``."""

    def test_py_file(self) -> None:
        assert (
            _is_supported_selector_path(raw="tests/test_a.py", normalized="tests/test_a.py") is True
        )

    def test_trailing_slash_directory(self) -> None:
        assert _is_supported_selector_path(raw="tests/", normalized="tests") is True

    def test_trailing_backslash_directory(self) -> None:
        assert _is_supported_selector_path(raw="tests\\", normalized="tests") is True

    def test_subpath_directory(self) -> None:
        """Line 284: slash in normalized → directory selector."""
        assert _is_supported_selector_path(raw="pkg/tests", normalized="pkg/tests") is True

    def test_bare_tests_keyword(self) -> None:
        """Line 285: bare ``tests`` is accepted."""
        assert _is_supported_selector_path(raw="tests", normalized="tests") is True

    def test_bare_test_keyword(self) -> None:
        """Line 285: bare ``test`` is accepted."""
        assert _is_supported_selector_path(raw="test", normalized="test") is True

    def test_bare_unknown_word(self) -> None:
        """Branch → False for unknown bare word."""
        assert _is_supported_selector_path(raw="foobar", normalized="foobar") is False


# ---------------------------------------------------------------------------
# _write_relative_file
# ---------------------------------------------------------------------------


class TestWriteRelativeFile:
    """Tests for ``_write_relative_file``."""

    def test_writes_file(self, tmp_path: Path) -> None:
        _write_relative_file(tmp_path, "a/b.py", "hello\n")
        assert (tmp_path / "a" / "b.py").read_text() == "hello\n"

    def test_unsafe_path_raises(self, tmp_path: Path) -> None:
        """Line 291: unsafe path → ValueError."""
        with pytest.raises(ValueError, match="unsafe relative path"):
            _write_relative_file(tmp_path, "../escape.py", "bad")

    def test_empty_path_raises(self, tmp_path: Path) -> None:
        """Line 291: empty path → ValueError."""
        with pytest.raises(ValueError, match="unsafe relative path"):
            _write_relative_file(tmp_path, "", "bad")


# ---------------------------------------------------------------------------
# write_native_case_files
# ---------------------------------------------------------------------------


class TestWriteNativeCaseFiles:
    """Tests for ``write_native_case_files``."""

    def test_contract_placeholder_written(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        write_native_case_files(
            case_dir,
            buggy_files={"src/demo.py": "x = 1\n"},
            test_files={
                "tests/test_demo.py": "def test_demo():\n    assert True\n",
            },
            oracle_patch="diff --git a/src/demo.py b/src/demo.py\n",
            metadata={"category": "logic_error"},
        )
        assert (case_dir / "src" / "demo.py").exists()
        assert (case_dir / "tests" / "test_demo.py").exists()
        assert (case_dir / "test_buggy.py").exists()
        loaded = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        assert loaded["category"] == "logic_error"

    def test_no_placeholder_when_test_buggy_exists(
        self,
        tmp_path: Path,
    ) -> None:
        """Line 181→187 branch: skip placeholder if test_buggy.py is
        already among the provided files."""
        case_dir = tmp_path / "case2"
        write_native_case_files(
            case_dir,
            buggy_files={"src/demo.py": "x = 1\n"},
            test_files={
                "test_buggy.py": "def test_it(): pass\n",
            },
            oracle_patch="patch\n",
            metadata={},
        )
        content = (case_dir / "test_buggy.py").read_text(encoding="utf-8")
        assert content == "def test_it(): pass\n"  # original, not stub


# ---------------------------------------------------------------------------
# ensure_git_repo
# ---------------------------------------------------------------------------


class TestEnsureGitRepo:
    """Tests for ``ensure_git_repo``, mocking subprocess."""

    def test_fresh_clone_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 128-129: clone succeeds on fresh directory."""
        clone_dir = tmp_path / "repo"

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok, skip = ensure_git_repo("https://x.com/r.git", clone_dir)
        assert ok is True
        assert skip is None

    def test_clone_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 136-137: clone returns non-zero → failure."""
        clone_dir = tmp_path / "repo"

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            return SimpleNamespace(returncode=1, stdout="", stderr="err")

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok, skip = ensure_git_repo("https://x.com/r.git", clone_dir)
        assert ok is False
        assert skip == SKIP_GIT_CLONE_FAILED

    def test_fetch_success_on_existing_repo(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 115→125 branch: existing .git → fetch succeeds."""
        clone_dir = tmp_path / "repo"
        clone_dir.mkdir()
        (clone_dir / ".git").mkdir()

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok, skip = ensure_git_repo("https://x.com/r.git", clone_dir)
        assert ok is True
        assert skip is None

    def test_fetch_failure_triggers_reclone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 126: fetch fails → rmtree → re-clone."""
        clone_dir = tmp_path / "repo"
        clone_dir.mkdir()
        (clone_dir / ".git").mkdir()

        call_count = 0

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # fetch fails
                return SimpleNamespace(
                    returncode=128,
                    stdout="",
                    stderr="err",
                )
            # clone succeeds
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok, skip = ensure_git_repo("https://x.com/r.git", clone_dir)
        assert ok is True
        assert skip is None
        assert call_count == 2

    def test_timeout_returns_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 139-140: TimeoutExpired → failure."""
        clone_dir = tmp_path / "repo"

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            raise subprocess.TimeoutExpired(cmd, 180)

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok, skip = ensure_git_repo("https://x.com/r.git", clone_dir)
        assert ok is False
        assert skip == SKIP_GIT_CLONE_FAILED

    def test_oserror_returns_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 139-140: OSError → failure."""
        clone_dir = tmp_path / "repo"

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            raise OSError("disk full")

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok, skip = ensure_git_repo("https://x.com/r.git", clone_dir)
        assert ok is False
        assert skip == SKIP_GIT_CLONE_FAILED


# ---------------------------------------------------------------------------
# git_show_file
# ---------------------------------------------------------------------------


class TestGitShowFile:
    """Tests for ``git_show_file``, mocking subprocess."""

    def test_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        req = GitShowFileRequest(commitish="abc123", rel_path="src/a.py")

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            return SimpleNamespace(
                returncode=0,
                stdout="file content\n",
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = git_show_file(tmp_path, req)
        assert result == "file content\n"

    def test_unsafe_path_returns_none(self, tmp_path: Path) -> None:
        """Line 147: bad rel_path → None (no subprocess call)."""
        req = GitShowFileRequest(commitish="abc", rel_path="../escape")
        result = git_show_file(tmp_path, req)
        assert result is None

    def test_nonzero_returncode_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 160: nonzero returncode → None."""
        req = GitShowFileRequest(commitish="abc", rel_path="src/a.py")

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            return SimpleNamespace(
                returncode=128,
                stdout="",
                stderr="not found",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert git_show_file(tmp_path, req) is None

    def test_timeout_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 157-158: TimeoutExpired → None."""
        req = GitShowFileRequest(commitish="abc", rel_path="src/a.py")

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            raise subprocess.TimeoutExpired(cmd, 30)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert git_show_file(tmp_path, req) is None

    def test_oserror_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 157-158: OSError → None."""
        req = GitShowFileRequest(commitish="abc", rel_path="src/a.py")

        def fake_run(cmd: list[str], **kw: Any) -> SimpleNamespace:
            raise OSError("git not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert git_show_file(tmp_path, req) is None


# ---------------------------------------------------------------------------
# Parametrized edge-case round-ups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("src/demo.py", "src/demo.py"),
        ("src\\demo.py", "src/demo.py"),
        ("   ", None),
        ("", None),
        ("/abs", None),
        ("../up", None),
        ("./a/b.py", "a/b.py"),
    ],
    ids=[
        "posix",
        "backslash",
        "whitespace",
        "empty",
        "absolute",
        "parent",
        "dot-prefix",
    ],
)
def test_normalize_relative_path_parametrized(
    raw: str,
    expected: str | None,
) -> None:
    """Parametrized smoke tests for normalize_relative_path."""
    assert normalize_relative_path(raw) == expected


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest tests/a.py ; echo x",
        "pytest tests/a.py > log.txt",
        "pytest tests/a.py < /dev/null",
        "pytest tests/a.py || true",
    ],
    ids=["semicolon", "redirect-out", "redirect-in", "or-chain"],
)
def test_shell_composition_rejected(cmd: str) -> None:
    """Various shell operators must be rejected."""
    assert normalize_native_pytest_command(cmd) is None
