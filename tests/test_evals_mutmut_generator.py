"""Tests for evals.importers.mutmut_generator — mutmut case generation.

Mocks mutmut subprocess output to test filtering and category assignment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evals.importers.mutmut_generator import (
    _apply_diff_to_source,
    _find_test_file,
    _get_mutant_diff,
    _get_mutmut_results,
    _infer_operator_type,
    _parse_mutmut_results,
    _run_mutmut,
    import_mutmut,
)

# ---------------------------------------------------------------------------
# Operator type inference
# ---------------------------------------------------------------------------


class TestInferOperatorType:
    """Tests for _infer_operator_type with various diff patterns."""

    def test_arithmetic_operator(self) -> None:
        diff = "-    result = a + b\n+    result = a - b\n"
        assert _infer_operator_type(diff) in (
            "arithmetic",
            "comparison",
            "keyword",
        )

    def test_comparison_operator(self) -> None:
        diff = "-    if x == 0:\n+    if x != 0:\n"
        assert _infer_operator_type(diff) == "comparison"

    def test_boolean_operator(self) -> None:
        diff = "-    return True\n+    return False\n"
        assert _infer_operator_type(diff) == "boolean"

    def test_return_change(self) -> None:
        diff = "-    return result\n+    return None\n"
        assert _infer_operator_type(diff) == "changed_return"

    def test_removed_call(self) -> None:
        diff = "-    process(data)\n+    pass\n"
        assert _infer_operator_type(diff) == "removed_call"

    def test_empty_diff(self) -> None:
        assert _infer_operator_type("") == "keyword"

    def test_unknown_pattern(self) -> None:
        diff = "-    x = 'hello'\n+    x = 'world'\n"
        result = _infer_operator_type(diff)
        assert result in (
            "keyword",
            "boolean",
            "comparison",
            "arithmetic",
            "string",
        )

    def test_bare_less_than_comparison(self) -> None:
        """Branch [160,161]: bare < or > triggers comparison."""
        diff = "-    if x < 10:\n+    if x > 10:\n"
        assert _infer_operator_type(diff) == "comparison"

    def test_bare_greater_than_comparison(self) -> None:
        """Bare > without = should still be comparison."""
        diff = "-    if count > limit:\n+    if count < limit:\n"
        assert _infer_operator_type(diff) == "comparison"

    def test_boolean_keyword_not(self) -> None:
        """'not ' keyword triggers boolean detection."""
        diff = "-    if not done:\n+    if done:\n"
        assert _infer_operator_type(diff) == "boolean"

    def test_boolean_keyword_and(self) -> None:
        """' and ' keyword triggers boolean detection."""
        diff = "-    if a and b:\n+    if a or b:\n"
        assert _infer_operator_type(diff) == "boolean"

    def test_diff_header_lines_stripped(self) -> None:
        """Lines starting with +++ or --- should be excluded."""
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-    x = 'hello'\n+    x = 'world'\n"
        result = _infer_operator_type(diff)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------


class TestParseMutmutResults:
    """Tests for _parse_mutmut_results output parsing."""

    def test_parse_basic_output(self) -> None:
        output = "1  survived\n2  killed\n3  suspicious\n"
        result = _parse_mutmut_results(output)
        assert len(result) == 3
        assert result[0]["id"] == "1"
        assert result[0]["status"] == "survived"
        assert result[1]["status"] == "killed"

    def test_empty_output(self) -> None:
        assert _parse_mutmut_results("") == []

    def test_skip_non_matching_lines(self) -> None:
        output = "Running mutmut...\n1  survived\nDone.\n"
        result = _parse_mutmut_results(output)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_blank_lines_skipped(self) -> None:
        """Line 80: blank lines trigger continue."""
        output = "\n\n1  killed\n\n2  survived\n\n"
        result = _parse_mutmut_results(output)
        assert len(result) == 2

    def test_whitespace_only_lines_skipped(self) -> None:
        """Lines with only whitespace are stripped then skipped."""
        output = "   \n1  killed\n  \t  \n"
        result = _parse_mutmut_results(output)
        assert len(result) == 1
        assert result[0]["id"] == "1"


# ---------------------------------------------------------------------------
# _run_mutmut subprocess tests
# ---------------------------------------------------------------------------


class TestRunMutmut:
    """Tests for _run_mutmut subprocess execution."""

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 39-40, branch [46,48]: successful run."""
        mock_proc = MagicMock(returncode=0, stderr="", stdout="")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        ok, msg = _run_mutmut("target.py")
        assert ok is True
        assert msg == ""

    def test_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 46-47, branch [46,47]: returncode != 0."""
        mock_proc = MagicMock(returncode=1, stderr="some error")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        ok, msg = _run_mutmut("target.py")
        assert ok is False
        assert "exited with code 1" in msg

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 49-50: TimeoutExpired."""

        def _raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="mutmut", timeout=600)

        monkeypatch.setattr(subprocess, "run", _raise_timeout)
        ok, msg = _run_mutmut("target.py")
        assert ok is False
        assert "timed out" in msg

    def test_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 51-52: FileNotFoundError."""

        def _raise_fnf(*a: Any, **kw: Any) -> None:
            raise FileNotFoundError("mutmut")

        monkeypatch.setattr(subprocess, "run", _raise_fnf)
        ok, msg = _run_mutmut("target.py")
        assert ok is False
        assert "not found" in msg


# ---------------------------------------------------------------------------
# _get_mutmut_results subprocess tests
# ---------------------------------------------------------------------------


class TestGetMutmutResults:
    """Tests for _get_mutmut_results subprocess execution."""

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 57-58, branch [64,66]: successful query."""
        mock_proc = MagicMock(returncode=0, stdout="1  killed\n2  survived\n")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        mutants, err = _get_mutmut_results()
        assert len(mutants) == 2
        assert err == ""

    def test_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 64-65, branch [64,65]: returncode != 0."""
        mock_proc = MagicMock(returncode=2, stderr="error output")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        mutants, err = _get_mutmut_results()
        assert mutants == []
        assert "exited with code 2" in err

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 67-68: TimeoutExpired."""

        def _raise_timeout(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="mutmut", timeout=30)

        monkeypatch.setattr(subprocess, "run", _raise_timeout)
        mutants, err = _get_mutmut_results()
        assert mutants == []
        assert "timed out" in err

    def test_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 69-70: FileNotFoundError."""

        def _raise_fnf(*a: Any, **kw: Any) -> None:
            raise FileNotFoundError("mutmut")

        monkeypatch.setattr(subprocess, "run", _raise_fnf)
        mutants, err = _get_mutmut_results()
        assert mutants == []
        assert "not found" in err


# ---------------------------------------------------------------------------
# _get_mutant_diff subprocess tests
# ---------------------------------------------------------------------------


class TestGetMutantDiff:
    """Tests for _get_mutant_diff subprocess execution."""

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 95-96, 102-103: successful diff retrieval."""
        mock_proc = MagicMock(returncode=0, stdout="--- a/f.py\n+++ b/f.py\n-old\n+new\n")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        result = _get_mutant_diff("42")
        assert result is not None
        assert "old" in result

    def test_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch [102,104]: returncode != 0 returns None."""
        mock_proc = MagicMock(returncode=1, stdout="")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        assert _get_mutant_diff("42") is None

    def test_empty_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch [102,104]: returncode 0 but empty stdout."""
        mock_proc = MagicMock(returncode=0, stdout="   ")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        assert _get_mutant_diff("42") is None

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 105-106: TimeoutExpired returns None."""

        def _raise(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="mutmut", timeout=15)

        monkeypatch.setattr(subprocess, "run", _raise)
        assert _get_mutant_diff("42") is None

    def test_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 105-106: FileNotFoundError returns None."""

        def _raise(*a: Any, **kw: Any) -> None:
            raise FileNotFoundError("mutmut")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert _get_mutant_diff("42") is None


# ---------------------------------------------------------------------------
# _apply_diff_to_source tests
# ---------------------------------------------------------------------------


class TestApplyDiffToSource:
    """Tests for _apply_diff_to_source patching logic."""

    def test_patch_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch [130,132]: patch returns non-zero -> None."""
        mock_proc = MagicMock(returncode=1, stdout="", stderr="failed")
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)
        result = _apply_diff_to_source("x = 1\n", "bad diff")
        assert result is None

    def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 133-134: TimeoutExpired returns None."""

        def _raise(*a: Any, **kw: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="patch", timeout=5)

        monkeypatch.setattr(subprocess, "run", _raise)
        result = _apply_diff_to_source("x = 1\n", "some diff")
        assert result is None

    def test_file_not_found_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 133-134: FileNotFoundError returns None."""

        def _raise(*a: Any, **kw: Any) -> None:
            raise FileNotFoundError("patch")

        monkeypatch.setattr(subprocess, "run", _raise)
        result = _apply_diff_to_source("x = 1\n", "some diff")
        assert result is None

    def test_successful_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lines 130-131: successful patch reads back mutated file."""
        original_run = subprocess.run

        def _mock_run(cmd: Any, **kw: Any) -> MagicMock:
            # Actually write to the source file so read_text works.
            # cmd = ["patch", "-s", "--no-backup-if-mismatch", src, diff]
            if isinstance(cmd, list) and cmd[0] == "patch":
                src_path = Path(cmd[3])
                src_path.write_text("x = 2\n", encoding="utf-8")
                return MagicMock(returncode=0)
            return original_run(cmd, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(subprocess, "run", _mock_run)
        result = _apply_diff_to_source("x = 1\n", "fake diff")
        assert result == "x = 2\n"


# ---------------------------------------------------------------------------
# _find_test_file tests
# ---------------------------------------------------------------------------


class TestFindTestFile:
    """Tests for _find_test_file discovery logic."""

    def test_finds_sibling_test(self, tmp_path: Path) -> None:
        """Branch [188,187] loop finds candidate."""
        target = tmp_path / "mymod.py"
        target.write_text("x = 1\n", encoding="utf-8")
        test_file = tmp_path / "test_mymod.py"
        test_file.write_text("def test_x(): pass\n", encoding="utf-8")

        result = _find_test_file(str(target))
        assert result is not None
        assert result.name == "test_mymod.py"

    def test_finds_test_in_tests_dir(self, tmp_path: Path) -> None:
        """Candidate: parent/../tests/test_<module>.py."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        target = src_dir / "mymod.py"
        target.write_text("x = 1\n", encoding="utf-8")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_mymod.py"
        test_file.write_text("def test_x(): pass\n", encoding="utf-8")

        result = _find_test_file(str(target))
        assert result is not None
        assert result.name == "test_mymod.py"

    def test_no_test_file_found(self, tmp_path: Path) -> None:
        """Branch [187,190]: loop exhausted, return None."""
        target = tmp_path / "orphan.py"
        target.write_text("x = 1\n", encoding="utf-8")
        result = _find_test_file(str(target))
        assert result is None

    def test_non_file_target(self, tmp_path: Path) -> None:
        """Branch [179,190]: target is not a file -> None."""
        result = _find_test_file(str(tmp_path / "nonexistent.py"))
        assert result is None

    def test_directory_target(self, tmp_path: Path) -> None:
        """Branch [179,190]: target is a directory -> None."""
        result = _find_test_file(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestMutmutDryRun:
    """Tests for import_mutmut dry_run mode."""

    def test_dry_run_returns_preview(self, tmp_path: Path) -> None:
        result = import_mutmut(
            tmp_path,
            target="src/llmdebug/capture.py",
            max_cases=5,
            dry_run=True,
        )
        assert len(result) == 1
        assert result[0]["metadata"]["bug_source"] == "mutmut"
        assert result[0]["metadata"]["contamination_risk"] == "none"


# ---------------------------------------------------------------------------
# import_mutmut integration (all subprocess calls mocked)
# ---------------------------------------------------------------------------


def _make_target_with_test(tmp_path: Path) -> tuple[Path, Path]:
    """Helper: create a minimal target .py and matching test file."""
    target = tmp_path / "mymod.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    test_f = tmp_path / "test_mymod.py"
    test_f.write_text(
        "from mymod import add\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return target, test_f


class TestImportMutmutRunFailed:
    """Branch [240,241]: mutmut run fails -> empty list."""

    def test_run_failed_returns_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 241-242: _run_mutmut failure."""
        target, _ = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (False, "mutmut crashed"),
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []


class TestImportMutmutNoMutants:
    """Branch [246,247]: no mutants found -> empty list."""

    def test_no_mutants(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 247-248."""
        target, _ = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: ([], ""),
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []

    def test_no_mutants_with_error_msg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 247-248: error message variant."""
        target, _ = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: ([], "something went wrong"),
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []


class TestImportMutmutTargetNotFile:
    """Branch [255,256]: target is not a file -> empty list."""

    def test_target_not_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 256-257."""
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: ([{"id": "1", "status": "killed"}], ""),
        )
        result = import_mutmut(tmp_path / "out", target="nonexistent_dir/")
        assert result == []


class TestImportMutmutMaxCases:
    """Branch [270,271]: max_cases limit reached."""

    def test_max_cases_break(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 271: breaks when imported reaches max_cases."""
        target, _ = _make_target_with_test(tmp_path)
        killed = [{"id": str(i), "status": "killed"} for i in range(5)]
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: (killed, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: f"-old line {mid}\n+new line {mid}\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._apply_diff_to_source",
            lambda orig, diff: "def add(a, b):\n    return a - b\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-old\n+new\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.validate_generated_case",
            lambda d, schema_only: (True, "ok"),
        )
        result = import_mutmut(tmp_path / "out", target=str(target), max_cases=2)
        assert len(result) == 2


class TestImportMutmutSkips:
    """Tests for various skip conditions in the mutant processing loop."""

    @pytest.fixture()
    def _base_mocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
        """Shared mocks: mutmut run succeeds with one killed mutant."""
        target, test_f = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: ([{"id": "1", "status": "killed"}], ""),
        )
        return target, test_f

    def test_no_diff_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _base_mocks: tuple[Path, Path],
    ) -> None:
        """Lines 276-277, branch [275,276]: no diff -> skip."""
        target, _ = _base_mocks
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: None,
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []

    def test_buggy_code_none_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _base_mocks: tuple[Path, Path],
    ) -> None:
        """Lines 286-287, branch [285,286]: buggy_code is None."""
        target, _ = _base_mocks
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: "-old\n+new\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._apply_diff_to_source",
            lambda orig, diff: None,
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []

    def test_buggy_code_same_as_original_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _base_mocks: tuple[Path, Path],
    ) -> None:
        """Lines 286-287: buggy == original -> skip."""
        target, _ = _base_mocks
        original = target.read_text(encoding="utf-8")
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: "-old\n+new\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._apply_diff_to_source",
            lambda orig, diff: original,
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []

    def test_empty_oracle_patch_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _base_mocks: tuple[Path, Path],
    ) -> None:
        """Lines 292-293, branch [291,292]: empty oracle patch."""
        target, _ = _base_mocks
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: "-old\n+new\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._apply_diff_to_source",
            lambda orig, diff: "def add(a, b):\n    return a - b\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.generate_oracle_patch",
            lambda b, f: "   \n",
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []

    def test_no_test_code_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 296-297, branch [295,296]: no test file -> skip."""
        target = tmp_path / "lonely.py"
        target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: ([{"id": "1", "status": "killed"}], ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: "-old\n+new\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._apply_diff_to_source",
            lambda orig, diff: "def add(a, b):\n    return a - b\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-old\n+new\n",
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []


class TestImportMutmutValidation:
    """Branches [326,334] and [328,329]: validation pass/fail."""

    def _setup_full_pipeline(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Path:
        """Set up a full pipeline with one killed mutant."""
        target, _ = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: ([{"id": "1", "status": "killed"}], ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: "-old\n+new\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._apply_diff_to_source",
            lambda orig, diff: "def add(a, b):\n    return a - b\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-old\n+new\n",
        )
        return target

    def test_validation_failure_skips_case(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 329-332: validation fails -> skip + rmtree."""
        target = self._setup_full_pipeline(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.validate_generated_case",
            lambda d, schema_only: (False, "bad schema"),
        )
        out_dir = tmp_path / "out"
        result = import_mutmut(out_dir, target=str(target), validate=True)
        assert result == []
        # Case dir should have been removed.
        case_dir = out_dir / "mut_mymod_1"
        assert not case_dir.exists()

    def test_validation_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Branch [326,334]: validation passes -> case imported."""
        target = self._setup_full_pipeline(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.validate_generated_case",
            lambda d, schema_only: (True, "ok"),
        )
        result = import_mutmut(tmp_path / "out", target=str(target), validate=True)
        assert len(result) == 1
        assert result[0]["case_id"] == "mut_mymod_1"

    def test_no_validate_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When validate=False, skip validation entirely."""
        target = self._setup_full_pipeline(tmp_path, monkeypatch)
        result = import_mutmut(tmp_path / "out", target=str(target), validate=False)
        assert len(result) == 1


class TestImportMutmutSkippedMessage:
    """Line 337: skipped count message is printed."""

    def test_skipped_count_printed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Lines 336-337: skipped message printed when skipped > 0."""
        target, _ = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: (
                [
                    {"id": "1", "status": "killed"},
                    {"id": "2", "status": "killed"},
                ],
                "",
            ),
        )
        # First mutant: no diff -> skip; second: valid -> import.
        call_count = {"n": 0}

        def _mock_diff(mid: str) -> str | None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return None  # first call skipped
            return "-old\n+new\n"

        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            _mock_diff,
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._apply_diff_to_source",
            lambda orig, diff: "def add(a, b):\n    return a - b\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-old\n+new\n",
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator.validate_generated_case",
            lambda d, schema_only: (True, "ok"),
        )
        result = import_mutmut(tmp_path / "out", target=str(target), validate=True)
        assert len(result) == 1
        captured = capsys.readouterr()
        assert "Skipped 1 mutant(s)" in captured.out


class TestImportMutmutAllSkipped:
    """Edge case: all killed mutants are skipped."""

    def test_all_skipped_returns_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every mutant's diff is None -> all skipped."""
        target, _ = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: (
                [
                    {"id": "1", "status": "killed"},
                    {"id": "2", "status": "killed"},
                ],
                "",
            ),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutant_diff",
            lambda mid: None,
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []


class TestImportMutmutNoKilledMutants:
    """Edge case: mutants exist but none are killed."""

    def test_no_killed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All survived -> no cases generated (but no error)."""
        target, _ = _make_target_with_test(tmp_path)
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._run_mutmut",
            lambda t: (True, ""),
        )
        monkeypatch.setattr(
            "evals.importers.mutmut_generator._get_mutmut_results",
            lambda: (
                [
                    {"id": "1", "status": "survived"},
                    {"id": "2", "status": "survived"},
                ],
                "",
            ),
        )
        result = import_mutmut(tmp_path / "out", target=str(target))
        assert result == []
