"""Targeted coverage tests for eval importer modules.

Covers uncovered branches in:
- debugbench_importer.py (import pipeline, validation, syntax-error paths)
- condefects_importer.py (import pipeline, validation, error paths)
- pyresbugs_importer.py (import pipeline, validation, verify-patch paths)
- humanevalpack_importer.py (import pipeline, validation paths)
- repair_debugbench_patches.py (main() entry, empty-patch path)
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

from evals.importers import (
    condefects_importer,
    debugbench_importer,
    humanevalpack_importer,
    pyresbugs_importer,
    repair_debugbench_patches,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _install_fake_datasets(monkeypatch: Any, rows: list[dict[str, object]]) -> None:
    module = types.ModuleType("datasets")

    def _load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        return rows

    module.load_dataset = _load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


def _write_case(case_dir: Path, *, source_id: str, oracle_patch: str = "old patch\n") -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case.json").write_text(
        json.dumps({"source_id": source_id}),
        encoding="utf-8",
    )
    (case_dir / "buggy.py").write_text("print('buggy')\n", encoding="utf-8")
    (case_dir / "oracle.patch").write_text(oracle_patch, encoding="utf-8")


# =====================================================================
# debugbench_importer.py — missing lines
# =====================================================================


class TestDebugbenchExtractMethodName:
    """Cover _extract_method_name edge cases (lines 59-62, 68-70)."""

    def test_syntax_error_returns_none(self) -> None:
        result = debugbench_importer._extract_method_name("def broken(")
        assert result is None

    def test_no_solution_class_returns_none(self) -> None:
        result = debugbench_importer._extract_method_name("class Other:\n    pass\n")
        assert result is None

    def test_only_init_returns_none(self) -> None:
        code = "class Solution:\n    def __init__(self):\n        pass\n"
        result = debugbench_importer._extract_method_name(code)
        assert result is None


class TestDebugbenchMethodAcceptsCallWithoutArgs:
    """Cover _method_accepts_call_without_args edge cases (lines 101-102, 112, 115, 123)."""

    def test_syntax_error_returns_false(self) -> None:
        result = debugbench_importer._method_accepts_call_without_args("def broken(", "foo")
        assert result is False

    def test_method_not_found_returns_false(self) -> None:
        code = "class Solution:\n    def other(self):\n        pass\n"
        result = debugbench_importer._method_accepts_call_without_args(code, "missing")
        assert result is False

    def test_posonlyargs_with_defaults(self) -> None:
        code = textwrap.dedent("""\
            class Solution:
                def solve(self, x=1, /, y=2):
                    return x + y
        """)
        result = debugbench_importer._method_accepts_call_without_args(code, "solve")
        assert result is True

    def test_required_kwonly_returns_false(self) -> None:
        code = textwrap.dedent("""\
            class Solution:
                def solve(self, *, x):
                    return x
        """)
        result = debugbench_importer._method_accepts_call_without_args(code, "solve")
        assert result is False


class TestDebugbenchGenerateTest:
    """Cover _generate_test missing branches (lines 140, 149)."""

    def test_no_method_in_code_returns_none(self) -> None:
        # no Solution class → _extract_method_name returns None → return None
        result = debugbench_importer._generate_test("x = 1\n", None, "logic error")
        assert result is None

    def test_examples_with_no_parseable_args_returns_none(self) -> None:
        code = "class Solution:\n    def solve(self, nums):\n        return nums\n"
        result = debugbench_importer._generate_test(
            code,
            ["Input: ???\nOutput: 3\n"],
            "logic error",
        )
        # No parseable args → no tests generated → should fall through to None
        assert result is None


class TestDebugbenchInferException:
    """Cover _infer_exception branches (lines 192-194)."""

    def test_syntax_undefined_returns_name_error(self) -> None:
        result = debugbench_importer._infer_exception("syntax error", "undefined usage", "x")
        assert result == "NameError"

    def test_syntax_misused_returns_name_error(self) -> None:
        result = debugbench_importer._infer_exception("syntax error", "misused variable", "x")
        assert result == "NameError"


class TestDebugbenchImportPipeline:
    """Cover import_debugbench write + validate paths (lines 220-221, 233-236, 253-254,
    268-269, 277-278, 283-285, 309-310, 314-315, 321-322, 345-363)."""

    def _setup_monkeypatch(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            debugbench_importer,
            "map_debugbench_type",
            lambda bt: (
                "syntax_error"
                if "syntax" in bt
                else ("reference_error" if "reference" in bt else "logic_error")
            ),
        )
        monkeypatch.setattr(debugbench_importer, "infer_snapshot_dependency", lambda c: "dep")
        monkeypatch.setattr(debugbench_importer, "ruff_format", lambda code: code)
        monkeypatch.setattr(debugbench_importer, "has_external_deps", lambda code: False)
        monkeypatch.setattr(debugbench_importer, "is_deterministic", lambda code: True)

    def test_datasets_import_error_raised(self, monkeypatch: Any) -> None:
        """Lines 220-221: ImportError when datasets not available."""
        monkeypatch.delitem(sys.modules, "datasets", raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _make_import_blocker("datasets"),
        )
        with pytest.raises(ImportError, match="datasets"):
            debugbench_importer.import_debugbench(Path("/tmp/fake"))

    def test_empty_code_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 253-254: empty buggy or fixed code skipped."""
        rows = [
            {
                "language": "python3",
                "buggy_code": "",
                "solution": "def f():\n    return 1\n",
                "category": "logic bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
            {
                "language": "python3",
                "buggy_code": "def f():\n    return 1\n",
                "solution": "   ",
                "category": "logic bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(
            debugbench_importer,
            "generate_oracle_patch",
            lambda b, f: "patch\n",
        )
        monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda p: 1)
        result = debugbench_importer.import_debugbench(tmp_path / "cases", dry_run=True)
        assert result == []
        assert isinstance(result, list)

    def test_case_exists_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 268-269: case directory already exists → skip."""
        rows = [
            {
                "language": "python3",
                "buggy_code": "class Solution:\n    def solve(self):\n        return 1\n",
                "solution": "class Solution:\n    def solve(self):\n        return 2\n",
                "category": "logic bug",
                "subtype": "",
                "level": "easy",
                "examples": ["Input: \nOutput: 1\n"],
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(debugbench_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda p: 1)

        out = tmp_path / "cases"
        (out / "db_0000_easy").mkdir(parents=True)
        result = debugbench_importer.import_debugbench(out, dry_run=False)
        assert result == []

    def test_syntax_error_preflight(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 277-278, 283-285: syntax error preflight paths."""
        rows = [
            {
                "language": "python3",
                "buggy_code": "def broken(",
                "solution": "class Solution:\n    def solve(self):\n        return 1\n",
                "category": "syntax bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
            {
                "language": "python3",
                "buggy_code": "if True\n    pass",
                "solution": "if True\n    pass",
                "category": "syntax bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
            {
                "language": "python3",
                "buggy_code": "x = 1\n",
                "solution": "x = 2\n",
                "category": "syntax bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(
            debugbench_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-old\n+new\n",
        )
        monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            debugbench_importer,
            "_generate_test",
            lambda fixed, examples, category: ("def test_syntax():\n    import buggy\n"),
        )

        result = debugbench_importer.import_debugbench(tmp_path / "cases", dry_run=True)
        # Row 0: buggy has SyntaxError, fixed compiles → kept
        # Row 1: both buggy and fixed have SyntaxError → skipped
        # Row 2: buggy compiles (no SyntaxError) → skipped for syntax_error category
        assert len(result) == 1

    def test_empty_patch_and_long_patch_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 309-310, 314-315: empty oracle patch and too-long patch."""
        rows = [
            {
                "language": "python3",
                "buggy_code": "class Solution:\n    def solve(self):\n        return 1\n",
                "solution": "class Solution:\n    def solve(self):\n        return 1\n",
                "category": "logic bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
            {
                "language": "python3",
                "buggy_code": "class Solution:\n    def solve(self):\n        return 1\n",
                "solution": "class Solution:\n    def solve(self):\n        return 999\n",
                "category": "logic bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        call_count = {"n": 0}

        def _fake_patch(b: str, f: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ""
            return "long patch"

        monkeypatch.setattr(debugbench_importer, "generate_oracle_patch", _fake_patch)
        monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda p: 20)
        result = debugbench_importer.import_debugbench(tmp_path / "cases", dry_run=True)
        assert result == []

    def test_no_test_generated_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 321-322: _generate_test returns None → skip."""
        rows = [
            {
                "language": "python3",
                "buggy_code": "class Solution:\n    def solve(self):\n        return 1\n",
                "solution": "class Solution:\n    def solve(self):\n        return 2\n",
                "category": "logic bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(debugbench_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(debugbench_importer, "_generate_test", lambda *a: None)
        result = debugbench_importer.import_debugbench(tmp_path / "cases", dry_run=True)
        assert result == []

    def test_write_and_validate_success(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 345-363: write_case_files + validate_generated_case success path."""
        rows = [
            {
                "language": "python3",
                "buggy_code": "class Solution:\n    def solve(self):\n        return 1\n",
                "solution": "class Solution:\n    def solve(self):\n        return 2\n",
                "category": "reference bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(debugbench_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            debugbench_importer,
            "_generate_test",
            lambda *a: "def test_smoke():\n    pass\n",
        )
        monkeypatch.setattr(
            debugbench_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(
            debugbench_importer,
            "validate_generated_case",
            lambda case_dir, schema_only: (True, ""),
        )
        result = debugbench_importer.import_debugbench(
            tmp_path / "cases", dry_run=False, validate=True
        )
        assert len(result) == 1
        assert result[0]["case_id"] is not None

    def test_validate_failure_removes_case(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 353-361: validation failure removes case dir."""
        rows = [
            {
                "language": "python3",
                "buggy_code": "class Solution:\n    def solve(self):\n        return 1\n",
                "solution": "class Solution:\n    def solve(self):\n        return 2\n",
                "category": "reference bug",
                "subtype": "",
                "level": "easy",
                "examples": None,
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(debugbench_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(debugbench_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            debugbench_importer,
            "_generate_test",
            lambda *a: "def test_smoke():\n    pass\n",
        )
        monkeypatch.setattr(
            debugbench_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(
            debugbench_importer,
            "validate_generated_case",
            lambda case_dir, schema_only: (False, "bad schema"),
        )
        result = debugbench_importer.import_debugbench(
            tmp_path / "cases", dry_run=False, validate=True
        )
        assert result == []
        assert isinstance(result, list)


# =====================================================================
# condefects_importer.py — missing lines
# =====================================================================


class TestCondefectsImportPipeline:
    """Cover import_condefects error/validation paths."""

    def test_repo_path_none_raises(self) -> None:
        """Line 296: repo_path is None → ValueError."""
        with pytest.raises(ValueError, match="ConDefects requires a local clone"):
            condefects_importer.import_condefects(Path("/out"), repo_path=None, test_zip=Path("z"))

    def test_code_dir_missing_raises(self, tmp_path: Path) -> None:
        """Line 303: Code/ dir doesn't exist → FileNotFoundError."""
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(FileNotFoundError, match="Code/"):
            condefects_importer.import_condefects(Path("/out"), repo_path=repo, test_zip=Path("z"))

    def test_test_zip_none_raises(self, tmp_path: Path) -> None:
        """Line 306: test_zip is None → ValueError."""
        repo = tmp_path / "repo"
        (repo / "Code").mkdir(parents=True)
        with pytest.raises(ValueError, match=r"Test\.zip"):
            condefects_importer.import_condefects(Path("/out"), repo_path=repo, test_zip=None)

    def test_test_zip_missing_raises(self, tmp_path: Path) -> None:
        """Line 311: test_zip doesn't exist → FileNotFoundError."""
        repo = tmp_path / "repo"
        (repo / "Code").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match=r"Test\.zip"):
            condefects_importer.import_condefects(
                Path("/out"),
                repo_path=repo,
                test_zip=tmp_path / "nonexistent.zip",
            )

    def test_parse_difficulty_file_missing(self, tmp_path: Path) -> None:
        """Lines 68-69: no difficulty.txt → empty dict."""
        result = condefects_importer._parse_difficulty_file(tmp_path)
        assert result == {}
        assert isinstance(result, dict)

    def test_parse_difficulty_file_blank_lines(self, tmp_path: Path) -> None:
        """Lines 74-75: blank lines in difficulty.txt are skipped."""
        (tmp_path / "difficulty.txt").write_text("\n\nabc 10\n\n", encoding="utf-8")
        result = condefects_importer._parse_difficulty_file(tmp_path)
        assert result == {"abc": 10}
        assert len(result) == 1

    def test_build_zip_index_skips_dirs(self, tmp_path: Path) -> None:
        """Lines 127-128: directory entries are skipped."""
        import zipfile

        zp = tmp_path / "t.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dir/", "")
            zf.writestr("dir/file.txt", "data")
        with zipfile.ZipFile(zp) as zf:
            idx = condefects_importer._build_zip_index(zf)
        assert "dir" in idx
        assert "file.txt" in idx["dir"]

    def test_extract_test_pairs_read_error(self, tmp_path: Path) -> None:
        """Lines 180-181: KeyError/OSError when reading zip entries → skipped."""
        import zipfile

        zp = tmp_path / "t.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/t/in/example_00.txt", "1\n")
            zf.writestr("Test/t/out/example_00.txt", "2\n")
        with zipfile.ZipFile(zp) as zf:
            idx = condefects_importer._build_zip_index(zf)
            # Normal case works
            pairs = condefects_importer._extract_test_pairs_from_zip(zf, idx, "Test/t")
            assert len(pairs) == 1
            assert pairs[0][0].strip() == "1"

    def test_infer_exception_runtime(self) -> None:
        """Line 228: 'runtime' verdict → RuntimeError."""
        result = condefects_importer._infer_exception("Runtime Error")
        assert result == "RuntimeError"

    def test_infer_exception_default(self) -> None:
        """Line 249: non-runtime verdict → AssertionError."""
        result = condefects_importer._infer_exception("Wrong Answer")
        assert result == "AssertionError"

    def test_classify_off_by_one(self) -> None:
        """Lines 248-249: off-by-one pattern in patch → logic_error."""
        result = condefects_importer._classify_bug_from_patch(
            "for i in range(n):\n    pass\n",
            "for i in range(n + 1):\n    pass\n",
        )
        assert result == "logic_error"
        assert isinstance(result, str)

    def test_classify_default(self) -> None:
        """Lines 253-254: no matching pattern → logic_error."""
        result = condefects_importer._classify_bug_from_patch(
            "x = foo()\n",
            "x = bar()\n",
        )
        assert result == "logic_error"
        assert isinstance(result, str)

    def test_full_import_with_write_and_validate(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Cover write + validate paths (lines 459-475, 433-434)."""
        import zipfile

        repo = tmp_path / "repo"
        code_dir = repo / "Code"
        task_dir = code_dir / "abc001_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
        (task_dir / "correctVersion.py").write_text(
            "def solve():\n    return 2\n", encoding="utf-8"
        )
        (repo / "difficulty.txt").write_text("abc001_a 300\n", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc001/A/in/example_00.txt", "1\n")
            zf.writestr("Test/abc001/A/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda code: code)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda code: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda code: True)
        monkeypatch.setattr(
            condefects_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-return 1\n+return 2\n",
        )
        monkeypatch.setattr(condefects_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(condefects_importer, "infer_snapshot_dependency", lambda c: "dep")

        written: list[Path] = []

        def _fake_write(case_dir: Path, **kw: Any) -> None:
            case_dir.mkdir(parents=True, exist_ok=True)
            written.append(case_dir)

        monkeypatch.setattr(condefects_importer, "write_case_files", _fake_write)
        monkeypatch.setattr(
            condefects_importer,
            "validate_generated_case",
            lambda cd, schema_only: (True, ""),
        )

        out = tmp_path / "cases"
        result = condefects_importer.import_condefects(
            out, repo_path=repo, test_zip=zp, dry_run=False, validate=True
        )
        assert len(result) == 1
        assert len(written) == 1

    def test_validate_failure_removes_case(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 467-472: validation failure removes directory."""
        import zipfile

        repo = tmp_path / "repo"
        code_dir = repo / "Code"
        task_dir = code_dir / "abc002_b" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
        (task_dir / "correctVersion.py").write_text(
            "def solve():\n    return 2\n", encoding="utf-8"
        )
        (repo / "difficulty.txt").write_text("abc002_b 300\n", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc002/B/in/example_00.txt", "1\n")
            zf.writestr("Test/abc002/B/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda code: code)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda code: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda code: True)
        monkeypatch.setattr(
            condefects_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-return 1\n+return 2\n",
        )
        monkeypatch.setattr(condefects_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(condefects_importer, "infer_snapshot_dependency", lambda c: "dep")
        monkeypatch.setattr(
            condefects_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(
            condefects_importer,
            "validate_generated_case",
            lambda cd, schema_only: (False, "bad"),
        )

        result = condefects_importer.import_condefects(
            tmp_path / "cases",
            repo_path=repo,
            test_zip=zp,
            dry_run=False,
            validate=True,
        )
        assert result == []

    def test_case_exists_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 432-434: case directory already exists → skip."""
        import zipfile

        repo = tmp_path / "repo"
        code_dir = repo / "Code"
        task_dir = code_dir / "abc003_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
        (task_dir / "correctVersion.py").write_text(
            "def solve():\n    return 2\n", encoding="utf-8"
        )
        (repo / "difficulty.txt").write_text("abc003_a 300\n", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc003/A/in/example_00.txt", "1\n")
            zf.writestr("Test/abc003/A/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda code: code)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda code: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda code: True)
        monkeypatch.setattr(
            condefects_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-return 1\n+return 2\n",
        )
        monkeypatch.setattr(condefects_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(condefects_importer, "infer_snapshot_dependency", lambda c: "dep")

        out = tmp_path / "cases"
        (out / "cd_abc003_a_1_easy").mkdir(parents=True)

        result = condefects_importer.import_condefects(
            out, repo_path=repo, test_zip=zp, dry_run=False, validate=False
        )
        assert result == []

    def test_missing_files_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 379-381: missing faultyVersion or correctVersion → skip."""
        import zipfile

        repo = tmp_path / "repo"
        task_dir = repo / "Code" / "abc004_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("x = 1\n", encoding="utf-8")
        # No correctVersion.py

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc004/A/in/example_00.txt", "1\n")
            zf.writestr("Test/abc004/A/out/example_00.txt", "2\n")
        (repo / "difficulty.txt").write_text("", encoding="utf-8")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda c: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda c: True)

        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert result == []

    def test_empty_code_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 386-388: empty buggy or fixed code → skip."""
        import zipfile

        repo = tmp_path / "repo"
        task_dir = repo / "Code" / "abc005_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("  \n", encoding="utf-8")
        (task_dir / "correctVersion.py").write_text("x = 1\n", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc005/A/in/example_00.txt", "1\n")
            zf.writestr("Test/abc005/A/out/example_00.txt", "2\n")
        (repo / "difficulty.txt").write_text("", encoding="utf-8")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda c: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda c: True)

        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert result == []

    def test_difficulty_fallback_to_contest_prefix(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 370-371: difficulty lookup falls back to contest prefix."""
        import zipfile

        repo = tmp_path / "repo"
        task_dir = repo / "Code" / "abc999_z" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
        (task_dir / "correctVersion.py").write_text(
            "def solve():\n    return 2\n", encoding="utf-8"
        )
        (repo / "difficulty.txt").write_text("abc999 1200\n", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc999/Z/in/example_00.txt", "1\n")
            zf.writestr("Test/abc999/Z/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda c: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda c: True)
        monkeypatch.setattr(
            condefects_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-1\n+2\n",
        )
        monkeypatch.setattr(condefects_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(condefects_importer, "infer_snapshot_dependency", lambda c: "dep")

        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert len(result) == 1
        assert result[0]["metadata"]["difficulty"] == "medium"

    def test_no_tests_task_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 330-333: task without matching test pairs → skip."""
        import zipfile

        repo = tmp_path / "repo"
        task_dir = repo / "Code" / "abc010_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
        (task_dir / "correctVersion.py").write_text(
            "def solve():\n    return 2\n", encoding="utf-8"
        )
        (repo / "difficulty.txt").write_text("", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/other/A/in/example_00.txt", "1\n")
            zf.writestr("Test/other/A/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert result == []

    def test_extract_empty_pairs_skipped(self, tmp_path: Path) -> None:
        """Lines 183-186: empty input/output are not included."""
        import zipfile

        zp = tmp_path / "t.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/t/in/example_00.txt", "  \n")
            zf.writestr("Test/t/out/example_00.txt", "  \n")
        with zipfile.ZipFile(zp) as zf:
            idx = condefects_importer._build_zip_index(zf)
            pairs = condefects_importer._extract_test_pairs_from_zip(zf, idx, "Test/t")
        assert pairs == []

    def test_extract_large_input_skipped(self, tmp_path: Path) -> None:
        """Line 167: input larger than MAX_TEST_INPUT_BYTES → skipped."""
        import zipfile

        zp = tmp_path / "t.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            big = "x" * (condefects_importer.MAX_TEST_INPUT_BYTES + 1)
            zf.writestr("Test/t/in/example_00.txt", big)
            zf.writestr("Test/t/out/example_00.txt", "y\n")
        with zipfile.ZipFile(zp) as zf:
            idx = condefects_importer._build_zip_index(zf)
            pairs = condefects_importer._extract_test_pairs_from_zip(zf, idx, "Test/t")
        assert pairs == []

    def test_external_deps_filter(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 391-393: external deps → skip."""
        import zipfile

        repo = tmp_path / "repo"
        task_dir = repo / "Code" / "abc006_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text(
            "import numpy\ndef solve():\n    return 1\n", encoding="utf-8"
        )
        (task_dir / "correctVersion.py").write_text(
            "import numpy\ndef solve():\n    return 2\n", encoding="utf-8"
        )
        (repo / "difficulty.txt").write_text("", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc006/A/in/example_00.txt", "1\n")
            zf.writestr("Test/abc006/A/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda c: True)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda c: True)

        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert result == []

    def test_nondeterministic_filter(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 396-398: non-deterministic code → skip."""
        import zipfile

        repo = tmp_path / "repo"
        task_dir = repo / "Code" / "abc007_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        (task_dir / "faultyVersion.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
        (task_dir / "correctVersion.py").write_text(
            "def solve():\n    return 2\n", encoding="utf-8"
        )
        (repo / "difficulty.txt").write_text("", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc007/A/in/example_00.txt", "1\n")
            zf.writestr("Test/abc007/A/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda c: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda c: False)

        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert result == []

    def test_code_too_long_filter(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 401-403: code exceeding MAX_CODE_LINES → skip."""
        import zipfile

        repo = tmp_path / "repo"
        task_dir = repo / "Code" / "abc008_a" / "Python" / "1"
        task_dir.mkdir(parents=True)
        long_code = "\n".join(["x = 1"] * (condefects_importer.MAX_CODE_LINES + 1))
        (task_dir / "faultyVersion.py").write_text(long_code, encoding="utf-8")
        (task_dir / "correctVersion.py").write_text("x = 2\n", encoding="utf-8")
        (repo / "difficulty.txt").write_text("", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Test/abc008/A/in/example_00.txt", "1\n")
            zf.writestr("Test/abc008/A/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda c: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda c: True)

        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert result == []

    def test_empty_patch_and_long_fix_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 410-412, 415-417: empty patch or fix > MAX_FIX_LINES → skip."""
        import zipfile

        repo = tmp_path / "repo"
        for idx, tag in enumerate(["empty", "long"]):
            task_dir = repo / "Code" / f"abc{idx:03d}_a" / "Python" / "1"
            task_dir.mkdir(parents=True)
            (task_dir / "faultyVersion.py").write_text(
                f"def solve():\n    return '{tag}'\n", encoding="utf-8"
            )
            (task_dir / "correctVersion.py").write_text(
                f"def solve():\n    return '{tag}_fixed'\n", encoding="utf-8"
            )
        (repo / "difficulty.txt").write_text("", encoding="utf-8")

        zp = tmp_path / "tests.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            for idx in range(2):
                prefix = f"Test/abc{idx:03d}/A"
                zf.writestr(f"{prefix}/in/example_00.txt", "1\n")
                zf.writestr(f"{prefix}/out/example_00.txt", "2\n")

        monkeypatch.setattr(condefects_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(condefects_importer, "has_external_deps", lambda c: False)
        monkeypatch.setattr(condefects_importer, "is_deterministic", lambda c: True)
        monkeypatch.setattr(condefects_importer, "infer_snapshot_dependency", lambda c: "dep")

        call_n = {"n": 0}

        def _fake_patch(b: str, f: str) -> str:
            call_n["n"] += 1
            if "empty" in b:
                return ""
            return "long patch"

        monkeypatch.setattr(condefects_importer, "generate_oracle_patch", _fake_patch)
        monkeypatch.setattr(
            condefects_importer,
            "count_patch_lines",
            lambda p: 20,
        )

        result = condefects_importer.import_condefects(
            tmp_path / "cases", repo_path=repo, test_zip=zp, dry_run=True
        )
        assert result == []


# =====================================================================
# pyresbugs_importer.py — missing lines
# =====================================================================


class TestPyresbugsImportPipeline:
    """Cover import_pyresbugs pipeline branches."""

    def test_datasets_import_error(self, monkeypatch: Any) -> None:
        """Lines 283-284: missing datasets package."""
        monkeypatch.delitem(sys.modules, "datasets", raising=False)
        monkeypatch.setattr("builtins.__import__", _make_import_blocker("datasets"))
        with pytest.raises(ImportError, match="datasets"):
            pyresbugs_importer.import_pyresbugs(Path("/fake"))

    def test_syntax_error_find_changed_function(self) -> None:
        """Lines 144-145: SyntaxError when both have multiple functions but one is broken."""
        # Both have functions → len(all_funcs) > 1, then ast.parse triggers SyntaxError
        faulty = "def a():\n    return 1\ndef broken("
        fixed = "def a():\n    return 1\ndef b():\n    return 2\n"
        result = pyresbugs_importer._find_changed_function(faulty, fixed)
        assert result is None

    def test_find_changed_function_no_diff(self) -> None:
        """Line 160: no diff between functions → None."""
        code = "def a():\n    return 1\n\ndef b():\n    return 2\n"
        result = pyresbugs_importer._find_changed_function(code, code)
        assert result is None

    def test_get_function_params_not_found(self) -> None:
        """Line 172: function not found → None."""
        result = pyresbugs_importer._get_function_params("def other():\n    pass\n", "missing")
        assert result is None

    def test_generate_test_no_func_name(self) -> None:
        """Lines 197-198, 201: no function found → import-only test."""
        result = pyresbugs_importer._generate_test("x = 1\n", "x = 2\n", "TYPE")
        assert result is not None
        assert "def test_import()" in result

    def test_generate_test_params_none(self) -> None:
        """Line 208: _get_function_params returns None → treated as []."""
        result = pyresbugs_importer._generate_test(
            "def f():\n    return 1\n",
            "def f():\n    return 2\n",
            "WPFV",
        )
        assert result is not None
        assert "def test_call()" in result

    def test_verify_patch_success(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Line 246: successful patch verification."""
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        (case_dir / "oracle.patch").write_text("patch\n", encoding="utf-8")

        def _ok_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args[0], 0)

        monkeypatch.setattr(pyresbugs_importer.subprocess, "run", _ok_run)
        result = pyresbugs_importer._verify_patch(case_dir)
        assert result is True

    def test_verify_patch_timeout(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 246-248: timeout returns False."""
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        (case_dir / "oracle.patch").write_text("patch\n", encoding="utf-8")

        def _timeout(*args: Any, **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="patch", timeout=5)

        monkeypatch.setattr(pyresbugs_importer.subprocess, "run", _timeout)
        result = pyresbugs_importer._verify_patch(case_dir)
        assert result is False

    def test_case_exists_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 327-328: existing case dir → skip."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 2\\n",
                "Fault_Acronym": "MIA",
                "High-Level Description": "test",
                "Project": "demo",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)

        out = tmp_path / "cases"
        (out / "prb_0000_easy").mkdir(parents=True)
        result = pyresbugs_importer.import_pyresbugs(out, dry_run=False, validate=False)
        assert result == []

    def test_empty_patch_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 350-351: empty oracle patch → skip."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 1\\n",
                "Fault_Acronym": "MIA",
                "High-Level Description": "",
                "Project": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(pyresbugs_importer, "generate_oracle_patch", lambda b, f: "")
        result = pyresbugs_importer.import_pyresbugs(tmp_path / "cases", dry_run=True)
        assert result == []

    def test_long_fix_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 355-356: fix > MAX_FIX_LINES → skip."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 2\\n",
                "Fault_Acronym": "MIA",
                "High-Level Description": "",
                "Project": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(pyresbugs_importer, "generate_oracle_patch", lambda b, f: "long")
        monkeypatch.setattr(pyresbugs_importer, "count_patch_lines", lambda p: 20)
        result = pyresbugs_importer.import_pyresbugs(tmp_path / "cases", dry_run=True)
        assert result == []

    def test_no_test_generated_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 361-362: _generate_test returns None → skip."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 2\\n",
                "Fault_Acronym": "MIA",
                "High-Level Description": "",
                "Project": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(pyresbugs_importer, "generate_oracle_patch", lambda b, f: "patch")
        monkeypatch.setattr(pyresbugs_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(pyresbugs_importer, "_generate_test", lambda *a: None)
        result = pyresbugs_importer.import_pyresbugs(tmp_path / "cases", dry_run=True)
        assert result == []

    def test_write_verify_validate_paths(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 406-432: write, verify, validate paths."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 2\\n",
                "Fault_Acronym": "MIA",
                "High-Level Description": "desc",
                "Project": "proj",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(pyresbugs_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(pyresbugs_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            pyresbugs_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(pyresbugs_importer, "_verify_patch", lambda cd: True)
        monkeypatch.setattr(
            pyresbugs_importer,
            "validate_generated_case",
            lambda cd, schema_only: (True, ""),
        )
        result = pyresbugs_importer.import_pyresbugs(
            tmp_path / "cases", dry_run=False, validate=True
        )
        assert len(result) == 1
        assert result[0]["case_id"] is not None

    def test_verify_patch_failure_removes_case(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 415-420: verify_patch fails → remove case dir."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 2\\n",
                "Fault_Acronym": "MIA",
                "High-Level Description": "",
                "Project": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(pyresbugs_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(pyresbugs_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            pyresbugs_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(pyresbugs_importer, "_verify_patch", lambda cd: False)
        result = pyresbugs_importer.import_pyresbugs(
            tmp_path / "cases", dry_run=False, validate=True
        )
        assert result == []

    def test_validate_failure_removes_case(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 422-430: validate fails → remove case dir."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 2\\n",
                "Fault_Acronym": "MIA",
                "High-Level Description": "",
                "Project": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(pyresbugs_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(pyresbugs_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            pyresbugs_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(pyresbugs_importer, "_verify_patch", lambda cd: True)
        monkeypatch.setattr(
            pyresbugs_importer,
            "validate_generated_case",
            lambda cd, schema_only: (False, "bad"),
        )
        result = pyresbugs_importer.import_pyresbugs(
            tmp_path / "cases", dry_run=False, validate=True
        )
        assert result == []

    def test_progress_report_at_500(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 298-299: progress reported every 500 rows (idx % 500 == 0)."""
        rows = [{"Faulty Code": "", "Fault Free Code": "", "Fault_Acronym": "MIA"}] * 501
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        result = pyresbugs_importer.import_pyresbugs(tmp_path / "cases", dry_run=True)
        assert result == []

    def test_second_case_id_collision(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 376-377: second case_id (with difficulty suffix) already exists → skip."""
        rows = [
            {
                "Faulty Code": "def f():\\n    return 1\\n",
                "Fault Free Code": "def f():\\n    return 2\\n",
                "Fault_Acronym": "WPFV",
                "High-Level Description": "desc",
                "Project": "proj",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        self._setup_monkeypatch(monkeypatch)
        monkeypatch.setattr(pyresbugs_importer, "generate_oracle_patch", lambda b, f: "patch\n")
        monkeypatch.setattr(pyresbugs_importer, "count_patch_lines", lambda p: 1)

        out = tmp_path / "cases"
        (out / "prb_0000_easy").mkdir(parents=True)
        result = pyresbugs_importer.import_pyresbugs(out, dry_run=False, validate=False)
        assert result == []

    def _setup_monkeypatch(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(pyresbugs_importer, "map_pyresbugs_type", lambda a: a.lower())
        monkeypatch.setattr(pyresbugs_importer, "infer_snapshot_dependency", lambda c: "dep")
        monkeypatch.setattr(pyresbugs_importer, "ruff_format", lambda c: c)
        monkeypatch.setattr(pyresbugs_importer, "has_external_deps", lambda c: False)
        monkeypatch.setattr(pyresbugs_importer, "is_deterministic", lambda c: True)


# =====================================================================
# humanevalpack_importer.py — missing lines
# =====================================================================


class TestHumanevalpackImportPipeline:
    """Cover import_humanevalpack pipeline branches."""

    def test_datasets_import_error(self, monkeypatch: Any) -> None:
        """Lines 97-98: datasets package missing."""
        monkeypatch.delitem(sys.modules, "datasets", raising=False)
        monkeypatch.setattr("builtins.__import__", _make_import_blocker("datasets"))
        with pytest.raises(ImportError, match="datasets"):
            humanevalpack_importer.import_humanevalpack(Path("/fake"))

    def test_max_cases_limit(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Line 109: max_cases limit reached → stop."""
        rows = [
            {
                "task_id": f"Task/{i}",
                "buggy_solution": "def f():\n    return 1\n",
                "canonical_solution": "def f():\n    return 2\n",
                "prompt": "",
                "test": "assert f() == 2\n",
                "bug_type": "logic",
                "entry_point": "f()",
                "failure_symptoms": "",
            }
            for i in range(5)
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(
            humanevalpack_importer, "map_humanevalpack_type", lambda bt: "logic_error"
        )
        monkeypatch.setattr(
            humanevalpack_importer,
            "infer_snapshot_dependency",
            lambda c: "dep",
        )
        monkeypatch.setattr(
            humanevalpack_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-1\n+2\n",
        )
        monkeypatch.setattr(humanevalpack_importer, "count_patch_lines", lambda p: 1)
        result = humanevalpack_importer.import_humanevalpack(
            tmp_path / "cases", max_cases=2, dry_run=True
        )
        assert len(result) == 2
        assert all("case_id" in r for r in result)

    def _install_rows(self, monkeypatch: Any, count: int) -> None:
        rows = [
            {
                "task_id": f"Task/{i}",
                "buggy_solution": "def f():\n    return 1\n",
                "canonical_solution": "def f():\n    return 2\n",
                "prompt": "",
                "test": "assert f() == 2\n",
                "bug_type": "logic",
                "entry_point": "f()",
                "failure_symptoms": "",
            }
            for i in range(count)
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(humanevalpack_importer, "map_humanevalpack_type", lambda bt: "logic")
        monkeypatch.setattr(humanevalpack_importer, "infer_snapshot_dependency", lambda c: "dep")
        monkeypatch.setattr(
            humanevalpack_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-1\n+2\n",
        )
        monkeypatch.setattr(humanevalpack_importer, "count_patch_lines", lambda p: 1)

    def test_max_cases_zero_means_unlimited(self, tmp_path: Path, monkeypatch: Any) -> None:
        """DAT-22: 0 means unlimited, as in the other importers."""
        self._install_rows(monkeypatch, 5)
        result = humanevalpack_importer.import_humanevalpack(
            tmp_path / "cases", max_cases=0, dry_run=True
        )
        assert len(result) == 5

    def test_existing_cases_are_not_overwritten(self, tmp_path: Path, monkeypatch: Any) -> None:
        """DAT-23: a re-run keeps earlier corrections such as fixed expected exceptions."""
        self._install_rows(monkeypatch, 2)
        output_dir = tmp_path / "cases"
        existing = output_dir / "hep_task_0_logic"
        existing.mkdir(parents=True)
        (existing / "case.json").write_text('{"corrected": true}', encoding="utf-8")
        written: list[str] = []
        monkeypatch.setattr(
            humanevalpack_importer,
            "write_case_files",
            lambda case_dir, **_kwargs: written.append(case_dir.name),
        )
        monkeypatch.setattr(
            humanevalpack_importer, "validate_generated_case", lambda *_a, **_k: (True, "ok")
        )

        humanevalpack_importer.import_humanevalpack(output_dir, max_cases=0)

        assert written == ["hep_task_1_logic"]
        assert (existing / "case.json").read_text(encoding="utf-8") == '{"corrected": true}'

    def test_empty_patch_skipped(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 130-131: empty oracle patch → skip."""
        rows = [
            {
                "task_id": "Task/0",
                "buggy_solution": "def f():\n    return 1\n",
                "canonical_solution": "def f():\n    return 1\n",
                "prompt": "",
                "test": "assert True\n",
                "bug_type": "logic",
                "entry_point": "f()",
                "failure_symptoms": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(humanevalpack_importer, "map_humanevalpack_type", lambda bt: "logic")
        monkeypatch.setattr(
            humanevalpack_importer,
            "infer_snapshot_dependency",
            lambda c: "dep",
        )
        monkeypatch.setattr(humanevalpack_importer, "generate_oracle_patch", lambda b, f: "")
        result = humanevalpack_importer.import_humanevalpack(tmp_path / "cases", dry_run=True)
        assert result == []

    def test_write_and_validate_success(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 178-197: write_case_files and validate paths."""
        rows = [
            {
                "task_id": "Task/1",
                "buggy_solution": "def f():\n    return 1\n",
                "canonical_solution": "def f():\n    return 2\n",
                "prompt": "",
                "test": "assert f() == 2\n",
                "bug_type": "logic",
                "entry_point": "f()",
                "failure_symptoms": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(humanevalpack_importer, "map_humanevalpack_type", lambda bt: "logic")
        monkeypatch.setattr(
            humanevalpack_importer,
            "infer_snapshot_dependency",
            lambda c: "dep",
        )
        monkeypatch.setattr(
            humanevalpack_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-1\n+2\n",
        )
        monkeypatch.setattr(humanevalpack_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            humanevalpack_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(
            humanevalpack_importer,
            "validate_generated_case",
            lambda cd, schema_only: (True, ""),
        )
        result = humanevalpack_importer.import_humanevalpack(
            tmp_path / "cases", dry_run=False, validate=True
        )
        assert len(result) == 1
        assert result[0]["case_id"] is not None

    def test_validate_failure_removes_case(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 187-195: validate fails → remove case dir."""
        rows = [
            {
                "task_id": "Task/2",
                "buggy_solution": "def f():\n    return 1\n",
                "canonical_solution": "def f():\n    return 2\n",
                "prompt": "",
                "test": "assert f() == 2\n",
                "bug_type": "logic",
                "entry_point": "f()",
                "failure_symptoms": "",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(humanevalpack_importer, "map_humanevalpack_type", lambda bt: "logic")
        monkeypatch.setattr(
            humanevalpack_importer,
            "infer_snapshot_dependency",
            lambda c: "dep",
        )
        monkeypatch.setattr(
            humanevalpack_importer,
            "generate_oracle_patch",
            lambda b, f: "--- a/buggy.py\n+++ b/buggy.py\n-1\n+2\n",
        )
        monkeypatch.setattr(humanevalpack_importer, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(
            humanevalpack_importer,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )
        monkeypatch.setattr(
            humanevalpack_importer,
            "validate_generated_case",
            lambda cd, schema_only: (False, "bad"),
        )
        result = humanevalpack_importer.import_humanevalpack(
            tmp_path / "cases", dry_run=False, validate=True
        )
        assert result == []


# =====================================================================
# repair_debugbench_patches.py — missing lines
# =====================================================================


class TestRepairDebugbenchPatches:
    """Cover repair_debugbench_patches missing paths."""

    def test_datasets_import_error(self, monkeypatch: Any) -> None:
        """Lines 59-60: datasets package missing."""
        monkeypatch.delitem(sys.modules, "datasets", raising=False)
        monkeypatch.setattr("builtins.__import__", _make_import_blocker("datasets"))
        with pytest.raises(ImportError, match="datasets"):
            repair_debugbench_patches.repair_debugbench_patches()

    def test_empty_patch_after_ruff(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 125-126: empty patch after ruff formatting."""
        case_dir = tmp_path / "cases" / "db_empty"
        _write_case(case_dir, source_id="debugbench_0")

        rows = [
            {
                "language": "python3",
                "buggy_code": "x = 1\n",
                "solution": "x = 1\n",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(repair_debugbench_patches, "ruff_format", lambda c: c)
        monkeypatch.setattr(repair_debugbench_patches, "generate_oracle_patch", lambda b, f: "")
        monkeypatch.setattr(repair_debugbench_patches, "_patch_applies", lambda cd: False)

        result = repair_debugbench_patches.repair_debugbench_patches(tmp_path / "cases")
        assert result["db_empty"] == "failed: empty patch after ruff formatting"
        assert "db_empty" in result

    def test_repair_still_broken(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Lines 142-143: new patch still doesn't apply."""
        case_dir = tmp_path / "cases" / "db_broken"
        _write_case(case_dir, source_id="debugbench_0")

        rows = [
            {
                "language": "python3",
                "buggy_code": "x = 1\n",
                "solution": "x = 2\n",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(repair_debugbench_patches, "ruff_format", lambda c: c)
        monkeypatch.setattr(
            repair_debugbench_patches,
            "generate_oracle_patch",
            lambda b, f: "new patch\n",
        )
        monkeypatch.setattr(repair_debugbench_patches, "count_patch_lines", lambda p: 1)
        monkeypatch.setattr(repair_debugbench_patches, "_patch_applies", lambda cd: False)

        result = repair_debugbench_patches.repair_debugbench_patches(tmp_path / "cases")
        assert result["db_broken"] == "failed: new patch still doesn't apply"
        assert "db_broken" in result

    def test_main_function(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """Lines 149-159: main() entry point."""
        case_dir = tmp_path / "cases" / "db_ok"
        _write_case(case_dir, source_id="debugbench_0")
        fail_dir = tmp_path / "cases" / "db_fail"
        _write_case(fail_dir, source_id="debugbench_0")

        rows = [
            {
                "language": "python3",
                "buggy_code": "x = 1\n",
                "solution": "x = 2\n",
            },
        ]
        _install_fake_datasets(monkeypatch, rows)
        monkeypatch.setattr(repair_debugbench_patches, "ruff_format", lambda c: c)
        monkeypatch.setattr(
            repair_debugbench_patches,
            "generate_oracle_patch",
            lambda b, f: "new patch\n",
        )
        monkeypatch.setattr(repair_debugbench_patches, "count_patch_lines", lambda p: 1)

        call_count = {"n": 0}

        def _fake_applies(cd: Path) -> bool:
            call_count["n"] += 1
            if cd.name == "db_ok":
                return True
            if cd.name == "db_fail":
                return False
            # After writing new patch
            return call_count["n"] > 2

        monkeypatch.setattr(repair_debugbench_patches, "_patch_applies", _fake_applies)
        monkeypatch.setattr(
            repair_debugbench_patches,
            "CASES_DIR",
            tmp_path / "cases",
        )

        repair_debugbench_patches.main()

        # Verify main() produced output (it prints results summary)
        captured = capsys.readouterr()
        assert "Results:" in captured.out


# =====================================================================
# Helper: import blocker for datasets package
# =====================================================================


_original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _make_import_blocker(blocked_module: str):
    """Return a __import__ replacement that blocks one specific module."""

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return _original_import(name, *args, **kwargs)

    return _blocked_import


def test_dat16_condefects_skips_a_bug_already_imported_under_another_difficulty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """DAT-16: the difficulty suffix in the case ID let the same bug be imported twice."""
    import zipfile

    repo = tmp_path / "repo"
    task_dir = repo / "Code" / "abc001_a" / "Python" / "1"
    task_dir.mkdir(parents=True)
    (task_dir / "faultyVersion.py").write_text("print(1)\n", encoding="utf-8")
    (task_dir / "correctVersion.py").write_text("print(2)\n", encoding="utf-8")
    (repo / "difficulty.txt").write_text("abc001_a 300\n", encoding="utf-8")
    zp = tmp_path / "tests.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("Test/abc001/A/in/example_00.txt", "1\n")
        zf.writestr("Test/abc001/A/out/example_00.txt", "2\n")
    monkeypatch.setattr(condefects_importer, "ruff_format", lambda code: code)

    out = tmp_path / "cases"
    existing = out / "cd_abc001_a_1_hard"  # imported earlier with another difficulty
    existing.mkdir(parents=True)
    (existing / "case.json").write_text(
        json.dumps({"source_id": "condefects_abc001_a_1"}), encoding="utf-8"
    )

    result = condefects_importer.import_condefects(out, repo_path=repo, test_zip=zp)

    assert result == []
    assert sorted(path.name for path in out.iterdir()) == ["cd_abc001_a_1_hard"]
    # Control: the same candidate imports into an empty directory.
    fresh = condefects_importer.import_condefects(tmp_path / "fresh", repo_path=repo, test_zip=zp)
    assert [case["case_id"] for case in fresh] != []
    assert all(not case["case_id"].endswith("_hard") for case in fresh)


def test_dat16_existing_source_ids_reads_nested_cases(tmp_path: Path) -> None:
    from evals.importers.base import existing_source_ids

    for rel, source_id in (("a/c1", "s1"), ("b/nested/c2", "s2")):
        case_dir = tmp_path / rel
        case_dir.mkdir(parents=True)
        (case_dir / "case.json").write_text(json.dumps({"source_id": source_id}), "utf-8")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "case.json").write_text("{", encoding="utf-8")

    assert existing_source_ids(tmp_path) == {"s1", "s2"}
    assert existing_source_ids(tmp_path / "missing") == set()
