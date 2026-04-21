"""Coverage-boost tests for evals/importers/quixbugs_importer.py.

Targets missing line and branches to push from 98% to 100%:
- Line 209: _strip_docstring_tail with non-zero col_offset
- Branch [314, 324]: graph program with no test file → test_code stays None
- Branch [321, 324]: JSON test cases file exists but is empty
- Branch [374, 384]: validate=True and validation fails → case removed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evals.importers.quixbugs_importer import (
    _strip_docstring_tail,
    import_quixbugs,
)

# ---------------------------------------------------------------------------
# _strip_docstring_tail: non-zero col_offset → line 209
# ---------------------------------------------------------------------------


class TestStripDocstringTailColOffset:
    def test_indented_string_literal_not_stripped(self) -> None:
        """A string constant indented inside a function has col_offset > 0."""
        code = 'def f():\n    "docstring inside function"\n'
        result = _strip_docstring_tail(code)
        # The trailing expression has col_offset=4 so it is NOT stripped.
        assert result == code

    def test_toplevel_docstring_is_stripped(self) -> None:
        """A module-level trailing triple-quoted docstring IS stripped."""
        code = 'def f():\n    pass\n\n"""This is a trailing doc"""\n'
        result = _strip_docstring_tail(code)
        assert '"""This is a trailing doc"""' not in result


# ---------------------------------------------------------------------------
# import_quixbugs: graph program with no test file → branch [314, 324]
# ---------------------------------------------------------------------------


class TestGraphProgramNoTestFile:
    def test_graph_program_without_test_file_skipped(self, tmp_path: Path) -> None:
        """A graph-based program with no pytest test file should be skipped."""
        repo = tmp_path / "repo"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)
        (repo / "python_testcases").mkdir(parents=True)

        # Use breadth_first_search (a GRAPH_BASED_PROGRAMS member)
        buggy_code = "def breadth_first_search(startnode, goalnode):\n    return False\n"
        fixed_code = "def breadth_first_search(startnode, goalnode):\n    return True\n"
        (repo / "python_programs" / "breadth_first_search.py").write_text(
            buggy_code, encoding="utf-8"
        )
        (repo / "correct_python_programs" / "breadth_first_search.py").write_text(
            fixed_code, encoding="utf-8"
        )
        # Deliberately do NOT create test_breadth_first_search.py in python_testcases

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = import_quixbugs(output_dir, repo_path=repo, dry_run=True)
        # Should be skipped because test_code is None
        assert len(result) == 0


# ---------------------------------------------------------------------------
# import_quixbugs: JSON test cases empty → branch [321, 324]
# ---------------------------------------------------------------------------


class TestJsonTestCasesEmpty:
    def test_empty_json_testcases_skips_program(self, tmp_path: Path) -> None:
        """When the JSON test case file exists but has no valid cases."""
        repo = tmp_path / "repo"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)
        (repo / "python_testcases").mkdir(parents=True)

        # Non-graph program
        buggy_code = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a % b, b)\n"
        fixed_code = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\n"
        (repo / "python_programs" / "gcd.py").write_text(buggy_code, encoding="utf-8")
        (repo / "correct_python_programs" / "gcd.py").write_text(fixed_code, encoding="utf-8")
        # JSON file exists but is empty (or has only blank lines)
        (repo / "json_testcases" / "gcd.json").write_text("\n\n", encoding="utf-8")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = import_quixbugs(output_dir, repo_path=repo, dry_run=True)
        # Skipped because test_cases list is empty
        assert len(result) == 0


# ---------------------------------------------------------------------------
# import_quixbugs: validation fails → branch [374, 384]
# ---------------------------------------------------------------------------


class TestValidationFailsRemovesCase:
    def test_failed_validation_removes_case_directory(self, tmp_path: Path) -> None:
        """When validate=True and validate_generated_case returns failure."""
        repo = tmp_path / "repo"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)
        (repo / "python_testcases").mkdir(parents=True)

        buggy_code = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a % b, b)\n"
        fixed_code = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\n"
        (repo / "python_programs" / "gcd.py").write_text(buggy_code, encoding="utf-8")
        (repo / "correct_python_programs" / "gcd.py").write_text(fixed_code, encoding="utf-8")
        (repo / "json_testcases" / "gcd.json").write_text(
            "[[10, 5], 5]\n[[14, 21], 7]\n", encoding="utf-8"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch(
            "evals.importers.quixbugs_importer.validate_generated_case",
            return_value=(False, "schema mismatch"),
        ):
            result = import_quixbugs(output_dir, repo_path=repo, validate=True, dry_run=False)

        # Case should NOT be in the result (validation failed)
        assert len(result) == 0
        # Case directory should have been removed
        case_dir = output_dir / "qb_gcd_easy"
        assert not case_dir.exists()


# ---------------------------------------------------------------------------
# import_quixbugs: validate=False skips validation → branch [374, 384]
# ---------------------------------------------------------------------------


class TestImportWithValidateFalse:
    def test_validate_false_skips_validation(self, tmp_path: Path) -> None:
        """When validate=False, the validation block is skipped entirely."""
        repo = tmp_path / "repo"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)
        (repo / "python_testcases").mkdir(parents=True)

        buggy_code = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(a % b, b)\n"
        fixed_code = "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\n"
        (repo / "python_programs" / "gcd.py").write_text(buggy_code, encoding="utf-8")
        (repo / "correct_python_programs" / "gcd.py").write_text(fixed_code, encoding="utf-8")
        (repo / "json_testcases" / "gcd.json").write_text(
            "[[10, 5], 5]\n[[14, 21], 7]\n", encoding="utf-8"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = import_quixbugs(output_dir, repo_path=repo, validate=False, dry_run=False)

        # Case should be in the result (validation was skipped)
        assert len(result) == 1
        assert result[0]["case_id"] == "qb_gcd_easy"
        # Case directory should exist
        case_dir = output_dir / "qb_gcd_easy"
        assert case_dir.exists()
