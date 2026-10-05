"""Tests for evals.importers.base — shared importer utilities.

Uses synthetic data only (no HuggingFace downloads).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from evals.importers import debugbench_importer, fixup_expected_exceptions, pyresbugs_importer
from evals.importers.base import (
    RuffFormatError,
    _extract_import_roots_fallback,
    _extract_import_roots_from_ast,
    count_patch_lines,
    generate_oracle_patch,
    has_external_deps,
    is_deterministic,
    ruff_format,
    validate_generated_case,
    write_case_files,
)
from llmdebug import _json as json

# ---------------------------------------------------------------------------
# generate_oracle_patch
# ---------------------------------------------------------------------------


class TestGenerateOraclePatch:
    def test_simple_change(self) -> None:
        buggy = "def f():\n    return 1\n"
        fixed = "def f():\n    return 2\n"
        diff_patch = generate_oracle_patch(buggy, fixed)
        assert "diff --git" in diff_patch
        assert "--- a/buggy.py" in diff_patch
        assert "+++ b/buggy.py" in diff_patch
        assert "-    return 1" in diff_patch
        assert "+    return 2" in diff_patch

    def test_custom_filename(self) -> None:
        buggy = "x = 1\n"
        fixed = "x = 2\n"
        diff_patch = generate_oracle_patch(buggy, fixed, filename="module.py")
        assert "a/module.py" in diff_patch
        assert "b/module.py" in diff_patch

    def test_no_diff_returns_empty(self) -> None:
        code = "def f():\n    return 1\n"
        diff_patch = generate_oracle_patch(code, code)
        assert diff_patch == ""

    def test_multiline_change(self) -> None:
        buggy = "a = 1\nb = 2\nc = 3\n"
        fixed = "a = 1\nb = 20\nc = 3\n"
        diff_patch = generate_oracle_patch(buggy, fixed)
        assert "-b = 2" in diff_patch
        assert "+b = 20" in diff_patch

    def test_addition_only(self) -> None:
        buggy = "a = 1\n"
        fixed = "a = 1\nb = 2\n"
        diff_patch = generate_oracle_patch(buggy, fixed)
        assert "+b = 2" in diff_patch


# ---------------------------------------------------------------------------
# write_case_files
# ---------------------------------------------------------------------------


class TestWriteCaseFiles:
    def test_creates_all_files(self, tmp_path) -> None:
        case_dir = tmp_path / "test_case"
        write_case_files(
            case_dir,
            buggy_code="def f():\n    return 1\n",
            test_code="def test_f():\n    assert True\n",
            oracle_patch="--- a/buggy.py\n+++ b/buggy.py\n",
            metadata={"category": "logic_error", "difficulty": "easy"},
        )

        assert (case_dir / "buggy.py").exists()
        assert (case_dir / "test_buggy.py").exists()
        assert (case_dir / "oracle.patch").exists()
        assert (case_dir / "case.json").exists()

        meta = json.loads((case_dir / "case.json").read_text())
        assert meta["category"] == "logic_error"
        assert meta["difficulty"] == "easy"

    def test_creates_nested_directories(self, tmp_path) -> None:
        case_dir = tmp_path / "nested" / "deep" / "case"
        write_case_files(
            case_dir,
            buggy_code="x = 1\n",
            test_code="assert True\n",
            oracle_patch="patch\n",
            metadata={},
        )
        assert case_dir.exists()

    def test_file_contents_match(self, tmp_path) -> None:
        case_dir = tmp_path / "case"
        write_case_files(
            case_dir,
            buggy_code="CODE",
            test_code="TEST",
            oracle_patch="PATCH",
            metadata={"key": "value"},
        )
        assert (case_dir / "buggy.py").read_text() == "CODE"
        assert (case_dir / "test_buggy.py").read_text() == "TEST"
        assert (case_dir / "oracle.patch").read_text() == "PATCH"


# ---------------------------------------------------------------------------
# validate_generated_case
# ---------------------------------------------------------------------------


class TestValidateGeneratedCase:
    def test_valid_case_passes(self, tmp_path) -> None:
        case_dir = tmp_path / "valid"
        write_case_files(
            case_dir,
            buggy_code="def f():\n    return 1\n",
            test_code="def test_f():\n    assert True\n",
            oracle_patch="diff --git a/buggy.py b/buggy.py\n-    return 1\n+    return 2\n",
            metadata={
                "category": "logic_error",
                "difficulty": "easy",
                "snapshot_dependency": "helpful",
                "expected_exception": "AssertionError",
            },
        )
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert ok, msg

    def test_missing_file_fails(self, tmp_path) -> None:
        case_dir = tmp_path / "bad"
        case_dir.mkdir()
        (case_dir / "buggy.py").write_text("x = 1\n")
        (case_dir / "case.json").write_text("{}")
        # Missing test_buggy.py and oracle.patch
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "missing required file" in msg

    def test_invalid_category_fails(self, tmp_path) -> None:
        case_dir = tmp_path / "bad_cat"
        write_case_files(
            case_dir,
            buggy_code="x = 1\n",
            test_code="assert True\n",
            oracle_patch="patch content\n",
            metadata={
                "category": "nonexistent_category",
                "difficulty": "easy",
                "snapshot_dependency": "helpful",
                "expected_exception": "AssertionError",
            },
        )
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "invalid category" in msg

    def test_empty_patch_fails(self, tmp_path) -> None:
        case_dir = tmp_path / "empty_patch"
        write_case_files(
            case_dir,
            buggy_code="x = 1\n",
            test_code="assert True\n",
            oracle_patch="",
            metadata={
                "category": "logic_error",
                "difficulty": "easy",
                "snapshot_dependency": "helpful",
                "expected_exception": "AssertionError",
            },
        )
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "empty" in msg


# ---------------------------------------------------------------------------
# count_patch_lines
# ---------------------------------------------------------------------------


class TestCountPatchLines:
    def test_simple_replacement(self) -> None:
        diff_patch = (
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def f():\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        assert count_patch_lines(diff_patch) == 2  # 1 addition + 1 deletion

    def test_addition_only(self) -> None:
        diff_patch = "+new_line\n+another_line\n"
        assert count_patch_lines(diff_patch) == 2

    def test_excludes_header_lines(self) -> None:
        diff_patch = "--- a/file.py\n+++ b/file.py\n-old\n+new\n"
        assert count_patch_lines(diff_patch) == 2  # Only -old and +new

    def test_empty_patch(self) -> None:
        assert count_patch_lines("") == 0


class TestHasExternalDeps:
    def test_stdlib_only_imports_are_allowed(self) -> None:
        code = "import math\nfrom collections import deque\n"
        assert has_external_deps(code) is False

    def test_multiple_imports_detect_external(self) -> None:
        code = "import math, requests\n"
        assert has_external_deps(code) is True

    def test_from_import_detects_external(self) -> None:
        code = "from pydantic import BaseModel\n"
        assert has_external_deps(code) is True

    def test_syntax_error_fallback_still_detects_imports(self) -> None:
        code = "import requests\nx = (\n"
        assert has_external_deps(code) is True


class TestDebugBenchFallbackTestGeneration:
    def test_reference_error_fallback_uses_zero_arg_method(self) -> None:
        fixed_code = "class Solution:\n    def solve(self):\n        return 1\n"
        generated = debugbench_importer._generate_test(
            fixed_code, examples=None, category="reference error"
        )
        assert generated is not None
        assert "sol.solve()" in generated
        assert "except TypeError" not in generated

    def test_reference_error_fallback_skips_required_args(self) -> None:
        fixed_code = "class Solution:\n    def solve(self, nums):\n        return nums\n"
        generated = debugbench_importer._generate_test(
            fixed_code, examples=None, category="reference error"
        )
        assert generated is None


class TestPyResBugsPatchVerification:
    def test_verify_patch_fails_closed_on_patch_tool_error(self, tmp_path, monkeypatch) -> None:
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        (case_dir / "oracle.patch").write_text(
            "diff --git a/buggy.py b/buggy.py\n", encoding="utf-8"
        )

        def _raise(*args, **kwargs):
            raise FileNotFoundError

        monkeypatch.setattr(pyresbugs_importer.subprocess, "run", _raise)
        assert pyresbugs_importer._verify_patch(case_dir) is False


class TestFixupExpectedExceptions:
    def test_delete_skipped_cases_is_opt_in_helper(self, tmp_path) -> None:
        cases_dir = tmp_path / "cases"
        skip_case = cases_dir / "case_skip"
        keep_case = cases_dir / "case_keep"
        skip_case.mkdir(parents=True)
        keep_case.mkdir(parents=True)

        deleted = fixup_expected_exceptions._delete_skipped_cases(
            cases_dir,
            {
                "case_skip": "skip: no exception found",
                "case_keep": "fixed: AssertionError -> NameError",
            },
        )

        assert deleted == 1
        assert not skip_case.exists()
        assert keep_case.exists()


# ---------------------------------------------------------------------------
# QuixBugs importer
# ---------------------------------------------------------------------------


class TestQuixBugsJsonTestGeneration:
    def test_generates_parametrized_tests(self) -> None:
        from evals.importers.quixbugs_importer import _generate_json_test

        cases = [([127], 7), ([128], 1)]
        test_code = _generate_json_test("bitcount", cases)
        assert "from buggy import bitcount" in test_code
        assert "def test_case_0():" in test_code
        assert "def test_case_1():" in test_code
        assert "bitcount(127)" in test_code
        assert "assert result == 7" in test_code

    def test_empty_cases_returns_minimal(self) -> None:
        from evals.importers.quixbugs_importer import _generate_json_test

        test_code = _generate_json_test("gcd", [])
        assert "from buggy import gcd" in test_code
        assert "def test_case_" not in test_code

    def test_multi_arg_cases(self) -> None:
        from evals.importers.quixbugs_importer import _generate_json_test

        cases = [([3, 5], 1), ([10, 2], 2)]
        test_code = _generate_json_test("gcd", cases)
        assert "gcd(3, 5)" in test_code
        assert "assert result == 1" in test_code


class TestQuixBugsGraphTestRewrite:
    def test_rewrites_imports(self) -> None:
        from evals.importers.quixbugs_importer import _rewrite_graph_test

        original = (
            "import pytest\n"
            "from node import Node\n"
            "\n"
            "if pytest.use_correct:\n"
            "    from correct_python_programs.breadth_first_search"
            " import breadth_first_search\n"
            "else:\n"
            "    from python_programs.breadth_first_search"
            " import breadth_first_search\n"
            "\n"
            "def test1():\n"
            "    node = Node('A')\n"
            "    assert True\n"
        )
        rewritten = _rewrite_graph_test(original, "breadth_first_search")
        assert rewritten is not None
        assert "from buggy import Node" in rewritten
        assert "from buggy import breadth_first_search" in rewritten
        assert "pytest.use_correct" not in rewritten
        assert "correct_python_programs" not in rewritten
        assert "def test1():" in rewritten

    def test_returns_none_for_empty(self) -> None:
        from evals.importers.quixbugs_importer import _rewrite_graph_test

        result = _rewrite_graph_test("import pytest\n", "bfs")
        assert result is None


class TestQuixBugsStripDocstring:
    def test_strips_trailing_docstring(self) -> None:
        from evals.importers.quixbugs_importer import _strip_docstring_tail

        code = 'def f():\n    return 1\n\n"""\nSome doc\n"""\n'
        result = _strip_docstring_tail(code)
        assert "return 1" in result
        assert "Some doc" not in result

    def test_strips_trailing_docstring_with_blank_lines(self) -> None:
        from evals.importers.quixbugs_importer import _strip_docstring_tail

        code = 'def f():\n    return 1\n\n"""\nSome doc\n\nMore doc\n"""\n'
        result = _strip_docstring_tail(code)
        assert result == "def f():\n    return 1\n"
        assert "Some doc" not in result
        assert "More doc" not in result
        compile(result, "<quixbugs>", "exec")

    def test_preserves_code_without_docstring(self) -> None:
        from evals.importers.quixbugs_importer import _strip_docstring_tail

        code = "def f():\n    return 1\n"
        result = _strip_docstring_tail(code)
        assert result == code

    def test_preserves_inner_docstring_when_stripping_trailing_docstring(self) -> None:
        from evals.importers.quixbugs_importer import _strip_docstring_tail

        code = 'def f():\n    """Doc inside"""\n    return 1\n\n"""Trailing doc"""\n'
        result = _strip_docstring_tail(code)
        assert result == 'def f():\n    """Doc inside"""\n    return 1\n'
        compile(result, "<quixbugs>", "exec")


class TestQuixBugsLoadJsonTestcases:
    def test_loads_standard_format(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import _load_json_testcases

        json_file = tmp_path / "test.json"
        json_file.write_text("[[3, 5], 1]\n[[10], 2]\n")
        cases = _load_json_testcases(json_file)
        assert len(cases) == 2
        assert cases[0] == ([3, 5], 1)
        assert cases[1] == ([10], 2)

    def test_wraps_scalar_input(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import _load_json_testcases

        json_file = tmp_path / "test.json"
        json_file.write_text("[42, 7]\n")
        cases = _load_json_testcases(json_file)
        assert cases[0] == ([42], 7)


class TestQuixBugsImportIntegration:
    """Integration test using a minimal synthetic QuixBugs-like repo."""

    def _build_fake_repo(self, tmp_path):
        """Create a minimal QuixBugs-like directory structure."""
        repo = tmp_path / "quixbugs"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)

        # Simple non-graph program.
        (repo / "python_programs" / "gcd.py").write_text(
            'def gcd(a, b):\n    if b:\n        return gcd(a % b, b)\n    return a\n\n"""\nGCD\n"""\n'
        )
        (repo / "correct_python_programs" / "gcd.py").write_text(
            "def gcd(a, b):\n    if b:\n        return gcd(b, a % b)\n    return a\n"
        )
        (repo / "json_testcases" / "gcd.json").write_text(
            "[[17, 0], 17]\n[[13, 13], 13]\n[[37, 600], 1]\n"
        )
        return repo

    def test_dry_run_discovers_cases(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_fake_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo, dry_run=True)
        assert len(results) == 1
        assert results[0]["case_id"] == "qb_gcd_easy"
        assert results[0]["metadata"]["category"] == "logic_error"
        assert results[0]["metadata"]["bug_source"] == "quixbugs"
        assert results[0]["metadata"]["difficulty"] == "easy"

    def test_full_import_writes_files(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_fake_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo, validate=False)
        assert len(results) == 1

        case_dir = output / "qb_gcd_easy"
        assert (case_dir / "buggy.py").exists()
        assert (case_dir / "test_buggy.py").exists()
        assert (case_dir / "oracle.patch").exists()
        assert (case_dir / "case.json").exists()

        meta = json.loads((case_dir / "case.json").read_text())
        assert meta["bug_source"] == "quixbugs"
        assert meta["contamination_risk"] == "high"

    def test_missing_repo_raises(self, tmp_path) -> None:
        import pytest

        from evals.importers.quixbugs_importer import import_quixbugs

        with pytest.raises(FileNotFoundError, match="QuixBugs"):
            import_quixbugs(tmp_path / "out", repo_path=tmp_path / "nonexistent")


class TestQuixBugsIntegrationGraphProgram:
    """End-to-end test for graph-based programs (Node class + pytest rewrite)."""

    def _build_graph_repo(self, tmp_path):
        """Create a QuixBugs-like repo with a graph-based program."""
        repo = tmp_path / "quixbugs"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)
        (repo / "python_testcases").mkdir(parents=True)
        # Provide node.py (should be skipped from import).
        (repo / "python_programs" / "node.py").write_text("class Node:\n    pass\n")
        # Buggy: reverse_linked_list with wrong successor assignment.
        (repo / "python_programs" / "reverse_linked_list.py").write_text(
            "def reverse_linked_list(node):\n"
            "    prevnode = None\n"
            "    while node:\n"
            "        nextnode = node.successor\n"
            "        node.successor = prevnode\n"
            "        node = nextnode\n"
            "    return prevnode\n"
        )
        # Correct: adds prevnode = node.
        (repo / "correct_python_programs" / "reverse_linked_list.py").write_text(
            "def reverse_linked_list(node):\n"
            "    prevnode = None\n"
            "    while node:\n"
            "        nextnode = node.successor\n"
            "        node.successor = prevnode\n"
            "        prevnode = node\n"
            "        node = nextnode\n"
            "    return prevnode\n"
        )
        # Graph test file.
        (repo / "python_testcases" / "test_reverse_linked_list.py").write_text(
            "import pytest\n"
            "from node import Node\n"
            "from python_programs.reverse_linked_list import reverse_linked_list\n"
            "\n"
            "def test_single():\n"
            "    n = Node(1)\n"
            "    result = reverse_linked_list(n)\n"
            "    assert result.value == 1\n"
        )
        return repo

    def test_graph_program_full_import(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_graph_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo, validate=False)
        assert len(results) == 1
        assert results[0]["case_id"] == "qb_reverse_linked_list_easy"
        assert "graph_algorithm" in results[0]["metadata"]["tags"]

    def test_graph_program_embeds_node_class(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_graph_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        import_quixbugs(output, repo_path=repo, validate=False)

        buggy_code = (output / "qb_reverse_linked_list_easy" / "buggy.py").read_text()
        assert "class Node:" in buggy_code
        assert "def reverse_linked_list" in buggy_code

    def test_graph_program_rewrites_test_imports(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_graph_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        import_quixbugs(output, repo_path=repo, validate=False)

        test_code = (output / "qb_reverse_linked_list_easy" / "test_buggy.py").read_text()
        assert "from buggy import" in test_code
        assert "from python_programs" not in test_code
        assert "from node import" not in test_code
        assert "import pytest" not in test_code

    def test_graph_program_oracle_patch_valid(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_graph_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        import_quixbugs(output, repo_path=repo, validate=False)

        diff_patch = (output / "qb_reverse_linked_list_easy" / "oracle.patch").read_text()
        assert "--- a/buggy.py" in diff_patch
        assert "+++ b/buggy.py" in diff_patch
        assert "prevnode = node" in diff_patch

    def test_node_py_is_not_imported_as_case(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_graph_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo)
        case_ids = [r["case_id"] for r in results]
        assert not any(cid.split("_")[1] == "node" for cid in case_ids)


class TestQuixBugsIntegrationMixedRepo:
    """End-to-end tests with a repo containing both graph and non-graph programs."""

    def _build_mixed_repo(self, tmp_path):
        """Create a repo with 2 non-graph and 1 graph program."""
        repo = tmp_path / "quixbugs"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)
        (repo / "python_testcases").mkdir(parents=True)

        # Non-graph 1: gcd.
        (repo / "python_programs" / "gcd.py").write_text(
            "def gcd(a, b):\n    if b:\n        return gcd(a % b, b)\n    return a\n"
        )
        (repo / "correct_python_programs" / "gcd.py").write_text(
            "def gcd(a, b):\n    if b:\n        return gcd(b, a % b)\n    return a\n"
        )
        (repo / "json_testcases" / "gcd.json").write_text("[[17, 0], 17]\n[[13, 13], 13]\n")

        # Non-graph 2: bitcount.
        (repo / "python_programs" / "bitcount.py").write_text(
            "def bitcount(n):\n    count = 0\n    while n:\n        n ^= n - 1\n        count += 1\n    return count\n"
        )
        (repo / "correct_python_programs" / "bitcount.py").write_text(
            "def bitcount(n):\n    count = 0\n    while n:\n        n &= n - 1\n        count += 1\n    return count\n"
        )
        (repo / "json_testcases" / "bitcount.json").write_text("[[127], 7]\n[[0], 0]\n")

        # Graph program: detect_cycle (minimal).
        (repo / "python_programs" / "detect_cycle.py").write_text(
            "def detect_cycle(node):\n    hare = tortoise = node\n    while True:\n"
            "        if hare.successor is None:\n            return False\n"
            "        tortoise = tortoise.successor\n"
            "        hare = hare.successor\n"
            "        if hare is tortoise:\n            return True\n"
        )
        (repo / "correct_python_programs" / "detect_cycle.py").write_text(
            "def detect_cycle(node):\n    hare = tortoise = node\n    while True:\n"
            "        if hare is None or hare.successor is None:\n            return False\n"
            "        tortoise = tortoise.successor\n"
            "        hare = hare.successor.successor\n"
            "        if hare is tortoise:\n            return True\n"
        )
        (repo / "python_testcases" / "test_detect_cycle.py").write_text(
            "from node import Node\n"
            "from python_programs.detect_cycle import detect_cycle\n"
            "\n"
            "def test_no_cycle():\n"
            "    a = Node(1, successor=Node(2))\n"
            "    assert detect_cycle(a) is False\n"
        )

        return repo

    def test_imports_both_graph_and_nongraph(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_mixed_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo, validate=False)
        case_ids = sorted(r["case_id"] for r in results)
        assert "qb_bitcount_easy" in case_ids
        assert "qb_detect_cycle_easy" in case_ids
        assert "qb_gcd_easy" in case_ids
        assert len(results) == 3

    def test_graph_case_has_node_class_nongraph_does_not(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_mixed_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        import_quixbugs(output, repo_path=repo, validate=False)

        gcd_buggy = (output / "qb_gcd_easy" / "buggy.py").read_text()
        detect_buggy = (output / "qb_detect_cycle_easy" / "buggy.py").read_text()
        assert "class Node:" not in gcd_buggy
        assert "class Node:" in detect_buggy

    def test_max_cases_limits_output(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_mixed_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo, max_cases=2, validate=False)
        assert len(results) == 2

    def test_dry_run_writes_no_files(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_mixed_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo, dry_run=True)
        assert len(results) == 3
        # No case directories should be created.
        assert list(output.iterdir()) == []

    def test_all_metadata_fields_present(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_mixed_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo, validate=False)
        required_keys = {
            "description",
            "category",
            "difficulty",
            "snapshot_dependency",
            "expected_exception",
            "tags",
            "python_args",
            "timeout_sec",
            "bug_source",
            "contamination_risk",
        }
        for r in results:
            assert required_keys <= set(r["metadata"]), (
                f"{r['case_id']} missing metadata keys: {required_keys - set(r['metadata'])}"
            )


class TestQuixBugsIntegrationEdgeCases:
    """End-to-end tests for skipping, dedup, and error paths."""

    def _build_basic_repo(self, tmp_path, *, extra_programs=None):
        """Create repo with gcd + optional extras."""
        repo = tmp_path / "quixbugs"
        (repo / "python_programs").mkdir(parents=True)
        (repo / "correct_python_programs").mkdir(parents=True)
        (repo / "json_testcases").mkdir(parents=True)

        (repo / "python_programs" / "gcd.py").write_text(
            "def gcd(a, b):\n    if b:\n        return gcd(a % b, b)\n    return a\n"
        )
        (repo / "correct_python_programs" / "gcd.py").write_text(
            "def gcd(a, b):\n    if b:\n        return gcd(b, a % b)\n    return a\n"
        )
        (repo / "json_testcases" / "gcd.json").write_text("[[17, 0], 17]\n")

        if extra_programs:
            for name, buggy, correct, tests_json in extra_programs:
                (repo / "python_programs" / f"{name}.py").write_text(buggy)
                (repo / "correct_python_programs" / f"{name}.py").write_text(correct)
                if tests_json is not None:
                    (repo / "json_testcases" / f"{name}.json").write_text(tests_json)

        return repo

    def test_skips_program_with_external_deps(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_basic_repo(
            tmp_path,
            extra_programs=[
                (
                    "bad_prog",
                    "import numpy\ndef bad_prog(x):\n    return numpy.sum(x)\n",
                    "import numpy\ndef bad_prog(x):\n    return numpy.sum(x) + 1\n",
                    "[[1], 2]\n",
                ),
            ],
        )
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo)
        case_ids = [r["case_id"] for r in results]
        assert "qb_bad_prog_easy" not in case_ids
        assert "qb_gcd_easy" in case_ids

    def test_skips_program_without_correct_version(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_basic_repo(tmp_path)
        # Add buggy-only program (no correct version).
        (repo / "python_programs" / "orphan.py").write_text("def orphan(x):\n    return x + 1\n")
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo)
        case_ids = [r["case_id"] for r in results]
        assert "qb_orphan_easy" not in case_ids
        assert "qb_gcd_easy" in case_ids

    def test_skips_program_without_test_cases(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_basic_repo(
            tmp_path,
            extra_programs=[
                (
                    "no_tests",
                    "def no_tests(x):\n    return x + 1\n",
                    "def no_tests(x):\n    return x - 1\n",
                    None,  # No JSON test file.
                ),
            ],
        )
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo)
        case_ids = [r["case_id"] for r in results]
        assert "qb_no_tests_easy" not in case_ids

    def test_skips_empty_buggy_file(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_basic_repo(
            tmp_path,
            extra_programs=[
                ("empty", "", "def empty(x):\n    return x\n", "[[1], 1]\n"),
            ],
        )
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo)
        case_ids = [r["case_id"] for r in results]
        assert "qb_empty_easy" not in case_ids

    def test_skips_identical_buggy_and_fixed(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        same = "def same(x):\n    return x\n"
        repo = self._build_basic_repo(
            tmp_path,
            extra_programs=[("same", same, same, "[[1], 1]\n")],
        )
        output = tmp_path / "output"
        output.mkdir()

        results = import_quixbugs(output, repo_path=repo)
        case_ids = [r["case_id"] for r in results]
        assert "qb_same_easy" not in case_ids

    def test_skips_existing_case_directory(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_basic_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        # First import succeeds.
        results1 = import_quixbugs(output, repo_path=repo)
        assert len(results1) == 1

        # Second import skips because case dir already exists.
        results2 = import_quixbugs(output, repo_path=repo)
        assert len(results2) == 0

    def test_case_json_is_valid_json(self, tmp_path) -> None:
        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_basic_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        import_quixbugs(output, repo_path=repo, validate=False)

        case_json = json.loads((output / "qb_gcd_easy" / "case.json").read_text())
        assert case_json["category"] == "logic_error"
        assert case_json["difficulty"] == "easy"
        assert case_json["bug_source"] == "quixbugs"
        assert case_json["contamination_risk"] == "high"
        assert case_json["timeout_sec"] == 30
        assert case_json["python_args"] == ["-m", "pytest", "-q"]

    def test_generated_test_runs_against_fixed_code(self, tmp_path) -> None:
        """Verify that the generated test actually passes against the fixed code."""
        import subprocess

        from evals.importers.quixbugs_importer import import_quixbugs

        repo = self._build_basic_repo(tmp_path)
        output = tmp_path / "output"
        output.mkdir()

        import_quixbugs(output, repo_path=repo, validate=False)

        case_dir = output / "qb_gcd_easy"
        oracle_patch = (case_dir / "oracle.patch").read_text()

        # Apply the oracle patch to get the fixed version.
        fixed_dir = tmp_path / "fixed_run"
        fixed_dir.mkdir()

        import shutil

        shutil.copy(case_dir / "test_buggy.py", fixed_dir / "test_buggy.py")

        # Read the buggy code and apply the patch manually by reading
        # the correct version from the repo (equivalent to applying oracle).
        correct_code = (repo / "correct_python_programs" / "gcd.py").read_text()
        (fixed_dir / "buggy.py").write_text(correct_code)

        # Ensure the oracle patch is non-empty.
        assert oracle_patch.strip()
        assert "--- a/buggy.py" in oracle_patch

        # Run the test against the fixed code.
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", str(fixed_dir / "test_buggy.py")],
            capture_output=True,
            text=True,
            cwd=str(fixed_dir),
            timeout=15,
        )
        assert result.returncode == 0, (
            f"Test failed against fixed code:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_generated_test_fails_against_buggy_code(self, tmp_path) -> None:
        """Verify that the generated test detects the bug in the buggy code."""
        import subprocess

        from evals.importers.quixbugs_importer import import_quixbugs

        # Use test cases that expose the gcd bug (a % b, b) vs (b, a % b).
        # gcd(13, 13) triggers infinite recursion in the buggy version.
        repo = self._build_basic_repo(tmp_path)
        (repo / "json_testcases" / "gcd.json").write_text("[[13, 13], 13]\n[[37, 600], 1]\n")
        output = tmp_path / "output"
        output.mkdir()

        import_quixbugs(output, repo_path=repo, validate=False)

        case_dir = output / "qb_gcd_easy"

        # Run the test against the buggy code (should fail).
        result = subprocess.run(
            ["python", "-m", "pytest", "-q", str(case_dir / "test_buggy.py")],
            capture_output=True,
            text=True,
            cwd=str(case_dir),
            timeout=15,
        )
        assert result.returncode != 0, "Test should fail against buggy code but passed"


class TestQuixBugsCategoryMapping:
    def test_all_programs_map_to_logic_error(self) -> None:
        from evals.category_mapping import map_quixbugs_type

        assert map_quixbugs_type("bitcount") == "logic_error"
        assert map_quixbugs_type("quicksort") == "logic_error"
        assert map_quixbugs_type("breadth_first_search") == "logic_error"


# ---------------------------------------------------------------------------
# Additional coverage for evals/importers/base.py — missing lines/branches
# ---------------------------------------------------------------------------


def _make_valid_case(
    case_dir: Path,
    *,
    meta_overrides: dict[str, Any] | None = None,
    oracle_patch: str = "diff --git a/b.py b/b.py\n- return 1\n+ return 2\n",
) -> Path:
    """Helper: write a fully-valid case, optionally overriding metadata."""
    meta: dict[str, Any] = {
        "category": "logic_error",
        "difficulty": "easy",
        "snapshot_dependency": "none",
        "expected_exception": "AssertionError",
    }
    if meta_overrides:
        meta.update(meta_overrides)
    write_case_files(
        case_dir,
        buggy_code="def f():\n    return 1\n",
        test_code="def test_f():\n    assert True\n",
        oracle_patch=oracle_patch,
        metadata=meta,
    )
    return case_dir


class TestGenerateOraclePatchEdgeCases:
    """Cover empty inputs, identical code, and large files (lines 41-42)."""

    def test_empty_buggy_and_fixed(self) -> None:
        """Both empty strings → no diff."""
        assert generate_oracle_patch("", "") == ""

    def test_empty_buggy_nonempty_fixed(self) -> None:
        """Entire file is an addition."""
        diff_patch = generate_oracle_patch("", "x = 1\n")
        assert "diff --git" in diff_patch
        assert "+x = 1" in diff_patch

    def test_nonempty_buggy_empty_fixed(self) -> None:
        """Entire file is a deletion."""
        diff_patch = generate_oracle_patch("x = 1\n", "")
        assert "diff --git" in diff_patch
        assert "-x = 1" in diff_patch

    def test_identical_large_files(self) -> None:
        """Large identical files still return empty string."""
        code = "".join(f"line_{i} = {i}\n" for i in range(2000))
        assert generate_oracle_patch(code, code) == ""

    def test_large_diff(self) -> None:
        """Large change produces a valid patch."""
        buggy = "".join(f"a_{i} = {i}\n" for i in range(500))
        fixed = "".join(f"b_{i} = {i}\n" for i in range(500))
        diff_patch = generate_oracle_patch(buggy, fixed)
        assert "diff --git" in diff_patch
        assert len(diff_patch) > 0


class TestValidateGeneratedCaseInvalidJson:
    """Lines 90-91: invalid case.json (JSONDecodeError / OSError)."""

    def test_malformed_json(self, tmp_path: Path) -> None:
        """case.json containing invalid JSON triggers JSONDecodeError."""
        case_dir = _make_valid_case(tmp_path / "bad_json")
        (case_dir / "case.json").write_text("{not valid json", encoding="utf-8")
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "invalid case.json" in msg

    def test_case_json_not_a_dict(self, tmp_path: Path) -> None:
        """Line 94: case.json that is a JSON array, not an object."""
        case_dir = _make_valid_case(tmp_path / "list_json")
        (case_dir / "case.json").write_text("[1, 2, 3]\n", encoding="utf-8")
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "not a JSON object" in msg


class TestValidateGeneratedCaseInvalidMetadata:
    """Lines 104, 108, 112: invalid difficulty / snapshot_dependency / expected_exception."""

    def test_invalid_difficulty(self, tmp_path: Path) -> None:
        """Line 104: difficulty not in VALID_DIFFICULTIES."""
        case_dir = _make_valid_case(
            tmp_path / "bad_diff",
            meta_overrides={"difficulty": "impossible"},
        )
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "invalid difficulty" in msg

    def test_missing_difficulty(self, tmp_path: Path) -> None:
        """difficulty key absent → None not in VALID_DIFFICULTIES."""
        case_dir = _make_valid_case(tmp_path / "no_diff")
        meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        del meta["difficulty"]
        (case_dir / "case.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "invalid difficulty" in msg

    def test_invalid_snapshot_dependency(self, tmp_path: Path) -> None:
        """Line 108: snapshot_dependency not in VALID_SNAPSHOT_DEPENDENCIES."""
        case_dir = _make_valid_case(
            tmp_path / "bad_snap",
            meta_overrides={"snapshot_dependency": "always"},
        )
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "invalid snapshot_dependency" in msg

    def test_missing_snapshot_dependency(self, tmp_path: Path) -> None:
        """snapshot_dependency key absent → None not in set."""
        case_dir = _make_valid_case(tmp_path / "no_snap")
        meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        del meta["snapshot_dependency"]
        (case_dir / "case.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "invalid snapshot_dependency" in msg

    def test_missing_expected_exception(self, tmp_path: Path) -> None:
        """Line 112: expected_exception absent → not isinstance(str)."""
        case_dir = _make_valid_case(tmp_path / "no_exc")
        meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        del meta["expected_exception"]
        (case_dir / "case.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "missing or empty expected_exception" in msg

    def test_empty_expected_exception(self, tmp_path: Path) -> None:
        """Line 112: expected_exception is empty string."""
        case_dir = _make_valid_case(
            tmp_path / "empty_exc",
            meta_overrides={"expected_exception": ""},
        )
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "missing or empty expected_exception" in msg

    def test_non_string_expected_exception(self, tmp_path: Path) -> None:
        """expected_exception is an int, not a string."""
        case_dir = _make_valid_case(
            tmp_path / "int_exc",
            meta_overrides={"expected_exception": 42},
        )
        ok, msg = validate_generated_case(case_dir, schema_only=True)
        assert not ok
        assert "missing or empty expected_exception" in msg


class TestValidateGeneratedCaseFullPath:
    """Lines 121, 123-125: schema_only=False delegates to validate_one_case."""

    def test_schema_only_false_calls_validate_one_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full validation delegates to validate_one_case and returns its result."""
        case_dir = _make_valid_case(
            tmp_path / "full",
            meta_overrides={
                "python_args": ["-m", "pytest", "-x"],
                "timeout_sec": 60,
            },
        )

        def fake_validate(
            case_dir: Path,
            expected_exception: str,
            python_args: list[str],
            timeout_sec: int,
            fmt: str,
        ) -> tuple[bool, str]:
            return True, "oracle fixes the bug"

        monkeypatch.setattr("evals.importers.base.validate_one_case", fake_validate, raising=False)
        # We must also ensure the import inside the function resolves:
        import evals.validate_cases

        monkeypatch.setattr(evals.validate_cases, "validate_one_case", fake_validate)

        ok, msg = validate_generated_case(case_dir, schema_only=False)
        assert ok
        assert "oracle fixes the bug" in msg

    def test_schema_only_false_returns_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full validation can also return failure from validate_one_case."""
        case_dir = _make_valid_case(tmp_path / "full_fail")

        def fake_validate(*args: Any, **kwargs: Any) -> tuple[bool, str]:
            return False, "baseline did not fail"

        import evals.validate_cases

        monkeypatch.setattr(evals.validate_cases, "validate_one_case", fake_validate)

        ok, msg = validate_generated_case(case_dir, schema_only=False)
        assert not ok
        assert "baseline did not fail" in msg

    def test_schema_only_false_uses_default_python_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lines 123-124: default python_args and timeout_sec are used."""
        case_dir = _make_valid_case(tmp_path / "defaults")
        captured: dict[str, Any] = {}

        def fake_validate(
            case_dir: Path,
            expected_exception: str,
            python_args: list[str],
            timeout_sec: int,
            fmt: str,
        ) -> tuple[bool, str]:
            captured["python_args"] = python_args
            captured["timeout_sec"] = timeout_sec
            return True, "ok"

        import evals.validate_cases

        monkeypatch.setattr(evals.validate_cases, "validate_one_case", fake_validate)

        validate_generated_case(case_dir, schema_only=False)
        assert captured["python_args"] == ["-m", "pytest", "-q"]
        assert captured["timeout_sec"] == 120


class TestExtractImportRootsFromAst:
    """Cover AST-based import extraction: relative imports, empty roots."""

    def test_relative_import_skipped(self) -> None:
        """Line 167: relative imports (level > 0) are skipped."""
        code = "from . import sibling\nfrom ..parent import thing\n"
        roots = _extract_import_roots_from_ast(code)
        assert roots == set()

    def test_import_from_with_no_module(self) -> None:
        """Branch [168,157]: `from . import x` has node.module = None."""
        code = "from . import foo\n"
        roots = _extract_import_roots_from_ast(code)
        assert roots == set()

    def test_stdlib_import(self) -> None:
        """Normal import produces the top-level root."""
        code = "import os.path\nfrom collections.abc import Mapping\n"
        roots = _extract_import_roots_from_ast(code)
        assert roots == {"os", "collections"}

    def test_mixed_imports(self) -> None:
        """Mix of absolute, relative, and dotted imports."""
        code = "import sys\nfrom . import local\nfrom os.path import join\nimport numpy.random\n"
        roots = _extract_import_roots_from_ast(code)
        assert "sys" in roots
        assert "os" in roots
        assert "numpy" in roots
        # relative import excluded
        assert "local" not in roots


class TestExtractImportRootsFallback:
    """Lines 180, 186, 192-197: fallback regex-based import parser."""

    def test_empty_lines_skipped(self) -> None:
        """Line 180: blank lines are skipped via continue."""
        roots = _extract_import_roots_fallback("import os\n\n\nimport sys\n")
        assert roots == {"os", "sys"}

    def test_from_import_standard(self) -> None:
        """Lines 191-197: `from X import Y` extracts root X."""
        roots = _extract_import_roots_fallback("from collections import deque\n")
        assert roots == {"collections"}

    def test_from_relative_import_skipped(self) -> None:
        """Branch [194,177]: `from .foo import bar` — module starts with '.'."""
        roots = _extract_import_roots_fallback("from .local import helper\n")
        assert roots == set()

    def test_from_dotted_import(self) -> None:
        """`from os.path import join` → root is 'os'."""
        roots = _extract_import_roots_fallback("from os.path import join\n")
        assert roots == {"os"}

    def test_import_with_comma_separated_modules(self) -> None:
        """Lines 183-189: `import a, b, c` extracts all roots."""
        roots = _extract_import_roots_fallback("import os, sys, json\n")
        assert roots == {"os", "sys", "json"}

    def test_import_with_alias(self) -> None:
        """Lines 184-186: `import numpy as np` — token splitting."""
        roots = _extract_import_roots_fallback("import numpy as np\n")
        assert roots == {"numpy"}

    def test_non_import_lines_ignored(self) -> None:
        """Non-import lines should be ignored."""
        code = "x = 1\nprint('hello')\nimport math\n"
        roots = _extract_import_roots_fallback(code)
        assert roots == {"math"}

    def test_empty_code(self) -> None:
        """Empty string returns empty set."""
        assert _extract_import_roots_fallback("") == set()

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("import ,,,\n", set()),  # degenerate commas → empty tokens
            ("from  import x\n", set()),  # empty module name
        ],
        ids=["degenerate-commas", "empty-module"],
    )
    def test_degenerate_fallback_inputs(self, code: str, expected: set[str]) -> None:
        """Edge cases for fallback parser: empty tokens, blank modules."""
        roots = _extract_import_roots_fallback(code)
        assert roots == expected


class TestHasExternalDepsExtended:
    """Lines 209, 215: allowed_externals param, syntax-error-no-imports."""

    def test_allowed_externals_permits_known_package(self) -> None:
        """Line 209: allowed_externals extends the allowlist."""
        code = "import numpy\n"
        # Without allowlist → external detected.
        assert has_external_deps(code) is True
        # With allowlist → permitted.
        assert has_external_deps(code, allowed_externals={"numpy"}) is False

    def test_allowed_externals_strips_whitespace(self) -> None:
        """Whitespace in allowed_externals is stripped."""
        code = "import pandas\n"
        assert has_external_deps(code, allowed_externals={" pandas "}) is False

    def test_allowed_externals_ignores_empty_strings(self) -> None:
        """Empty strings in allowed_externals don't pollute the allowlist."""
        code = "import numpy\n"
        assert has_external_deps(code, allowed_externals={"", "  "}) is True

    def test_syntax_error_no_imports_returns_false(self) -> None:
        """Line 215: SyntaxError code with no import lines → False."""
        code = "x = (\n"  # SyntaxError, no imports
        assert has_external_deps(code) is False

    def test_syntax_error_with_stdlib_import_returns_false(self) -> None:
        """Syntax error + stdlib import only → no external deps."""
        code = "import os\nx = (\n"
        assert has_external_deps(code) is False

    def test_future_import_never_external(self) -> None:
        """__future__ imports are always in the allow set."""
        code = "from __future__ import annotations\n"
        assert has_external_deps(code) is False

    def test_empty_code(self) -> None:
        """Empty code has no external deps."""
        assert has_external_deps("") is False


class TestIsDeterministic:
    """Lines 223-225: heuristic determinism check."""

    @pytest.mark.parametrize(
        "code",
        [
            "x = 1\ny = x + 2\n",
            "import math\nresult = math.sqrt(4)\n",
            "def f():\n    return [1, 2, 3]\n",
        ],
        ids=["arithmetic", "math-sqrt", "list-return"],
    )
    def test_deterministic_code(self, code: str) -> None:
        """Safe code is considered deterministic."""
        assert is_deterministic(code) is True

    @pytest.mark.parametrize(
        ("code", "reason"),
        [
            ("import requests\nrequests.get('http://x')\n", "requests."),
            ("import urllib\nurllib.request.urlopen('x')\n", "urllib."),
            ("from http.client import HTTPConnection\n", "http."),
            ("import socket\nsocket.create_connection()\n", "socket."),
            ("import time\ntime.sleep(1)\n", "time.sleep"),
            ("import random\nx = random.random()\n", "random.random"),
        ],
        ids=["requests", "urllib", "http", "socket", "time-sleep", "random"],
    )
    def test_nondeterministic_code(self, code: str, reason: str) -> None:
        """Risky patterns are flagged as non-deterministic."""
        assert is_deterministic(code) is False


class TestRuffFormat:
    """Lines 240-242: ruff_format failure paths."""

    def test_returns_formatted_code_on_success(self) -> None:
        """Happy path: the ruff installed with this interpreter formats the code."""
        assert ruff_format("x=1\n") == "x = 1\n"

    def test_formatting_ignores_the_working_directory_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DAT-15: a pyproject.toml in the cwd used to change imported case content."""
        (tmp_path / "pyproject.toml").write_text(
            "[tool.ruff]\nline-length = 20\n[tool.ruff.format]\nquote-style = 'single'\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        code = 'value = call_something("a", "b", "c")\n'
        assert ruff_format(code) == code

    def test_raises_when_ruff_cannot_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DAT-15: a missing ruff used to return unformatted code silently."""

        def fake_run(*args: Any, **kwargs: Any) -> None:
            raise FileNotFoundError("python not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuffFormatError):
            ruff_format("x=1\n")

    def test_raises_when_ruff_module_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["python"], returncode=1, stdout="", stderr="No module named ruff"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuffFormatError, match="not installed"):
            ruff_format("x=1\n")

    def test_raises_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="ruff", timeout=30)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuffFormatError):
            ruff_format("y = 2\n")

    def test_unparsable_code_is_returned_unchanged(self) -> None:
        code = "def broken(:\n    pass\n"
        assert ruff_format(code) == code

    def test_returns_original_on_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch [238,242]: ruff exits non-zero → return original."""

        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["ruff"], returncode=1, stdout="", stderr="error"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        code = "z = 3\n"
        assert ruff_format(code) == code

    def test_returns_original_on_empty_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Branch [238,242]: ruff returns 0 but empty stdout → return original."""

        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=["ruff"], returncode=0, stdout="   ", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        code = "w = 4\n"
        assert ruff_format(code) == code


class TestWriteCaseFilesEdgeCases:
    """Additional write_case_files edge-case coverage."""

    def test_unicode_content(self, tmp_path: Path) -> None:
        """UTF-8 content is written correctly."""
        case_dir = tmp_path / "unicode_case"
        write_case_files(
            case_dir,
            buggy_code="# héllo wörld\nx = '日本語'\n",
            test_code="# tëst\nassert True\n",
            oracle_patch="# pätch\n",
            metadata={"description": "ñ café"},
        )
        buggy = (case_dir / "buggy.py").read_text(encoding="utf-8")
        assert "héllo" in buggy
        assert "日本語" in buggy

    def test_overwrites_existing_files(self, tmp_path: Path) -> None:
        """Re-calling write_case_files overwrites previous content."""
        case_dir = tmp_path / "overwrite"
        write_case_files(
            case_dir,
            buggy_code="old",
            test_code="old",
            oracle_patch="old",
            metadata={"v": 1},
        )
        write_case_files(
            case_dir,
            buggy_code="new",
            test_code="new",
            oracle_patch="new",
            metadata={"v": 2},
        )
        assert (case_dir / "buggy.py").read_text() == "new"
        meta = json.loads((case_dir / "case.json").read_text())
        assert meta["v"] == 2


@pytest.mark.parametrize(
    ("code", "reasons"),
    [
        ("import time\nprint(time.time())\n", ["clock"]),
        ("from datetime import datetime\nprint(datetime.now())\n", ["clock"]),
        ("import random\nprint(random.randint(1, 3))\n", ["unseeded_random"]),
        ("from random import choice, shuffle, sample\nshuffle([1])\n", ["unseeded_random"]),
        ("import random\nrandom.seed(0)\nprint(random.choice([1, 2]))\n", []),
        ("import uuid\nprint(uuid.uuid4())\n", ["entropy"]),
        ("import os\nprint(os.urandom(4))\n", ["entropy"]),
        ("import secrets\nprint(secrets.token_hex())\n", ["entropy"]),
        ("for x in set('ab'):\n    print(x)\n", ["set_iteration"]),
        ("print([x for x in {'a', 'b'}])\n", ["set_iteration"]),
        ("print(sorted(set('ab')))\n", []),
        ("import numpy as np\nprint(np.random.rand(3))\n", ["unseeded_random"]),
        ("import numpy as np\nrng = np.random.default_rng(0)\nprint(rng.random())\n", []),
        ("import numpy as np\nrng = np.random.default_rng()\n", ["unseeded_random"]),
    ],
    ids=[
        "time-time",
        "datetime-now",
        "randint",
        "shuffle-from-import",
        "seeded-random",
        "uuid4",
        "urandom",
        "secrets",
        "set-loop",
        "set-comprehension",
        "sorted-set",
        "numpy-random",
        "numpy-seeded-generator",
        "numpy-unseeded-generator",
    ],
)
def test_dat14_nondeterminism_reasons(code: str, reasons: list[str]) -> None:
    """DAT-14: the substring filter let all of these through."""
    from evals.importers.base import nondeterminism_reasons

    assert nondeterminism_reasons(code) == reasons


def test_dat14_scan_cases_reports_flagged_files(tmp_path: Path, capsys) -> None:
    from evals.importers import determinism

    for case_id, buggy in (("clean", "x = 1\n"), ("flaky", "import time\nt = time.time()\n")):
        case_dir = tmp_path / case_id
        case_dir.mkdir()
        (case_dir / "case.json").write_text("{}", encoding="utf-8")
        (case_dir / "buggy.py").write_text(buggy, encoding="utf-8")
        (case_dir / "test_buggy.py").write_text("def test():\n    pass\n", encoding="utf-8")

    assert determinism.scan_cases(tmp_path) == {"flaky": ["buggy.py:clock"]}
    assert determinism.main([str(tmp_path)]) == 1
    assert "flaky: buggy.py:clock" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("buggy", "fixed"),
    [
        ("def f():\n    return 1", "def f():\n    return 2"),  # neither ends in a newline
        ("x = 1\n", "x = 2"),
        ("x = 1", "x = 1\ny = 2\n"),
    ],
    ids=["no-final-newline", "newline-removed", "newline-added"],
)
def test_dat10_oracle_patch_without_trailing_newline_applies(
    tmp_path: Path, buggy: str, fixed: str
) -> None:
    """DAT-10: the last removed and added lines used to merge into one line."""
    from evals.eval_patches import apply_diff

    (tmp_path / "buggy.py").write_bytes(buggy.encode())
    patch = generate_oracle_patch(buggy, fixed)

    ok, message = apply_diff(tmp_path, patch)
    assert ok, message
    assert message == ""
    assert (tmp_path / "buggy.py").read_bytes() == fixed.encode()


def test_dat10_count_patch_lines_counts_the_no_newline_case() -> None:
    patch = generate_oracle_patch("def f():\n    return 1", "def f():\n    return 2")
    assert count_patch_lines(patch) == 2


_IMPORTERS = (
    "bugsinpy_importer",
    "condefects_importer",
    "debugbench_importer",
    "humanevalpack_importer",
    "pybughive_importer",
    "pyresbugs_importer",
    "quixbugs_importer",
    "trickybugs_importer",
)


@pytest.mark.parametrize("module_name", _IMPORTERS)
def test_dat05_importers_validate_by_execution_when_asked(module_name: str) -> None:
    """DAT-05: seven importers hardcoded schema_only=True, so --validate never ran a case."""
    import ast

    source = (Path("evals/importers") / f"{module_name}.py").read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_generated_case"
    ]
    assert calls, module_name
    for call in calls:
        schema_only = next(kw.value for kw in call.keywords if kw.arg == "schema_only")
        assert ast.unparse(schema_only) == "not validate", module_name


@pytest.mark.parametrize(("validate", "schema_only"), [(False, True), (True, False)])
def test_dat05_quixbugs_schema_always_checked_execution_on_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validate: bool, schema_only: bool
) -> None:
    from evals.importers import quixbugs_importer

    seen: list[bool] = []

    def _spy(case_dir: Path, *, schema_only: bool = True) -> tuple[bool, str]:
        del case_dir
        seen.append(schema_only)
        return True, "ok"

    monkeypatch.setattr(quixbugs_importer, "validate_generated_case", _spy)
    repo = TestQuixBugsImportIntegration()._build_fake_repo(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    quixbugs_importer.import_quixbugs(output, repo_path=repo, validate=validate)

    assert seen == [schema_only]


def _native_case(tmp_path: Path, *, with_source: bool) -> Path:
    case_dir = tmp_path / "native_case"
    (case_dir / "src").mkdir(parents=True)
    if with_source:
        (case_dir / "src" / "demo.py").write_text("x = 1\n", encoding="utf-8")
    (case_dir / "oracle.patch").write_text(
        generate_oracle_patch("x = 1\n", "x = 2\n", filename="src/demo.py"), encoding="utf-8"
    )
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "category": "logic_error",
                "difficulty": "easy",
                "snapshot_dependency": "helpful",
                "expected_exception": "AssertionError",
                "repro_mode": "native_dataset_test",
            }
        ),
        encoding="utf-8",
    )
    return case_dir


def test_dat05_native_repro_case_passes_schema_without_buggy_py(tmp_path: Path) -> None:
    """The schema check required buggy.py, so --validate rejected every native case."""
    assert validate_generated_case(_native_case(tmp_path, with_source=True)) == (
        True,
        "schema valid",
    )


def test_dat05_native_repro_case_needs_the_oracle_target(tmp_path: Path) -> None:
    ok, message = validate_generated_case(_native_case(tmp_path, with_source=False))
    assert not ok
    assert message == "missing required file: src/demo.py"


_DEBUGBENCH_FIXED = "class Solution:\n    def solve(self, nums, flag):\n        return nums\n"


def test_dat20_debugbench_values_are_emitted_as_literals() -> None:
    """DAT-20: LeetCode JSON values become Python literals instead of raw source."""
    from evals.importers import debugbench_importer

    generated = debugbench_importer._generate_test(
        _DEBUGBENCH_FIXED,
        ["Input: nums = [1,2], flag = true\nOutput: null"],
        "logic error",
    )

    assert generated is not None
    assert "sol.solve([1, 2], True)" in generated
    assert "assert result == None" in generated
    compile(generated, "test_buggy.py", "exec")


@pytest.mark.parametrize(
    "example",
    [
        "Input: nums = __import__('os').system('echo pwned'), flag = 1\nOutput: 1",
        "Input: nums = [1], flag = 1\nOutput: open('/etc/passwd').read()",
        "Input: nums = (lambda: 1)(), flag = 1\nOutput: 1",
    ],
    ids=["code-in-input", "code-in-output", "lambda"],
)
def test_dat20_debugbench_rejects_values_that_are_not_literals(example: str) -> None:
    from evals.importers import debugbench_importer

    assert debugbench_importer._generate_test(_DEBUGBENCH_FIXED, [example], "logic error") is None


def test_dat20_condefects_embeds_stdin_and_expected_output_as_strings() -> None:
    """ConDefects already used repr(); pin it so dataset text stays data."""
    import ast

    from evals.importers import condefects_importer

    payload = "__import__('os').system('echo pwned')"
    generated = condefects_importer._generate_test([(payload, payload)])

    tree = ast.parse(generated)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "system" not in calls
    assert repr(payload) in generated


def test_both_validators_accept_a_native_repro_case(tmp_path: Path) -> None:
    """DAT-06: importers and validate_cases apply one file contract."""
    from evals.case_contract import validate_case_contract
    from evals.importers.base import validate_generated_case

    case_dir = tmp_path / "bip_demo"
    (case_dir / "src").mkdir(parents=True)
    (case_dir / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (case_dir / "oracle.patch").write_text(
        "--- a/src/demo.py\n+++ b/src/demo.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
        encoding="utf-8",
    )
    (case_dir / "case.json").write_text(
        '{"repro_mode": "native_dataset_test", "category": "logic_error"}', encoding="utf-8"
    )

    assert validate_case_contract(case_dir) == (True, "ok")
    files_ok, message = validate_generated_case(case_dir)
    assert "missing required file" not in message
    assert files_ok or "test_buggy.py" not in message
