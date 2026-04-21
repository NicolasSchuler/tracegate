"""Tests for evals.importers.quixbugs_importer — QuixBugs dataset importer.

Uses synthetic data only (no QuixBugs repo clone needed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.importers.quixbugs_importer import (
    GRAPH_BASED_PROGRAMS,
    MAX_CODE_LINES,
    MAX_FIX_LINES,
    _generate_json_test,
    _load_json_testcases,
    _prepend_node_class,
    _rewrite_graph_test,
    _strip_docstring_tail,
    import_quixbugs,
)

# ---------------------------------------------------------------------------
# _rewrite_graph_test
# ---------------------------------------------------------------------------


class TestRewriteGraphTest:
    def test_basic_rewrite(self) -> None:
        src = (
            "from python_programs.bfs import bfs\n"
            "from node import Node\n"
            "\n"
            "def test_bfs():\n"
            "    pass\n"
        )
        result = _rewrite_graph_test(src, "bfs")
        assert result is not None
        assert "from buggy import bfs" in result
        assert "from buggy import Node" in result
        assert "python_programs" not in result

    def test_skips_pytest_use_correct(self) -> None:
        src = (
            "import pytest\n"
            "if pytest.use_correct:\n"
            "    from correct_python_programs.bfs import bfs\n"
            "else:\n"
            "    from python_programs.bfs import bfs\n"
            "def test_bfs():\n"
            "    pass\n"
        )
        result = _rewrite_graph_test(src, "bfs")
        assert result is not None
        assert "pytest.use_correct" not in result
        assert "correct_python_programs" not in result
        assert "import pytest" not in result

    def test_skips_pytest_run_slow(self) -> None:
        src = (
            "from python_programs.bfs import bfs\n"
            "if not pytest.run_slow:\n"
            "    pass\n"
            "def test_bfs():\n"
            "    pass\n"
        )
        result = _rewrite_graph_test(src, "bfs")
        assert result is not None
        assert "pytest.run_slow" not in result

    def test_skips_pytest_mark_skip(self) -> None:
        src = (
            "from python_programs.bfs import bfs\n"
            "@pytest.mark.skip(reason='slow')\n"
            "def test_bfs():\n"
            "    pass\n"
        )
        result = _rewrite_graph_test(src, "bfs")
        assert result is not None
        assert "pytest.mark.skip" not in result

    def test_empty_body_returns_none(self) -> None:
        src = (
            "import pytest\n"
            "if pytest.use_correct:\n"
            "    from correct_python_programs.bfs import bfs\n"
        )
        result = _rewrite_graph_test(src, "bfs")
        assert result is None

    def test_import_node_module_style(self) -> None:
        src = "import node\ndef test_x():\n    pass\n"
        result = _rewrite_graph_test(src, "bfs")
        assert result is not None
        assert "from buggy import Node" in result


# ---------------------------------------------------------------------------
# _load_json_testcases
# ---------------------------------------------------------------------------


class TestLoadJsonTestcases:
    def test_valid_cases(self, tmp_path: Path) -> None:
        content = "[[1, 2], 3]\n[[[4, 5]], 9]\n"
        p = tmp_path / "test.json"
        p.write_text(content, encoding="utf-8")
        cases = _load_json_testcases(p)
        assert len(cases) == 2
        assert cases[0] == ([1, 2], 3)
        assert cases[1] == ([[4, 5]], 9)

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        content = "[[1], 2]\n\n  \n[[3], 4]\n"
        p = tmp_path / "test.json"
        p.write_text(content, encoding="utf-8")
        cases = _load_json_testcases(p)
        assert len(cases) == 2

    def test_scalar_input_wrapped_in_list(self, tmp_path: Path) -> None:
        content = "[5, 25]\n"
        p = tmp_path / "test.json"
        p.write_text(content, encoding="utf-8")
        cases = _load_json_testcases(p)
        assert cases[0] == ([5], 25)

    def test_non_pair_skipped(self, tmp_path: Path) -> None:
        content = "[1, 2, 3]\n[[1], 2]\n"
        p = tmp_path / "test.json"
        p.write_text(content, encoding="utf-8")
        cases = _load_json_testcases(p)
        assert len(cases) == 1

    def test_whitespace_only_lines(self, tmp_path: Path) -> None:
        content = "   \n\t\n"
        p = tmp_path / "test.json"
        p.write_text(content, encoding="utf-8")
        cases = _load_json_testcases(p)
        assert cases == []


# ---------------------------------------------------------------------------
# _strip_docstring_tail
# ---------------------------------------------------------------------------


class TestStripDocstringTail:
    def test_strips_trailing_triple_double_quoted(self) -> None:
        code = 'def f():\n    return 1\n\n"""Algorithm description."""\n'
        result = _strip_docstring_tail(code)
        assert "Algorithm description" not in result
        assert "def f():" in result

    def test_strips_trailing_triple_single_quoted(self) -> None:
        code = "def f():\n    return 1\n\n'''Algorithm description.'''\n"
        result = _strip_docstring_tail(code)
        assert "Algorithm description" not in result

    def test_preserves_code_without_trailing_docstring(self) -> None:
        code = "def f():\n    return 1\n"
        assert _strip_docstring_tail(code) == code

    def test_syntax_error_returns_original(self) -> None:
        code = "def f(\n    broken syntax!!\n"
        assert _strip_docstring_tail(code) == code

    def test_empty_module_body(self) -> None:
        code = ""
        assert _strip_docstring_tail(code) == code

    def test_last_stmt_not_expr(self) -> None:
        code = "x = 1\n"
        assert _strip_docstring_tail(code) == code

    def test_last_expr_not_string_constant(self) -> None:
        code = "def f():\n    return 1\n\n42\n"
        assert _strip_docstring_tail(code) == code

    def test_indented_string_not_stripped(self) -> None:
        code = 'def f():\n    """docstring"""\n    return 1\n'
        assert _strip_docstring_tail(code) == code

    def test_prefixed_string_literal(self) -> None:
        code = 'def f():\n    return 1\n\nr"""Raw docstring."""\n'
        result = _strip_docstring_tail(code)
        assert "Raw docstring" not in result

    def test_byte_prefix_string(self) -> None:
        code = 'def f():\n    return 1\n\nb"""Not a real docstring."""\n'
        # b"..." is bytes, not str → should NOT be stripped
        assert _strip_docstring_tail(code) == code

    def test_non_triple_quoted_string(self) -> None:
        # Single-quoted single-line string at top level — get_source_segment
        # should return it, but it's not triple-quoted, so it stays.
        code = 'def f():\n    return 1\n\n"short"\n'
        assert _strip_docstring_tail(code) == code


# ---------------------------------------------------------------------------
# _generate_json_test
# ---------------------------------------------------------------------------


class TestGenerateJsonTest:
    def test_basic_output(self) -> None:
        code = _generate_json_test("gcd", [([12, 8], 4), ([7, 7], 7)])
        assert "from buggy import gcd" in code
        assert "def test_case_0():" in code
        assert "def test_case_1():" in code
        assert "result = gcd(12, 8)" in code
        assert "assert result == 4" in code


# ---------------------------------------------------------------------------
# _prepend_node_class
# ---------------------------------------------------------------------------


class TestPrependNodeClass:
    def test_prepends_class(self) -> None:
        code = "def bfs(graph):\n    pass\n"
        result = _prepend_node_class(code)
        assert result.startswith("class Node:")
        assert "def bfs(graph):" in result


# ---------------------------------------------------------------------------
# import_quixbugs (integration-style with synthetic repo layout)
# ---------------------------------------------------------------------------


def _make_fake_repo(
    tmp_path: Path,
    *,
    programs: dict[str, tuple[str, str]],
    json_testcases: dict[str, str] | None = None,
    graph_tests: dict[str, str] | None = None,
) -> Path:
    """Build a minimal QuixBugs-like directory tree in tmp_path."""
    repo = tmp_path / "quixbugs"
    (repo / "python_programs").mkdir(parents=True)
    (repo / "correct_python_programs").mkdir(parents=True)
    (repo / "json_testcases").mkdir(parents=True)
    (repo / "python_testcases").mkdir(parents=True)

    for name, (buggy, fixed) in programs.items():
        (repo / "python_programs" / f"{name}.py").write_text(buggy, encoding="utf-8")
        (repo / "correct_python_programs" / f"{name}.py").write_text(fixed, encoding="utf-8")

    if json_testcases:
        for name, content in json_testcases.items():
            (repo / "json_testcases" / f"{name}.json").write_text(content, encoding="utf-8")

    if graph_tests:
        for name, content in graph_tests.items():
            (repo / "python_testcases" / f"test_{name}.py").write_text(content, encoding="utf-8")

    return repo


# Stubs for base functions to avoid side-effects (ruff subprocess, etc.)
def _noop_validate(case_dir: Path, *, schema_only: bool = True) -> tuple[bool, str]:
    return (True, "ok")


class TestImportQuixbugs:
    """Integration tests for import_quixbugs with synthetic data."""

    def _stub_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub out ruff_format, has_external_deps, validate_generated_case."""
        import evals.importers.quixbugs_importer as mod

        monkeypatch.setattr(mod, "ruff_format", lambda code: code)
        monkeypatch.setattr(mod, "has_external_deps", lambda code: False)
        monkeypatch.setattr(mod, "validate_generated_case", _noop_validate)
        monkeypatch.setattr(
            mod,
            "write_case_files",
            lambda case_dir, **kw: case_dir.mkdir(parents=True, exist_ok=True),
        )

    def test_missing_repo_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="QuixBugs buggy programs"):
            import_quixbugs(tmp_path / "out", repo_path=tmp_path / "nonexistent")

    def test_dry_run_returns_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_base(monkeypatch)
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("def gcd(a,b):\n  return a\n", "def gcd(a,b):\n  return b\n")},
            json_testcases={"gcd": "[[12, 8], 4]\n"},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert len(result) == 1
        assert result[0]["case_id"] == "qb_gcd_easy"

    def test_skips_missing_correct_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_base(monkeypatch)
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("def gcd(a,b):\n  return a\n", "def gcd(a,b):\n  return b\n")},
            json_testcases={"gcd": "[[12, 8], 4]\n"},
        )
        # Remove correct file
        (repo / "correct_python_programs" / "gcd.py").unlink()
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert result == []

    def test_skips_empty_buggy_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_base(monkeypatch)
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("", "def gcd(a,b):\n  return b\n")},
            json_testcases={"gcd": "[[12, 8], 4]\n"},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert result == []

    def test_skips_code_exceeding_max_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_base(monkeypatch)
        big_code = "\n".join(f"x_{i} = {i}" for i in range(MAX_CODE_LINES + 1)) + "\n"
        small_code = "x = 1\n"
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": (big_code, small_code)},
            json_testcases={"gcd": "[[1], 2]\n"},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert result == []

    def test_skips_too_many_fix_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_base(monkeypatch)
        import evals.importers.quixbugs_importer as mod

        # Make count_patch_lines return more than MAX_FIX_LINES
        monkeypatch.setattr(mod, "count_patch_lines", lambda p: MAX_FIX_LINES + 1)
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("def gcd(a,b):\n  return a\n", "def gcd(a,b):\n  return b\n")},
            json_testcases={"gcd": "[[12, 8], 4]\n"},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert result == []

    def test_skips_empty_oracle_patch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_base(monkeypatch)
        # Identical buggy and fixed code → empty patch
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("def gcd():\n  return 1\n", "def gcd():\n  return 1\n")},
            json_testcases={"gcd": "[[1], 1]\n"},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert result == []

    def test_skips_no_test_code(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_base(monkeypatch)
        # No json testcases and not graph-based
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("def gcd(a,b):\n  return a\n", "def gcd(a,b):\n  return b\n")},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert result == []

    def test_validation_failure_removes_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_base(monkeypatch)
        import evals.importers.quixbugs_importer as mod

        monkeypatch.setattr(
            mod,
            "validate_generated_case",
            lambda case_dir, *, schema_only=True: (False, "bad schema"),
        )
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("def gcd(a,b):\n  return a\n", "def gcd(a,b):\n  return b\n")},
            json_testcases={"gcd": "[[12, 8], 4]\n"},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, validate=True)
        assert result == []
        assert not (out / "qb_gcd_easy").exists()

    def test_graph_program_import(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_base(monkeypatch)
        graph_name = next(iter(GRAPH_BASED_PROGRAMS))
        buggy_code = f"def {graph_name}(g):\n  return None\n"
        fixed_code = f"def {graph_name}(g):\n  return g\n"
        test_src = (
            f"from python_programs.{graph_name} import {graph_name}\ndef test_it():\n    pass\n"
        )
        repo = _make_fake_repo(
            tmp_path,
            programs={graph_name: (buggy_code, fixed_code)},
            graph_tests={graph_name: test_src},
        )
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True)
        assert len(result) == 1
        assert "graph_algorithm" in result[0]["metadata"]["tags"]

    def test_max_cases_limits_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_base(monkeypatch)
        programs = {
            f"prog{i}": (f"def prog{i}():\n  return {i}\n", f"def prog{i}():\n  return 0\n")
            for i in range(5)
        }
        json_tc = {f"prog{i}": f"[[], {i}]\n" for i in range(5)}
        repo = _make_fake_repo(tmp_path, programs=programs, json_testcases=json_tc)
        out = tmp_path / "out"
        out.mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=True, max_cases=2)
        assert len(result) == 2

    def test_skips_existing_case_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_base(monkeypatch)
        repo = _make_fake_repo(
            tmp_path,
            programs={"gcd": ("def gcd(a,b):\n  return a\n", "def gcd(a,b):\n  return b\n")},
            json_testcases={"gcd": "[[12, 8], 4]\n"},
        )
        out = tmp_path / "out"
        out.mkdir()
        (out / "qb_gcd_easy").mkdir()
        result = import_quixbugs(out, repo_path=repo, dry_run=False)
        assert result == []
