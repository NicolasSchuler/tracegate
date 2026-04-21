"""Tests for evals/importers/fixup_expected_exceptions.py.

Covers:
- _select_best_exception() with various traceback formats
- _delete_skipped_cases() deleting only skip-prefixed entries
- fixup_cases() pipeline with mocked run_pytest / load_cases / apply_diff
- main() CLI argument parsing, execution config, and summary printing
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evals.eval_types import CaseSpec, Condition, RunResult
from evals.importers.fixup_expected_exceptions import (
    _delete_skipped_cases,
    _select_best_exception,
    fixup_cases,
    main,
)
from llmdebug import _json as json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_spec(
    case_id: str,
    path: Path,
    *,
    bug_source: str | None = None,
    expected_exception: str = "AssertionError",
    python_args: list[str] | None = None,
    timeout_sec: int = 30,
) -> CaseSpec:
    """Build a minimal CaseSpec for testing."""
    return CaseSpec(
        case_id=case_id,
        path=path,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception=expected_exception,
        tags=["test"],
        python_args=python_args or ["-m", "pytest", "-q"],
        timeout_sec=timeout_sec,
        bug_source=bug_source,
    )


def _make_run_result(
    *,
    returncode: int = 1,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> RunResult:
    """Build a minimal RunResult for testing."""
    return RunResult(
        returncode=returncode,
        duration_sec=0.5,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


def _write_case_files(
    case_dir: Path,
    *,
    expected_exception: str = "AssertionError",
    bug_source: str | None = None,
) -> None:
    """Write the minimal case.json and oracle.patch into *case_dir*."""
    case_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "description": "fixture",
        "category": "traceback_controls",
        "difficulty": "easy",
        "snapshot_dependency": "none",
        "expected_exception": expected_exception,
        "tags": ["test"],
        "python_args": ["-m", "pytest", "-q"],
        "timeout_sec": 30,
    }
    if bug_source is not None:
        meta["bug_source"] = bug_source
    (case_dir / "case.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (case_dir / "oracle.patch").write_text(
        textwrap.dedent("""\
            --- a/buggy.py
            +++ b/buggy.py
            @@ -1 +1 @@
            -bad
            +good
        """),
        encoding="utf-8",
    )


# ===================================================================
# _select_best_exception
# ===================================================================


class TestSelectBestException:
    """Unit tests for _select_best_exception()."""

    def test_returns_none_for_empty_output(self) -> None:
        """No exception token in the output → None."""
        assert _select_best_exception("") is None
        assert _select_best_exception("all tests passed\n") is None

    def test_picks_exception_near_buggy_py_frame(self) -> None:
        """Exceptions near a buggy.py traceback frame are preferred."""
        output = textwrap.dedent("""\
            FAILED test_buggy.py::test_foo
            File "buggy.py", line 10, in foo
                x = 1 / 0
            E   ZeroDivisionError: division by zero
        """)
        assert _select_best_exception(output) == "ZeroDivisionError"

    def test_picks_last_buggy_nearby_when_multiple(self) -> None:
        """When multiple exceptions appear near buggy.py, pick the last."""
        output = textwrap.dedent("""\
            File "buggy.py", line 5, in f
            E   ValueError: bad value
            E   TypeError: wrong type
            other stuff
            File "buggy.py", line 20, in g
            E   KeyError: 'missing'
        """)
        # The last buggy_nearby append wins.
        assert _select_best_exception(output) == "KeyError"

    def test_no_colon_or_E_prefix_near_buggy_skipped(self) -> None:
        """Lines near buggy.py without ':' or 'E' prefix are skipped.

        Falls back to extract_exception_candidates.
        """
        output = textwrap.dedent("""\
            File "buggy.py", line 5, in f
            some random NameError mention without colon or E
            E   RuntimeError: actual error
        """)
        # "NameError mention without colon or E" has no colon and
        # doesn't start with E.  The line "E   RuntimeError: ..." IS
        # within 8 lines of the buggy.py frame, starts with E, AND has
        # a colon → it's picked as buggy_nearby.
        result = _select_best_exception(output)
        assert result == "RuntimeError"

    def test_falls_back_to_extract_exception_candidates(self) -> None:
        """When no buggy.py frame, fall back to generic extraction."""
        output = textwrap.dedent("""\
            E   AttributeError: no attribute 'x'
        """)
        assert _select_best_exception(output) == "AttributeError"

    def test_no_buggy_frame_no_E_line_returns_none(self) -> None:
        """No buggy.py frame, no E-lines → None from candidates."""
        output = "FAILED test_foo.py::test_bar\nno useful info\n"
        assert _select_best_exception(output) is None

    def test_buggy_frame_but_no_match_in_window(self) -> None:
        """buggy.py frame present but no exception token nearby.

        Falls back to extract_exception_candidates.
        """
        output = textwrap.dedent("""\
            File "buggy.py", line 1, in f
            just regular text
            more regular text
            nothing to see
            and more
            and more
            and more
            and more
            and more again beyond 8 lines
            E   NameError: name 'x'
        """)
        # NameError line is at index 9 (0-based), buggy.py at 0 →
        # window is lines[1:8] = indices 1..7 which doesn't include 9.
        # Falls back to extract_exception_candidates which picks it.
        assert _select_best_exception(output) == "NameError"

    def test_colon_in_nearby_without_E_still_matches(self) -> None:
        """A line with ':' (but no 'E' prefix) near buggy.py matches."""
        output = textwrap.dedent("""\
            File "buggy.py", line 1, in f
            TypeError: something went wrong
        """)
        assert _select_best_exception(output) == "TypeError"


# ===================================================================
# _delete_skipped_cases
# ===================================================================


class TestDeleteSkippedCases:
    """Unit tests for _delete_skipped_cases()."""

    def test_deletes_only_skip_prefixed(self, tmp_path: Path) -> None:
        """Directories are removed only when status starts with 'skip'."""
        (tmp_path / "case_a").mkdir()
        (tmp_path / "case_b").mkdir()
        (tmp_path / "case_c").mkdir()
        results = {
            "case_a": "skip: timed out",
            "case_b": "fixed: X -> Y",
            "case_c": "skip: no exception",
        }
        deleted = _delete_skipped_cases(tmp_path, results)
        assert deleted == 2
        assert not (tmp_path / "case_a").exists()
        assert (tmp_path / "case_b").exists()
        assert not (tmp_path / "case_c").exists()

    def test_nonexistent_dir_is_harmless(self, tmp_path: Path) -> None:
        """A skip entry whose directory doesn't exist doesn't crash."""
        results = {"ghost": "skip: vanished"}
        deleted = _delete_skipped_cases(tmp_path, results)
        assert deleted == 0

    def test_empty_results(self, tmp_path: Path) -> None:
        """Empty results dict → 0 deletions."""
        assert _delete_skipped_cases(tmp_path, {}) == 0


# ===================================================================
# fixup_cases - integration with mocks
# ===================================================================

_MOD = "evals.importers.fixup_expected_exceptions"


class TestFixupCases:
    """Integration tests for fixup_cases() with mocked externals."""

    def test_skip_by_bug_source_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cases not matching the bug_source filter are skipped (line 94)."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, bug_source="other")
        spec = _make_case_spec("case1", case_dir, bug_source="other")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])

        results = fixup_cases(tmp_path, bug_source="debugbench")
        assert results == {}

    def test_skip_timed_out(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timed-out baseline → skip: timed out."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir)
        spec = _make_case_spec("case1", case_dir)
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])
        monkeypatch.setattr(
            f"{_MOD}.run_pytest",
            lambda *a, **kw: _make_run_result(timed_out=True),
        )
        results = fixup_cases(tmp_path)
        assert results["case1"] == "skip: timed out"

    def test_skip_baseline_passed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Baseline passes (rc=0) → skip: baseline passed."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir)
        spec = _make_case_spec("case1", case_dir)
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])
        monkeypatch.setattr(
            f"{_MOD}.run_pytest",
            lambda *a, **kw: _make_run_result(returncode=0),
        )
        results = fixup_cases(tmp_path)
        assert results["case1"] == "skip: baseline passed"

    def test_already_correct_and_oracle_fixes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Expected exception already in output + oracle fixes → ok."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, expected_exception="AssertionError")
        spec = _make_case_spec("case1", case_dir, expected_exception="AssertionError")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])

        call_count = 0

        def mock_run_pytest(*args: Any, **kwargs: Any) -> RunResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Baseline fails with the expected exception.
                return _make_run_result(
                    returncode=1,
                    stdout="E   AssertionError: wrong\n",
                )
            # Patched run passes.
            return _make_run_result(returncode=0)

        monkeypatch.setattr(f"{_MOD}.run_pytest", mock_run_pytest)
        monkeypatch.setattr(f"{_MOD}.apply_diff", lambda _w, _p: (True, "ok"))
        results = fixup_cases(tmp_path)
        assert results["case1"] == "ok: already correct"

    def test_already_correct_but_oracle_didnt_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Expected exc in output but oracle didn't fix → skip (line 142)."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, expected_exception="AssertionError")
        spec = _make_case_spec("case1", case_dir, expected_exception="AssertionError")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])

        def mock_run_pytest(*args: Any, **kwargs: Any) -> RunResult:
            # Both baseline and patched fail.
            return _make_run_result(
                returncode=1,
                stdout="E   AssertionError: wrong\n",
            )

        monkeypatch.setattr(f"{_MOD}.run_pytest", mock_run_pytest)
        monkeypatch.setattr(f"{_MOD}.apply_diff", lambda _w, _p: (True, "ok"))
        results = fixup_cases(tmp_path)
        assert results["case1"] == "skip: oracle didn't fix"

    def test_already_correct_but_oracle_patch_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Expected exc in output but apply_diff fails → skip (line 144)."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, expected_exception="AssertionError")
        spec = _make_case_spec("case1", case_dir, expected_exception="AssertionError")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])
        monkeypatch.setattr(
            f"{_MOD}.run_pytest",
            lambda *a, **kw: _make_run_result(
                returncode=1,
                stdout="E   AssertionError: bad\n",
            ),
        )
        monkeypatch.setattr(f"{_MOD}.apply_diff", lambda _w, _p: (False, "fail"))
        results = fixup_cases(tmp_path)
        assert results["case1"] == "skip: oracle patch failed"

    def test_no_exception_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No exception candidate in output → skip (lines 150-151)."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, expected_exception="AssertionError")
        spec = _make_case_spec("case1", case_dir, expected_exception="AssertionError")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])
        monkeypatch.setattr(
            f"{_MOD}.run_pytest",
            lambda *a, **kw: _make_run_result(
                returncode=1,
                stdout="some random failure text\n",
            ),
        )
        results = fixup_cases(tmp_path)
        assert results["case1"] == "skip: no exception found"

    def test_oracle_patch_failed_after_new_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New exception found but oracle patch can't apply (157-158)."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, expected_exception="AssertionError")
        spec = _make_case_spec("case1", case_dir, expected_exception="AssertionError")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])
        monkeypatch.setattr(
            f"{_MOD}.run_pytest",
            lambda *a, **kw: _make_run_result(
                returncode=1,
                stdout="E   NameError: name 'x'\n",
            ),
        )
        monkeypatch.setattr(f"{_MOD}.apply_diff", lambda _w, _p: (False, "failed"))
        results = fixup_cases(tmp_path)
        assert results["case1"] == "skip: oracle patch failed"

    def test_oracle_didnt_fix_after_new_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New exception found, oracle applied but doesn't fix (164-165)."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, expected_exception="AssertionError")
        spec = _make_case_spec("case1", case_dir, expected_exception="AssertionError")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])

        call_count = 0

        def mock_run_pytest(*args: Any, **kwargs: Any) -> RunResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Baseline with NameError (not AssertionError).
                return _make_run_result(
                    returncode=1,
                    stdout="E   NameError: name 'x'\n",
                )
            # Patched run still fails.
            return _make_run_result(returncode=1, stdout="still broken\n")

        monkeypatch.setattr(f"{_MOD}.run_pytest", mock_run_pytest)
        monkeypatch.setattr(f"{_MOD}.apply_diff", lambda _w, _p: (True, "ok"))
        results = fixup_cases(tmp_path)
        assert results["case1"] == "skip: oracle didn't fix"

    def test_fixed_writes_updated_case_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full fix: exception corrected and case.json updated."""
        case_dir = tmp_path / "case1"
        _write_case_files(case_dir, expected_exception="AssertionError")
        spec = _make_case_spec("case1", case_dir, expected_exception="AssertionError")
        monkeypatch.setattr(f"{_MOD}.load_cases", lambda _d: [spec])

        call_count = 0

        def mock_run_pytest(*args: Any, **kwargs: Any) -> RunResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Baseline fails with NameError (mismatch).
                return _make_run_result(
                    returncode=1,
                    stdout="E   NameError: name 'x'\n",
                )
            # Oracle-patched run passes.
            return _make_run_result(returncode=0)

        monkeypatch.setattr(f"{_MOD}.run_pytest", mock_run_pytest)
        monkeypatch.setattr(f"{_MOD}.apply_diff", lambda _w, _p: (True, "ok"))

        results = fixup_cases(tmp_path)
        assert results["case1"] == "fixed: AssertionError -> NameError"

        # Verify case.json was rewritten.
        updated = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        assert updated["expected_exception"] == "NameError"


# ===================================================================
# main() CLI
# ===================================================================


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_main_with_relative_cases_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative --cases-dir is resolved against _REPO_ROOT (202-203)."""
        case_dir = tmp_path / "cases"
        case_dir.mkdir()

        captured: dict[str, Any] = {}

        def mock_fixup(cases_dir: Path, **kw: Any) -> dict[str, str]:
            captured["cases_dir"] = cases_dir
            return {"c1": "fixed: A -> B", "c2": "ok: already correct"}

        monkeypatch.setattr(f"{_MOD}.fixup_cases", mock_fixup)
        monkeypatch.setattr(f"{_MOD}.set_active_execution_config", lambda c: c)
        monkeypatch.setattr(f"{_MOD}.build_execution_config_from_args", lambda a: object())
        monkeypatch.setattr(f"{_MOD}._REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--cases-dir", "cases"],
        )

        main()

        assert captured["cases_dir"] == tmp_path / "cases"

    def test_main_with_absolute_cases_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absolute --cases-dir is used as-is (branch 202→205)."""
        case_dir = tmp_path / "abs_cases"
        case_dir.mkdir()

        captured: dict[str, Any] = {}

        def mock_fixup(cases_dir: Path, **kw: Any) -> dict[str, str]:
            captured["cases_dir"] = cases_dir
            return {}

        monkeypatch.setattr(f"{_MOD}.fixup_cases", mock_fixup)
        monkeypatch.setattr(f"{_MOD}.set_active_execution_config", lambda c: c)
        monkeypatch.setattr(f"{_MOD}.build_execution_config_from_args", lambda a: object())
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--cases-dir", str(case_dir)],
        )

        main()
        assert captured["cases_dir"] == case_dir

    def test_main_prints_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """main() prints a results summary line (line 220)."""
        monkeypatch.setattr(
            f"{_MOD}.fixup_cases",
            lambda *a, **kw: {
                "c1": "fixed: A -> B",
                "c2": "ok: already correct",
                "c3": "skip: timed out",
            },
        )
        monkeypatch.setattr(f"{_MOD}.set_active_execution_config", lambda c: c)
        monkeypatch.setattr(f"{_MOD}.build_execution_config_from_args", lambda a: object())
        monkeypatch.setattr(f"{_MOD}._REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--cases-dir", str(tmp_path)],
        )
        main()
        captured = capsys.readouterr()
        assert "1 fixed" in captured.out
        assert "1 already correct" in captured.out
        assert "1 skipped" in captured.out

    def test_main_no_delete_skipped_prints_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """Without --delete-skipped we get the 'kept' message (line 230)."""
        monkeypatch.setattr(
            f"{_MOD}.fixup_cases",
            lambda *a, **kw: {"c1": "skip: timed out"},
        )
        monkeypatch.setattr(f"{_MOD}.set_active_execution_config", lambda c: c)
        monkeypatch.setattr(f"{_MOD}.build_execution_config_from_args", lambda a: object())
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--cases-dir", str(tmp_path)],
        )
        main()
        captured = capsys.readouterr()
        assert "Skipped cases were kept" in captured.out

    def test_main_delete_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """--delete-skipped invokes deletion and prints count (222-228)."""
        skip_dir = tmp_path / "c1"
        skip_dir.mkdir()

        monkeypatch.setattr(
            f"{_MOD}.fixup_cases",
            lambda *a, **kw: {"c1": "skip: timed out"},
        )
        monkeypatch.setattr(f"{_MOD}.set_active_execution_config", lambda c: c)
        monkeypatch.setattr(f"{_MOD}.build_execution_config_from_args", lambda a: object())
        monkeypatch.setattr(f"{_MOD}._REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--cases-dir",
                str(tmp_path),
                "--delete-skipped",
            ],
        )

        main()
        captured = capsys.readouterr()
        assert "Deleted 1 skipped cases" in captured.out
        assert not skip_dir.exists()

    def test_main_delete_skipped_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """--delete-skipped with relative --cases-dir (branch 224→225)."""
        cases = tmp_path / "mycases"
        skip_dir = cases / "c1"
        skip_dir.mkdir(parents=True)

        monkeypatch.setattr(
            f"{_MOD}.fixup_cases",
            lambda *a, **kw: {"c1": "skip: no exception"},
        )
        monkeypatch.setattr(f"{_MOD}.set_active_execution_config", lambda c: c)
        monkeypatch.setattr(f"{_MOD}.build_execution_config_from_args", lambda a: object())
        monkeypatch.setattr(f"{_MOD}._REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--cases-dir", "mycases", "--delete-skipped"],
        )

        main()
        captured = capsys.readouterr()
        assert "Deleted 1 skipped cases" in captured.out

    def test_main_delete_skipped_zero_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """--delete-skipped with no actual dirs → no 'Deleted' line (227→179)."""
        monkeypatch.setattr(
            f"{_MOD}.fixup_cases",
            lambda *a, **kw: {"ghost": "skip: vanished"},
        )
        monkeypatch.setattr(f"{_MOD}.set_active_execution_config", lambda c: c)
        monkeypatch.setattr(f"{_MOD}.build_execution_config_from_args", lambda a: object())
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--cases-dir", str(tmp_path), "--delete-skipped"],
        )
        main()
        captured = capsys.readouterr()
        assert "Deleted" not in captured.out

    def test_main_invalid_args_raises_system_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """build_execution_config_from_args raising ValueError → SystemExit."""
        monkeypatch.setattr(
            f"{_MOD}.build_execution_config_from_args",
            lambda a: (_ for _ in ()).throw(ValueError("bad")),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--cases-dir", "/tmp/fake"],
        )
        with pytest.raises(SystemExit, match="Invalid arguments"):
            main()
