"""High-yield unit coverage for small eval/importer helper modules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.importers import fixup_expected_exceptions as fixup
from evals.importers import mutmut_generator as mutmut
from evals.importers import native_repro_helpers as native
from evals.patchers.heuristic_patcher import HeuristicSnapshotPatcher
from evals.run_eval import CaseSpec, Condition, RunResult


def _write_case_json(case_dir: Path, *, expected_exception: str = "AssertionError") -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text("def test_f():\n    assert True\n", encoding="utf-8")
    (case_dir / "oracle.patch").write_text(
        "--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-def f():\n+def f():\n",
        encoding="utf-8",
    )
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "description": "fixture",
                "category": "traceback_controls",
                "difficulty": "easy",
                "snapshot_dependency": "none",
                "expected_exception": expected_exception,
                "tags": [],
                "python_args": ["-m", "pytest", "-q"],
                "timeout_sec": 30,
            }
        ),
        encoding="utf-8",
    )


def test_heuristic_patcher_covers_failure_reasons_and_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = HeuristicSnapshotPatcher()
    cwd = tmp_path
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    target = cwd / "buggy.py"
    target.write_text("def f(xs, i):\n    return xs[i]\n", encoding="utf-8")

    failure_cases = [
        (None, "no_snapshot", None),
        ({"frames": []}, "no_frames", None),
        ({"frames": [1]}, "bad_crash_frame", None),
        ({"frames": [{}]}, "no_crash_file", None),
        (
            {"frames": [{"file_rel": "src/demo.py", "locals": {"i": 1, "xs": [1]}}]},
            "not_toy_case",
            None,
        ),
        (
            {"frames": [{"file_rel": "buggy.py", "locals": [1]}]},
            "no_locals",
            None,
        ),
        (
            {"frames": [{"file_rel": "buggy.py", "locals": {"i": "x", "xs": []}}]},
            "unexpected_locals",
            None,
        ),
        (
            {"frames": [{"file_rel": "buggy.py", "locals": {"i": 1, "xs": [1, 2]}}]},
            "not_off_by_one",
            None,
        ),
        (
            {"frames": [{"file_rel": "buggy.py", "locals": {"i": 1, "xs": [1]}}]},
            "pattern_not_found",
            "def f(xs, i):\n    return xs[i + 1]\n",
        ),
    ]

    for snapshot, expected_reason, source_text in failure_cases:
        if source_text is not None:
            target.write_text(source_text, encoding="utf-8")
        else:
            target.write_text("def f(xs, i):\n    return xs[i]\n", encoding="utf-8")
        if snapshot is None:
            snapshot_path = context_dir / "snapshot.json"
            if snapshot_path.exists():
                snapshot_path.unlink()
        else:
            (context_dir / "snapshot.json").write_text(
                json.dumps(snapshot, ensure_ascii=False),
                encoding="utf-8",
            )
        result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=cwd)
        assert result.diff is None
        assert result.meta is not None
        assert result.meta["reason"] == expected_reason

    target.write_text("def f(xs, i):\n    return xs[i]\n", encoding="utf-8")

    (context_dir / "snapshot.json").write_text(
        json.dumps({"frames": [{"file_rel": "buggy.py", "locals": {"i": 1, "xs": [1]}}]}),
        encoding="utf-8",
    )
    result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=cwd)
    assert result.diff is not None
    assert result.meta is not None
    assert result.meta["reason"] == "patched"
    assert "return xs[i - 1]" in result.diff

    original_read_text = Path.read_text

    def _boom(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.name == "buggy.py":
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _boom, raising=False)
    result = patcher.propose_patch(prompt="ignored", context_dir=context_dir, cwd=cwd)
    assert result.diff is None
    assert result.meta is not None
    assert result.meta["reason"] == "cannot_read_target"


def test_fixup_expected_exception_helpers_and_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = 'File "buggy.py", line 1, in <module>\nE   ValueError: boom\n'
    assert fixup._select_best_exception(output) == "ValueError"
    assert fixup._select_best_exception("E   IndexError: boom\n") == "IndexError"

    cases_dir = tmp_path / "cases"
    skip_a = cases_dir / "skip_a"
    skip_b = cases_dir / "skip_b"
    skip_a.mkdir(parents=True, exist_ok=True)
    skip_b.mkdir(parents=True, exist_ok=True)
    deleted = fixup._delete_skipped_cases(cases_dir, {"skip_a": "skip: timeout", "skip_b": "ok"})
    assert deleted == 1
    assert not skip_a.exists()
    assert skip_b.exists()

    case_timeout = tmp_path / "case_timeout"
    case_pass = tmp_path / "case_pass"
    case_fixed = tmp_path / "case_fixed"
    case_bad_patch = tmp_path / "case_bad_patch"
    case_ok = tmp_path / "case_ok"
    for case in [case_timeout, case_pass, case_fixed, case_bad_patch, case_ok]:
        _write_case_json(case)

    cases = [
        CaseSpec(
            case_id="case_timeout",
            path=case_timeout,
            category="traceback_controls",
            difficulty="easy",
            snapshot_dependency="none",
            expected_exception="AssertionError",
            tags=[],
            python_args=["-m", "pytest", "-q"],
            timeout_sec=30,
        ),
        CaseSpec(
            case_id="case_pass",
            path=case_pass,
            category="traceback_controls",
            difficulty="easy",
            snapshot_dependency="none",
            expected_exception="AssertionError",
            tags=[],
            python_args=["-m", "pytest", "-q"],
            timeout_sec=30,
        ),
        CaseSpec(
            case_id="case_fixed",
            path=case_fixed,
            category="traceback_controls",
            difficulty="easy",
            snapshot_dependency="none",
            expected_exception="AssertionError",
            tags=[],
            python_args=["-m", "pytest", "-q"],
            timeout_sec=30,
        ),
        CaseSpec(
            case_id="case_bad_patch",
            path=case_bad_patch,
            category="traceback_controls",
            difficulty="easy",
            snapshot_dependency="none",
            expected_exception="AssertionError",
            tags=[],
            python_args=["-m", "pytest", "-q"],
            timeout_sec=30,
        ),
        CaseSpec(
            case_id="case_ok",
            path=case_ok,
            category="traceback_controls",
            difficulty="easy",
            snapshot_dependency="none",
            expected_exception="AssertionError",
            tags=[],
            python_args=["-m", "pytest", "-q"],
            timeout_sec=30,
        ),
    ]

    monkeypatch.setattr(fixup, "load_cases", lambda _cases_dir: cases)

    run_counts: dict[str, int] = {}

    def _run_pytest(workdir, python_args, condition, timeout_sec, output_format):  # type: ignore[no-untyped-def]
        del python_args, condition, timeout_sec, output_format
        case_id = Path(workdir).name.split("-", 3)[2]
        run_counts[case_id] = run_counts.get(case_id, 0) + 1
        if case_id == "case_timeout":
            return RunResult(1, 0.1, "", "timed out", timed_out=True, timeout_sec=1)
        if case_id == "case_pass":
            return RunResult(0, 0.1, "", "", timed_out=False, timeout_sec=None)
        if case_id in {"case_fixed", "case_bad_patch", "case_ok"} and run_counts[case_id] == 1:
            stderr = (
                'File "buggy.py", line 1, in <module>\nE   ValueError: boom\n'
                if case_id == "case_fixed"
                else "E   AssertionError: boom\n"
            )
            return RunResult(1, 0.1, "", stderr, timed_out=False, timeout_sec=None)
        return RunResult(0, 0.1, "", "", timed_out=False, timeout_sec=None)

    monkeypatch.setattr(fixup, "run_pytest", _run_pytest)

    def _apply_diff(workdir, patch_text):  # type: ignore[no-untyped-def]
        case_id = Path(workdir).name.split("-", 3)[2]
        return case_id != "case_bad_patch", "ok"

    monkeypatch.setattr(fixup, "apply_diff", _apply_diff)

    results = fixup.fixup_cases(tmp_path, timeout_sec=30, output_format="json_compact")
    assert results["case_timeout"] == "skip: timed out"
    assert results["case_pass"] == "skip: baseline passed"
    assert results["case_fixed"] == "fixed: AssertionError -> ValueError"
    assert results["case_bad_patch"] == "skip: oracle patch failed"
    assert results["case_ok"] == "ok: already correct"
    updated = json.loads((case_fixed / "case.json").read_text(encoding="utf-8"))
    assert updated["expected_exception"] == "ValueError"


def test_mutmut_generator_helpers_and_import_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert mutmut._parse_mutmut_results("1 killed\n2 survived\nignored") == [
        {"id": "1", "status": "killed"},
        {"id": "2", "status": "survived"},
    ]
    assert mutmut._infer_operator_type("diff --git a/x.py b/x.py\n- a + b\n") == "arithmetic"
    assert mutmut._infer_operator_type("diff --git a/x.py b/x.py\n- x == y\n") == "comparison"
    assert (
        mutmut._infer_operator_type("diff --git a/x.py b/x.py\n- return value\n")
        == "changed_return"
    )
    assert mutmut._infer_operator_type("") == "keyword"

    original = "def f():\n    return 1\n"
    diff = "--- a/demo.py\n+++ b/demo.py\n@@ -1,2 +1,2 @@\n-def f():\n+def f():\n"

    original_run = mutmut.subprocess.run

    def _fake_patch_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        src_path = Path(cmd[2])
        src_path.write_text("def f():\n    return 2\n", encoding="utf-8")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(mutmut.subprocess, "run", _fake_patch_run)
    mutated = mutmut._apply_diff_to_source(original, diff)
    monkeypatch.setattr(mutmut.subprocess, "run", original_run)
    assert mutated is not None
    assert "def f():" in mutated

    target = tmp_path / "src" / "demo.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(original, encoding="utf-8")
    test_file = tmp_path / "src" / "test_demo.py"
    test_file.write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    assert mutmut._find_test_file(str(target)) == test_file

    dry_run = mutmut.import_mutmut(tmp_path / "cases", target=str(target), dry_run=True)
    assert dry_run[0]["case_id"] == "mut_demo_preview"

    monkeypatch.setattr(mutmut, "_run_mutmut", lambda target: (True, ""))
    monkeypatch.setattr(
        mutmut,
        "_get_mutmut_results",
        lambda: ([{"id": "1", "status": "killed"}, {"id": "2", "status": "survived"}], ""),
    )
    monkeypatch.setattr(mutmut, "_find_test_file", lambda target: test_file)
    monkeypatch.setattr(
        mutmut,
        "_get_mutant_diff",
        lambda mutant_id: diff if mutant_id == "1" else None,
    )
    monkeypatch.setattr(
        mutmut,
        "_apply_diff_to_source",
        lambda original, diff: "def f():\n    return 2\n" if diff else original,
    )
    monkeypatch.setattr(mutmut, "generate_oracle_patch", lambda buggy, fixed: "oracle patch")
    monkeypatch.setattr(mutmut, "write_case_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mutmut, "validate_generated_case", lambda case_dir, schema_only=True: (True, "ok")
    )
    monkeypatch.setattr(mutmut, "map_mutmut_operator", lambda operator: f"mapped:{operator}")
    monkeypatch.setattr(mutmut, "infer_snapshot_dependency", lambda category: f"dep:{category}")

    imported = mutmut.import_mutmut(
        tmp_path / "cases",
        target=str(target),
        max_cases=5,
        validate=True,
        dry_run=False,
    )
    assert imported and imported[0]["case_id"] == "mut_demo_1"


def test_native_repro_helpers_cover_command_and_git_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert native.normalize_relative_path("pkg/../bad.py") is None
    assert native.normalize_relative_path("pkg/module.py") == "pkg/module.py"
    assert native._extract_single_command_line("# comment\npytest -q\n") == "pytest -q"
    assert native._starts_with_env_assignment(["FOO=1", "pytest"])
    assert native._collect_selector_tokens(["-q", "tests/test_demo.py::test_case", "--", "x"]) == [
        "tests/test_demo.py::test_case",
        "x",
    ]
    assert native._option_arity("--maxfail") == 2
    assert native._option_arity("--flag") == 1
    assert native._extract_test_paths(["tests/test_demo.py::test_case", "tests/"]) == [
        "tests/test_demo.py",
        "tests",
    ]
    assert native._is_supported_selector_path(raw="tests/", normalized="tests")

    parsed = native.normalize_native_pytest_command("python -m pytest -q tests/test_demo.py")
    assert parsed is not None
    assert parsed.python_args == ["-m", "pytest", "-q", "tests/test_demo.py"]
    assert parsed.test_paths == ["tests/test_demo.py"]
    assert native.normalize_native_pytest_command("pipenv run pytest tests/test_demo.py") is None

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        if cmd[:2] == ["git", "fetch"]:
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if cmd[:2] == ["git", "clone"]:
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if cmd[:2] == ["git", "show"]:
            return type("P", (), {"returncode": 0, "stdout": "content", "stderr": ""})()
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(native.subprocess, "run", _fake_run)
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    ok, err = native.ensure_git_repo("https://example.com/repo.git", repo_dir)
    assert ok is True and err is None
    assert (
        native.git_show_file(repo_dir, native.GitShowFileRequest("HEAD", "pkg/demo.py"))
        == "content"
    )
    assert calls

    case_dir = tmp_path / "case"
    native.write_native_case_files(
        case_dir,
        buggy_files={"src/demo.py": "print('buggy')\n"},
        test_files={"tests/test_demo.py": "def test_demo():\n    assert True\n"},
        oracle_patch="patch",
        metadata={"source_test_command": "pytest -q"},
    )
    assert (case_dir / "src" / "demo.py").exists()
    assert (case_dir / "tests" / "test_demo.py").exists()
    assert (case_dir / "oracle.patch").exists()
    assert (case_dir / "case.json").exists()
