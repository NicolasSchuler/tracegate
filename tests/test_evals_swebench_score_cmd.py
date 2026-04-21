"""Tests for SWE-bench scoring command construction and prerequisite checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.swebench.io import write_instance_ids, write_predictions
from evals.swebench.score_official import (
    build_run_evaluation_command,
    check_scoring_prerequisites,
    run_official_scoring,
)
from evals.swebench.types import PredictionRecord


def _write_predictions(path: Path, ids: list[str], *, model_tag: str) -> None:
    write_predictions(
        path,
        [
            PredictionRecord(
                instance_id=instance_id,
                model_patch="diff --git a/a.py b/a.py\n",
                model_name_or_path=model_tag,
            )
            for instance_id in ids
        ],
    )


def test_build_run_evaluation_command_handles_clean_flag() -> None:
    base_kwargs = {
        "predictions_path": Path("/tmp/predictions.jsonl"),
        "run_id": "run-baseline",
        "dataset_name": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "workers": 4,
        "cache_level": "env",
        "python_executable": "python",
        "module": "swebench.harness.run_evaluation",
    }

    without_clean = build_run_evaluation_command(clean=False, **base_kwargs)
    with_clean = build_run_evaluation_command(clean=True, **base_kwargs)

    assert "--clean" not in without_clean
    assert "--clean" in with_clean


def test_run_official_scoring_routes_conditions_and_outputs(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    write_instance_ids(run_dir / "instance_ids.txt", ["inst-1", "inst-2"])
    _write_predictions(
        run_dir / "baseline" / "predictions.jsonl",
        ["inst-1", "inst-2"],
        model_tag="baseline-model",
    )
    _write_predictions(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["inst-1", "inst-2"],
        model_tag="llmdebug-model",
    )

    calls: list[tuple[list[str], Path]] = []

    class _Proc:
        def __init__(self, returncode: int = 0) -> None:
            self.returncode = returncode

    def _fake_run(command, *, cwd, stdout=None, stderr=None, text=True, check=False):
        del stdout, stderr, text, check
        calls.append((list(command), cwd))
        return _Proc(returncode=0)

    monkeypatch.setattr("evals.swebench.score_official.subprocess.run", _fake_run)
    summary = run_official_scoring(
        run_dir=run_dir,
        conditions="baseline,llmdebug",
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        workers=2,
        cache_level="env",
        clean=False,
        check_prereqs=False,
    )

    assert len(calls) == 2
    assert summary.run_dir == run_dir
    assert {item.condition for item in summary.condition_summaries} == {"baseline", "llmdebug"}
    assert any("--run_id" in cmd and "demo-run-baseline" in cmd for cmd, _ in calls)
    assert any("--run_id" in cmd and "demo-run-llmdebug" in cmd for cmd, _ in calls)
    assert any(cwd == run_dir / "official_eval" / "baseline" for _, cwd in calls)
    assert any(cwd == run_dir / "official_eval" / "llmdebug" for _, cwd in calls)


def test_run_official_scoring_rejects_single_condition(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="conditions must include exactly baseline,llmdebug"):
        run_official_scoring(
            run_dir=run_dir,
            conditions="baseline",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            workers=2,
            cache_level="env",
            clean=False,
            check_prereqs=False,
        )


def test_check_scoring_prerequisites_rejects_version_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "evals.swebench.score_official.shutil.which", lambda _name: "/usr/bin/docker"
    )
    monkeypatch.setattr(
        "evals.swebench.score_official.importlib.metadata.version", lambda _pkg: "4.0.5"
    )

    with pytest.raises(RuntimeError, match="version mismatch"):
        check_scoring_prerequisites(
            python_executable="python",
            module="swebench.harness.run_evaluation",
        )


# ---------------------------------------------------------------------------
# build_run_evaluation_command validation (lines 36, 39)
# ---------------------------------------------------------------------------


def test_build_run_evaluation_command_rejects_workers_below_one() -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        build_run_evaluation_command(
            predictions_path=Path("/x/predictions.jsonl"),
            run_id="run-1",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            workers=0,
            cache_level="env",
            clean=False,
            python_executable="python",
            module="swebench.harness.run_evaluation",
        )


def test_build_run_evaluation_command_rejects_invalid_cache_level() -> None:
    with pytest.raises(ValueError, match="unsupported cache level"):
        build_run_evaluation_command(
            predictions_path=Path("/x/predictions.jsonl"),
            run_id="run-1",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            workers=2,
            cache_level="bogus",
            clean=False,
            python_executable="python",
            module="swebench.harness.run_evaluation",
        )


# ---------------------------------------------------------------------------
# check_scoring_prerequisites edge cases (lines 66, 71-72, 86, 92-94)
# ---------------------------------------------------------------------------


def test_check_scoring_prerequisites_rejects_missing_docker(
    monkeypatch,
) -> None:
    monkeypatch.setattr("evals.swebench.score_official.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="docker is required"):
        check_scoring_prerequisites(
            python_executable="python",
            module="swebench.harness.run_evaluation",
        )


def test_check_scoring_prerequisites_rejects_missing_package(
    monkeypatch,
) -> None:
    import importlib.metadata as _ilm

    monkeypatch.setattr(
        "evals.swebench.score_official.shutil.which",
        lambda _name: "/usr/bin/docker",
    )

    def _raise_not_found(_pkg: str):
        raise _ilm.PackageNotFoundError(_pkg)

    monkeypatch.setattr(
        "evals.swebench.score_official.importlib.metadata.version",
        _raise_not_found,
    )
    with pytest.raises(RuntimeError, match="SWE-Bench package is not installed"):
        check_scoring_prerequisites(
            python_executable="python",
            module="swebench.harness.run_evaluation",
        )


def test_check_scoring_prerequisites_rejects_module_import_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "evals.swebench.score_official.shutil.which",
        lambda _name: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        "evals.swebench.score_official.importlib.metadata.version",
        lambda _pkg: "4.1.0",
    )

    class _FailedProc:
        returncode = 1
        stdout = ""
        stderr = "ModuleNotFoundError: No module named 'swebench'"

    monkeypatch.setattr(
        "evals.swebench.score_official.subprocess.run",
        lambda *a, **kw: _FailedProc(),
    )
    with pytest.raises(RuntimeError, match="harness module is unavailable"):
        check_scoring_prerequisites(
            python_executable="python",
            module="swebench.harness.run_evaluation",
        )


# ---------------------------------------------------------------------------
# run_official_scoring edge cases (lines 114, 126, 137, 144, 146, 197-198)
# ---------------------------------------------------------------------------


def test_run_official_scoring_rejects_missing_run_dir(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="run directory does not exist"):
        run_official_scoring(
            run_dir=tmp_path / "nonexistent",
            conditions="baseline,llmdebug",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            workers=2,
            cache_level="env",
            clean=False,
            check_prereqs=False,
        )


def test_run_official_scoring_missing_ids_file_sets_none(monkeypatch, tmp_path) -> None:
    """When instance_ids.txt is absent, expected_ids is None (line 126)."""
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    _write_predictions(
        run_dir / "baseline" / "predictions.jsonl",
        ["inst-1"],
        model_tag="baseline-model",
    )
    _write_predictions(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["inst-1"],
        model_tag="llmdebug-model",
    )
    # No instance_ids.txt written

    class _Proc:
        returncode = 0

    monkeypatch.setattr(
        "evals.swebench.score_official.subprocess.run",
        lambda *a, **kw: _Proc(),
    )
    summary = run_official_scoring(
        run_dir=run_dir,
        conditions="baseline,llmdebug",
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        workers=1,
        cache_level="env",
        clean=False,
        check_prereqs=False,
    )
    assert summary.run_dir == run_dir


def test_run_official_scoring_calls_check_prereqs(monkeypatch, tmp_path) -> None:
    """check_prereqs=True → calls check_scoring_prerequisites (line 137)."""
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    _write_predictions(
        run_dir / "baseline" / "predictions.jsonl",
        ["inst-1"],
        model_tag="baseline-model",
    )
    _write_predictions(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["inst-1"],
        model_tag="llmdebug-model",
    )

    prereq_called = []

    def _fake_check(**_kw):
        prereq_called.append(True)

    monkeypatch.setattr(
        "evals.swebench.score_official.check_scoring_prerequisites",
        _fake_check,
    )

    class _Proc:
        returncode = 0

    monkeypatch.setattr(
        "evals.swebench.score_official.subprocess.run",
        lambda *a, **kw: _Proc(),
    )
    run_official_scoring(
        run_dir=run_dir,
        conditions="baseline,llmdebug",
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        workers=1,
        cache_level="env",
        clean=False,
        check_prereqs=True,
    )
    assert prereq_called


def test_run_official_scoring_raises_on_missing_predictions(monkeypatch, tmp_path) -> None:
    """Missing predictions file during iteration → RuntimeError (line 144)."""
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    _write_predictions(
        run_dir / "baseline" / "predictions.jsonl",
        ["inst-1"],
        model_tag="baseline-model",
    )
    _write_predictions(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["inst-1"],
        model_tag="llmdebug-model",
    )

    # Monkeypatch validate to pass, then remove a file before the loop
    _real_validate = __import__(
        "evals.swebench.io", fromlist=["validate_paired_predictions"]
    ).validate_paired_predictions
    monkeypatch.setattr(
        "evals.swebench.score_official.validate_paired_predictions",
        lambda *a, **kw: None,
    )
    # Remove predictions for baseline so the loop hits line 144
    (run_dir / "baseline" / "predictions.jsonl").unlink()

    with pytest.raises(RuntimeError, match="missing predictions file"):
        run_official_scoring(
            run_dir=run_dir,
            conditions="baseline,llmdebug",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            workers=1,
            cache_level="env",
            clean=False,
            check_prereqs=False,
        )


def test_run_official_scoring_raises_on_empty_predictions(monkeypatch, tmp_path) -> None:
    """Empty predictions file during iteration → RuntimeError (line 146)."""
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    # Write a valid file for llmdebug, empty for baseline
    (run_dir / "baseline" / "predictions.jsonl").write_text("", encoding="utf-8")
    _write_predictions(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["inst-1"],
        model_tag="llmdebug-model",
    )
    monkeypatch.setattr(
        "evals.swebench.score_official.validate_paired_predictions",
        lambda *a, **kw: None,
    )
    with pytest.raises(RuntimeError, match="predictions file is empty"):
        run_official_scoring(
            run_dir=run_dir,
            conditions="baseline,llmdebug",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            workers=1,
            cache_level="env",
            clean=False,
            check_prereqs=False,
        )


def test_run_official_scoring_raises_on_failed_conditions(monkeypatch, tmp_path) -> None:
    """Non-zero exit code → RuntimeError (lines 197-198)."""
    run_dir = tmp_path / "demo-run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (run_dir / "llmdebug").mkdir(parents=True, exist_ok=True)
    write_instance_ids(run_dir / "instance_ids.txt", ["inst-1"])
    _write_predictions(
        run_dir / "baseline" / "predictions.jsonl",
        ["inst-1"],
        model_tag="baseline-model",
    )
    _write_predictions(
        run_dir / "llmdebug" / "predictions.jsonl",
        ["inst-1"],
        model_tag="llmdebug-model",
    )

    class _FailProc:
        returncode = 1

    monkeypatch.setattr(
        "evals.swebench.score_official.subprocess.run",
        lambda *a, **kw: _FailProc(),
    )
    with pytest.raises(RuntimeError, match="official scoring failed"):
        run_official_scoring(
            run_dir=run_dir,
            conditions="baseline,llmdebug",
            dataset_name="princeton-nlp/SWE-bench_Lite",
            split="test",
            workers=1,
            cache_level="env",
            clean=False,
            check_prereqs=False,
        )


# ---------------------------------------------------------------------------
# _shell_quote edge case (line 208)
# ---------------------------------------------------------------------------


def test_shell_quote_returns_empty_quoted_for_blank_string() -> None:
    from evals.swebench.score_official import _shell_quote

    assert _shell_quote("") == "''"
