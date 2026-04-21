"""Tests for parallel job execution and thread-safety in eval runs."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from evals import run_eval, validate_cases
from llmdebug import _json as json


def _make_case(repo_root: Path, case_id: str) -> run_eval.CaseSpec:
    case_dir = repo_root / "synthetic_cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=["parallel"],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def _result_record(
    *,
    run_id: str,
    case: run_eval.CaseSpec,
    condition: run_eval.Condition,
    run_metadata: dict[str, object],
    success: bool,
    attempt: int = 1,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "case_id": case.case_id,
        "condition": condition.value,
        "attempt": attempt,
        "success": success,
        "run_metadata": dict(run_metadata),
    }


def _transport_failure_record(
    *,
    run_id: str,
    case: run_eval.CaseSpec,
    condition: run_eval.Condition,
    run_metadata: dict[str, object],
    attempt: int = 1,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "case_id": case.case_id,
        "condition": condition.value,
        "attempt": attempt,
        "success": False,
        "note": "patcher_request_failed",
        "patch_meta": {
            "error": "request_failed",
            "patcher_request_failed": True,
            "patcher_request_failure_kind": "transport",
        },
        "run_metadata": dict(run_metadata),
    }


def test_run_eval_jobs_argument_validation() -> None:
    exit_code = run_eval.main(["--jobs", "0"])
    assert exit_code == 2


def test_run_eval_transport_fail_fast_streak_argument_validation() -> None:
    exit_code = run_eval.main(["--transport-fail-fast-streak", "-1"])
    assert exit_code == 2


def test_run_eval_progress_verbosity_argument_validation() -> None:
    exit_code = run_eval.main(["--progress-verbosity", "verbose"])
    assert exit_code == 2


def test_run_eval_progress_heartbeat_argument_validation() -> None:
    exit_code = run_eval.main(["--progress-heartbeat-sec", "-0.1"])
    assert exit_code == 2


def test_run_eval_context_compaction_scope_argument_validation() -> None:
    exit_code = run_eval.main(["--context-compaction-scope", "snapshot,typo"])
    assert exit_code == 2


def test_run_eval_defaults_adaptive_openai_helper_response_mode_to_inherit(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "default_adaptive_mode_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    observed_mode: dict[str, str] = {}

    def _run(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        adaptive_openai_helper_response_mode,
        **_kwargs,
    ):
        observed_mode["value"] = adaptive_openai_helper_response_mode
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "default-adaptive-mode",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert observed_mode["value"] == "inherit"


def test_run_eval_accepts_openai_full_file_response_mode(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "full_file_mode_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    captured: dict[str, str] = {}

    def _make_patcher(args):
        captured["mode"] = str(args.openai_response_mode)
        return SimpleNamespace(name="stub")

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "make_patcher", _make_patcher)
    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "full-file-mode",
            "--patcher",
            "openai",
            "--openai-model",
            "dummy",
            "--openai-response-mode",
            "full_file",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert captured["mode"] == "full_file"


def test_run_eval_openai_auto_context_probe_sets_effective_tokens(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "auto_context_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    class _ProbePatcher:
        name = "openai"

        def probe_context_window_tokens(self):
            return 8192, {
                "context_window_source": "loaded_instance.context_length",
                "context_window_probe_provider": "lmstudio_api_v1",
            }

    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: _ProbePatcher())

    captured: dict[str, object] = {}

    def _run(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        openai_context_window_tokens,
        openai_context_window_source,
        openai_context_window_probe_provider,
        openai_context_window_probe_error,
        **_kwargs,
    ):
        captured["tokens"] = openai_context_window_tokens
        captured["source"] = openai_context_window_source
        captured["provider"] = openai_context_window_probe_provider
        captured["probe_error"] = openai_context_window_probe_error
        captured["run_metadata"] = run_metadata
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "auto-context-probe",
            "--patcher",
            "openai",
            "--openai-model",
            "dummy",
            "--openai-auto-context-window",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert captured["tokens"] == 8192
    assert captured["source"] == "loaded_instance.context_length"
    assert captured["provider"] == "lmstudio_api_v1"
    assert captured["probe_error"] == ""
    run_metadata = captured["run_metadata"]
    assert isinstance(run_metadata, dict)
    patcher_config = run_metadata["patcher_config"]
    assert isinstance(patcher_config, dict)
    assert patcher_config["openai_context_window_tokens_effective"] == 8192
    assert patcher_config["openai_context_window_source"] == "loaded_instance.context_length"


def test_run_eval_openai_manual_context_window_overrides_auto_probe(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "manual_context_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    probe_called = {"value": False}

    class _ProbePatcher:
        name = "openai"

        def probe_context_window_tokens(self):
            probe_called["value"] = True
            return 12345, {"context_window_source": "auto_probe"}

    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: _ProbePatcher())

    observed_tokens: dict[str, int] = {}
    observed_source: dict[str, str] = {}

    def _run(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        openai_context_window_tokens,
        openai_context_window_source,
        **_kwargs,
    ):
        observed_tokens["value"] = int(openai_context_window_tokens)
        observed_source["value"] = str(openai_context_window_source)
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "manual-context-overrides-auto",
            "--patcher",
            "openai",
            "--openai-model",
            "dummy",
            "--openai-context-window-tokens",
            "4096",
            "--openai-auto-context-window",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 0
    assert probe_called["value"] is False
    assert observed_tokens["value"] == 4096
    assert observed_source["value"] == "manual_flag"


def test_run_eval_parallel_best_effort_continues_after_runtime_error(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case_ok = _make_case(repo_root, "case_ok")
    case_err = _make_case(repo_root, "case_err")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case_ok, case_err])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        if case.case_id == "case_err":
            raise RuntimeError("simulated interruption")
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)
    exit_code = run_eval.main(
        [
            "--run-id",
            "parallel-best-effort",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--jobs",
            "2",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "ERROR case_err/base_repair" in out
    assert "DONE case_ok/base_repair" in out

    results_path = repo_root / "evals" / "results" / "parallel-best-effort.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert any(row["case_id"] == "case_ok" for row in rows)
    assert all(row["case_id"] != "case_err" for row in rows)


def test_run_eval_transport_fail_fast_stops_serial_after_default_threshold(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    cases = [_make_case(repo_root, f"case_{idx:02d}") for idx in range(8)]
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    calls: list[str] = []

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        calls.append(case.case_id)
        return [
            _transport_failure_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "transport-fail-fast-serial",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--jobs",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    assert len(calls) == 5
    out = capsys.readouterr().out
    assert "Fail-fast triggered" in out
    results_path = repo_root / "evals" / "results" / "transport-fail-fast-serial.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5


def test_run_eval_transport_fail_fast_stops_parallel_submission(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    cases = [_make_case(repo_root, f"case_p_{idx:02d}") for idx in range(10)]
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    lock = threading.Lock()
    calls: list[str] = []

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        time.sleep(0.05)
        with lock:
            calls.append(case.case_id)
        return [
            _transport_failure_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "transport-fail-fast-parallel",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--jobs",
            "3",
            "--transport-fail-fast-streak",
            "2",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "Fail-fast triggered" in out
    assert out.count("Fail-fast: skipping") == 1
    assert len(calls) < len(cases)
    results_path = repo_root / "evals" / "results" / "transport-fail-fast-parallel.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(calls)


def test_run_eval_transport_fail_fast_triggers_on_segfault_runtime_errors(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    cases = [_make_case(repo_root, f"segfault_case_{idx:02d}") for idx in range(4)]
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    calls: list[str] = []

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        del condition, run_id, run_metadata, _kwargs
        calls.append(case.case_id)
        raise RuntimeError("Segmentation fault (core dumped)")

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    exit_code = run_eval.main(
        [
            "--run-id",
            "transport-fail-fast-segfault",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--jobs",
            "1",
            "--transport-fail-fast-streak",
            "2",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    assert len(calls) == 2
    out = capsys.readouterr().out
    assert "Fail-fast triggered" in out


def test_run_eval_parallel_resume_retries_runtime_error_units(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case_ok = _make_case(repo_root, "case_ok")
    case_err = _make_case(repo_root, "case_err")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case_ok, case_err])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_first(*, case, condition, run_id, run_metadata, **_kwargs):
        if case.case_id == "case_err":
            raise RuntimeError("simulated interruption")
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_first)
    first_exit = run_eval.main(
        [
            "--run-id",
            "parallel-resume",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--jobs",
            "2",
            "--executor",
            "local",
        ]
    )
    assert first_exit == 2
    capsys.readouterr()

    calls: list[str] = []

    def _run_second(*, case, condition, run_id, run_metadata, **_kwargs):
        calls.append(case.case_id)
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_second)
    second_exit = run_eval.main(
        [
            "--run-id",
            "parallel-resume",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--jobs",
            "2",
            "--executor",
            "local",
        ]
    )
    assert second_exit == 0
    assert calls == ["case_err"]
    assert "SKIP (resumed) case_ok/base_repair" in capsys.readouterr().out


def test_run_eval_jobs1_handles_unit_setup_error_as_runtime_error(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case_ok = _make_case(repo_root, "case_ok")
    case_err = _make_case(repo_root, "case_err")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case_ok, case_err])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)
    original_copytree = run_eval.shutil.copytree

    def _copytree(src, dst, *args, **kwargs):
        if Path(src).name == "case_err":
            raise OSError("copy fail")
        return original_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(run_eval.shutil, "copytree", _copytree)
    exit_code = run_eval.main(
        [
            "--run-id",
            "jobs1-setup-error",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
            "--jobs",
            "1",
            "--executor",
            "local",
        ]
    )

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "DONE case_ok/base_repair" in out
    assert "ERROR case_err/base_repair" in out

    results_path = repo_root / "evals" / "results" / "jobs1-setup-error.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert any(row["case_id"] == "case_ok" for row in rows)
    assert all(row["case_id"] != "case_err" for row in rows)


def test_validate_cases_jobs_argument_validation() -> None:
    exit_code = validate_cases.main(["--jobs", "0"])
    assert exit_code == 2


def test_validate_cases_parallel_uses_multiple_workers(tmp_path, monkeypatch) -> None:
    cases = [
        _make_case(tmp_path, "validate_case_a"),
        _make_case(tmp_path, "validate_case_b"),
        _make_case(tmp_path, "validate_case_c"),
    ]
    monkeypatch.setattr(validate_cases, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_cases, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(validate_cases, "check_pytest_available", lambda: (True, "ok"))
    monkeypatch.setattr(validate_cases, "validate_case_contract", lambda _path: (True, "ok"))

    lock = threading.Lock()
    active = {"value": 0}
    max_seen = {"value": 0}

    def _validate(*_args, **_kwargs):
        with lock:
            active["value"] += 1
            max_seen["value"] = max(max_seen["value"], active["value"])
        try:
            time.sleep(0.05)
        finally:
            with lock:
                active["value"] -= 1
        return True, "baseline fails and oracle patch passes"

    monkeypatch.setattr(validate_cases, "validate_one_case", _validate)
    exit_code = validate_cases.main(
        ["--cases-dir", str(tmp_path), "--executor", "local", "--jobs", "2"]
    )

    assert exit_code == 0
    assert max_seen["value"] >= 2


def test_validate_cases_parallel_internal_error_is_reported(tmp_path, monkeypatch, capsys) -> None:
    cases = [_make_case(tmp_path, "validate_case_a"), _make_case(tmp_path, "validate_case_b")]
    monkeypatch.setattr(validate_cases, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_cases, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(validate_cases, "check_pytest_available", lambda: (True, "ok"))
    monkeypatch.setattr(validate_cases, "validate_case_contract", lambda _path: (True, "ok"))
    monkeypatch.setattr(
        validate_cases,
        "validate_one_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = validate_cases.main(
        ["--cases-dir", str(tmp_path), "--executor", "local", "--jobs", "2"]
    )

    assert exit_code == 1
    assert "internal error: RuntimeError: boom" in capsys.readouterr().out
