"""Tests for eval run resume-from-progress and deduplication logic."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import eval_attempt_flow, eval_recording, eval_types, progress_output, run_eval
from evals.patchers.base import ContextRequestResult, PatchResult
from llmdebug import _json as json


def _run_eval_main(argv: list[str]) -> int:
    return run_eval.main([*argv, "--executor", "local"])


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
        tags=["resume"],
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


def _infra_event_record(
    *,
    run_id: str,
    case: run_eval.CaseSpec,
    condition: run_eval.Condition,
    run_metadata: dict[str, object],
    attempt: int = 1,
) -> dict[str, object]:
    return {
        "record_type": "infra_event",
        "run_id": run_id,
        "case_id": case.case_id,
        "condition": condition.value,
        "attempt": attempt,
        "success": False,
        "run_metadata": dict(run_metadata),
    }


def test_run_eval_resume_skips_completed_units(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    cases = [_make_case(repo_root, "case_a"), _make_case(repo_root, "case_b")]
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_first(*, case, condition, run_id, run_metadata, **_kwargs):
        if case.case_id == "case_a":
            return [
                _result_record(
                    run_id=run_id,
                    case=case,
                    condition=condition,
                    run_metadata=run_metadata,
                    success=False,
                )
            ]
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(run_eval, "run_one_condition", _run_first)
    exit_first = _run_eval_main(
        [
            "--run-id",
            "resume-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
        ]
    )
    assert exit_first == 2
    capsys.readouterr()

    resumed_calls: list[tuple[str, str]] = []

    def _run_resume(*, case, condition, run_id, run_metadata, **_kwargs):
        resumed_calls.append((case.case_id, condition.value))
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=True,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_resume)
    exit_resume = _run_eval_main(
        [
            "--run-id",
            "resume-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
        ]
    )
    assert exit_resume == 0
    assert resumed_calls == [("case_b", "base_repair")]

    out = capsys.readouterr().out
    assert "SKIP (resumed) case_a/base_repair" in out
    assert "DONE case_b/base_repair" in out


def test_run_eval_resume_warns_on_settings_mismatch_and_continues(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "case_only")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_once(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)
    first_exit = _run_eval_main(
        [
            "--run-id",
            "mismatch-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
        ]
    )
    assert first_exit == 0
    capsys.readouterr()

    resumed_calls: list[tuple[str, str]] = []

    def _run_resume(*, case, condition, run_id, run_metadata, **_kwargs):
        resumed_calls.append((case.case_id, condition.value))
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_resume)
    resume_exit = _run_eval_main(
        [
            "--allow-settings-drift",  # ORC-04: drift is now opt-in
            "--run-id",
            "mismatch-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair,layered_adaptive_gated",
            "--k",
            "1",
        ]
    )
    assert resume_exit == 0
    assert resumed_calls == [("case_only", "layered_adaptive_gated")]
    out = capsys.readouterr().out
    assert "Resume warning: run settings fingerprint mismatch for existing run_id" in out
    assert "SKIP (resumed) case_only/base_repair" in out


def test_run_eval_resume_does_not_complete_units_with_only_infra_events(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "infra_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    captured_run_metadata: dict[str, object] = {}

    def _run_once(*, case, condition, run_id, run_metadata, **_kwargs):
        captured_run_metadata.update(dict(run_metadata))
        return [
            _infra_event_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)
    first_exit = _run_eval_main(
        [
            "--run-id",
            "infra-resume-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
        ]
    )
    assert first_exit == 0

    completed_units, _, _, _, _ = run_eval._load_existing_run_state(
        results_path=repo_root / "evals" / "results" / "infra-resume-run.jsonl",
        run_id="infra-resume-run",
        max_attempts_per_unit=1,
    )

    assert completed_units == set()


def test_run_eval_resume_does_not_complete_infra_resume_eligible_attempt_rows(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "infra_attempt_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    fingerprint = "infra-attempt-fp"

    def _run_once(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 1,
                "success": False,
                "record_type": "attempt",
                "note": "forced_noop_patch",
                "patch_meta": {"infra_resume_eligible": True},
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)
    monkeypatch.setattr(
        run_eval,
        "_build_run_settings_fingerprint",
        lambda **_kwargs: fingerprint,
    )

    first_exit = _run_eval_main(
        [
            "--run-id",
            "infra-attempt-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "1",
        ]
    )
    assert first_exit == 0

    completed_units, _, _, _, _ = run_eval._load_existing_run_state(
        results_path=repo_root / "evals" / "results" / "infra-attempt-run.jsonl",
        run_id="infra-attempt-run",
        max_attempts_per_unit=1,
    )

    assert completed_units == set()


def test_run_eval_resume_warns_on_retry_context_mode_mismatch(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "retry_mode_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_once(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)
    first_exit = _run_eval_main(
        [
            "--run-id",
            "retry-mode-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--retry-context-mode",
            "history",
            "--k",
            "1",
        ]
    )
    assert first_exit == 0
    capsys.readouterr()

    calls = {"count": 0}

    def _should_not_run(**_kwargs):
        calls["count"] += 1
        return []

    monkeypatch.setattr(run_eval, "run_one_condition", _should_not_run)
    resume_exit = _run_eval_main(
        [
            "--allow-settings-drift",  # ORC-04: drift is now opt-in
            "--run-id",
            "retry-mode-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--retry-context-mode",
            "fresh",
            "--k",
            "1",
        ]
    )
    assert resume_exit == 0
    assert calls["count"] == 0
    assert (
        "Resume warning: run settings fingerprint mismatch for existing run_id"
        in capsys.readouterr().out
    )


def test_run_eval_resume_warns_on_retry_history_budget_mismatch(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "retry_history_budget_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_once(*, case, condition, run_id, run_metadata, **_kwargs):
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)
    first_exit = _run_eval_main(
        [
            "--run-id",
            "retry-history-budget-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--prompt-budget-retry-history-chars",
            "12000",
            "--k",
            "1",
        ]
    )
    assert first_exit == 0
    capsys.readouterr()

    calls = {"count": 0}

    def _should_not_run(**_kwargs):
        calls["count"] += 1
        return []

    monkeypatch.setattr(run_eval, "run_one_condition", _should_not_run)
    resume_exit = _run_eval_main(
        [
            "--allow-settings-drift",  # ORC-04: drift is now opt-in
            "--run-id",
            "retry-history-budget-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--prompt-budget-retry-history-chars",
            "2000",
            "--k",
            "1",
        ]
    )
    assert resume_exit == 0
    assert calls["count"] == 0
    assert (
        "Resume warning: run settings fingerprint mismatch for existing run_id"
        in capsys.readouterr().out
    )


def test_run_eval_progress_prints_start_finish_and_eta(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "progress_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_progress(*, case, condition, run_id, run_metadata, **_kwargs):
        success = condition == run_eval.Condition.BASE_REPAIR
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=success,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_progress)
    exit_code = _run_eval_main(
        [
            "--run-id",
            "progress-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair,layered_adaptive_gated",
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "START progress_case/base_repair" in out
    assert "DONE progress_case/base_repair status=success" in out
    assert "DONE progress_case/layered_adaptive_gated status=failed reason=failed" in out
    assert "chars=in:0ch,out:0ch" in out
    assert "tokens=n/a" in out
    assert "eta=" in out
    assert "Progress: finished" in out
    assert "Outcome summary: success=1 failed=1 interrupted=0 runtime_errors=0 skipped=0" in out
    assert "Top failure reasons: failed=1" in out


def test_run_eval_progress_summary_reports_top_failure_reasons(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    cases = [
        _make_case(repo_root, "summary_success"),
        _make_case(repo_root, "summary_budget_a"),
        _make_case(repo_root, "summary_budget_b"),
        _make_case(repo_root, "summary_crash"),
    ]
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_summary(*, case, condition, run_id, run_metadata, **_kwargs):
        if case.case_id == "summary_success":
            return [
                _result_record(
                    run_id=run_id,
                    case=case,
                    condition=condition,
                    run_metadata=run_metadata,
                    success=True,
                )
            ]
        if case.case_id in {"summary_budget_a", "summary_budget_b"}:
            record = _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
            record["note"] = "tests_still_failing"
            record["patch_meta"] = {"case_fail_fast_reason": "request_budget_exceeded"}
            record["attempt_metadata"] = {"case_fail_fast_reason": "request_budget_exceeded"}
            return [record]
        raise RuntimeError("simulated runtime failure")

    monkeypatch.setattr(run_eval, "run_one_condition", _run_summary)
    exit_code = _run_eval_main(
        [
            "--run-id",
            "summary-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
        ]
    )
    assert exit_code == 2

    out = capsys.readouterr().out
    assert "DONE summary_budget_a/base_repair status=failed reason=request budget exceeded" in out
    assert "Outcome summary: success=1 failed=2 interrupted=0 runtime_errors=1 skipped=0" in out
    assert "Top failure reasons: request budget exceeded=2, runtime error=1" in out
    assert "\x1b[" not in out


def test_format_progress_token_usage_explains_coverage_state() -> None:
    assert (
        run_eval._format_progress_token_usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            available_records=0,
            unavailable_records=0,
        )
        == "tokens=n/a"
    )
    assert (
        run_eval._format_progress_token_usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            available_records=0,
            unavailable_records=1,
        )
        == "tokens=unavailable"
    )
    assert (
        run_eval._format_progress_token_usage(
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            available_records=1,
            unavailable_records=0,
        )
        == "tokens=prompt:100,completion:40,total:140"
    )
    assert (
        run_eval._format_progress_token_usage(
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            available_records=1,
            unavailable_records=1,
        )
        == "tokens=prompt:100,completion:40,total:140,coverage=partial"
    )


def test_record_token_usage_state_treats_timeout_without_usage_as_unavailable() -> None:
    assert (
        run_eval._record_token_usage_state(
            attempt_meta={"llm_call_count": 0, "tokens_prompt": 0, "tokens_total": 0},
            patch_meta={
                "tokens_prompt": 0,
                "tokens_total": 0,
                "baseline_meta": {"error": "timeout", "timeout_sec": 120.0},
                "baseline_metrics": {
                    "llm_call_count": 0,
                    "usage_available": False,
                    "usage_source": "timeout_no_report",
                },
            },
            prompt_tokens=0,
            total_tokens=0,
        )
        == "unavailable"
    )


def test_run_eval_resume_repairs_malformed_trailing_jsonl_line_durably(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "broken_json_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    results_dir = repo_root / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = "broken-fingerprint"
    valid_line = {
        "run_id": "broken-run",
        "case_id": "broken_json_case",
        "condition": "base_repair",
        "attempt": 1,
        "success": False,
        "run_metadata": {
            "run_id": "broken-run",
            "run_settings_fingerprint": fingerprint,
        },
    }
    (results_dir / "broken-run.jsonl").write_text(
        f"{json.dumps(valid_line)}\n{{not-json}}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_eval,
        "_build_run_settings_fingerprint",
        lambda **_kwargs: fingerprint,
    )
    calls = {"count": 0}

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        calls["count"] += 1
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 2,
                "success": True,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)

    first_exit = _run_eval_main(
        [
            "--run-id",
            "broken-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "3",
        ]
    )
    assert first_exit == 0
    assert calls["count"] == 1
    first_out = capsys.readouterr().out
    assert "Resume repair:" in first_out
    assert "truncated malformed trailing JSON record" in first_out
    sidecars = sorted(results_dir.glob("broken-run.jsonl.corrupt-tail-*.log"))
    assert len(sidecars) == 1
    sidecar_bytes = sidecars[0].read_bytes()
    assert b"{not-json}" in sidecar_bytes

    second_exit = _run_eval_main(
        [
            "--run-id",
            "broken-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "3",
        ]
    )
    assert second_exit == 0
    assert calls["count"] == 1
    second_out = capsys.readouterr().out
    assert "Resume aborted" not in second_out
    assert "Nothing to do: all planned case/condition units are already complete." in second_out


def test_run_eval_resume_invalid_non_trailing_jsonl_still_fails_fast(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "broken_non_trailing_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    results_dir = repo_root / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    good_line = {
        "run_id": "broken-non-trailing-run",
        "case_id": case.case_id,
        "condition": "base_repair",
        "attempt": 1,
        "success": False,
        "run_metadata": {
            "run_id": "broken-non-trailing-run",
            "run_settings_fingerprint": "fp",
        },
    }
    (results_dir / "broken-non-trailing-run.jsonl").write_text(
        "{not-json}\n" + json.dumps(good_line) + "\n",
        encoding="utf-8",
    )

    exit_code = _run_eval_main(
        [
            "--run-id",
            "broken-non-trailing-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
        ]
    )
    assert exit_code == 2
    assert "invalid JSON" in capsys.readouterr().out


def test_run_eval_resume_reruns_partially_written_unit(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "partial_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    results_dir = repo_root / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = "run-fp"
    existing_line = {
        "run_id": "partial-run",
        "case_id": "partial_case",
        "condition": "base_repair",
        "attempt": 1,
        "success": False,
        "run_metadata": {
            "run_id": "partial-run",
            "run_settings_fingerprint": fingerprint,
        },
    }
    (results_dir / "partial-run.jsonl").write_text(
        f"{json.dumps(existing_line)}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_eval,
        "_build_run_settings_fingerprint",
        lambda **_kwargs: fingerprint,
    )

    calls = {"count": 0}

    def _run_partial(*, case, condition, run_id, run_metadata, **_kwargs):
        calls["count"] += 1
        return [
            {
                "run_id": run_id,
                "case_id": case.case_id,
                "condition": condition.value,
                "attempt": 2,
                "success": True,
                "run_metadata": dict(run_metadata),
            }
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_partial)

    exit_code = _run_eval_main(
        [
            "--run-id",
            "partial-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--k",
            "3",
        ]
    )
    assert exit_code == 0
    assert calls["count"] == 1
    assert "SKIP (resumed) partial_case/base_repair" not in capsys.readouterr().out


def test_run_eval_resume_recomputes_fingerprint_across_repeated_main_calls(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "fingerprint_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    results_dir = repo_root / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_line = {
        "run_id": "fingerprint-run",
        "case_id": case.case_id,
        "condition": "base_repair",
        "attempt": 1,
        "success": True,
        "run_metadata": {
            "run_id": "fingerprint-run",
            "run_settings_fingerprint": "fp-one",
        },
    }
    (results_dir / "fingerprint-run.jsonl").write_text(
        f"{json.dumps(existing_line)}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(run_eval, "_build_run_settings_fingerprint", lambda **_kwargs: "fp-one")
    first_exit = _run_eval_main(
        [
            "--run-id",
            "fingerprint-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
        ]
    )
    assert first_exit == 0
    first_out = capsys.readouterr().out
    assert "Resume warning: run settings fingerprint mismatch" not in first_out
    assert "Nothing to do: all planned case/condition units are already complete." in first_out

    monkeypatch.setattr(run_eval, "_build_run_settings_fingerprint", lambda **_kwargs: "fp-two")
    second_exit = _run_eval_main(
        [
            "--allow-settings-drift",  # ORC-04: drift is now opt-in
            "--run-id",
            "fingerprint-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
        ]
    )
    assert second_exit == 0
    second_out = capsys.readouterr().out
    assert "Resume warning: run settings fingerprint mismatch" in second_out
    assert "Nothing to do: all planned case/condition units are already complete." in second_out


def test_run_eval_streamed_record_append_avoids_duplicate_lines(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "stream_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_streamed(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        record_sink,
        progress_update,
        **_kwargs,
    ):
        progress_update("attempt_start", {"attempt": 1, "max_attempts": 1})
        record = _result_record(
            run_id=run_id,
            case=case,
            condition=condition,
            run_metadata=run_metadata,
            success=True,
            attempt=1,
        )
        record_sink(record)
        return [record]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_streamed)
    exit_code = _run_eval_main(
        [
            "--run-id",
            "streamed-no-dup",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--progress-verbosity",
            "detailed",
        ]
    )
    assert exit_code == 0
    results_path = repo_root / "evals" / "results" / "streamed-no-dup.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["case_id"] == "stream_case"
    assert rows[0]["attempt"] == 1
    out = capsys.readouterr().out
    assert "[ATTEMPT] stream_case/base_repair" in out
    assert "chars=in:0ch,out:0ch" in out
    assert "tokens=n/a" in out


def test_run_eval_streamed_retry_gate_record_is_persisted_once(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "retry_gate_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])

    class _SameEvidencePatcher:
        name = "same_evidence"

        def __init__(self) -> None:
            self.propose_calls = 0

        def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
            del prompt, context_dir, cwd
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "buggy.py"}],
                    "sufficiency_rationale": "need buggy file",
                },
                raw_response='{"need_more_context":true,"requests":[{"type":"file","path":"buggy.py"}]}',
                meta={},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            self.propose_calls += 1
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _SameEvidencePatcher()
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: patcher)
    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda *args, **kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="tests/test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (
            {
                "exception": {"type": "AssertionError", "message": "expected 2 got 1"},
                "frames": [
                    {
                        "file_rel": "buggy.py",
                        "line": 1,
                        "function": "f",
                        "code": "return 1",
                        "locals": {},
                    }
                ],
                "pytest": {
                    "nodeid": "test_buggy.py::test_f",
                    "repro": ["python", "-m", "pytest", "-q"],
                },
            },
            None,
        ),
    )

    exit_code = _run_eval_main(
        [
            "--run-id",
            "streamed-retry-gate",
            "--patcher",
            "null",
            "--conditions",
            "layered_adaptive_gated",
            "--k",
            "2",
            "--context-protocol",
            "staged",
            "--adaptive-gate-policy",
            "aggressive",
            "--adaptive-max-helper-calls",
            "2",
        ]
    )
    assert exit_code == 0
    assert patcher.propose_calls == 2  # ATT-05 / D9: the blocked attempt is retried once

    results_path = repo_root / "evals" / "results" / "streamed-retry-gate.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["attempt"] == 1
    assert rows[1]["attempt"] == 2
    assert rows[1]["note"] == "retry_gate_no_new_evidence"
    assert rows[1]["record_type"] == "infra_event"
    assert rows[1]["patch_meta"]["retry_gate_blocked"] is True
    assert rows[2]["attempt"] == 2 and rows[2].get("record_type", "attempt") == "attempt"
    out = capsys.readouterr().out
    assert "DONE retry_gate_case/layered_adaptive_gated status=failed" in out


def test_run_eval_detailed_progress_emits_heartbeat_while_unit_is_running(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "heartbeat_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_slow(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        progress_update,
        **_kwargs,
    ):
        progress_update("attempt_start", {"attempt": 1, "max_attempts": 1})
        time.sleep(0.08)
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=True,
                attempt=1,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_slow)
    exit_code = _run_eval_main(
        [
            "--run-id",
            "heartbeat-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--progress-verbosity",
            "detailed",
            "--progress-heartbeat-sec",
            "0.02",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    heartbeat_line = next(line for line in out.splitlines() if "[HEARTBEAT]" in line)
    assert "[HEARTBEAT] heartbeat_case/base_repair" in heartbeat_line
    assert "prompt=" not in heartbeat_line
    assert "tokens=" not in heartbeat_line


def test_run_eval_detailed_progress_labels_openai_waits_in_heartbeat(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "openai_wait_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_waiting(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        progress_update,
        **_kwargs,
    ):
        progress_update(
            "llm_request",
            {
                "attempt": 1,
                "max_attempts": 1,
                "phase_label": "llm: propose patch",
                "owner": "openai",
                "source": "phase",
                "waiting_for": "openai_server",
                "external_operation": "propose_patch",
                "external_provider": "openai_server",
                "note": "propose patch [gpt-5.4-mini]",
            },
        )
        time.sleep(0.08)
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=True,
                attempt=1,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_waiting)
    exit_code = _run_eval_main(
        [
            "--run-id",
            "openai-wait-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--progress-verbosity",
            "detailed",
            "--progress-heartbeat-sec",
            "0.02",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    heartbeat_line = next(line for line in out.splitlines() if "[HEARTBEAT]" in line)
    assert "phase=llm: propose patch" in heartbeat_line
    assert "owner=openai" in heartbeat_line
    assert "source=phase" in heartbeat_line
    assert "wait=openai_server" in heartbeat_line
    assert "op=propose_patch" in heartbeat_line


def test_run_eval_detailed_progress_labels_baseline_subprocess_waits(
    tmp_path, monkeypatch, capsys
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "tracecoder_wait_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    def _run_waiting(
        *,
        case,
        condition,
        run_id,
        run_metadata,
        progress_update,
        **_kwargs,
    ):
        progress_update(
            "baseline_request",
            {
                "attempt": 1,
                "max_attempts": 1,
                "phase_label": "baseline: tracecoder subprocess",
                "owner": "baseline",
                "source": "sidecar",
                "waiting_for": "tracecoder_subprocess",
                "external_operation": "baseline_repair",
                "external_provider": "tracecoder",
                "attempts_completed": 2,
                "llm_calls_observed": 3,
                "eval_duration_sec_observed": 0.75,
                "note": "status=running",
            },
        )
        time.sleep(0.08)
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=True,
                attempt=1,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_waiting)
    exit_code = _run_eval_main(
        [
            "--run-id",
            "tracecoder-wait-run",
            "--patcher",
            "null",
            "--conditions",
            "base_repair",
            "--progress-verbosity",
            "detailed",
            "--progress-heartbeat-sec",
            "0.02",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    heartbeat_line = next(line for line in out.splitlines() if "[HEARTBEAT]" in line)
    assert "phase=baseline: tracecoder subprocess" in heartbeat_line
    assert "owner=baseline" in heartbeat_line
    assert "source=sidecar" in heartbeat_line
    assert "wait=tracecoder_subprocess" in heartbeat_line
    assert "op=baseline_repair" in heartbeat_line
    assert "iters=2" in heartbeat_line
    assert "llm_calls=3" in heartbeat_line


class _TtyStream:
    def isatty(self) -> bool:
        return True


def test_build_progress_palette_uses_color_on_tty(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    palette = progress_output.build_progress_palette(stream=_TtyStream())
    assert palette.success_count(2) == "\x1b[32m2\x1b[0m"
    assert palette.failure_count(3) == "\x1b[31m3\x1b[0m"


def test_build_progress_palette_respects_no_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    palette = progress_output.build_progress_palette(stream=_TtyStream())
    assert palette.success_count(2) == "2"
    assert palette.failure_count(3) == "3"


def test_run_results_lock_rejects_second_acquire(tmp_path) -> None:
    results_path = tmp_path / "results.jsonl"
    lock_one, error_one, warning_one = run_eval._RunResultsLock.acquire(
        results_path=results_path,
        run_id="lock-run",
    )
    assert error_one is None
    if warning_one is not None:
        assert lock_one is None
        assert "continuing without cross-process lock" in warning_one
        return
    assert lock_one is not None

    lock_two, error_two, warning_two = run_eval._RunResultsLock.acquire(
        results_path=results_path,
        run_id="lock-run",
    )
    assert lock_two is None
    assert isinstance(error_two, str)
    assert "already held" in error_two
    assert warning_two is None

    lock_one.release()
    lock_three, error_three, warning_three = run_eval._RunResultsLock.acquire(
        results_path=results_path,
        run_id="lock-run",
    )
    assert error_three is None
    assert warning_three is None
    assert lock_three is not None
    lock_three.release()


def test_run_eval_continues_when_lock_is_unavailable(tmp_path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "lock_fallback_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
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

    original_acquire = run_eval._RunResultsLock.acquire

    def _acquire_without_lock(*, results_path, run_id):
        del results_path, run_id
        return None, None, "simulated-missing-fcntl"

    monkeypatch.setattr(run_eval._RunResultsLock, "acquire", _acquire_without_lock)
    try:
        exit_code = _run_eval_main(
            [
                "--run-id",
                "lock-fallback-run",
                "--patcher",
                "null",
                "--conditions",
                "base_repair",
            ]
        )
    finally:
        monkeypatch.setattr(run_eval._RunResultsLock, "acquire", original_acquire)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Run lock warning: simulated-missing-fcntl" in out


def test_run_eval_resume_aborts_on_settings_mismatch_without_opt_in(
    tmp_path, monkeypatch, capsys
) -> None:
    """ORC-04: records written under new settings must not claim the old configuration."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    case = _make_case(repo_root, "case_only")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))
    calls: list[str] = []

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        calls.append(condition.value)
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
    base = ["--run-id", "drift-run", "--patcher", "null", "--k", "1"]
    assert _run_eval_main([*base, "--conditions", "base_repair"]) == 0
    capsys.readouterr()
    calls.clear()

    exit_code = _run_eval_main([*base, "--conditions", "base_repair,layered_adaptive_gated"])

    assert exit_code == 2
    assert calls == []
    assert "run settings fingerprint mismatch" in capsys.readouterr().out


def test_run_eval_patcher_config_error_is_a_clean_cli_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    """ORC-12: a patcher config error exits 2 with a message, not a raw SystemExit."""
    from evals.patcher_factory import PatcherConfigError

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [_make_case(repo_root, "c")])

    def _bad_patcher(_args):
        raise PatcherConfigError("--openai-model is required")

    monkeypatch.setattr(run_eval, "make_patcher", _bad_patcher)
    exit_code = _run_eval_main(
        ["--run-id", "bad-patcher", "--patcher", "null", "--conditions", "base_repair"]
    )

    assert exit_code == 2
    assert "Invalid arguments: --openai-model is required" in capsys.readouterr().out


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_every_unit_builds_its_own_patcher(tmp_path, monkeypatch, jobs: str) -> None:
    """ORC-11: --jobs 1 and --jobs N use the same patcher policy, one per unit."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    cases = [_make_case(repo_root, "case_a"), _make_case(repo_root, "case_b")]
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: cases)

    built: list[object] = []

    def _make_patcher(_args):
        patcher = SimpleNamespace(name="stub")
        built.append(patcher)
        return patcher

    used: list[object] = []

    def _run(*, case, condition, run_id, run_metadata, patcher, **_kwargs):
        used.append(patcher)
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=True,
            )
        ]

    monkeypatch.setattr(run_eval, "make_patcher", _make_patcher)
    monkeypatch.setattr(run_eval, "run_one_condition", _run)
    exit_code = _run_eval_main(
        [
            *("--run-id", f"jobs-{jobs}", "--patcher", "null"),
            *("--conditions", "base_repair", "--k", "1", "--jobs", jobs),
        ]
    )

    assert exit_code == 0
    # One run-level patcher (config check, context-window probe) plus one per unit.
    assert len(built) == 3
    assert len(used) == 2
    assert len({id(p) for p in used}) == 2
    assert id(built[0]) not in {id(p) for p in used}


def test_ctrl_c_in_a_parallel_run_reports_and_waits_for_in_flight_units(
    tmp_path, monkeypatch, capsys
) -> None:
    """ORC-09: Ctrl-C stops scheduling and says how many units are still running."""
    import threading

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    cases = [_make_case(repo_root, f"case_{name}") for name in "abc"]
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: cases)
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    slow_started = threading.Event()
    finished: list[str] = []

    def _run(*, case, condition, run_id, run_metadata, **_kwargs):
        if case.case_id == "case_a":
            slow_started.wait(timeout=5)
            raise KeyboardInterrupt
        slow_started.set()
        threading.Event().wait(0.2)
        finished.append(case.case_id)
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=True,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run)
    with pytest.raises(KeyboardInterrupt):
        _run_eval_main(
            [
                *("--run-id", "interrupted", "--patcher", "null"),
                *("--conditions", "base_repair", "--k", "1", "--jobs", "2"),
            ]
        )

    # The in-flight unit finished; the third unit was never scheduled.
    assert finished == ["case_b"]
    assert "waiting for 1 in-flight unit(s)" in capsys.readouterr().out


def test_resume_rejects_a_run_from_an_earlier_policy_defaults_version(
    tmp_path, monkeypatch, capsys
) -> None:
    """POL-01: the policy behaviour generation is part of the settings fingerprint.

    A run recorded under `policy_defaults_version: 1` used different features, decay and
    tie-breaking, so resuming it with today's defaults would stamp new records with the
    old run metadata (codex 3-1#1).
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "policy_version_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    recorded_metadata: list[dict[str, object]] = []

    def _run_once(*, case, condition, run_id, run_metadata, **_kwargs):
        recorded_metadata.append(dict(run_metadata))
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)
    argv = [
        "--run-id",
        "policy-version-run",
        "--patcher",
        "null",
        "--conditions",
        "layered_adaptive_gated",
        "--k",
        "1",
    ]

    # The earlier generation, written by the same real fingerprint seam.
    monkeypatch.setattr(
        eval_recording,
        "POLICY_DEFAULTS_VERSION",
        eval_types.POLICY_DEFAULTS_VERSION - 1,
    )
    assert _run_eval_main(argv) == 0
    capsys.readouterr()

    monkeypatch.undo()
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))
    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)

    assert _run_eval_main(argv) != 0
    out = capsys.readouterr().out
    assert "Resume aborted: run settings fingerprint mismatch" in out
    assert len(recorded_metadata) == 1
    patcher_config = recorded_metadata[0]["patcher_config"]
    assert isinstance(patcher_config, dict)
    assert patcher_config["policy_defaults_version"] == eval_types.POLICY_DEFAULTS_VERSION - 1


def test_resume_rejects_pre_remediation_records_without_a_fingerprint(
    tmp_path, monkeypatch, capsys
) -> None:
    """A v6-or-older run carries no settings fingerprint, so it cannot be resumed into a
    v7 run — not even with `--allow-settings-drift` (codex 3-1#4, 1d#4)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "legacy_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    results_dir = repo_root / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    legacy_record = {
        "schema_version": 6,
        "run_id": "legacy-run",
        "case_id": case.case_id,
        "condition": "base_repair",
        "attempt": 1,
        "success": False,
        "run_metadata": {"run_id": "legacy-run"},
    }
    (results_dir / "legacy-run.jsonl").write_text(
        f"{json.dumps(legacy_record)}\n", encoding="utf-8"
    )

    def _unexpected_run(**_kwargs):
        raise AssertionError("the legacy run must not be resumed")

    monkeypatch.setattr(run_eval, "run_one_condition", _unexpected_run)
    argv = [
        "--run-id",
        "legacy-run",
        "--patcher",
        "null",
        "--conditions",
        "base_repair",
        "--k",
        "1",
    ]

    # The record-schema guard (codex 1d#4) rejects the v6 file before the fingerprint check.
    assert _run_eval_main(argv) != 0
    assert "predates this harness" in capsys.readouterr().out
    assert _run_eval_main(["--allow-settings-drift", *argv]) != 0
    assert "predates this harness" in capsys.readouterr().out


def test_run_eval_resume_continues_after_permitted_settings_drift(
    tmp_path, monkeypatch, capsys
) -> None:
    """ORC-04 (codex 1d#1): a linked drift history stays resumable on later invocations."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    case = _make_case(repo_root, "drift_case")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: SimpleNamespace(name="stub"))

    executed: list[tuple[str, str]] = []

    def _run_once(*, case, condition, run_id, run_metadata, **_kwargs):
        executed.append((case.case_id, condition.value))
        return [
            _result_record(
                run_id=run_id,
                case=case,
                condition=condition,
                run_metadata=run_metadata,
                success=False,
            )
        ]

    monkeypatch.setattr(run_eval, "run_one_condition", _run_once)

    base_argv = ["--run-id", "drift-run", "--patcher", "null", "--k", "1"]
    assert _run_eval_main([*base_argv, "--conditions", "base_repair"]) == 0
    capsys.readouterr()

    drifted = [*base_argv, "--conditions", "base_repair,layered_adaptive_gated"]
    assert _run_eval_main(["--allow-settings-drift", *drifted]) == 0
    out = capsys.readouterr().out
    assert "Resume warning: run settings fingerprint mismatch for existing run_id" in out
    assert executed == [("drift_case", "base_repair"), ("drift_case", "layered_adaptive_gated")]

    # Third invocation: same settings as the drifted one. The results file now holds both
    # fingerprints, which used to abort resume with "inconsistent run_settings_fingerprint".
    assert _run_eval_main(drifted) == 0
    out = capsys.readouterr().out
    assert "Resume aborted" not in out
    assert "inconsistent run_settings_fingerprint" not in out
    assert "Resume warning: run settings fingerprint mismatch" not in out
    assert "all planned case/condition units are already complete" in out
    assert len(executed) == 2

    # Fourth invocation: the chain extends from the drifted generation, so unfinished units
    # still resume and run.
    assert (
        _run_eval_main(
            [
                "--allow-settings-drift",
                *base_argv,
                "--conditions",
                "base_repair,layered_adaptive_gated,cot_repair",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Resume aborted" not in out
    assert executed[-1] == ("drift_case", "cot_repair")
    assert "SKIP (resumed) drift_case/base_repair" in out

    records = [
        json.loads(line)
        for line in (repo_root / "evals" / "results" / "drift-run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    fingerprints = {str(record["run_metadata"]["run_settings_fingerprint"]) for record in records}
    assert len(fingerprints) == 3
    drift_links = {
        str(record["run_metadata"].get("settings_drift_from_fingerprint") or "")
        for record in records
    } - {""}
    assert len(drift_links) == 2
    assert drift_links < fingerprints


def test_run_eval_resume_rejects_an_unlinked_settings_change(tmp_path, monkeypatch) -> None:
    """ORC-04: only an explicitly linked drift chain is resumable."""
    from evals.run_state import _load_existing_run_state

    results = tmp_path / "unlinked.jsonl"
    rows = [
        {
            "run_id": "r",
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 1,
            "success": False,
            "run_metadata": {"run_id": "r", "run_settings_fingerprint": "fp-one"},
        },
        {
            "run_id": "r",
            "case_id": "c1",
            "condition": "cot_repair",
            "attempt": 1,
            "success": False,
            "run_metadata": {"run_id": "r", "run_settings_fingerprint": "fp-two"},
        },
    ]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent run_settings_fingerprint"):
        _load_existing_run_state(results_path=results, run_id="r", max_attempts_per_unit=1)


def test_run_eval_resume_rejects_an_older_record_schema(tmp_path) -> None:
    """Resuming a pre-remediation results file would mix record generations in one run."""
    from evals.eval_types import EVAL_RECORD_SCHEMA_VERSION
    from evals.run_state import _load_existing_run_state

    results = tmp_path / "legacy.jsonl"
    row = {
        "schema_version": EVAL_RECORD_SCHEMA_VERSION - 1,
        "run_id": "r",
        "case_id": "c1",
        "condition": "base_repair",
        "attempt": 1,
        "success": False,
        "run_metadata": {"run_id": "r", "run_settings_fingerprint": "fp-old"},
    }
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predates this harness"):
        _load_existing_run_state(results_path=results, run_id="r", max_attempts_per_unit=1)

    row["schema_version"] = EVAL_RECORD_SCHEMA_VERSION
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")
    completed, _metadata, fingerprint, count, _issue = _load_existing_run_state(
        results_path=results, run_id="r", max_attempts_per_unit=1
    )
    assert completed == {("c1", "base_repair")}
    assert fingerprint == "fp-old"
    assert count == 1


def test_resume_of_a_pre_remediation_run_is_rejected(tmp_path, monkeypatch, capsys) -> None:
    """A v6-era run cannot be continued as a v7 run (LLM-02/LLM-22 bookkeeping).

    Pre-remediation records carry a fingerprint computed without the keys this revision
    added, so identical CLI arguments still abort the resume instead of mixing old and
    new repair semantics under one run_id.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    case = _make_case(repo_root, "case_only")
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
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
    argv = [
        "--run-id",
        "legacy-run",
        "--patcher",
        "null",
        "--conditions",
        "base_repair",
        "--k",
        "1",
    ]
    assert _run_eval_main(argv) == 0
    capsys.readouterr()

    # Rewrite the results as a pre-remediation (v6) run with its own fingerprint.
    results_path = repo_root / "evals" / "results" / "legacy-run.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["schema_version"] = 6
        row["run_metadata"]["run_settings_fingerprint"] = "pre_remediation_fingerprint"
    results_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8"
    )

    exit_code = _run_eval_main(argv)

    # The record-schema guard (codex 1d#4) rejects the v6 rows before the fingerprint check.
    assert exit_code != 0
    assert "predates this harness" in capsys.readouterr().out
