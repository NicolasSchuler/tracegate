"""Tests for eval summary report generation from run result records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.eval_summary import main
from llmdebug import _json as json


def _record(
    *,
    run_id: str,
    case_id: str,
    condition: str,
    success: bool,
    snapshot_dependency: str = "required",
    category: str = "logic_error",
    difficulty: str = "easy",
    attempt: int = 1,
    schema_version: int = 5,
    tokens_prompt: int = 10,
    tokens_completion: int = 2,
    tokens_total: int = 12,
    llm_call_count: int = 1,
    cumulative_runtime_sec: float = 1.0,
    attempt_metadata_updates: dict[str, Any] | None = None,
    patch_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempt_metadata: dict[str, Any] = {
        "attempt": attempt,
        "max_attempts": 3,
        "attempt_ordinal_label": f"{attempt}/3",
        "llm_call_count": llm_call_count,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "tokens_total": tokens_total,
        "retry_history_budget_chars": 12000,
        "retry_history_budget_applied": False,
        "prompt_sha256": "prompt-hash",
        "prompt_path": None,
        "evidence_sha256": "evidence-hash",
        "evidence_path": None,
    }
    if attempt_metadata_updates:
        attempt_metadata.update(attempt_metadata_updates)

    record: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": attempt,
        "success": success,
        "category": category,
        "difficulty": difficulty,
        "snapshot_dependency": snapshot_dependency,
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": "2026-03-01T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.11.0",
            "platform": "test",
            "llmdebug_version": "2.11.4",
            "patcher_name": "openai",
            "patcher_config_fingerprint": "fp",
        },
        "attempt_metadata": attempt_metadata,
        "model_metadata": {
            "backend": "openai",
            "name": "openai",
            "base_url": "http://127.0.0.1:12000/v1",
            "model": "dummy",
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_mode": "edits",
            "api_compat_seed_requested": None,
            "api_compat_seed_acknowledged": False,
        },
        "cumulative_runtime_sec": cumulative_runtime_sec,
        "note": "success" if success else "tests_still_failing",
    }
    if patch_meta is not None:
        record["patch_meta"] = patch_meta
    return record


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_eval_summary_missing_file(tmp_path: Path, capsys) -> None:
    rc = main(["missing-run", "--results-dir", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "results file not found" in out


def test_eval_summary_rejects_legacy_schema_by_default(tmp_path: Path, capsys) -> None:
    run_id = "legacy-run"
    _write_jsonl(
        tmp_path / f"{run_id}.jsonl",
        [
            _record(
                run_id=run_id,
                case_id="case-legacy",
                condition="base_repair",
                success=False,
                schema_version=2,
            )
        ],
    )
    rc = main([run_id, "--results-dir", str(tmp_path)])
    assert rc == 2
    out = capsys.readouterr().out
    assert "unsupported schema_version=2" in out


def test_eval_summary_base_repair_core_output(tmp_path: Path, capsys) -> None:
    run_id = "tb-only-run"
    _write_jsonl(
        tmp_path / f"{run_id}.jsonl",
        [
            _record(
                run_id=run_id,
                case_id="case-a",
                condition="base_repair",
                success=True,
                tokens_total=30,
                cumulative_runtime_sec=2.0,
            ),
            _record(
                run_id=run_id,
                case_id="case-b",
                condition="base_repair",
                success=False,
                tokens_total=45,
                cumulative_runtime_sec=3.0,
            ),
        ],
    )
    rc = main([run_id, "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Run Summary" in out
    assert "Condition Effectiveness & Efficiency" in out
    assert "- base_repair:" in out
    assert "Pairwise Uplift Matrix" in out
    assert "Reliability & Input Quality" in out
    assert "Controls & Guardrails" in out


def test_eval_summary_prints_pairwise_uplift_matrix(tmp_path: Path, capsys) -> None:
    run_id = "pairwise-run"
    _write_jsonl(
        tmp_path / f"{run_id}.jsonl",
        [
            _record(
                run_id=run_id,
                case_id="case-1",
                condition="base_repair",
                success=False,
            ),
            _record(
                run_id=run_id,
                case_id="case-1",
                condition="cot_repair",
                success=True,
            ),
            _record(
                run_id=run_id,
                case_id="case-2",
                condition="base_repair",
                success=True,
            ),
            _record(
                run_id=run_id,
                case_id="case-2",
                condition="cot_repair",
                success=True,
            ),
        ],
    )
    rc = main([run_id, "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "base_repair -> cot_repair: +50.0%" in out


def test_eval_summary_prints_adaptive_signals_when_present(tmp_path: Path, capsys) -> None:
    run_id = "adaptive-run"
    _write_jsonl(
        tmp_path / f"{run_id}.jsonl",
        [
            _record(
                run_id=run_id,
                case_id="case-adaptive",
                condition="layered_adaptive_llm",
                success=True,
                attempt_metadata_updates={
                    "adaptive_mode": "llm_discretion",
                    "adaptive_helper_invoked": True,
                    "adaptive_helper_need_more_context": True,
                    "adaptive_helper_budget_exhausted": True,
                    "adaptive_online_policy_mode": "on",
                    "adaptive_online_policy_algo": "linucb",
                    "adaptive_online_decision_id": "d1",
                    "adaptive_online_action_selected": "invoke_helper",
                    "adaptive_online_action_applied": "invoke_helper",
                    "adaptive_online_updated": True,
                    "adaptive_online_reward_scope": "helper_local_v1",
                    "adaptive_online_reward": 0.4,
                    "adaptive_online_propensity": 0.8,
                    "adaptive_budget_decision_id": "b1",
                    "adaptive_budget_action_selected": "increase_budget",
                    "adaptive_budget_action_applied": "increase_budget",
                    "adaptive_budget_updated": True,
                    "adaptive_budget_reward_scope": "budget_local_v1",
                    "adaptive_budget_reward": 0.3,
                    "adaptive_budget_propensity": 0.9,
                    "adaptive_budget_applied_multiplier": 1.5,
                },
                patch_meta={
                    "patch_verifier_enabled": True,
                    "patch_verifier_source_scope_matched": True,
                },
            )
        ],
    )
    rc = main([run_id, "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Adaptive Signals" in out
    assert "online policy: decisions=1" in out
    assert "budget policy: decisions=1" in out
    assert "patch verifier source-scope match rate: 100.0%" in out


def test_eval_summary_prints_baseline_diagnostics_when_present(tmp_path: Path, capsys) -> None:
    run_id = "baseline-metrics-run"
    _write_jsonl(
        tmp_path / f"{run_id}.jsonl",
        [
            _record(
                run_id=run_id,
                case_id="tracecoder-case",
                condition="tracecoder",
                success=True,
                patch_meta={
                    "baseline_hybrid_guidance_mode": "fair_v1",
                    "baseline_hybrid_guidance_chars": 120,
                    "baseline_metrics": {
                        "direct_gen_passed": False,
                        "debug_added_value": True,
                        "method_instrumentation": True,
                        "method_two_step_repair": True,
                        "stopped_reason": "Debug cycle completed",
                    },
                },
            ),
            _record(
                run_id=run_id,
                case_id="intervenor-case",
                condition="intervenor",
                success=False,
                patch_meta={
                    "baseline_metrics": {
                        "teacher_duration_sec_total": 1.2,
                        "learner_duration_sec_total": 0.8,
                        "postcheck_duration_sec_total": 0.4,
                        "learner_empty_response_count": 1,
                    }
                },
            ),
            _record(
                run_id=run_id,
                case_id="ldb-case",
                condition="ldb",
                success=True,
                patch_meta={
                    "baseline_metrics": {
                        "is_passing": True,
                        "test_feedback_improvement_steps": 2,
                        "test_feedback_regression_steps": 0,
                        "test_feedback_final_fail_count": 0,
                    }
                },
            ),
            _record(
                run_id=run_id,
                case_id="ldb-case-missing",
                condition="ldb",
                success=False,
            ),
            _record(
                run_id=run_id,
                case_id="non-baseline-case",
                condition="base_repair",
                success=True,
                patch_meta={
                    "baseline_metrics": {
                        "debug_added_value": True,
                    }
                },
            ),
        ],
    )

    rc = main([run_id, "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "baseline flow:" in out
    assert "debug_added_value_rate=100.0% (coverage 1/1)" in out
    assert "hybrid_guidance=fair_v1:1" in out
    assert "baseline timings:" in out
    assert "guidance_chars_mean=120" in out
    assert "teacher_sec=1.20" in out
    assert "baseline feedback:" in out
    assert "is_passing_rate=100.0% (coverage 1/2)" in out
    assert "base_repair: 1/1 (100.0%)" in out


def test_eval_summary_prints_reasoning_token_coverage_when_available(
    tmp_path: Path, capsys
) -> None:
    run_id = "reasoning-run"
    _write_jsonl(
        tmp_path / f"{run_id}.jsonl",
        [
            _record(
                run_id=run_id,
                case_id="case-r",
                condition="base_repair",
                success=True,
                attempt_metadata_updates={
                    "token_usage_available": True,
                    "reasoning_token_usage_available": True,
                    "tokens_reasoning": 6,
                },
            )
        ],
    )

    rc = main([run_id, "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mean_tokens_reasoning=6" in out
    assert "reasoning_coverage=1/1" in out
