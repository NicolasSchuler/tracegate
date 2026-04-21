"""Tests for eval run reproducibility guarantees and deterministic record hashing."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import pytest

from evals import run_eval
from evals.patchers.base import ContextRequestResult, PatchResult
from llmdebug import _json as json


@dataclass(frozen=True)
class _OpenAIDiffPatcher:
    name: str = "openai"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "dummy-model"
    temperature: float = 0.0
    max_tokens: int = 128
    response_mode: str = "edits"
    seed: int | None = None

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        diff = (
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def f() -> int:\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        return PatchResult(
            diff=diff,
            meta={
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "dummy-model",
                "temperature": 0.0,
                "max_tokens": 128,
                "response_mode": "edits",
                "api_compat_seed_requested": None,
                "api_compat_seed_acknowledged": False,
                "api_compat_seed_response": None,
            },
        )


@dataclass(frozen=True)
class _OpenAINoAckPatcher:
    name: str = "openai"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "dummy-model"
    temperature: float = 0.0
    max_tokens: int = 128
    response_mode: str = "edits"
    seed: int | None = 123

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        return PatchResult(
            diff=None,
            meta={
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "dummy-model",
                "temperature": 0.0,
                "max_tokens": 128,
                "response_mode": "edits",
                "api_compat_seed_requested": 123,
                "api_compat_seed_acknowledged": False,
                "api_compat_seed_response": None,
            },
        )


@dataclass(frozen=True)
class _OpenAIUsageAggregatePatcher:
    name: str = "openai"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "dummy-model"
    temperature: float = 0.0
    max_tokens: int = 128
    response_mode: str = "edits"
    seed: int | None = None

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        return ContextRequestResult(
            request_json={
                "need_more_context": False,
                "requests": [],
                "sufficiency_rationale": "enough",
            },
            raw_response='{"need_more_context":false,"requests":[],"sufficiency_rationale":"enough"}',
            meta={
                "llm_call_count": 1,
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            },
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=None,
            meta={
                "llm_call_count": 2,
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                "reason": "no_diff",
            },
        )


@dataclass(frozen=True)
class _OpenAIHelperRetryAggregatePatcher:
    name: str = "openai"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "dummy-model"
    temperature: float = 0.0
    max_tokens: int = 128
    response_mode: str = "edits"
    seed: int | None = None

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        return ContextRequestResult(
            request_json={
                "need_more_context": False,
                "requests": [],
                "sufficiency_rationale": "enough",
            },
            raw_response='{"need_more_context":false,"requests":[],"sufficiency_rationale":"enough"}',
            meta={
                "llm_call_count": 3,
                "request_context_retry_count": 2,
                "request_context_transport_retry_count": 1,
                "usage": {"input_tokens": 21, "output_tokens": 3, "total_tokens": 24},
            },
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=None,
            meta={
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                "reason": "no_diff",
            },
        )


def _make_case(tmp_path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="repro_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=["repro"],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def test_run_eval_main_deterministic_requires_seed() -> None:
    exit_code = run_eval.main(["--deterministic"])
    assert exit_code == 2


def test_default_openai_api_key_prefers_z_api(monkeypatch) -> None:
    monkeypatch.setenv("Z_API", "z-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert run_eval._default_openai_api_key() == "z-key"


def test_build_model_metadata_keeps_v4_base_url() -> None:
    patcher = _OpenAIDiffPatcher(base_url="https://api.z.ai/api/coding/paas/v4")
    model_meta = run_eval._build_model_metadata(patcher=patcher, patch_meta=None)
    assert model_meta["base_url"] == "https://api.z.ai/api/coding/paas/v4"


def test_build_model_metadata_includes_baseline_attribution() -> None:
    patcher = type("BaselinePatcher", (), {"name": "tracecoder"})()
    model_meta = run_eval._build_model_metadata(
        patcher=patcher,
        patch_meta={
            "baseline": "tracecoder",
            "model": "test-model",
            "base_url": "http://127.0.0.1:9999",
        },
    )
    assert model_meta == {
        "backend": "tracecoder",
        "name": "tracecoder",
        "baseline_name": "tracecoder",
        "model": "test-model",
        "base_url": "http://127.0.0.1:9999",
    }


def test_run_one_condition_emits_current_schema_metadata(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir"
    context_dir = tmp_path / "context"
    workdir.mkdir(parents=True, exist_ok=True)

    call_state = {"count": 0}

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="E   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=0,
            duration_sec=0.01,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(run_eval, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(run_eval, "apply_diff", lambda cwd, diff: (True, ""))

    run_metadata = {
        "run_id": "run-meta",
        "started_at_utc": "2026-02-10T00:00:00Z",
        "deterministic": False,
        "seed": None,
        "python_version": "3.12.0",
        "platform": "test",
        "llmdebug_version": "2.1.0",
        "patcher_name": "openai",
        "patcher_config_fingerprint": "fp",
    }

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_OpenAIDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-meta",
        include_files_context=False,
        force_patch_attempt=True,
        run_metadata=run_metadata,
        record_prompt="hash",
        record_evidence="hash",
        deterministic=False,
        seed=None,
    )

    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == run_eval.EVAL_RECORD_SCHEMA_VERSION
    assert record["note"] == "success"
    assert "run_metadata" in record
    assert "attempt_metadata" in record
    assert "model_metadata" in record
    assert record["run_metadata"]["patcher_config_fingerprint"] == "fp"
    assert record["attempt_metadata"]["attempt"] == 1
    assert record["attempt_metadata"]["max_attempts"] == 1
    assert record["attempt_metadata"]["attempt_ordinal_label"] == "1/1"
    assert record["attempt_metadata"]["llm_call_count"] == 1
    assert record["attempt_metadata"]["tokens_total"] >= 0
    assert record["attempt_metadata"]["retry_history_budget_chars"] >= 0
    assert record["attempt_metadata"]["retry_history_budget_applied"] is False
    assert isinstance(record["attempt_metadata"]["prompt_xxh3_128"], str)
    assert isinstance(record["attempt_metadata"]["evidence_xxh3_128"], str)
    assert "prompt_sha256" not in record["attempt_metadata"]
    assert "evidence_sha256" not in record["attempt_metadata"]
    assert record["model_metadata"]["backend"] == "openai"
    assert record["model_metadata"]["response_mode"] == "edits"
    assert json.loads(json.dumps(record))["schema_version"] == run_eval.EVAL_RECORD_SCHEMA_VERSION


def test_run_one_condition_aggregates_stage_a_and_patch_usage(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_usage_aggregate"
    context_dir = tmp_path / "context_usage_aggregate"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        run_eval,
        "run_pytest",
        lambda *args, **kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(run_eval, "read_latest_snapshot", lambda *_args, **_kwargs: (None, None))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_OpenAIUsageAggregatePatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-usage-aggregate",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "no_patch_proposed"
    attempt_meta = record["attempt_metadata"]
    assert int(attempt_meta["llm_call_count"]) == 3
    assert int(attempt_meta["tokens_prompt"]) == 18
    assert int(attempt_meta["tokens_completion"]) == 7
    assert int(attempt_meta["tokens_total"]) == 25
    assert int(attempt_meta["stage_a_prompt_chars"]) > 0
    assert int(attempt_meta["stage_b_prompt_chars"]) > 0
    assert int(attempt_meta["prompt_chars_total"]) == (
        int(attempt_meta["stage_a_prompt_chars"])
        + int(attempt_meta["stage_b_prompt_chars"])
        + int(attempt_meta.get("context_compaction_prompt_chars") or 0)
    )
    retry_summary = attempt_meta.get("retry_summary")
    assert isinstance(retry_summary, dict)
    assert retry_summary["attempt"] == 1
    assert retry_summary["outcome"] == "no_patch_proposed"
    assert retry_summary["hypothesis_tested"]
    assert retry_summary["evidence_used"]
    patch_meta = record["patch_meta"]
    assert int(patch_meta["llm_call_count"]) == 3
    assert int(patch_meta["prompt_chars_total"]) == int(attempt_meta["prompt_chars_total"])
    usage = patch_meta.get("usage")
    assert isinstance(usage, dict)
    assert int(usage["prompt_tokens"]) == 18
    assert int(usage["completion_tokens"]) == 7
    assert int(usage["total_tokens"]) == 25


def test_run_one_condition_counts_stage_a_prompt_chars_for_helper_retries(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    monkeypatch.setattr(
        run_eval,
        "run_pytest",
        lambda cwd, python_args, condition, timeout_sec, output_format: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(run_eval, "read_latest_snapshot", lambda *_args, **_kwargs: (None, None))

    single_records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=tmp_path / "workdir_single",
        context_dir_base=tmp_path / "context_single",
        patcher=_OpenAIUsageAggregatePatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-prompt-single",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
    )
    retry_records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=tmp_path / "workdir_retry",
        context_dir_base=tmp_path / "context_retry",
        patcher=_OpenAIHelperRetryAggregatePatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-prompt-retry",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
    )

    single_meta = single_records[0]["attempt_metadata"]
    retry_meta = retry_records[0]["attempt_metadata"]

    assert int(single_meta["stage_a_prompt_chars"]) > 0
    assert int(retry_meta["stage_a_prompt_chars"]) == int(single_meta["stage_a_prompt_chars"]) * 3
    assert int(retry_meta["prompt_chars_total"]) == (
        int(retry_meta["stage_a_prompt_chars"])
        + int(retry_meta["stage_b_prompt_chars"])
        + int(retry_meta.get("context_compaction_prompt_chars") or 0)
    )
    assert int(retry_meta["llm_call_count"]) == 4


def test_run_one_condition_deterministic_requires_seed_ack(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_noack"
    context_dir = tmp_path / "context_noack"
    workdir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        run_eval,
        "run_pytest",
        lambda cwd, python_args, condition, timeout_sec, output_format: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="deterministic mode failed"):
        run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=_OpenAINoAckPatcher(),
            k=1,
            output_format="json_compact",
            run_id="run-noack",
            include_files_context=False,
            force_patch_attempt=False,
            deterministic=True,
            seed=123,
        )
