"""Tests for adaptive condition selection during eval runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import evals.patchers.baseline_patcher as baseline_patcher_module
from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineConfig,
    BaselineResult,
)
from evals import attempt_proposal, eval_attempt_flow, run_eval
from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_CONTEXT,
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.adaptive_budget_online_policy import (
    STATE_SCHEMA_VERSION as BUDGET_STATE_SCHEMA_VERSION,
)
from evals.adaptive_online_policy import (
    ACTION_COMPACT_THEN_PATCH,
    ACTION_FILE_ONLY,
    ACTION_FRAME_LOCALS_LITE,
    ACTION_FRAME_ONLY,
    ACTION_INVOKE_HELPER,
    ACTION_RETRY_HELPER_STRICT_JSON,
    ACTION_SEARCH_ONLY,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_FULL_LITE,
    ACTION_SNAPSHOT_LITE,
    ONLINE_POLICY_ACTIONS,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)
from evals.adaptive_online_policy import (
    EVENT_SCHEMA_VERSION as ONLINE_EVENT_SCHEMA_VERSION,
)
from evals.adaptive_online_policy import (
    STATE_SCHEMA_VERSION as ONLINE_STATE_SCHEMA_VERSION,
)
from evals.decision_model.config import DecisionModelPolicyConfig
from evals.decision_model.policy import DecisionModelAdvice, DecisionModelBinding
from evals.decision_model.runtime import DecisionStateSink
from evals.decision_model.schema import ParsedAnswers, get_profile
from evals.decision_model.state import build_decision_state
from evals.eval_types import _filter_context_request_payload_for_action
from evals.lambda_rlm_backend import LambdaRLMDecompositionResult
from evals.patchers.base import (
    ContextCompactionResult,
    ContextRequestResult,
    Patcher,
    PatchResult,
)
from evals.patchers.baseline_patcher import BaselinePatcher
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from llmdebug import _json as json


def _make_case(tmp_path: Path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "import helper\n\n\ndef f() -> int:\n    return helper.value()\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="adaptive_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="helpful",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def _snapshot_payload() -> dict:
    return {
        "exception": {"type": "AssertionError", "message": "expected 2 got 1"},
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 4,
                "function": "f",
                "code": "return helper.value()",
                "locals": {"x": 1},
            }
        ],
        "pytest": {"nodeid": "test_buggy.py::test_f", "repro": ["python", "-m", "pytest", "-q"]},
    }


class _HybridBaselineAdapter(BaselineAdapter):
    seen_case_context: BaselineCaseContext | None
    seen_case_input: dict[str, object] | None

    def __init__(self) -> None:
        self.seen_case_context = None
        self.seen_case_input = None

    @property
    def name(self) -> str:
        return "intervenor"

    def setup(self, config: BaselineConfig) -> None:
        pass

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> dict[str, object]:
        self.seen_case_context = case_context
        return {
            "task_id": case_dir.name,
            "buggy_code": (case_dir / "buggy.py").read_text(encoding="utf-8"),
        }

    def run_repair(
        self,
        case_input: dict[str, object],
        config: BaselineConfig,
    ) -> BaselineResult:
        self.seen_case_input = dict(case_input)
        return BaselineResult(
            repaired_code=("import helper\n\n\ndef f() -> int:\n    return helper.value() + 1\n"),
            success=True,
            iterations_used=1,
        )


def _online_policy_state_payload(
    *,
    mode: str,
    preferred_action: str,
    fallback_action: str | None = None,
) -> dict[str, object]:
    """State where *preferred_action* scores best and *fallback_action* second.

    Masking tests name the fallback arm explicitly: greedy ties are broken at random
    since POL-24, so the next arm no longer comes from the action order.
    """
    b_values = dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0)
    if fallback_action is not None:
        b_values[fallback_action] = 1.0
    b_values[preferred_action] = 2.0
    return {
        "schema_version": ONLINE_STATE_SCHEMA_VERSION,
        "mode": mode,
        "algorithm": "linucb",
        "alpha": 0.0,
        "epsilon": 0.0,
        "l2": 1.0,
        "warmup_decisions": 0,
        "save_every": 1,
        "feature_names": ["bias"],
        "decisions": 0,
        "updates": 0,
        "action_counts": dict.fromkeys(ONLINE_POLICY_ACTIONS, 0),
        "A": {action: [[1.0]] for action in ONLINE_POLICY_ACTIONS},
        "b": {action: [b_values[action]] for action in ONLINE_POLICY_ACTIONS},
    }


def _skip_biased_online_controller(
    tmp_path: Path, *, mode: str
) -> tuple[AdaptiveOnlinePolicyController, Path]:
    state_path = tmp_path / f"policy_state_{mode}.json"
    events_path = tmp_path / f"policy_events_{mode}.jsonl"
    state_path.write_text(
        json.dumps(
            _online_policy_state_payload(mode=mode, preferred_action=ACTION_SKIP_HELPER),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    return AdaptiveOnlinePolicyController(config), events_path


def _invoke_biased_online_controller(
    tmp_path: Path, *, mode: str
) -> tuple[AdaptiveOnlinePolicyController, Path]:
    state_path = tmp_path / f"policy_state_invoke_{mode}.json"
    events_path = tmp_path / f"policy_events_invoke_{mode}.jsonl"
    state_path.write_text(
        json.dumps(
            _online_policy_state_payload(mode=mode, preferred_action=ACTION_INVOKE_HELPER),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    return AdaptiveOnlinePolicyController(config), events_path


def _preferred_action_online_controller(
    tmp_path: Path, *, mode: str, preferred_action: str, fallback_action: str | None = None
) -> tuple[AdaptiveOnlinePolicyController, Path]:
    state_path = tmp_path / f"policy_state_{preferred_action}_{mode}.json"
    events_path = tmp_path / f"policy_events_{preferred_action}_{mode}.jsonl"
    state_path.write_text(
        json.dumps(
            _online_policy_state_payload(
                mode=mode, preferred_action=preferred_action, fallback_action=fallback_action
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    return AdaptiveOnlinePolicyController(config), events_path


def test_adaptive_baseline_hybrid_writes_controller_packet_and_case_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir"
    context_dir = tmp_path / "context"
    shutil.copytree(case.path, workdir)
    context_dir.mkdir()
    adapter = _HybridBaselineAdapter()
    patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

    monkeypatch.setattr(
        eval_attempt_flow, "read_latest_snapshot", lambda _path: (_snapshot_payload(), None)
    )
    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="diff",
            run_id="hybrid-baseline-run",
            include_files_context=False,
            force_patch_attempt=False,
            retry_context_mode="history",
        )
    finally:
        run_eval.set_active_execution_config(previous_execution)

    assert records
    assert (
        records[-1]["patch_meta"]["adaptive_solver_architecture"] == "hybrid_baseline_controller_v1"
    )
    assert adapter.seen_case_context is not None
    assert adapter.seen_case_context.case_id == "adaptive_case"
    assert adapter.seen_case_context.python_args == ("-m", "pytest", "-q")
    assert adapter.seen_case_input is not None
    controller_packet = adapter.seen_case_input["llmdebug_controller_packet"]
    assert isinstance(controller_packet, dict)
    assert controller_packet["condition"] == "layered_adaptive_llm"
    assert controller_packet["solver_hints"]["strategy_summary"]
    assert controller_packet["controller_state"]["retry_context_mode"] == "history"
    assert records[-1]["patch_meta"]["adaptive_controller_summarizer_invoked"] is True


def test_baseline_run_clears_stale_adaptive_sidecars_before_adapter_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_stale_sidecars"
    context_dir = tmp_path / "context_stale_sidecars"
    shutil.copytree(case.path, workdir)
    stale_attempt_dir = context_dir / "attempt_01"
    stale_attempt_dir.mkdir(parents=True)
    (stale_attempt_dir / "llmdebug_controller_packet.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "condition": "layered_adaptive_llm",
                "controller_state": {"helper_action_executed": "stale"},
            }
        ),
        encoding="utf-8",
    )
    adapter = _HybridBaselineAdapter()
    patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())

    monkeypatch.setattr(
        eval_attempt_flow, "read_latest_snapshot", lambda _path: (_snapshot_payload(), None)
    )
    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.INTERVENOR,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="diff",
            run_id="baseline-stale-sidecars",
            include_files_context=False,
            force_patch_attempt=False,
        )
    finally:
        run_eval.set_active_execution_config(previous_execution)

    assert records
    assert adapter.seen_case_input is not None
    assert "llmdebug_controller_packet" not in adapter.seen_case_input
    assert not (stale_attempt_dir / "llmdebug_controller_packet.json").exists()


def test_adaptive_baseline_hybrid_invokes_lambda_rlm_controller_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_lambda_rlm"
    context_dir = tmp_path / "context_lambda_rlm"
    shutil.copytree(case.path, workdir)
    context_dir.mkdir()
    adapter = _HybridBaselineAdapter()
    patcher = BaselinePatcher(
        adapter=adapter,
        config=BaselineConfig(
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
        ),
    )
    lambda_repo = tmp_path / "lambda-RLM"
    lambda_repo.mkdir()
    lambda_python = tmp_path / "lambda-python"
    lambda_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    lambda_python.chmod(0o755)

    monkeypatch.setattr(
        eval_attempt_flow, "read_latest_snapshot", lambda _path: (_snapshot_payload(), None)
    )
    monkeypatch.setattr(
        eval_attempt_flow,
        "invoke_lambda_rlm_decomposition",
        lambda **_kwargs: LambdaRLMDecompositionResult(
            summary="First localize the return path, then keep the patch local.",
            steps=[
                {
                    "title": "constrain_edit",
                    "question": "Edit buggy.py:4 without widening the patch scope.",
                    "supporting_routes": ["retry_plan", "history_lessons"],
                    "evidence_ids": [],
                }
            ],
            usage_summary={
                "total_input_tokens": 11,
                "total_output_tokens": 7,
                "model_usage_summaries": {"root": {"total_calls": 1}},
            },
            execution_time_sec=0.25,
            raw_response="{}",
            root_model="test-root-model",
        ),
    )
    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="diff",
            run_id="hybrid-baseline-lambda-rlm",
            include_files_context=False,
            force_patch_attempt=False,
            retry_context_mode="history",
            adaptive_controller_lambda_rlm=True,
            adaptive_controller_lambda_rlm_repo_dir=str(lambda_repo),
            adaptive_controller_lambda_rlm_python=str(lambda_python),
        )
    finally:
        run_eval.set_active_execution_config(previous_execution)

    assert records
    assert adapter.seen_case_input is not None
    controller_packet = adapter.seen_case_input["llmdebug_controller_packet"]
    assert controller_packet["solver_hints"]["strategy_summary"].startswith("First localize")
    assert controller_packet["solver_hints"]["decomposition"][0]["title"] == "constrain_edit"
    patch_meta = records[-1]["patch_meta"]
    assert patch_meta["adaptive_controller_lambda_rlm_invoked"] is True
    assert patch_meta["adaptive_controller_lambda_rlm_success"] is True
    assert patch_meta["adaptive_controller_lambda_rlm_root_model"] == "test-root-model"
    assert patch_meta["adaptive_controller_lambda_rlm_step_count"] == 1
    assert patch_meta["adaptive_controller_lambda_rlm_call_count"] == 1
    assert patch_meta["adaptive_controller_lambda_rlm_total_tokens"] == 18


def test_adaptive_baseline_hybrid_lambda_rlm_uses_effective_baseline_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_lambda_rlm_default_model"
    context_dir = tmp_path / "context_lambda_rlm_default_model"
    shutil.copytree(case.path, workdir)
    context_dir.mkdir()
    adapter = _HybridBaselineAdapter()
    patcher = BaselinePatcher(
        adapter=adapter,
        config=BaselineConfig(
            model="",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
        ),
    )
    lambda_repo = tmp_path / "lambda-RLM"
    lambda_repo.mkdir()
    lambda_python = tmp_path / "lambda-python"
    lambda_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    lambda_python.chmod(0o755)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        eval_attempt_flow, "read_latest_snapshot", lambda _path: (_snapshot_payload(), None)
    )

    def _fake_invoke_lambda_rlm_decomposition(**kwargs) -> LambdaRLMDecompositionResult:
        observed.update(kwargs)
        return LambdaRLMDecompositionResult(
            summary="Keep the repair focused on the failing return path.",
            steps=[],
            usage_summary={},
            execution_time_sec=0.1,
            raw_response="{}",
            root_model=str(kwargs.get("model") or ""),
        )

    monkeypatch.setattr(
        eval_attempt_flow,
        "invoke_lambda_rlm_decomposition",
        _fake_invoke_lambda_rlm_decomposition,
    )
    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="diff",
            run_id="hybrid-baseline-lambda-rlm-default-model",
            include_files_context=False,
            force_patch_attempt=False,
            retry_context_mode="history",
            adaptive_controller_lambda_rlm=True,
            adaptive_controller_lambda_rlm_repo_dir=str(lambda_repo),
            adaptive_controller_lambda_rlm_python=str(lambda_python),
        )
    finally:
        run_eval.set_active_execution_config(previous_execution)

    assert records
    assert observed["model"] == "gpt-4o-mini"
    patch_meta = records[-1]["patch_meta"]
    assert patch_meta["adaptive_controller_lambda_rlm_invoked"] is True
    assert patch_meta["adaptive_controller_lambda_rlm_success"] is True
    assert patch_meta["adaptive_controller_lambda_rlm_root_model"] == "gpt-4o-mini"


def test_adaptive_baseline_hybrid_acceptance_uses_helper_delegate_strict_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance test: baseline-backed adaptive runs recover from invalid helper payloads via the shared Stage-A helper delegate."""

    class _FailingBaselineAdapter(BaselineAdapter):
        seen_case_input: dict[str, object] | None
        run_calls: int

        def __init__(self) -> None:
            self.seen_case_input = None
            self.run_calls = 0

        @property
        def name(self) -> str:
            return "tracecoder"

        def setup(self, config: BaselineConfig) -> None:
            del config

        def convert_case(
            self,
            case_dir: Path,
            *,
            case_context: BaselineCaseContext | None = None,
        ) -> dict[str, object]:
            del case_context
            return {
                "task_id": case_dir.name,
                "buggy_code": (case_dir / "buggy.py").read_text(encoding="utf-8"),
            }

        def run_repair(
            self,
            case_input: dict[str, object],
            config: BaselineConfig,
        ) -> BaselineResult:
            del config
            self.run_calls += 1
            self.seen_case_input = dict(case_input)
            return BaselineResult(
                repaired_code=str(case_input["buggy_code"]),
                success=False,
                iterations_used=1,
            )

    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_baseline_helper_delegate_acceptance"
    context_dir = tmp_path / "context_baseline_helper_delegate_acceptance"
    shutil.copytree(case.path, workdir)
    context_dir.mkdir()
    adapter = _FailingBaselineAdapter()
    patcher = BaselinePatcher(
        adapter=adapter,
        config=BaselineConfig(
            model="baseline-model",
            api_base_url="http://127.0.0.1:4321/v1",
            api_key="configured-key",
            api_timeout_sec=9.0,
            extra={
                "baseline_openai_compat_profile": "lmstudio_model_yaml",
                "baseline_openai_allow_reasoning_fallback": False,
                "baseline_openai_enable_thinking": False,
            },
        ),
    )
    observed_init: list[dict[str, object]] = []
    observed_calls: list[dict[str, object]] = []

    class _FakeOpenAICompatPatcher:
        def __init__(self, **kwargs: object) -> None:
            observed_init.append(dict(kwargs))

        def request_context(
            self,
            *,
            prompt: str,
            context_dir: Path,
            cwd: Path,
            strict_json_first: bool = False,
        ) -> ContextRequestResult:
            del prompt, context_dir, cwd
            observed_calls.append({"strict_json_first": bool(strict_json_first)})
            if len(observed_calls) == 1:
                return ContextRequestResult(
                    request_json=None,
                    raw_response="{invalid-json",
                    meta={},
                )
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "helper.py"}],
                    "sufficiency_rationale": "need helper implementation",
                },
                raw_response='{"need_more_context":true}',
                meta={},
            )

    monkeypatch.setattr(
        baseline_patcher_module,
        "OpenAICompatPatcher",
        _FakeOpenAICompatPatcher,
    )
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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )
    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=2,
            output_format="json_compact",
            run_id="baseline-helper-delegate-acceptance",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_max_helper_calls=1,
            adaptive_budget_topup_max_calls=1,
        )
    finally:
        run_eval.set_active_execution_config(previous_execution)

    assert len(records) == 2
    assert adapter.run_calls == 2
    assert adapter.seen_case_input is not None
    assert observed_init == [
        {
            "base_url": "http://127.0.0.1:4321/v1",
            "model": "baseline-model",
            "api_key": "configured-key",
            "timeout_sec": 9.0,
            "compat_profile": "lmstudio_model_yaml",
            "allow_reasoning_fallback": False,
            "enable_thinking": False,
            "name": "tracecoder_stage_a_helper",
        }
    ]
    assert observed_calls == [
        {"strict_json_first": False},
        {"strict_json_first": True},
    ]
    assert records[0]["attempt_metadata"]["adaptive_helper_failure_kind"] == "invalid_payload"
    assert records[1]["attempt_metadata"]["adaptive_budget_topup_granted"] is True
    assert (
        records[1]["attempt_metadata"]["adaptive_budget_topup_reason"]
        == "invalid_payload_recovery_once"
    )
    assert (
        records[1]["attempt_metadata"]["adaptive_helper_action_executed"]
        == ACTION_RETRY_HELPER_STRICT_JSON
    )
    assert (
        records[1]["attempt_metadata"]["adaptive_online_override_reason"]
        == "constraint_invalid_payload_force_strict_retry"
    )


def _increase_biased_budget_controller(
    tmp_path: Path, *, mode: str
) -> tuple[AdaptiveBudgetPolicyController, Path]:
    state_path = tmp_path / f"budget_policy_state_{mode}.json"
    events_path = tmp_path / f"budget_policy_events_{mode}.jsonl"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": BUDGET_STATE_SCHEMA_VERSION,
                "mode": mode,
                "algorithm": "linucb",
                "architecture": "tier_binary",
                "alpha": 0.0,
                "epsilon": 0.0,
                "l2": 1.0,
                "warmup_decisions": 0,
                "save_every": 1,
                "gate_state": {
                    "feature_names": ["bias"],
                    "decisions": 0,
                    "updates": 0,
                    "action_counts": {
                        ACTION_KEEP_BUDGET: 0,
                        ACTION_INCREASE_BUDGET: 0,
                    },
                    "A": {
                        ACTION_KEEP_BUDGET: [[1.0]],
                        ACTION_INCREASE_BUDGET: [[1.0]],
                    },
                    "b": {
                        ACTION_KEEP_BUDGET: [0.0],
                        ACTION_INCREASE_BUDGET: [2.0],
                    },
                },
                "family_state": {
                    "feature_names": [],
                    "decisions": 0,
                    "updates": 0,
                    "action_counts": {"time": 0, "tokens": 0, "context": 0},
                    "A": {},
                    "b": {},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    return AdaptiveBudgetPolicyController(config), events_path


def _increase_biased_hierarchical_budget_controller(
    tmp_path: Path, *, mode: str, preferred_family: str
) -> tuple[AdaptiveBudgetPolicyController, Path]:
    state_path = tmp_path / f"budget_policy_hierarchical_state_{preferred_family}_{mode}.json"
    events_path = tmp_path / f"budget_policy_hierarchical_events_{preferred_family}_{mode}.jsonl"
    family_actions = (
        ACTION_BUDGET_FAMILY_TIME,
        ACTION_BUDGET_FAMILY_TOKENS,
        ACTION_BUDGET_FAMILY_CONTEXT,
    )
    state_path.write_text(
        json.dumps(
            {
                "schema_version": BUDGET_STATE_SCHEMA_VERSION,
                "mode": mode,
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
                "alpha": 0.0,
                "epsilon": 0.0,
                "l2": 1.0,
                "warmup_decisions": 0,
                "save_every": 1,
                "gate_state": {
                    "feature_names": ["bias"],
                    "decisions": 0,
                    "updates": 0,
                    "action_counts": {
                        ACTION_KEEP_BUDGET: 0,
                        ACTION_INCREASE_BUDGET: 0,
                    },
                    "A": {
                        ACTION_KEEP_BUDGET: [[1.0]],
                        ACTION_INCREASE_BUDGET: [[1.0]],
                    },
                    "b": {
                        ACTION_KEEP_BUDGET: [0.0],
                        ACTION_INCREASE_BUDGET: [2.0],
                    },
                },
                "family_state": {
                    "feature_names": ["bias"],
                    "decisions": 0,
                    "updates": 0,
                    "action_counts": dict.fromkeys(family_actions, 0),
                    "A": {family: [[1.0]] for family in family_actions},
                    "b": {
                        family: [2.0 if family == preferred_family else 0.0]
                        for family in family_actions
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    return AdaptiveBudgetPolicyController(config), events_path


def _default_first_budget_up_profile(
    *, is_openai_patcher: bool
) -> tuple[int, run_eval.AdaptiveBudgetControlProfile, run_eval.AdaptiveBudgetControlProfile]:
    tier_multipliers = run_eval._parse_adaptive_budget_policy_tiers(
        run_eval.DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS
    )
    next_tier_index = eval_attempt_flow._adaptive_budget_next_tier_index(
        current_tier_index=0,
        tier_multipliers=tier_multipliers,
    )
    default_profile = run_eval._build_adaptive_budget_control_profile(
        tier_index=0,
        base_tier_index=0,
        tier_multipliers=tier_multipliers,
        base_propose_budget_sec=10.0,
        base_openai_max_tokens=400,
        base_helper_max_calls=1,
        base_stage_a_max_requests=6,
        base_stage_a_max_chars=120_000,
        base_retry_history_max_chars=48_000,
        base_context_compaction_max_calls=0,
        context_compaction_enabled=False,
        base_response_mode="edits",
        is_openai_patcher=is_openai_patcher,
    )
    next_profile = run_eval._build_adaptive_budget_control_profile(
        tier_index=next_tier_index,
        base_tier_index=0,
        tier_multipliers=tier_multipliers,
        base_propose_budget_sec=10.0,
        base_openai_max_tokens=400,
        base_helper_max_calls=1,
        base_stage_a_max_requests=6,
        base_stage_a_max_chars=120_000,
        base_retry_history_max_chars=48_000,
        base_context_compaction_max_calls=0,
        context_compaction_enabled=False,
        base_response_mode="edits",
        is_openai_patcher=is_openai_patcher,
    )
    return next_tier_index, default_profile, next_profile


def test_hierarchical_budget_context_family_scales_compaction_with_capped_widening() -> None:
    """Verifies that the context-family profile now uses the same capped widening envelope for compaction as other discrete limits."""
    profile = run_eval._build_adaptive_budget_family_profile(
        time_level=0,
        tokens_level=0,
        context_level=2,
        base_tier_index=0,
        tier_multipliers=[1.0, 1.5, 2.0],
        base_propose_budget_sec=10.0,
        base_openai_max_tokens=400,
        base_helper_max_calls=1,
        base_stage_a_max_requests=6,
        base_stage_a_max_chars=120_000,
        base_retry_history_max_chars=48_000,
        base_context_compaction_max_calls=1,
        context_compaction_enabled=True,
        base_response_mode="edits",
        is_openai_patcher=True,
    )

    assert profile.helper_max_calls == 3
    assert profile.stage_a_max_requests == 8
    assert profile.context_compaction_max_calls == 3


@dataclass
class _AdaptiveGatedCapturePatcher:
    name: str = "layered_adaptive_gated_capture"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need code context",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDiscretionDeclinePatcher:
    name: str = "adaptive_discretion_decline"
    request_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": False,
                "requests": [],
                "sufficiency_rationale": "enough evidence",
            },
            raw_response='{"need_more_context":false,"requests":[]}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDiscretionInvalidPatcher:
    name: str = "adaptive_discretion_invalid"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(request_json=None, raw_response="{invalid-json", meta={})

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDiscretionBackendUnavailablePatcher:
    name: str = "adaptive_discretion_backend_unavailable"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json=None,
            raw_response=None,
            meta={
                "error": "request_failed",
                "exc": "ConnectionRefusedError: [Errno 61] Connection refused",
                "request_context_failure_kind": "backend_unavailable",
            },
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDiscretionTransportFailurePatcher:
    name: str = "adaptive_discretion_transport_failure"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json=None,
            raw_response=None,
            meta={
                "error": "request_failed",
                "exc": "ReadTimeout: timed out while waiting for response",
                "request_context_retry_count": 1,
                "request_context_transport_retry_count": 1,
            },
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDiscretionRequestBudgetPatcher:
    name: str = "adaptive_discretion_request_budget"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json=None,
            raw_response=None,
            meta={"error": "request_budget_exceeded"},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDiscretionCompatReasoningOnlyPatcher:
    name: str = "adaptive_discretion_compat_reasoning_only"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json=None,
            raw_response=None,
            meta={
                "error": "openai_compat_reasoning_only",
                "compatibility_reason": "openai_compat_reasoning_only",
            },
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDiscretionSnapshotGenericPatcher:
    name: str = "adaptive_discretion_snapshot_generic"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": False,
                "need_snapshot": True,
                "snapshot_justification": "Need more context.",
                "requests": [],
                "sufficiency_rationale": "need state",
            },
            raw_response='{"need_more_context":false,"need_snapshot":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveToggleDiffPatcher:
    name: str = "adaptive_toggle_diff"
    request_calls: int = 0
    patch_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need source",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        self.patch_calls += 1
        if self.patch_calls % 2 == 1:
            diff = (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            )
        else:
            diff = (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 2\n"
                "+    return 1\n"
            )
        return PatchResult(diff=diff, meta={"reason": "toggle_diff"})


@dataclass
class _AdaptiveTestEditPatcher:
    name: str = "adaptive_test_edit"
    request_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need source",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        diff = (
            "diff --git a/test_buggy.py b/test_buggy.py\n"
            "--- a/test_buggy.py\n"
            "+++ b/test_buggy.py\n"
            "@@ -3,4 +3,4 @@\n"
            " \n"
            " \n"
            " def test_f() -> None:\n"
            "-    assert f() == 2\n"
            "+    assert f() == 1\n"
        )
        return PatchResult(diff=diff, meta={"reason": "test_edit"})


@dataclass
class _AdaptiveNeighborEditPatcher:
    name: str = "adaptive_neighbor_edit"
    request_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need source",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        diff = (
            "diff --git a/helper.py b/helper.py\n"
            "--- a/helper.py\n"
            "+++ b/helper.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def value() -> int:\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        return PatchResult(diff=diff, meta={"reason": "neighbor_edit"})


@dataclass
class _SyntaxCyclingPatcher:
    name: str = "syntax_cycling"
    diffs: list[str] = field(default_factory=list)
    metas: list[dict[str, object]] = field(default_factory=list)
    patch_calls: int = 0

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        self.patch_calls += 1
        if not self.diffs:
            return PatchResult(diff=None, meta={"reason": "no_diff"})
        idx = min(self.patch_calls - 1, len(self.diffs) - 1)
        meta: dict[str, object] = {"reason": "cycle_diff", "llm_call_count": 1}
        if self.metas:
            meta.update(self.metas[min(idx, len(self.metas) - 1)])
        return PatchResult(diff=self.diffs[idx], meta=meta)


@dataclass
class _AdaptiveSequentialPatchResultPatcher:
    name: str = "adaptive_sequential_patch_result"
    max_propose_sec: float = 0.0
    response_mode: str = "diff"
    patch_results: list[PatchResult] = field(default_factory=list)
    request_calls: int = 0
    patch_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need source",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.patch_calls += 1
        self.prompts.append(prompt)
        if not self.patch_results:
            return PatchResult(diff=None, meta={"reason": "no_diff"})
        idx = min(self.patch_calls - 1, len(self.patch_results) - 1)
        patch_result = self.patch_results[idx]
        meta = dict(patch_result.meta) if isinstance(patch_result.meta, dict) else patch_result.meta
        return PatchResult(diff=patch_result.diff, meta=meta)


@dataclass
class _AdaptiveBudgetNoDiffPatcher:
    name: str = "openai"
    max_propose_sec: float = 10.0
    max_tokens: int = 400
    response_mode: str = "edits"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveBudgetRetryGatePatcher:
    name: str = "openai"
    max_propose_sec: float = 10.0
    max_tokens: int = 400
    response_mode: str = "edits"
    request_calls: int = 0
    patch_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
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
        self.patch_calls += 1
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveBudgetExceededPatcher:
    name: str = "openai"
    max_propose_sec: float = 10.0
    max_tokens: int = 400
    response_mode: str = "edits"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff=None, meta={"error": "request_budget_exceeded"})


class _AdaptiveBudgetNoDiffNonDataclassPatcher:
    name = "openai"
    max_propose_sec = 10.0
    response_mode = "edits"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _AdaptiveDelayedCreditPatcher:
    name: str = "adaptive_delayed_credit"
    patch_outcomes: tuple[str, ...] = ("no_diff", "fix")
    request_calls: int = 0
    patch_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        # Vary the requests per call so the evidence fingerprint changes,
        # preventing the retry gate from blocking subsequent propose_patch calls.
        requests: list[dict[str, str]] = [{"type": "file", "path": "helper.py"}]
        if self.request_calls >= 2:
            requests.append({"type": "file", "path": "buggy.py"})
        if self.request_calls >= 3:
            requests.append({"type": "file", "path": "test_buggy.py"})
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": requests,
                "sufficiency_rationale": "need helper implementation",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        self.patch_calls += 1
        outcome = self.patch_outcomes[min(self.patch_calls - 1, len(self.patch_outcomes) - 1)]
        if outcome == "no_diff":
            return PatchResult(diff=None, meta={"reason": "no_diff"})
        if outcome == "fix":
            return PatchResult(diff=_helper_fix_diff(), meta={"reason": "delayed_credit_fix"})
        raise AssertionError(f"unexpected patch outcome: {outcome}")


@dataclass
class _AdaptiveActionCapturePatcher:
    name: str = "adaptive_action_capture"
    request_payload: dict[str, object] | None = None
    request_meta: dict[str, object] = field(default_factory=dict)
    compacted_text: str | None = None
    compact_meta: dict[str, object] = field(default_factory=dict)
    request_calls: int = 0
    compact_calls: int = 0
    patch_calls: int = 0
    request_kwargs: list[dict[str, object]] = field(default_factory=list)
    request_prompts: list[str] = field(default_factory=list)
    patch_prompts: list[str] = field(default_factory=list)

    def request_context(
        self,
        *,
        prompt: str,
        context_dir,
        cwd,
        strict_json_first: bool = False,
    ) -> ContextRequestResult:
        del context_dir, cwd
        self.request_calls += 1
        self.request_prompts.append(prompt)
        self.request_kwargs.append({"strict_json_first": bool(strict_json_first)})
        return ContextRequestResult(
            request_json=self.request_payload,
            raw_response="{}" if self.request_payload is None else json.dumps(self.request_payload),
            meta=dict(self.request_meta),
        )

    def compact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir,
        cwd,
        target_chars: int,
    ) -> ContextCompactionResult:
        del context_dir, cwd, target_chars
        self.compact_calls += 1
        self.patch_prompts.append(f"COMPACTION\n{prompt}\n{context_text}")
        return ContextCompactionResult(
            compacted_text=self.compacted_text,
            raw_response=self.compacted_text,
            meta=dict(self.compact_meta),
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.patch_calls += 1
        self.patch_prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def _helper_fix_diff() -> str:
    return (
        "diff --git a/helper.py b/helper.py\n"
        "--- a/helper.py\n"
        "+++ b/helper.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )


def _test_only_expectation_diff(*, expected: int) -> str:
    return (
        "diff --git a/test_buggy.py b/test_buggy.py\n"
        "--- a/test_buggy.py\n"
        "+++ b/test_buggy.py\n"
        "@@ -3,4 +3,4 @@\n"
        " \n"
        " \n"
        " def test_f() -> None:\n"
        "-    assert f() == 2\n"
        f"+    assert f() == {expected}\n"
    )


def _helper_non_substantive_diff() -> str:
    return (
        "diff --git a/helper.py b/helper.py\n"
        "--- a/helper.py\n"
        "+++ b/helper.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value() -> int:\n"
        "     return 1\n"
    )


def _run_pytest_for_helper_fix(cwd, *args, **kwargs) -> run_eval.RunResult:
    del args, kwargs
    helper_text = (cwd / "helper.py").read_text(encoding="utf-8")
    fixed = "return 2" in helper_text
    return run_eval.RunResult(
        returncode=0 if fixed else 1,
        duration_sec=0.01,
        stdout=(
            "=== 1 passed in 0.01s ==="
            if fixed
            else "tests/test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n"
        ),
        stderr="",
    )


def _run_adaptive_delayed_credit_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    patch_outcomes: tuple[str, ...],
    adaptive_max_helper_calls: int,
    adaptive_budget_topup_max_calls: int = 1,
    k: int,
    run_id: str,
    extra_run_kwargs: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], _AdaptiveDelayedCreditPatcher, Path]:
    case = _make_case(tmp_path)
    workdir = tmp_path / f"{run_id}_workdir"
    context_dir = tmp_path / f"{run_id}_context"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDelayedCreditPatcher(patch_outcomes=patch_outcomes)
    controller, events_path = _invoke_biased_online_controller(tmp_path, mode="on")

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _run_pytest_for_helper_fix)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    run_kwargs: dict[str, object] = {}
    if extra_run_kwargs:
        run_kwargs.update(extra_run_kwargs)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=k,
        output_format="json_compact",
        run_id=run_id,
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_max_helper_calls=adaptive_max_helper_calls,
        adaptive_budget_topup_max_calls=adaptive_budget_topup_max_calls,
        adaptive_online_policy_controller=controller,
        adaptive_online_policy_mode="on",
        **run_kwargs,
    )
    return records, patcher, events_path


def _run_adaptive_verifier_case(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patcher: object,
    run_id: str,
    apply_result: tuple[bool, str] = (False, "simulated apply failure"),
    **run_one_condition_kwargs: object,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, int]]:
    case = _make_case(tmp_path)
    workdir = tmp_path / f"{run_id}_workdir"
    context_dir = tmp_path / f"{run_id}_context"
    shutil.copytree(case.path, workdir)

    run_calls = {"count": 0}
    apply_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_calls["count"] += 1
        return apply_result

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id=run_id,
        include_files_context=False,
        force_patch_attempt=False,
        **run_one_condition_kwargs,
    )
    return records, run_calls, apply_calls


def _run_adaptive_action_case(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patcher: object,
    preferred_action: str,
    run_id: str,
    k: int = 1,
    adaptive_helper_max_calls_per_case: int | None = None,
    snapshot_payload: dict[str, object] | None = None,
    include_snapshot: bool = True,
    extra_run_kwargs: dict[str, object] | None = None,
    decision_model: DecisionModelBinding | None = None,
    fallback_action: str | None = None,
    include_files_context: bool = False,
) -> tuple[list[dict[str, object]], Path]:
    case = _make_case(tmp_path)
    workdir = tmp_path / f"{run_id}_workdir"
    context_dir = tmp_path / f"{run_id}_context"
    shutil.copytree(case.path, workdir)
    controller, events_path = _preferred_action_online_controller(
        tmp_path, mode="on", preferred_action=preferred_action, fallback_action=fallback_action
    )
    controller.decision_model = decision_model

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
    effective_snapshot_payload = (
        (_snapshot_payload() if snapshot_payload is None else snapshot_payload)
        if include_snapshot
        else None
    )
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (effective_snapshot_payload, None),
    )

    run_kwargs: dict[str, object] = {}
    if adaptive_helper_max_calls_per_case is not None:
        run_kwargs["adaptive_max_helper_calls"] = adaptive_helper_max_calls_per_case
    if extra_run_kwargs:
        run_kwargs.update(extra_run_kwargs)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=k,
        output_format="json_compact",
        run_id=run_id,
        include_files_context=include_files_context,
        force_patch_attempt=False,
        adaptive_online_policy_controller=controller,
        adaptive_online_policy_mode="on",
        **run_kwargs,
    )
    return records, events_path


def test_layered_adaptive_gated_invokes_helper_after_repeated_signature(
    tmp_path, monkeypatch
) -> None:
    """Verifies that the gated adaptive condition invokes the Stage-A helper on the second attempt when the failure signature is unchanged."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_layered_adaptive_gated"
    context_dir = tmp_path / "context_layered_adaptive_gated"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher()

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-adaptive-gated",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_gate_policy="balanced",
        adaptive_stagnation_window=1,
    )

    assert len(records) == 2
    assert patcher.request_calls == 1
    first_attempt_meta = records[0]["attempt_metadata"]
    second_attempt_meta = records[1]["attempt_metadata"]
    assert first_attempt_meta["adaptive_helper_invoked"] is False
    assert second_attempt_meta["adaptive_helper_invoked"] is True
    assert first_attempt_meta["context_protocol"] == "monolithic"
    assert second_attempt_meta["context_protocol"] == "staged"
    assert records[1]["patch_meta"]["adaptive_helper_decision_reason"] in {
        "gate_balanced_repeated_signature",
        "gate_requested_helper",
    }
    assert len(patcher.prompts) == 2
    assert "Pytest output (stdout):" in patcher.prompts[1]
    assert "Stage-A evidence index:" not in patcher.prompts[1]
    assert second_attempt_meta["retry_delta_prompt"] is False


def test_adaptive_llm_discretion_decline_keeps_monolithic_attempt(tmp_path, monkeypatch) -> None:
    """Verifies that when the LLM declines additional context the attempt stays in monolithic mode and no staged context is materialised."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_decline"
    context_dir = tmp_path / "context_adaptive_discretion_decline"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionDeclinePatcher()

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-discretion-decline",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    assert patcher.request_calls == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_invoked"] is True
    assert attempt_meta["adaptive_helper_materialized_context"] is False
    assert attempt_meta["adaptive_helper_outcome"] == "semantic_no_context"
    assert attempt_meta["adaptive_helper_failure_kind"] == ""
    assert attempt_meta["adaptive_helper_decision_reason"] == "llm_declined_helper"
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 1
    assert attempt_meta["adaptive_helper_retry_count"] == 0
    assert attempt_meta["adaptive_helper_transport_retry_count"] == 0
    assert attempt_meta["context_protocol"] == "monolithic"
    assert records[0]["patch_meta"]["stage_a_request_success"] is True


@pytest.mark.parametrize(
    ("preferred_action", "expected_type"),
    [
        (ACTION_FRAME_ONLY, "frame"),
        (ACTION_FILE_ONLY, "file"),
        (ACTION_SEARCH_ONLY, "search"),
    ],
)
def test_adaptive_llm_discretion_restricted_actions_filter_stage_a_requests(
    tmp_path,
    monkeypatch,
    preferred_action: str,
    expected_type: str,
) -> None:
    """Verifies that frame-only, file-only, and search-only policy actions each filter Stage-A requests to their respective type."""
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "need_snapshot": True,
            "need_compact_context": True,
            "requests": [
                {"type": "file", "path": "buggy.py"},
                {"type": "search", "path": "buggy.py", "pattern": "helper.value"},
                {"type": "frame", "index": 0},
            ],
            "sufficiency_rationale": "need targeted evidence",
        }
    )

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=preferred_action,
        run_id=f"run-{preferred_action}",
    )

    attempt_meta = records[0]["attempt_metadata"]
    patch_meta = records[0]["patch_meta"]

    assert patcher.request_calls == 1
    assert attempt_meta["adaptive_helper_action_selected"] == preferred_action
    assert attempt_meta["adaptive_helper_action_executed"] == preferred_action
    assert attempt_meta["adaptive_helper_action_fallback_reason"] == ""
    assert attempt_meta["adaptive_helper_outcome"] == "materialized"
    assert patch_meta["stage_a_requests_count"] == 1
    assert patch_meta["stage_a_need_compact_context"] is False
    if expected_type == "frame":
        assert "Stage-A requested context:\n[e_" in patcher.patch_prompts[0]
        assert "frame:frame:0" in patcher.patch_prompts[0]
    elif expected_type == "file":
        assert "file:buggy.py" in patcher.patch_prompts[0]
    else:
        assert "search:buggy.py:helper.value" in patcher.patch_prompts[0]


def test_adaptive_llm_discretion_snapshot_lite_injects_snapshot_index_without_helper_budget(
    tmp_path, monkeypatch
) -> None:
    """Verifies that the snapshot-lite action injects a compact snapshot index into the patch prompt without consuming a helper call budget unit."""
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SNAPSHOT_LITE,
        run_id="run-snapshot-lite",
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert patcher.request_calls == 0
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_SNAPSHOT_LITE
    assert attempt_meta["adaptive_snapshot_lite_included"] is True
    assert attempt_meta["adaptive_helper_invoked"] is False
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 0
    assert attempt_meta["adaptive_helper_outcome"] == "materialized"
    assert attempt_meta["adaptive_lite_replaced_snapshot"] is True
    assert "Compact snapshot index:" in patcher.patch_prompts[0]
    assert "=== llmdebug Snapshot Index ===" in patcher.patch_prompts[0]
    # Under full reveal the lite section replaces the full debug snapshot (lite replaces).
    assert "=== llmdebug Debug Snapshot ===" not in patcher.patch_prompts[0]


@pytest.mark.parametrize(
    ("preferred_action", "expected_section"),
    (
        (ACTION_FRAME_LOCALS_LITE, "=== llmdebug Crash Frame ==="),
        (ACTION_SNAPSHOT_FULL_LITE, "=== llmdebug Debug Snapshot ==="),
    ),
)
def test_adaptive_llm_discretion_lite_snapshot_actions_credit_materialized_context(
    tmp_path, monkeypatch, preferred_action: str, expected_section: str
) -> None:
    """Policy lite snapshot actions should be credited as materialized context."""
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=preferred_action,
        run_id=f"run-{preferred_action}",
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert patcher.request_calls == 0
    assert attempt_meta["adaptive_helper_action_executed"] == preferred_action
    assert attempt_meta["adaptive_snapshot_lite_included"] is True
    assert attempt_meta["adaptive_helper_invoked"] is False
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 0
    assert attempt_meta["adaptive_helper_outcome"] == "materialized"
    assert attempt_meta["adaptive_lite_replaced_snapshot"] is True
    assert "Compact snapshot index:" in patcher.patch_prompts[0]
    assert expected_section in patcher.patch_prompts[0]
    # The full debug snapshot is replaced, so its header appears at most once (from the lite
    # section itself for snapshot_full_lite).
    expected_header_count = 1 if preferred_action == ACTION_SNAPSHOT_FULL_LITE else 0
    assert patcher.patch_prompts[0].count("=== llmdebug Debug Snapshot ===") == (
        expected_header_count
    )


def test_adaptive_llm_discretion_lite_snapshot_supplement_mode_keeps_full_snapshot(
    tmp_path, monkeypatch
) -> None:
    """The legacy supplement mode adds the lite section on top of the full snapshot."""
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_FRAME_LOCALS_LITE,
        run_id="run-frame-locals-lite-supplement",
        extra_run_kwargs={"adaptive_lite_snapshot_mode": "supplement"},
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_FRAME_LOCALS_LITE
    assert attempt_meta["adaptive_lite_replaced_snapshot"] is False
    assert "=== llmdebug Crash Frame ===" in patcher.patch_prompts[0]
    assert "=== llmdebug Debug Snapshot ===" in patcher.patch_prompts[0]


def test_adaptive_llm_discretion_lite_snapshot_traceback_first_never_replaces(
    tmp_path, monkeypatch
) -> None:
    """Replacement only applies under full reveal; traceback_first keeps its own reveal logic."""
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_FRAME_LOCALS_LITE,
        run_id="run-frame-locals-lite-traceback-first",
        extra_run_kwargs={"adaptive_reveal_mode": "traceback_first"},
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert attempt_meta["adaptive_lite_replaced_snapshot"] is False
    assert "=== llmdebug Crash Frame ===" in patcher.patch_prompts[0]


class _RecordingAdvisor:
    def __init__(self, advice: DecisionModelAdvice) -> None:
        self.advice = advice
        self.requests: list[Any] = []

    def advise(self, request: Any) -> DecisionModelAdvice:
        self.requests.append(request)
        return self.advice


def test_adaptive_llm_discretion_decision_model_policy_role_drives_helper_action(
    tmp_path, monkeypatch
) -> None:
    """The policy role overrides LinUCB's preference and logs to rows, events and states."""
    patcher = _AdaptiveActionCapturePatcher()
    advisor = _RecordingAdvisor(
        DecisionModelAdvice(
            status="ok",
            profile="evidence_v1",
            answers=ParsedAnswers(action_probs={ACTION_FILE_ONLY: 0.9, ACTION_SKIP_HELPER: 0.1}),
            state_sha256="sha",
            state_text='{"0_note":"state"}',
        )
    )
    states_path = tmp_path / "decision_model_states.jsonl"
    binding = DecisionModelBinding(
        config=DecisionModelPolicyConfig(role="policy"),
        advisor=advisor,
        state_sink=DecisionStateSink(states_path),
    )

    records, events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SKIP_HELPER,
        run_id="run-decision-model-policy",
        decision_model=binding,
    )

    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_FILE_ONLY
    assert attempt_meta["adaptive_online_decision_strategy"] == "decision_model_greedy"
    assert attempt_meta["adaptive_online_dm_role"] == "policy"
    assert attempt_meta["adaptive_online_dm_status"] == "ok"
    assert attempt_meta["adaptive_online_dm_top_action"] == ACTION_FILE_ONLY

    request = advisor.requests[0]
    assert request.state_inputs.attempt == 1
    assert request.state_inputs.reveal_mode == "full"
    assert ACTION_FILE_ONLY in request.available_actions
    assert "run-decision-model-policy" in request.state_inputs.redact_tokens
    built = build_decision_state(request.state_inputs, profile=get_profile("evidence_v1"))
    assert "run-decision-model-policy" not in built.rendered
    assert str(tmp_path) not in built.rendered

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["schema_version"] == ONLINE_EVENT_SCHEMA_VERSION  # 7 since POL-21
    assert events[0]["decision_model"]["role"] == "policy"
    states = [json.loads(line) for line in states_path.read_text(encoding="utf-8").splitlines()]
    assert states[0]["decision_id"] == attempt_meta["adaptive_online_decision_id"]
    assert states[0]["case_id"] == records[0]["case_id"]


def test_adaptive_llm_discretion_without_decision_model_keeps_dm_defaults(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SKIP_HELPER,
        run_id="run-no-decision-model",
    )

    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_online_decision_strategy"] == "linucb_greedy"
    assert attempt_meta["adaptive_online_dm_role"] == "off"
    assert attempt_meta["adaptive_online_dm_status"] == ""


def test_adaptive_llm_discretion_snapshot_lite_not_credited_when_budget_drops_it(
    tmp_path, monkeypatch
) -> None:
    """Ensures that the snapshot-lite action is recorded as not included when the prompt budget is too small to accommodate the snapshot index."""
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SNAPSHOT_LITE,
        run_id="run-snapshot-lite-budget-drop",
        extra_run_kwargs={
            "prompt_budget_total_chars": 300,
            "prompt_budget_snapshot_chars": 0,
        },
    )

    attempt_meta = records[0]["attempt_metadata"]
    patch_meta = records[0]["patch_meta"]

    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_SNAPSHOT_LITE
    assert attempt_meta["adaptive_snapshot_lite_included"] is False
    assert attempt_meta["adaptive_helper_outcome"] == "semantic_no_context"
    assert int(patch_meta["prompt_budget_used_chars"] or 0) <= 300
    assert "Compact snapshot index:" not in patcher.patch_prompts[0]


def test_adaptive_reveal_full_omits_snapshot_without_policy_grant(tmp_path, monkeypatch) -> None:
    """ATT-06 / D5: under full reveal the policy, not the reveal mode, decides the snapshot.

    This test previously asserted the full snapshot was always present while
    ``adaptive_snapshot_included`` said False; updated deliberately.
    """
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "need_snapshot": False,
            "requests": [{"type": "file", "path": "buggy.py"}],
            "sufficiency_rationale": "need helper implementation",
        }
    )

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_INVOKE_HELPER,
        run_id="run-adaptive-reveal-full",
        extra_run_kwargs={"adaptive_reveal_mode": "full"},
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert patcher.request_calls == 1
    assert attempt_meta["adaptive_snapshot_requested"] is False
    assert attempt_meta["stage_b_snapshot_in_prompt"] is False
    assert attempt_meta["adaptive_snapshot_included"] is False
    assert attempt_meta["adaptive_reveal_mode"] == "full"
    assert "Stage-A requested context:" in patcher.patch_prompts[0]
    assert "=== llmdebug Debug Snapshot ===" not in patcher.patch_prompts[0]
    assert attempt_meta["evidence_chars_by_type"]["snapshot"] == 0
    assert attempt_meta["evidence_chars_by_type"]["file"] > 0


def test_adaptive_reveal_full_includes_snapshot_when_stage_a_requests_it(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "need_snapshot": True,
            "snapshot_justification": "need the local value of x in f() to see the off-by-one",
            "requests": [{"type": "file", "path": "buggy.py"}],
            "sufficiency_rationale": "need helper implementation",
        }
    )

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_INVOKE_HELPER,
        run_id="run-adaptive-reveal-full-granted",
        extra_run_kwargs={"adaptive_reveal_mode": "full"},
    )

    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_snapshot_requested"] is True
    assert attempt_meta["stage_b_snapshot_in_prompt"] is True
    assert attempt_meta["adaptive_snapshot_included"] is True
    assert "=== llmdebug Debug Snapshot ===" in patcher.patch_prompts[0]
    assert attempt_meta["evidence_chars_by_type"]["snapshot"] > 0


def test_adaptive_reveal_traceback_first_uses_snapshot_index_without_full_snapshot(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SKIP_HELPER,
        run_id="run-adaptive-reveal-traceback-first",
        extra_run_kwargs={"adaptive_reveal_mode": "traceback_first"},
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert patcher.request_calls == 0
    assert attempt_meta["adaptive_reveal_mode"] == "traceback_first"
    assert attempt_meta["adaptive_reveal_applied"] is True
    assert attempt_meta["adaptive_reveal_trigger"] == "traceback_only"
    assert "Compact snapshot index:" in patcher.patch_prompts[0]
    assert "=== llmdebug Snapshot Index ===" in patcher.patch_prompts[0]
    assert "=== llmdebug Debug Snapshot ===" not in patcher.patch_prompts[0]


def test_adaptive_llm_discretion_snapshot_lite_masks_unavailable_snapshot_action(
    tmp_path, monkeypatch
) -> None:
    """Ensures that snapshot-lite is removed from available actions and the policy falls back to the next-best arm (invoke-helper) when no snapshot is present."""
    patcher = _AdaptiveActionCapturePatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SNAPSHOT_LITE,
        run_id="run-snapshot-lite-fallback",
        fallback_action=ACTION_INVOKE_HELPER,
        include_snapshot=False,
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert attempt_meta["adaptive_helper_action_selected"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_helper_action_fallback_reason"] == ""
    assert ACTION_SNAPSHOT_LITE not in attempt_meta["adaptive_helper_available_actions"]
    assert (
        attempt_meta["adaptive_helper_unavailable_action_reasons"][ACTION_SNAPSHOT_LITE]
        == "snapshot_unavailable"
    )
    assert attempt_meta["adaptive_snapshot_lite_included"] is False
    assert patcher.request_calls == 1


def test_adaptive_llm_discretion_compact_then_patch_uses_compaction_budget_only(
    tmp_path, monkeypatch
) -> None:
    """Verifies that the compact-then-patch action invokes context compaction and charges only the compaction budget, not the helper call budget."""
    patcher = _AdaptiveActionCapturePatcher(
        compacted_text="crash in buggy.py line 4; helper.value() returns 1 but test expects 2",
        compact_meta={
            "llm_call_count": 1,
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            "compact_context_duration_sec": 0.3,
        },
    )

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_COMPACT_THEN_PATCH,
        run_id="run-compact-then-patch",
    )

    attempt_meta = records[0]["attempt_metadata"]
    patch_meta = records[0]["patch_meta"]

    assert patcher.request_calls == 0
    assert patcher.compact_calls == 1
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_COMPACT_THEN_PATCH
    assert attempt_meta["adaptive_helper_invoked"] is False
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 0
    assert attempt_meta["context_compaction_invoked"] is True
    assert attempt_meta["context_compaction_trigger"] == "policy_action"
    assert attempt_meta["adaptive_helper_outcome"] == "materialized"
    assert attempt_meta["adaptive_online_local_tokens_total"] == 15
    assert patch_meta["context_compaction_success"] is True
    assert "Compacted context summary:" in patcher.patch_prompts[-1]


def test_adaptive_llm_discretion_compact_then_patch_supports_legacy_patcher_subclass(
    tmp_path, monkeypatch
) -> None:
    @dataclass
    class _LegacyCompactionPatcher(Patcher):
        name: str = "legacy_compaction_stage_b"
        compact_calls: int = 0
        patch_prompts: list[str] = field(default_factory=list)

        def compact_context(
            self,
            *,
            prompt: str,
            context_text: str,
            context_dir,
            cwd,
            target_chars: int,
        ) -> ContextCompactionResult:
            del prompt, context_text, context_dir, cwd, target_chars
            self.compact_calls += 1
            return ContextCompactionResult(
                compacted_text="legacy summary: helper.value() still returns 1",
                raw_response="legacy summary: helper.value() still returns 1",
                meta={"compact_context_duration_sec": 0.2},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del context_dir, cwd
            self.patch_prompts.append(prompt)
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _LegacyCompactionPatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_COMPACT_THEN_PATCH,
        run_id="run-legacy-compact-then-patch",
    )

    attempt_meta = records[0]["attempt_metadata"]
    patch_meta = records[0]["patch_meta"]

    assert patcher.compact_calls == 1
    assert attempt_meta["adaptive_helper_action_selected"] == ACTION_COMPACT_THEN_PATCH
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_COMPACT_THEN_PATCH
    assert attempt_meta["context_compaction_invoked"] is True
    assert patch_meta["context_compaction_success"] is True
    assert "Compacted context summary:" in patcher.patch_prompts[-1]


def test_context_compaction_policy_action_respects_mode_off() -> None:
    """Verifies that context compaction is skipped with reason 'mode_off' when compaction mode is explicitly disabled, regardless of other flags."""
    should_compact, reason = run_eval._should_invoke_context_compaction(
        mode="off",
        supports_context_compaction=True,
        stage_a_requested=False,
        policy_action_requested=True,
        auto_overflow_triggered=True,
        calls_used=0,
        max_calls=1,
        bundle_input_chars=128,
    )

    assert should_compact is False
    assert reason == "mode_off"


def test_adaptive_llm_discretion_compact_then_patch_masks_disabled_compaction_action(
    tmp_path, monkeypatch
) -> None:
    """Ensures that compact-then-patch is excluded from available actions and the policy falls back to the next-best arm (invoke-helper) when compaction mode is 'off'."""
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "requests": [{"type": "file", "path": "helper.py"}],
            "sufficiency_rationale": "need helper implementation",
        }
    )

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_COMPACT_THEN_PATCH,
        run_id="run-compact-then-patch-mode-off",
        fallback_action=ACTION_INVOKE_HELPER,
        extra_run_kwargs={"context_compaction_mode": "off"},
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert attempt_meta["adaptive_helper_action_selected"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_helper_action_fallback_reason"] == ""
    assert ACTION_COMPACT_THEN_PATCH not in attempt_meta["adaptive_helper_available_actions"]
    assert (
        attempt_meta["adaptive_helper_unavailable_action_reasons"][ACTION_COMPACT_THEN_PATCH]
        == "context_compaction_disabled"
    )
    assert attempt_meta["context_compaction_invoked"] is False
    assert patcher.compact_calls == 0
    assert patcher.request_calls == 1


def test_adaptive_llm_discretion_compact_then_patch_masks_unsupported_compaction_action(
    tmp_path, monkeypatch
) -> None:
    """Ensures that compact-then-patch is excluded from available actions when the patcher does not implement compact_context."""

    @dataclass
    class _NoCompactionPatcher:
        name: str = "openai"
        request_calls: int = 0

        def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
            del prompt, context_dir, cwd
            self.request_calls += 1
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "helper.py"}],
                    "sufficiency_rationale": "need helper implementation",
                },
                raw_response='{"need_more_context":true}',
                meta={},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _NoCompactionPatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_COMPACT_THEN_PATCH,
        run_id="run-compact-then-patch-fallback",
        fallback_action=ACTION_INVOKE_HELPER,
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert attempt_meta["adaptive_helper_action_selected"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_helper_action_fallback_reason"] == ""
    assert ACTION_COMPACT_THEN_PATCH not in attempt_meta["adaptive_helper_available_actions"]
    assert (
        attempt_meta["adaptive_helper_unavailable_action_reasons"][ACTION_COMPACT_THEN_PATCH]
        == "context_compaction_unsupported"
    )
    assert attempt_meta["context_compaction_invoked"] is False
    assert patcher.request_calls == 1


def test_adaptive_llm_discretion_retry_helper_strict_json_masks_until_prior_invalid_payload(
    tmp_path, monkeypatch
) -> None:
    """Verifies that the strict-JSON retry action is unavailable on the first attempt and becomes available after an invalid-payload failure."""

    @dataclass
    class _StrictJsonRetryPatcher:
        name: str = "openai"
        request_calls: int = 0
        request_kwargs: list[dict[str, object]] = field(default_factory=list)

        def request_context(
            self,
            *,
            prompt: str,
            context_dir,
            cwd,
            strict_json_first: bool = False,
        ) -> ContextRequestResult:
            del prompt, context_dir, cwd
            self.request_calls += 1
            self.request_kwargs.append({"strict_json_first": bool(strict_json_first)})
            if self.request_calls == 1:
                return ContextRequestResult(
                    request_json=None, raw_response="{invalid-json", meta={}
                )
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "helper.py"}],
                    "sufficiency_rationale": "need helper implementation",
                },
                raw_response='{"need_more_context":true}',
                meta={},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _StrictJsonRetryPatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_RETRY_HELPER_STRICT_JSON,
        run_id="run-retry-helper-strict-json",
        fallback_action=ACTION_INVOKE_HELPER,
        k=2,
        adaptive_helper_max_calls_per_case=2,
    )

    first_attempt = records[0]["attempt_metadata"]
    second_attempt = records[1]["attempt_metadata"]

    assert patcher.request_calls == 2
    assert first_attempt["adaptive_helper_action_selected"] == ACTION_INVOKE_HELPER
    assert first_attempt["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert first_attempt["adaptive_helper_action_fallback_reason"] == ""
    assert ACTION_RETRY_HELPER_STRICT_JSON not in first_attempt["adaptive_helper_available_actions"]
    assert (
        first_attempt["adaptive_helper_unavailable_action_reasons"][ACTION_RETRY_HELPER_STRICT_JSON]
        == "strict_json_requires_prior_invalid_payload"
    )
    assert patcher.request_kwargs[0]["strict_json_first"] is False
    assert second_attempt["adaptive_helper_action_executed"] == ACTION_RETRY_HELPER_STRICT_JSON
    assert patcher.request_kwargs[1]["strict_json_first"] is True
    assert second_attempt["adaptive_helper_outcome"] == "materialized"


def test_adaptive_llm_discretion_retry_helper_strict_json_supports_non_openai_stage_b(
    tmp_path, monkeypatch
) -> None:
    @dataclass
    class _SupportedNonOpenAIPatcher(Patcher):
        name: str = "custom_stage_b"
        request_calls: int = 0
        request_kwargs: list[dict[str, object]] = field(default_factory=list)

        def supports_request_context(self) -> bool:
            return True

        def supports_strict_json_first(self) -> bool:
            return True

        def request_context(
            self,
            *,
            prompt: str,
            context_dir,
            cwd,
            strict_json_first: bool = False,
        ) -> ContextRequestResult:
            del prompt, context_dir, cwd
            self.request_calls += 1
            self.request_kwargs.append({"strict_json_first": bool(strict_json_first)})
            if self.request_calls == 1:
                return ContextRequestResult(
                    request_json=None,
                    raw_response="{invalid-json",
                    meta={},
                )
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "helper.py"}],
                    "sufficiency_rationale": "need helper implementation",
                },
                raw_response='{"need_more_context":true}',
                meta={},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _SupportedNonOpenAIPatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_RETRY_HELPER_STRICT_JSON,
        run_id="run-retry-helper-strict-json-non-openai",
        fallback_action=ACTION_INVOKE_HELPER,
        k=2,
        adaptive_helper_max_calls_per_case=2,
    )

    first_attempt = records[0]["attempt_metadata"]
    second_attempt = records[1]["attempt_metadata"]

    assert patcher.request_calls == 2
    assert first_attempt["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert second_attempt["adaptive_helper_action_executed"] == ACTION_RETRY_HELPER_STRICT_JSON
    assert patcher.request_kwargs == [
        {"strict_json_first": False},
        {"strict_json_first": True},
    ]
    assert second_attempt["adaptive_helper_outcome"] == "materialized"


def test_adaptive_llm_discretion_retry_helper_strict_json_supports_legacy_patcher_subclass(
    tmp_path, monkeypatch
) -> None:
    @dataclass
    class _LegacyStrictJsonPatcher(Patcher):
        name: str = "legacy_custom_stage_b"
        request_calls: int = 0
        request_kwargs: list[dict[str, object]] = field(default_factory=list)

        def request_context(
            self,
            *,
            prompt: str,
            context_dir,
            cwd,
            strict_json_first: bool = False,
        ) -> ContextRequestResult:
            del prompt, context_dir, cwd
            self.request_calls += 1
            self.request_kwargs.append({"strict_json_first": bool(strict_json_first)})
            if self.request_calls == 1:
                return ContextRequestResult(
                    request_json=None,
                    raw_response="{invalid-json",
                    meta={},
                )
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "helper.py"}],
                    "sufficiency_rationale": "need helper implementation",
                },
                raw_response='{"need_more_context":true}',
                meta={},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _LegacyStrictJsonPatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_RETRY_HELPER_STRICT_JSON,
        run_id="run-legacy-strict-json-retry",
        fallback_action=ACTION_INVOKE_HELPER,
        k=2,
        adaptive_helper_max_calls_per_case=2,
    )

    first_attempt = records[0]["attempt_metadata"]
    second_attempt = records[1]["attempt_metadata"]

    assert patcher.request_calls == 2
    assert first_attempt["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert second_attempt["adaptive_helper_action_executed"] == ACTION_RETRY_HELPER_STRICT_JSON
    assert patcher.request_kwargs == [
        {"strict_json_first": False},
        {"strict_json_first": True},
    ]
    assert second_attempt["adaptive_helper_outcome"] == "materialized"


def test_adaptive_llm_discretion_masks_stage_a_actions_when_request_context_unsupported(
    tmp_path, monkeypatch
) -> None:
    """Ensures that all Stage-A actions are masked and skip-helper is selected when the patcher does not implement request_context."""

    @dataclass
    class _NoRequestContextPatcher:
        name: str = "unsupported_context"

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=_NoRequestContextPatcher(),
        preferred_action=ACTION_FILE_ONLY,
        run_id="run-file-only-fallback",
        fallback_action=ACTION_SKIP_HELPER,
    )

    attempt_meta = records[0]["attempt_metadata"]

    assert attempt_meta["adaptive_helper_action_selected"] == ACTION_SKIP_HELPER
    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_SKIP_HELPER
    assert attempt_meta["adaptive_helper_action_fallback_reason"] == ""
    assert ACTION_FILE_ONLY not in attempt_meta["adaptive_helper_available_actions"]
    assert (
        attempt_meta["adaptive_helper_unavailable_action_reasons"][ACTION_FILE_ONLY]
        == "request_context_unsupported"
    )


def test_inherited_noop_request_context_is_treated_as_unsupported(tmp_path, monkeypatch) -> None:
    class _InheritedNoRequestContextPatcher(Patcher):
        name = "inherited_no_request_context"

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=_InheritedNoRequestContextPatcher(),
        preferred_action=ACTION_FILE_ONLY,
        run_id="run-inherited-request-context-noop",
        fallback_action=ACTION_SKIP_HELPER,
    )

    attempt_meta = records[0]["attempt_metadata"]
    patch_meta = records[0]["patch_meta"]

    assert attempt_meta["adaptive_helper_action_executed"] == ACTION_SKIP_HELPER
    assert (
        attempt_meta["adaptive_helper_unavailable_action_reasons"][ACTION_FILE_ONLY]
        == "request_context_unsupported"
    )
    assert attempt_meta["adaptive_helper_decision_reason"] == "request_context_unavailable"
    assert attempt_meta["adaptive_helper_outcome"] == "skipped_unsupported"
    assert attempt_meta["adaptive_helper_failure_kind"] == ""
    assert patch_meta["adaptive_helper_outcome"] == "skipped_unsupported"
    assert patch_meta["adaptive_helper_failure_kind"] == ""
    assert patch_meta.get("stage_a_reason") != "helper_invalid_payload"


def test_restricted_helper_actions_apply_action_specific_mini_budgets() -> None:
    """Verifies that frame-only and search-only filtering apply per-action item caps and strip disallowed request types from the payload."""
    frame_payload = _filter_context_request_payload_for_action(
        action=ACTION_FRAME_ONLY,
        request_payload={
            "need_more_context": True,
            "requests": [
                {"type": "frame", "index": 0},
                {"type": "local_var", "frame_index": 0, "name": "first"},
                {"type": "local_var", "frame_index": 0, "name": "second"},
                {"type": "local_var", "frame_index": 0, "name": "third"},
                {"type": "search", "path": "buggy.py", "pattern": "value"},
            ],
        },
    )
    search_payload = _filter_context_request_payload_for_action(
        action=ACTION_SEARCH_ONLY,
        request_payload={
            "need_more_context": True,
            "requests": [
                {"type": "search", "path": "a.py", "pattern": "alpha"},
                {"type": "search", "path": "b.py", "pattern": "beta"},
                {"type": "search", "path": "c.py", "pattern": "gamma"},
                {"type": "file", "path": "ignored.py"},
            ],
        },
    )

    assert frame_payload is not None
    assert [entry["type"] for entry in frame_payload["requests"]] == [
        "frame",
        "local_var",
        "local_var",
    ]
    assert frame_payload["need_snapshot"] is False
    assert frame_payload["need_compact_context"] is False

    assert search_payload is not None
    assert [entry["type"] for entry in search_payload["requests"]] == ["search", "search"]


@pytest.mark.parametrize("override_mode", ["diff", "full_file"])
def test_adaptive_openai_helper_response_mode_override(
    tmp_path, monkeypatch, override_mode: str
) -> None:
    """Verifies that adaptive_openai_helper_response_mode overrides the patcher's default response mode for the patch proposal call."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_openai_override"
    context_dir = tmp_path / "context_adaptive_openai_override"
    shutil.copytree(case.path, workdir)

    captured_response_modes: list[str] = []

    def _fake_request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del self, prompt, context_dir, cwd
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need source",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def _fake_propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        captured_response_modes.append(str(self.response_mode))
        return PatchResult(
            diff=None, meta={"reason": "no_diff", "response_mode": self.response_mode}
        )

    monkeypatch.setattr(OpenAICompatPatcher, "request_context", _fake_request_context)
    monkeypatch.setattr(OpenAICompatPatcher, "propose_patch", _fake_propose_patch)
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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-openai-override",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_openai_helper_response_mode=override_mode,
    )

    assert len(records) == 1
    assert captured_response_modes == [override_mode]
    assert records[0]["patch_meta"]["adaptive_openai_response_mode_override"] == override_mode
    assert records[0]["model_metadata"]["response_mode"] == override_mode


def test_adaptive_llm_discretion_invalid_response_counts_as_consult_without_context(
    tmp_path, monkeypatch
) -> None:
    """Verifies that a malformed Stage-A response counts as a consumed helper call but does not inject any context into the patch prompt."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_invalid"
    context_dir = tmp_path / "context_adaptive_discretion_invalid"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionInvalidPatcher()

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-discretion-invalid",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    assert patcher.request_calls == 1
    assert len(patcher.prompts) == 1
    assert "Stage-A requested context:" not in patcher.prompts[0]
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_invoked"] is True
    assert attempt_meta["adaptive_helper_materialized_context"] is False
    assert attempt_meta["adaptive_helper_outcome"] == "invalid_payload"
    assert attempt_meta["adaptive_helper_failure_kind"] == "invalid_payload"
    assert attempt_meta["adaptive_helper_decision_reason"] == "llm_invalid_response_no_helper"
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 1
    assert attempt_meta["adaptive_helper_retry_count"] == 0
    assert attempt_meta["adaptive_helper_transport_retry_count"] == 0
    assert attempt_meta["context_protocol"] == "monolithic"
    assert records[0]["patch_meta"]["stage_a_request_success"] is False
    assert records[0]["patch_meta"]["stage_a_reason"] == "helper_invalid_payload"


def test_adaptive_llm_discretion_attempt1_snapshot_denial_syncs_helper_reason(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_snapshot_denial"
    context_dir = tmp_path / "context_adaptive_discretion_snapshot_denial"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionSnapshotGenericPatcher()

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-discretion-snapshot-denial",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    assert patcher.request_calls == 1
    attempt_meta = records[0]["attempt_metadata"]
    patch_meta = records[0]["patch_meta"]
    assert attempt_meta["adaptive_snapshot_attempt1_denied"] is True
    assert attempt_meta["adaptive_snapshot_included"] is False
    assert attempt_meta["adaptive_helper_invoked"] is True
    assert attempt_meta["adaptive_helper_materialized_context"] is False
    assert attempt_meta["adaptive_helper_outcome"] == "semantic_no_context"
    assert attempt_meta["adaptive_helper_decision_reason"] == "llm_snapshot_request_denied_attempt1"
    assert patch_meta["adaptive_helper_decision_reason"] == "llm_snapshot_request_denied_attempt1"
    assert attempt_meta["context_protocol"] == "monolithic"


def test_adaptive_llm_discretion_backend_unavailable_does_not_spend_helper_budget(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_transport_failure"
    context_dir = tmp_path / "context_adaptive_discretion_transport_failure"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionBackendUnavailablePatcher()

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-discretion-transport-failure",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_invoked"] is True
    assert attempt_meta["adaptive_helper_materialized_context"] is False
    assert attempt_meta["adaptive_helper_outcome"] == "unavailable_backend"
    assert attempt_meta["adaptive_helper_failure_kind"] == "backend_unavailable"
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 0
    assert attempt_meta["adaptive_helper_retry_count"] == 0
    assert attempt_meta["adaptive_helper_transport_retry_count"] == 0
    assert attempt_meta["context_protocol"] == "monolithic"
    assert records[0]["patch_meta"]["stage_a_reason"] == "helper_unavailable_backend"


def test_adaptive_llm_discretion_transport_failure_does_not_spend_helper_budget(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_transport_failure_retryable"
    context_dir = tmp_path / "context_adaptive_discretion_transport_failure_retryable"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionTransportFailurePatcher()

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-discretion-transport-failure-retryable",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_invoked"] is True
    assert attempt_meta["adaptive_helper_materialized_context"] is False
    assert attempt_meta["adaptive_helper_outcome"] == "transport_failed"
    assert attempt_meta["adaptive_helper_failure_kind"] == "transport"
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 0
    assert attempt_meta["adaptive_helper_retry_count"] == 1
    assert attempt_meta["adaptive_helper_transport_retry_count"] == 1
    assert attempt_meta["context_protocol"] == "monolithic"
    assert records[0]["patch_meta"]["stage_a_reason"] == "helper_transport_failed"


def test_adaptive_llm_discretion_compat_reasoning_only_does_not_spend_helper_budget(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_compat_reasoning_only"
    context_dir = tmp_path / "context_adaptive_discretion_compat_reasoning_only"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionCompatReasoningOnlyPatcher()

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-discretion-compat-reasoning-only",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_invoked"] is True
    assert attempt_meta["adaptive_helper_materialized_context"] is False
    assert attempt_meta["adaptive_helper_outcome"] == "compat_reasoning_only"
    assert attempt_meta["adaptive_helper_failure_kind"] == "compat_reasoning_only"
    assert attempt_meta["adaptive_helper_decision_reason"] != "llm_declined_helper"
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 0
    assert attempt_meta["adaptive_helper_retry_count"] == 0
    assert attempt_meta["adaptive_helper_transport_retry_count"] == 0
    assert attempt_meta["context_protocol"] == "monolithic"
    assert records[0]["patch_meta"]["stage_a_reason"] == "helper_compat_reasoning_only"


def test_adaptive_llm_discretion_caps_helper_calls_and_stops_on_no_progress(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_cap"
    context_dir = tmp_path / "context_adaptive_discretion_cap"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher(name="adaptive_discretion_cap")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=5,
        output_format="json_compact",
        run_id="run-adaptive-discretion-cap",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
        adaptive_budget_topup_max_calls=0,
    )

    assert len(records) == 3
    assert patcher.request_calls == 1
    assert len(patcher.prompts) == 3
    assert records[1]["attempt_metadata"]["adaptive_helper_budget_exhausted"] is True
    assert records[2]["attempt_metadata"]["adaptive_progress_stop_triggered"] is True
    assert records[2]["patch_meta"]["adaptive_progress_stop_reason"] == "helper_budget_exhausted"
    assert records[2]["note"] == "no_patch_proposed"


def test_adaptive_llm_discretion_grants_one_budget_topup(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_discretion_topup"
    context_dir = tmp_path / "context_adaptive_discretion_topup"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher(name="adaptive_discretion_topup")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=4,
        output_format="json_compact",
        run_id="run-adaptive-discretion-topup",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
        adaptive_budget_topup_max_calls=1,
    )

    assert len(records) == 4
    assert patcher.request_calls == 2
    topup_attempts = [
        record
        for record in records
        if bool(record["attempt_metadata"]["adaptive_budget_topup_granted"])
    ]
    assert len(topup_attempts) == 1
    assert (
        topup_attempts[0]["attempt_metadata"]["adaptive_budget_topup_reason"]
        == "no_progress_recovery_once"
    )
    assert records[-1]["attempt_metadata"]["adaptive_helper_budget_exhausted"] is True
    assert records[-1]["attempt_metadata"]["adaptive_progress_stop_triggered"] is True


def test_adaptive_llm_discretion_invalid_payload_topup_forces_strict_retry(
    tmp_path, monkeypatch
) -> None:
    @dataclass
    class _InvalidPayloadRecoveryPatcher(Patcher):
        name: str = "invalid_payload_recovery"
        request_calls: int = 0
        request_kwargs: list[dict[str, object]] = field(default_factory=list)

        def supports_request_context(self) -> bool:
            return True

        def supports_strict_json_first(self) -> bool:
            return True

        def request_context(
            self,
            *,
            prompt: str,
            context_dir,
            cwd,
            strict_json_first: bool = False,
        ) -> ContextRequestResult:
            del prompt, context_dir, cwd
            self.request_calls += 1
            self.request_kwargs.append({"strict_json_first": bool(strict_json_first)})
            if self.request_calls == 1:
                return ContextRequestResult(
                    request_json=None,
                    raw_response="{invalid-json",
                    meta={},
                )
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "helper.py"}],
                    "sufficiency_rationale": "need helper implementation",
                },
                raw_response='{"need_more_context":true}',
                meta={},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _InvalidPayloadRecoveryPatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_FILE_ONLY,
        run_id="run-invalid-payload-topup",
        k=2,
        adaptive_helper_max_calls_per_case=1,
        extra_run_kwargs={"adaptive_budget_topup_max_calls": 1},
    )

    first_attempt = records[0]["attempt_metadata"]
    second_attempt = records[1]["attempt_metadata"]

    assert first_attempt["adaptive_helper_failure_kind"] == "invalid_payload"
    assert patcher.request_calls == 2
    assert second_attempt["adaptive_budget_topup_granted"] is True
    assert second_attempt["adaptive_budget_topup_reason"] == "invalid_payload_recovery_once"
    assert second_attempt["adaptive_helper_action_executed"] == ACTION_RETRY_HELPER_STRICT_JSON
    assert second_attempt["adaptive_online_override_reason"] == (
        "constraint_invalid_payload_force_strict_retry"
    )
    assert patcher.request_kwargs == [
        {"strict_json_first": False},
        {"strict_json_first": True},
    ]


def test_adaptive_llm_discretion_no_progress_topup_forces_helper_invoke(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "requests": [{"type": "file", "path": "helper.py"}],
            "sufficiency_rationale": "need helper implementation",
        }
    )

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_FILE_ONLY,
        run_id="run-no-progress-topup",
        k=3,
        adaptive_helper_max_calls_per_case=2,
        extra_run_kwargs={"adaptive_budget_topup_max_calls": 1},
    )

    second_attempt = records[1]["attempt_metadata"]
    third_attempt = records[2]["attempt_metadata"]

    assert second_attempt["adaptive_progress_reason"] == "no_progress_v2"
    assert patcher.request_calls == 3
    assert third_attempt["adaptive_budget_topup_granted"] is True
    assert third_attempt["adaptive_budget_topup_reason"] == "no_progress_recovery_once"
    assert third_attempt["adaptive_helper_action_selected"] == ACTION_INVOKE_HELPER
    assert third_attempt["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert third_attempt["adaptive_online_override_reason"] == "constraint_no_progress_force_helper"


def test_adaptive_llm_discretion_repair_failure_topup_forces_helper_invoke(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_repair_failure_topup"
    context_dir = tmp_path / "context_adaptive_repair_failure_topup"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveTestEditPatcher()
    controller, _events_path = _preferred_action_online_controller(
        tmp_path,
        mode="on",
        preferred_action=ACTION_FILE_ONLY,
    )

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-adaptive-repair-failure-topup",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
        adaptive_budget_topup_max_calls=1,
        adaptive_online_policy_controller=controller,
        adaptive_online_policy_mode="on",
        patch_verifier_scope=run_eval.PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
    )

    second_attempt = records[1]["attempt_metadata"]

    assert records[0]["note"] == "patch_verifier_rejected"
    assert patcher.request_calls == 2
    assert second_attempt["adaptive_budget_topup_granted"] is True
    assert second_attempt["adaptive_budget_topup_reason"] == "repair_failure_recovery_once"
    assert second_attempt["adaptive_helper_action_selected"] == ACTION_INVOKE_HELPER
    assert second_attempt["adaptive_helper_action_executed"] == ACTION_INVOKE_HELPER
    assert second_attempt["adaptive_online_override_reason"] == (
        "constraint_repair_failure_force_helper"
    )


def test_adaptive_llm_discretion_topup_does_not_fire_when_pregate_skips(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_topup_pregate_skip"
    context_dir = tmp_path / "context_adaptive_topup_pregate_skip"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "requests": [{"type": "file", "path": "helper.py"}],
            "sufficiency_rationale": "need helper implementation",
        }
    )
    controller, _events_path = _preferred_action_online_controller(
        tmp_path,
        mode="on",
        preferred_action=ACTION_FILE_ONLY,
    )
    snapshots = iter(
        [
            _snapshot_payload(),
            {
                "exception": {"type": "SyntaxError", "message": "invalid syntax"},
                "frames": [],
            },
        ]
    )

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
        lambda *_args, **_kwargs: (next(snapshots), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-adaptive-topup-pregate-skip",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
        adaptive_budget_topup_max_calls=1,
        adaptive_online_policy_controller=controller,
        adaptive_online_policy_mode="on",
        adaptive_informativeness_pregate=True,
    )

    second_attempt = records[1]["attempt_metadata"]

    assert patcher.request_calls == 1
    assert second_attempt["adaptive_informativeness_pregate_fired"] is True
    assert second_attempt["adaptive_budget_topup_granted"] is False
    assert second_attempt["adaptive_helper_decision_reason"] == "informativeness_pregate_skip"


def test_adaptive_llm_discretion_topup_does_not_fire_when_saturation_guard_blocks(
    tmp_path, monkeypatch
) -> None:
    del tmp_path, monkeypatch

    recovery = eval_attempt_flow._decide_adaptive_helper_recovery_topup(
        is_adaptive_condition=True,
        helper_budget_exhausted=True,
        adaptive_budget_unlimited=False,
        adaptive_budget_topup_calls_used=0,
        adaptive_budget_topup_max_calls=1,
        pregate_should_skip=False,
        helper_saturation_guarded=True,
        supports_context_request=True,
        supports_strict_json_first=True,
        previous_note="retry_gate_no_new_evidence",
        previous_helper_failure_kind="",
        previous_progress_reason="no_progress_v2",
        previous_helper_need_more_context=True,
        last_helper_need_more_context=True,
        failure_signature_repeat=True,
        adaptive_no_progress_streak=1,
    )

    assert recovery == ("", "", "", "")


def test_adaptive_llm_discretion_topup_does_not_refire_after_single_use(
    tmp_path, monkeypatch
) -> None:
    @dataclass
    class _SingleTopupPatcher(Patcher):
        name: str = "single_topup_reuse_guard"
        request_calls: int = 0
        request_kwargs: list[dict[str, object]] = field(default_factory=list)

        def supports_request_context(self) -> bool:
            return True

        def supports_strict_json_first(self) -> bool:
            return True

        def request_context(
            self,
            *,
            prompt: str,
            context_dir,
            cwd,
            strict_json_first: bool = False,
        ) -> ContextRequestResult:
            del prompt, context_dir, cwd
            self.request_calls += 1
            self.request_kwargs.append({"strict_json_first": bool(strict_json_first)})
            if self.request_calls == 1:
                return ContextRequestResult(
                    request_json=None,
                    raw_response="{invalid-json",
                    meta={},
                )
            return ContextRequestResult(
                request_json={
                    "need_more_context": True,
                    "requests": [{"type": "file", "path": "helper.py"}],
                    "sufficiency_rationale": "need helper implementation",
                },
                raw_response='{"need_more_context":true}',
                meta={},
            )

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _SingleTopupPatcher()

    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_FILE_ONLY,
        run_id="run-single-topup-reuse-guard",
        k=3,
        adaptive_helper_max_calls_per_case=1,
        extra_run_kwargs={"adaptive_budget_topup_max_calls": 1},
    )

    second_attempt = records[1]["attempt_metadata"]
    third_attempt = records[2]["attempt_metadata"]

    assert patcher.request_calls == 2
    assert second_attempt["adaptive_budget_topup_granted"] is True
    assert second_attempt["adaptive_budget_topup_reason"] == "invalid_payload_recovery_once"
    assert third_attempt["adaptive_budget_topup_granted"] is False
    assert third_attempt["adaptive_helper_budget_exhausted"] is True


def test_snapshot_justification_generic_detector_handles_variants() -> None:
    assert run_eval._is_generic_snapshot_justification("Need more context.")
    assert run_eval._is_generic_snapshot_justification("Need the snapshot, please!")
    assert not run_eval._is_generic_snapshot_justification(
        "Need snapshot to inspect locals in frame 0 for None-path divergence."
    )


def test_adaptive_reintroduces_stage_a_index_when_context_is_truncated(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_index_fallback"
    context_dir = tmp_path / "context_adaptive_index_fallback"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher(name="adaptive_index_fallback")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-index-fallback",
        include_files_context=False,
        force_patch_attempt=False,
        prompt_budget_total_chars=900,
        prompt_budget_evidence_chars=80,
    )

    assert len(records) == 1
    assert patcher.request_calls == 1
    assert len(patcher.prompts) == 1
    assert records[0]["attempt_metadata"]["stage_a_index_included"] is True
    assert records[0]["attempt_metadata"]["stage_a_index_reason"] == "context_truncated_fallback"


def test_adaptive_postpatch_recompute_avoids_premature_stop_on_signature_change(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_postpatch_recompute"
    context_dir = tmp_path / "context_adaptive_postpatch_recompute"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveToggleDiffPatcher()

    run_outputs = [
        "sig_a",  # initial run
        "sig_a",  # after attempt 1
        "sig_a",  # after attempt 2
        "sig_b",  # after attempt 3 (changed)
        "sig_b",  # after attempt 4
    ]
    call_counter = {"idx": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        idx = call_counter["idx"]
        call_counter["idx"] += 1
        signature = run_outputs[min(idx, len(run_outputs) - 1)]
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout=f"{signature}\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )
    monkeypatch.setattr(
        eval_attempt_flow,
        "_build_failure_signature_for_attempt",
        lambda *, run, snapshot: (run.stdout or "").strip(),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=4,
        output_format="json_compact",
        run_id="run-adaptive-postpatch-recompute",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
    )

    assert len(records) == 4
    third_attempt = records[2]
    assert third_attempt["attempt"] == 3
    assert third_attempt["patch_meta"]["adaptive_progress_recomputed_after_patch"] is True
    assert third_attempt["attempt_metadata"]["adaptive_progress_stop_triggered"] is False


def test_adaptive_patch_verifier_rejects_test_only_patch_before_apply(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_verifier_reject"
    context_dir = tmp_path / "context_adaptive_verifier_reject"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveTestEditPatcher()

    run_calls = {"count": 0}
    apply_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_calls["count"] += 1
        return True, ""

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-verifier-reject",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 0
    assert records[0]["note"] == "patch_verifier_rejected"
    assert records[0]["patch_meta"]["patch_verifier_enabled"] is True
    assert (
        records[0]["patch_meta"]["patch_verifier_scope"] == "all"
    )  # ATT-14 / D7: uniform default scope
    assert records[0]["patch_meta"]["patch_verifier_passed"] is False
    assert records[0]["attempt_metadata"]["patch_verifier_passed"] is False


def test_patch_verifier_scope_disabled_skips_verifier_on_adaptive(tmp_path, monkeypatch) -> None:
    patcher = _AdaptiveTestEditPatcher()
    records, run_calls, apply_calls = _run_adaptive_verifier_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        run_id="run-adaptive-verifier-scope-disabled",
        patch_verifier_scope=run_eval.PATCH_VERIFIER_SCOPE_DISABLED,
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 1
    assert records[0]["note"] == "patch_apply_failed"
    assert records[0]["patch_meta"]["patch_verifier_enabled"] is False
    assert (
        records[0]["patch_meta"]["patch_verifier_scope"] == run_eval.PATCH_VERIFIER_SCOPE_DISABLED
    )
    assert records[0]["attempt_metadata"]["patch_verifier_scope"] == (
        run_eval.PATCH_VERIFIER_SCOPE_DISABLED
    )
    assert records[0]["patch_meta"]["patch_verifier_min_score"] == run_eval.PATCH_VERIFIER_MIN_SCORE
    assert records[0]["attempt_metadata"]["patch_verifier_min_score"] == (
        run_eval.PATCH_VERIFIER_MIN_SCORE
    )


def test_patch_verifier_scope_all_enables_verifier_on_baseline_condition(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_base_verifier_scope_all"
    context_dir = tmp_path / "context_base_verifier_scope_all"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveTestEditPatcher()

    run_calls = {"count": 0}
    apply_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_calls["count"] += 1
        return False, "simulated apply failure"

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LDB,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-ldb-verifier-scope-all",
        include_files_context=False,
        force_patch_attempt=False,
        patch_verifier_scope=run_eval.PATCH_VERIFIER_SCOPE_ALL,
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 0
    assert records[0]["note"] == "patch_verifier_rejected"
    assert records[0]["patch_meta"]["patch_verifier_enabled"] is True
    assert records[0]["patch_meta"]["patch_verifier_scope"] == run_eval.PATCH_VERIFIER_SCOPE_ALL
    assert (
        records[0]["attempt_metadata"]["patch_verifier_scope"] == run_eval.PATCH_VERIFIER_SCOPE_ALL
    )
    assert records[0]["patch_meta"]["patch_verifier_min_score"] == run_eval.PATCH_VERIFIER_MIN_SCORE
    assert records[0]["attempt_metadata"]["patch_verifier_min_score"] == (
        run_eval.PATCH_VERIFIER_MIN_SCORE
    )


def test_patch_verifier_min_score_changes_pass_threshold(tmp_path) -> None:
    case = _make_case(tmp_path)
    snapshot = _snapshot_payload()
    diff = (
        "diff --git a/notes.txt b/notes.txt\n"
        "--- a/notes.txt\n"
        "+++ b/notes.txt\n"
        "@@ -0,0 +1 @@\n"
        "+debug\n"
    )
    stdout = "test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n"

    passed_default, meta_default = run_eval._score_patch_relevance(
        diff=diff,
        cwd=case.path,
        snapshot=snapshot,
        stdout=stdout,
        stderr="",
        min_score=2,
    )
    passed_relaxed, meta_relaxed = run_eval._score_patch_relevance(
        diff=diff,
        cwd=case.path,
        snapshot=snapshot,
        stdout=stdout,
        stderr="",
        min_score=1,
    )

    assert meta_default["patch_verifier_score"] == 1
    assert meta_relaxed["patch_verifier_score"] == 1
    assert passed_default is False
    assert passed_relaxed is True
    assert meta_default["patch_verifier_min_score"] == 2
    assert meta_relaxed["patch_verifier_min_score"] == 1


def test_adaptive_patch_verifier_allows_neighbor_patch_and_reaches_apply(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_verifier_pass"
    context_dir = tmp_path / "context_adaptive_verifier_pass"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveNeighborEditPatcher()

    run_calls = {"count": 0}
    apply_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_calls["count"] += 1
        return False, "simulated apply failure"

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-verifier-pass",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 1
    assert records[0]["note"] == "patch_apply_failed"
    assert records[0]["patch_meta"]["patch_verifier_enabled"] is True
    assert records[0]["patch_meta"]["patch_verifier_passed"] is True


def test_adaptive_patch_verifier_repair_passes_and_reaches_apply(tmp_path, monkeypatch) -> None:
    patcher = _AdaptiveSequentialPatchResultPatcher(
        patch_results=[
            PatchResult(diff=_test_only_expectation_diff(expected=1), meta={"reason": "test_edit"}),
            PatchResult(diff=_helper_fix_diff(), meta={"reason": "verifier_repair"}),
        ]
    )

    records, run_calls, apply_calls = _run_adaptive_verifier_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        run_id="run-adaptive-verifier-repair-pass",
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 1
    assert patcher.patch_calls == 2
    assert "Verifier-guided repair request" in patcher.prompts[1]
    assert "test_buggy.py" in patcher.prompts[1]
    assert records[0]["note"] == "patch_apply_failed"
    assert records[0]["patch_meta"]["patch_verifier_passed"] is True
    assert records[0]["patch_meta"]["patch_file_paths"] == ["helper.py"]
    assert records[0]["patch_meta"]["patch_verifier_recovery_eligible"] is True
    assert records[0]["patch_meta"]["patch_verifier_recovery_invoked"] is True
    assert records[0]["patch_meta"]["patch_verifier_recovery_result"] == "passed_verifier"
    assert (
        records[0]["patch_meta"]["patch_verifier_recovery_score_before"]
        < run_eval.PATCH_VERIFIER_MIN_SCORE
    )
    assert records[0]["patch_meta"]["patch_verifier_recovery_patch_files_before"] == [
        "test_buggy.py"
    ]
    assert records[0]["attempt_metadata"]["patch_verifier_recovery_terminal_note"] == (
        "patch_apply_failed"
    )


def test_adaptive_patch_verifier_repair_no_diff_keeps_rejection(tmp_path, monkeypatch) -> None:
    patcher = _AdaptiveSequentialPatchResultPatcher(
        patch_results=[
            PatchResult(diff=_test_only_expectation_diff(expected=1), meta={"reason": "test_edit"}),
            PatchResult(diff=None, meta={"reason": "no_diff", "llm_call_count": 1}),
        ]
    )

    records, run_calls, apply_calls = _run_adaptive_verifier_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        run_id="run-adaptive-verifier-repair-no-diff",
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 0
    assert patcher.patch_calls == 2
    assert records[0]["note"] == "patch_verifier_rejected"
    assert records[0]["patch_meta"]["patch_verifier_passed"] is False
    assert records[0]["patch_meta"]["patch_file_paths"] == ["test_buggy.py"]
    assert records[0]["patch_meta"]["patch_verifier_recovery_result"] == "no_diff"
    assert records[0]["patch_meta"]["patcher_request_failed"] is False


def test_adaptive_patch_verifier_repair_request_failure_keeps_main_attempt_state(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveSequentialPatchResultPatcher(
        patch_results=[
            PatchResult(diff=_test_only_expectation_diff(expected=1), meta={"reason": "test_edit"}),
            PatchResult(
                diff=None,
                meta={
                    "error": "request_failed",
                    "exc": "ReadTimeout: timed out while waiting for response",
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
                },
            ),
        ]
    )

    records, run_calls, apply_calls = _run_adaptive_verifier_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        run_id="run-adaptive-verifier-repair-request-failed",
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 0
    assert patcher.patch_calls == 2
    assert records[0]["note"] == "patch_verifier_rejected"
    assert records[0]["patch_meta"]["patch_verifier_recovery_result"] == "request_failed"
    assert records[0]["patch_meta"]["patcher_request_failed"] is False
    assert records[0]["attempt_metadata"]["patcher_request_failed"] is False
    assert records[0]["patch_meta"]["patch_verifier_recovery_raw_meta"]["error"] == (
        "request_failed"
    )
    assert (
        records[0]["patch_meta"]["patch_verifier_recovery_raw_meta"]["usage"]["total_tokens"] == 11
    )
    tokens = records[0]["attempt_metadata"]["token_breakdown"]
    assert tokens["recovery_tokens_total"] == 11
    assert tokens["stage_b_tokens_total"] == 0


def test_adaptive_patch_verifier_repair_still_rejected_updates_final_candidate(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveSequentialPatchResultPatcher(
        patch_results=[
            PatchResult(diff=_test_only_expectation_diff(expected=1), meta={"reason": "test_edit"}),
            PatchResult(diff=_helper_non_substantive_diff(), meta={"reason": "non_substantive"}),
        ]
    )

    records, run_calls, apply_calls = _run_adaptive_verifier_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        run_id="run-adaptive-verifier-repair-still-rejected",
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 0
    assert patcher.patch_calls == 2
    assert records[0]["note"] == "patch_verifier_rejected"
    assert records[0]["patch_meta"]["patch_verifier_recovery_result"] == "still_rejected"
    assert records[0]["patch_meta"]["patch_file_paths"] == ["helper.py"]
    assert "no_substantive_change" in records[0]["patch_meta"]["patch_verifier_reasons"]
    assert records[0]["patch_meta"]["patch_verifier_recovery_patch_files_before"] == [
        "test_buggy.py"
    ]
    assert (
        "touches_failing_test_file"
        in records[0]["patch_meta"]["patch_verifier_recovery_reasons_before"]
    )


def test_adaptive_patch_verifier_repair_skips_non_substantive_initial_reject(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveSequentialPatchResultPatcher(
        patch_results=[
            PatchResult(diff=_helper_non_substantive_diff(), meta={"reason": "non_substantive"})
        ]
    )

    records, run_calls, apply_calls = _run_adaptive_verifier_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        run_id="run-adaptive-verifier-repair-not-eligible",
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 0
    assert patcher.patch_calls == 1
    assert records[0]["note"] == "patch_verifier_rejected"
    assert records[0]["patch_meta"]["patch_verifier_recovery_eligible"] is False
    assert records[0]["patch_meta"]["patch_verifier_recovery_invoked"] is False
    assert records[0]["patch_meta"]["patch_verifier_recovery_result"] == "not_eligible"


def test_adaptive_patch_verifier_repair_skips_when_attempt_budget_is_exhausted(
    tmp_path, monkeypatch
) -> None:
    patcher = _AdaptiveSequentialPatchResultPatcher(
        name="openai",
        max_propose_sec=5.0,
        patch_results=[
            PatchResult(diff=_test_only_expectation_diff(expected=1), meta={"reason": "test_edit"})
        ],
    )
    remaining_budget_values = iter([5.0, 0.0])

    def _fake_remaining_attempt_budget(_deadline) -> float:
        return next(remaining_budget_values, 0.0)

    # The attempt flow and the shared-budget proposal helpers both read the deadline.
    for module in (eval_attempt_flow, attempt_proposal):
        monkeypatch.setattr(
            module, "_remaining_attempt_propose_budget_sec", _fake_remaining_attempt_budget
        )

    records, run_calls, apply_calls = _run_adaptive_verifier_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        run_id="run-adaptive-verifier-repair-budget-exhausted",
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert apply_calls["count"] == 0
    assert records[0]["note"] == "patch_verifier_rejected"
    assert records[0]["patch_meta"]["patch_file_paths"] == ["test_buggy.py"]
    assert records[0]["patch_meta"]["patch_verifier_recovery_eligible"] is True
    assert records[0]["patch_meta"]["patch_verifier_recovery_invoked"] is False
    assert records[0]["patch_meta"]["patch_verifier_recovery_result"] == (
        "skipped_deadline_exhausted"
    )


def test_adaptive_patch_verifier_infers_source_scope_from_failing_test_file(tmp_path) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_verifier_scope_infer"
    shutil.copytree(case.path, workdir)

    diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -2,4 +2,4 @@\n"
        " \n"
        " \n"
        " def f() -> int:\n"
        "-    return helper.value()\n"
        "+    return helper.value() + 0\n"
    )
    snapshot = {
        "exception": {"type": "AssertionError", "message": "expected 2 got 1"},
        "frames": [
            {
                "file_rel": "tests/test_buggy.py",
                "line": 4,
                "function": "test_f",
                "code": "assert f() == 2",
                "locals": {},
            }
        ],
    }
    stdout = "test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n"

    passed_without_infer, meta_without_infer = run_eval._score_patch_relevance(
        diff=diff,
        cwd=workdir,
        snapshot=snapshot,
        stdout=stdout,
        stderr="",
        infer_test_source_scope=False,
    )
    passed_with_infer, meta_with_infer = run_eval._score_patch_relevance(
        diff=diff,
        cwd=workdir,
        snapshot=snapshot,
        stdout=stdout,
        stderr="",
        infer_test_source_scope=True,
    )

    assert passed_without_infer is False
    assert meta_without_infer["patch_verifier_source_scope_matched"] is False
    assert passed_with_infer is True
    assert meta_with_infer["patch_verifier_source_scope_matched"] is True
    assert "buggy.py" in meta_with_infer["patch_verifier_inferred_source_scope"]


def test_adaptive_patch_verifier_infers_scope_from_subprocess_target(tmp_path) -> None:
    workdir = tmp_path / "workdir_adaptive_verifier_subprocess_scope"
    (workdir / "tests").mkdir(parents=True, exist_ok=True)
    (workdir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (workdir / "tests" / "test_cli.py").write_text(
        (
            "import subprocess\n\n"
            "def test_cli() -> None:\n"
            "    subprocess.run(['python', 'buggy.py'])\n"
            "    assert True\n"
        ),
        encoding="utf-8",
    )

    diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    snapshot = {
        "exception": {"type": "AssertionError", "message": "cli mismatch"},
        "frames": [
            {
                "file_rel": "tests/test_cli.py",
                "line": 4,
                "function": "test_cli",
                "code": "assert True",
                "locals": {},
            }
        ],
    }
    stdout = "tests/test_cli.py::test_cli\nE   AssertionError: cli mismatch\n"

    passed_without_infer, _meta_without_infer = run_eval._score_patch_relevance(
        diff=diff,
        cwd=workdir,
        snapshot=snapshot,
        stdout=stdout,
        stderr="",
        infer_test_source_scope=False,
    )
    passed_with_infer, meta_with_infer = run_eval._score_patch_relevance(
        diff=diff,
        cwd=workdir,
        snapshot=snapshot,
        stdout=stdout,
        stderr="",
        infer_test_source_scope=True,
    )

    assert passed_without_infer is False
    assert passed_with_infer is True
    assert meta_with_infer["patch_verifier_source_scope_matched"] is True
    assert "buggy.py" in meta_with_infer["patch_verifier_inferred_source_scope"]


def test_patch_verifier_scope_match_avoids_ambiguous_basename_false_positive() -> None:
    ambiguous_scope = {"pkg/a/utils.py", "pkg/b/utils.py"}
    assert (
        run_eval._patch_mentions_scope_files(
            patch_files=["other/utils.py"],
            scope_files=ambiguous_scope,
        )
        is False
    )
    assert (
        run_eval._patch_mentions_scope_files(
            patch_files=["utils.py"],
            scope_files=ambiguous_scope,
        )
        is False
    )

    unambiguous_scope = {"pkg/a/buggy.py"}
    assert (
        run_eval._patch_mentions_scope_files(
            patch_files=["buggy.py"],
            scope_files=unambiguous_scope,
        )
        is True
    )
    assert (
        run_eval._patch_mentions_scope_files(
            patch_files=["other/buggy.py"],
            scope_files=unambiguous_scope,
        )
        is False
    )


def test_syntax_preflight_retries_within_same_attempt(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_syntax_preflight_retry"
    context_dir = tmp_path / "context_syntax_preflight_retry"
    shutil.copytree(case.path, workdir)
    patcher = _SyntaxCyclingPatcher(
        name="syntax_preflight_retry",
        diffs=[
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return (\n"
            ),
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
        ],
    )

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        helper_text = (workdir / "helper.py").read_text(encoding="utf-8")
        if "return 2" in helper_text and "return (" not in helper_text:
            return run_eval.RunResult(
                returncode=0,
                duration_sec=0.01,
                stdout="1 passed\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   SyntaxError: '(' was never closed\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-syntax-preflight-retry",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
    )

    assert len(records) == 1
    assert run_calls["count"] == 2
    assert patcher.patch_calls == 2
    assert records[0]["success"] is True
    assert records[0]["note"] == "success"
    assert records[0]["attempt"] == 1
    assert records[0]["attempt_metadata"]["syntax_preflight_cycles_used"] == 2
    assert records[0]["attempt_metadata"]["syntax_preflight_failures"] == 1
    events = records[0]["patch_meta"]["syntax_preflight_events"]
    assert len(events) == 2
    assert events[0]["passed"] is False
    assert events[1]["passed"] is True


def test_adaptive_baseline_syntax_preflight_retries_accumulate_bridge_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_baseline_syntax_timing_retry"
    context_dir = tmp_path / "context_baseline_syntax_timing_retry"
    shutil.copytree(case.path, workdir)
    context_dir.mkdir()

    invalid_diff = (
        "diff --git a/helper.py b/helper.py\n"
        "--- a/helper.py\n"
        "+++ b/helper.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value() -> int:\n"
        "-    return 1\n"
        "+    return (\n"
    )
    valid_diff = (
        "diff --git a/helper.py b/helper.py\n"
        "--- a/helper.py\n"
        "+++ b/helper.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    class _RetryTimingBaselineAdapter(BaselineAdapter):
        def __init__(self) -> None:
            self.run_calls = 0

        @property
        def name(self) -> str:
            return "intervenor"

        def setup(self, config: BaselineConfig) -> None:
            del config

        def convert_case(
            self,
            case_dir: Path,
            *,
            case_context: BaselineCaseContext | None = None,
        ) -> dict[str, object]:
            del case_context
            return {"task_id": case_dir.name}

        def run_repair(
            self,
            case_input: dict[str, object],
            config: BaselineConfig,
        ) -> BaselineResult:
            del case_input, config
            diff = invalid_diff if self.run_calls == 0 else valid_diff
            self.run_calls += 1
            return BaselineResult(
                patch_diff=diff,
                success=True,
                iterations_used=1,
            )

    class _FakeSidecarWriteResult:
        def __init__(self, build_sec: float, write_sec: float) -> None:
            self.written_sidecar_names = ["llmdebug_controller_packet.json"]
            self.sidecar_error = ""
            self.sidecar_build_sec = build_sec
            self.sidecar_write_sec = write_sec

    class _BaselinePerfClock:
        def perf_counter(self) -> float:
            return next(perf_counter_values)

    adapter = _RetryTimingBaselineAdapter()
    patcher = BaselinePatcher(adapter=adapter, config=BaselineConfig())
    sidecar_timings = iter(((0.1, 0.15), (0.2, 0.25)))
    # Per propose call: conversion start/end, sidecar start/end, run_repair start/end (BAS-08).
    perf_counter_values = iter((0.0, 0.2, 0.2, 0.5, 0.5, 0.6, 1.0, 1.4, 1.4, 1.9, 1.9, 2.0))
    run_calls = {"count": 0}

    def _fake_write_controller_sidecars_for_cycle(**kwargs: object) -> _FakeSidecarWriteResult:
        del kwargs
        build_sec, write_sec = next(sidecar_timings)
        return _FakeSidecarWriteResult(build_sec=build_sec, write_sec=write_sec)

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        helper_text = (workdir / "helper.py").read_text(encoding="utf-8")
        if "return 2" in helper_text and "return (" not in helper_text:
            return run_eval.RunResult(
                returncode=0,
                duration_sec=0.01,
                stdout="1 passed\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   SyntaxError: '(' was never closed\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )
    monkeypatch.setattr(
        eval_attempt_flow,
        "write_controller_sidecars_for_cycle",
        _fake_write_controller_sidecars_for_cycle,
    )
    monkeypatch.setattr(
        baseline_patcher_module,
        "time",
        _BaselinePerfClock(),
    )
    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-baseline-syntax-preflight-timing-retry",
            include_files_context=False,
            force_patch_attempt=False,
            syntax_preflight_enabled=True,
            syntax_preflight_max_cycles=2,
        )
    finally:
        run_eval.set_active_execution_config(previous_execution)

    assert len(records) == 1
    assert run_calls["count"] == 2
    assert adapter.run_calls == 2
    assert records[0]["success"] is True
    assert records[0]["attempt_metadata"]["syntax_preflight_cycles_used"] == 2
    assert records[0]["attempt_metadata"]["syntax_preflight_failures"] == 1
    assert records[0]["patch_meta"]["adaptive_controller_sidecar_build_sec"] == pytest.approx(0.3)
    assert records[0]["patch_meta"]["adaptive_controller_sidecar_write_sec"] == pytest.approx(0.4)
    assert records[0]["patch_meta"]["baseline_case_conversion_sec"] == pytest.approx(0.6)
    assert records[0]["patch_meta"]["baseline_sidecar_load_sec"] == pytest.approx(0.8)
    timing = records[0]["attempt_metadata"]["timing_breakdown"]
    assert timing["adaptive_controller_sidecar_build_sec"] == pytest.approx(0.3)
    assert timing["adaptive_controller_sidecar_write_sec"] == pytest.approx(0.4)
    assert timing["baseline_case_conversion_sec"] == pytest.approx(0.6)
    assert timing["baseline_sidecar_load_sec"] == pytest.approx(0.8)


def test_syntax_preflight_exhaustion_marks_attempt_failed(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_syntax_preflight_failed"
    context_dir = tmp_path / "context_syntax_preflight_failed"
    shutil.copytree(case.path, workdir)
    patcher = _SyntaxCyclingPatcher(
        name="syntax_preflight_failed",
        diffs=[
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return (\n"
            )
        ],
    )

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-syntax-preflight-failed",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        syntax_preflight_fallthrough=False,
    )

    assert len(records) == 1
    assert run_calls["count"] == 1
    assert patcher.patch_calls == 2
    assert records[0]["success"] is False
    assert records[0]["note"] == "syntax_preflight_failed"
    assert records[0]["attempt_metadata"]["syntax_preflight_cycles_used"] == 2
    assert records[0]["attempt_metadata"]["syntax_preflight_failures"] == 2
    assert records[0]["attempt_metadata"]["syntax_preflight_last_error_type"] == "PyCompileError"
    events = records[0]["patch_meta"]["syntax_preflight_events"]
    assert len(events) == 2
    assert all(event["passed"] is False for event in events)


def test_syntax_preflight_fallthrough_runs_tests_on_exhaustion(tmp_path, monkeypatch) -> None:
    """With fallthrough enabled, preflight exhaustion falls through to test execution."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_syntax_preflight_fallthrough"
    context_dir = tmp_path / "context_syntax_preflight_fallthrough"
    shutil.copytree(case.path, workdir)
    patcher = _SyntaxCyclingPatcher(
        name="syntax_preflight_fallthrough",
        diffs=[
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return (\n"
            )
        ],
    )

    run_calls = {"count": 0}
    helper_versions: list[str] = []

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        helper_versions.append((workdir / "helper.py").read_text(encoding="utf-8"))
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="SyntaxError: unexpected EOF while parsing\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-syntax-preflight-fallthrough",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        syntax_preflight_fallthrough=True,
    )

    assert len(records) == 1
    # With fallthrough, tests are executed despite preflight failure.
    assert run_calls["count"] == 2  # before + after
    assert patcher.patch_calls == 2
    assert records[0]["success"] is False
    assert records[0]["note"] == "tests_still_failing"
    assert records[0]["attempt_metadata"]["syntax_preflight_exhausted_fallthrough"] is True
    assert records[0]["attempt_metadata"]["syntax_preflight_failures"] == 2
    assert helper_versions[0] == "def value() -> int:\n    return 1\n"
    assert helper_versions[1] == "def value() -> int:\n    return (\n"
    events = records[0]["patch_meta"]["syntax_preflight_events"]
    assert len(events) == 2
    assert all(event["passed"] is False for event in events)


def test_syntax_preflight_clears_transient_patch_meta_between_cycles(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_syntax_preflight_meta_cleanup"
    context_dir = tmp_path / "context_syntax_preflight_meta_cleanup"
    shutil.copytree(case.path, workdir)
    patcher = _SyntaxCyclingPatcher(
        name="syntax_preflight_meta_cleanup",
        diffs=[
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return (\n"
            ),
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
        ],
        metas=[
            {"runner_forced_patch_attempt": True},
            {},
        ],
    )

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=0,
            duration_sec=0.01,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-syntax-preflight-meta-cleanup",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
    )

    assert len(records) == 1
    assert records[0]["success"] is True
    assert patcher.patch_calls == 2
    assert "runner_forced_patch_attempt" not in records[0]["patch_meta"]


def test_patch_sanitize_reject_then_retry_success(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_patch_sanitize_retry"
    context_dir = tmp_path / "context_patch_sanitize_retry"
    shutil.copytree(case.path, workdir)
    patcher = _SyntaxCyclingPatcher(
        name="patch_sanitize_retry",
        diffs=[
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,3 @@\n"
                " def value() -> int:\n"
                "+    # keep \\n marker\n"
                "     return 1\n"
            ),
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
        ],
    )

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=0,
            duration_sec=0.01,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-patch-sanitize-retry",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        patch_sanitize_enabled=True,
    )

    assert len(records) == 1
    assert records[0]["success"] is True
    assert run_calls["count"] == 2
    assert patcher.patch_calls == 2
    assert records[0]["attempt_metadata"]["syntax_preflight_failures"] == 1
    events = records[0]["patch_meta"]["syntax_preflight_events"]
    assert len(events) == 2
    assert events[0]["passed"] is False
    assert events[0]["syntax_preflight_error_type"] == "PatchSanitizationError"
    assert events[1]["passed"] is True


def test_patch_sanitize_exhaustion_marks_attempt_failed(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_patch_sanitize_failed"
    context_dir = tmp_path / "context_patch_sanitize_failed"
    shutil.copytree(case.path, workdir)
    patcher = _SyntaxCyclingPatcher(
        name="patch_sanitize_failed",
        diffs=[
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,3 @@\n"
                " def value() -> int:\n"
                "+    # keep \\n marker\n"
                "     return 1\n"
            )
        ],
    )

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-patch-sanitize-failed",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        patch_sanitize_enabled=True,
    )

    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["note"] == "syntax_preflight_failed"
    assert run_calls["count"] == 1
    assert patcher.patch_calls == 2
    assert records[0]["attempt_metadata"]["syntax_preflight_failures"] == 2
    assert (
        records[0]["attempt_metadata"]["syntax_preflight_last_error_type"]
        == "PatchSanitizationError"
    )
    assert records[0]["patch_meta"]["patch_sanitize_rejected"] is True
    assert records[0]["patch_meta"]["patch_sanitize_reasons"]


def test_patch_sanitize_disabled_allows_patch_to_apply(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_patch_sanitize_disabled"
    context_dir = tmp_path / "context_patch_sanitize_disabled"
    shutil.copytree(case.path, workdir)
    patcher = _SyntaxCyclingPatcher(
        name="patch_sanitize_disabled",
        diffs=[
            (
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,3 @@\n"
                " def value() -> int:\n"
                "+    # keep \\n marker\n"
                "     return 1\n"
            )
        ],
    )

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=0,
            duration_sec=0.01,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-patch-sanitize-disabled",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=False,
        patch_sanitize_enabled=False,
    )

    assert len(records) == 1
    assert records[0]["success"] is True
    assert run_calls["count"] == 2
    assert patcher.patch_calls == 1
    assert records[0]["patch_meta"]["patch_sanitize_enabled"] is False
    assert records[0]["patch_meta"]["patch_sanitize_rejected"] is False


def test_openai_response_mode_autoswitch_on_patch_sanitize_retry(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_openai_autoswitch_sanitize"
    context_dir = tmp_path / "context_openai_autoswitch_sanitize"
    shutil.copytree(case.path, workdir)

    captured_response_modes: list[str] = []
    diffs = [
        (
            "diff --git a/helper.py b/helper.py\n"
            "--- a/helper.py\n"
            "+++ b/helper.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def value() -> int:\n"
            "+    # keep \\n marker\n"
            "     return 1\n"
        ),
        (
            "diff --git a/helper.py b/helper.py\n"
            "--- a/helper.py\n"
            "+++ b/helper.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def value() -> int:\n"
            "-    return 1\n"
            "+    return 2\n"
        ),
    ]

    def _fake_propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        captured_response_modes.append(str(self.response_mode))
        idx = min(len(captured_response_modes) - 1, len(diffs) - 1)
        return PatchResult(diff=diffs[idx], meta={"response_mode": self.response_mode})

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=0,
            duration_sec=0.01,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(OpenAICompatPatcher, "propose_patch", _fake_propose_patch)
    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-openai-autoswitch-sanitize",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        patch_sanitize_enabled=True,
        openai_response_mode_autoswitch="retry_to_diff",
    )

    assert len(records) == 1
    assert records[0]["success"] is True
    assert run_calls["count"] == 2
    assert captured_response_modes == ["edits", "diff"]
    assert records[0]["patch_meta"]["openai_response_mode_autoswitched"] is True
    assert records[0]["patch_meta"]["openai_response_mode_autoswitch_reason"] == (
        "patch_sanitize_rejected"
    )
    assert records[0]["model_metadata"]["response_mode"] == "diff"


def test_openai_response_mode_autoswitch_on_syntax_preflight_retry(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_openai_autoswitch_syntax"
    context_dir = tmp_path / "context_openai_autoswitch_syntax"
    shutil.copytree(case.path, workdir)

    captured_response_modes: list[str] = []
    diffs = [
        (
            "diff --git a/helper.py b/helper.py\n"
            "--- a/helper.py\n"
            "+++ b/helper.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def value() -> int:\n"
            "-    return 1\n"
            "+    return (\n"
        ),
        (
            "diff --git a/helper.py b/helper.py\n"
            "--- a/helper.py\n"
            "+++ b/helper.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def value() -> int:\n"
            "-    return 1\n"
            "+    return 2\n"
        ),
    ]

    def _fake_propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        captured_response_modes.append(str(self.response_mode))
        idx = min(len(captured_response_modes) - 1, len(diffs) - 1)
        return PatchResult(diff=diffs[idx], meta={"response_mode": self.response_mode})

    run_calls = {"count": 0}

    def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
        del args, kwargs
        run_calls["count"] += 1
        if run_calls["count"] == 1:
            return run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
                stderr="",
            )
        helper_text = (workdir / "helper.py").read_text(encoding="utf-8")
        if "return 2" in helper_text and "return (" not in helper_text:
            return run_eval.RunResult(
                returncode=0,
                duration_sec=0.01,
                stdout="1 passed\n",
                stderr="",
            )
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   SyntaxError: '(' was never closed\n",
            stderr="",
        )

    monkeypatch.setattr(OpenAICompatPatcher, "propose_patch", _fake_propose_patch)
    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-openai-autoswitch-syntax",
        include_files_context=False,
        force_patch_attempt=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        patch_sanitize_enabled=False,
        openai_response_mode_autoswitch="retry_to_diff",
    )

    assert len(records) == 1
    assert records[0]["success"] is True
    assert run_calls["count"] == 2
    assert captured_response_modes == ["edits", "diff"]
    assert records[0]["patch_meta"]["openai_response_mode_autoswitched"] is True
    assert records[0]["patch_meta"]["openai_response_mode_autoswitch_reason"] == (
        "syntax_preflight_failed"
    )
    assert records[0]["model_metadata"]["response_mode"] == "diff"


def test_adaptive_online_policy_shadow_keeps_baseline_action(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_shadow"
    context_dir = tmp_path / "context_adaptive_online_shadow"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher(name="adaptive_online_shadow")
    controller, events_path = _skip_biased_online_controller(tmp_path, mode="shadow")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-shadow",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    assert patcher.request_calls == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_online_policy_mode"] == "shadow"
    assert attempt_meta["adaptive_online_action_selected"] == ACTION_SKIP_HELPER
    assert attempt_meta["adaptive_online_action_applied"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_online_override_reason"] == "shadow_mode_baseline"
    assert attempt_meta["adaptive_online_behavior_action"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_online_behavior_propensity"] == 1.0
    assert attempt_meta["adaptive_online_feature_scaling_profile"] == "v1"
    assert isinstance(attempt_meta["adaptive_online_feature_vector_raw"], dict)
    assert attempt_meta["adaptive_online_updated"] is False
    assert attempt_meta["adaptive_online_update_reason"] == "shadow_no_update"
    assert attempt_meta["adaptive_online_event_logged"] is True
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_online_policy_on_can_skip_helper(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_on"
    context_dir = tmp_path / "context_adaptive_online_on"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher(name="adaptive_online_on")
    controller, events_path = _skip_biased_online_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-on",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    assert patcher.request_calls == 0
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_decision_reason"] == "online_policy_skip_helper"
    assert attempt_meta["adaptive_online_policy_mode"] == "on"
    assert attempt_meta["adaptive_online_action_selected"] == ACTION_SKIP_HELPER
    assert attempt_meta["adaptive_online_action_applied"] == ACTION_SKIP_HELPER
    assert attempt_meta["adaptive_online_behavior_action"] == ACTION_SKIP_HELPER
    assert attempt_meta["adaptive_online_behavior_propensity"] > 0.0
    assert attempt_meta["adaptive_online_feature_scaling_profile"] == "v1"
    assert isinstance(attempt_meta["adaptive_online_feature_vector_raw"], dict)
    assert isinstance(attempt_meta["adaptive_online_feature_vector"], dict)
    assert attempt_meta["adaptive_online_updated"] is True
    assert attempt_meta["adaptive_online_update_reason"] == "on_policy_update"
    assert attempt_meta["adaptive_online_event_logged"] is True
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_online_policy_suppresses_update_for_invalid_helper_payload(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_invalid_payload"
    context_dir = tmp_path / "context_adaptive_online_invalid_payload"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionInvalidPatcher()
    controller, events_path = _invoke_biased_online_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-invalid-payload",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_outcome"] == "invalid_payload"
    assert attempt_meta["adaptive_online_updated"] is False
    assert attempt_meta["adaptive_online_update_reason"] == "helper_invalid_payload_no_update"
    assert attempt_meta["adaptive_online_event_logged"] is True
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_online_policy_suppresses_update_for_helper_backend_unavailable(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_transport_failure"
    context_dir = tmp_path / "context_adaptive_online_transport_failure"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionBackendUnavailablePatcher()
    controller, events_path = _invoke_biased_online_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-transport-failure",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_outcome"] == "unavailable_backend"
    assert attempt_meta["adaptive_online_updated"] is False
    assert attempt_meta["adaptive_online_update_reason"] == "helper_backend_unavailable_no_update"
    assert attempt_meta["adaptive_online_event_logged"] is True
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_online_policy_suppresses_update_for_helper_transport_failure(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_transport_failure_retryable"
    context_dir = tmp_path / "context_adaptive_online_transport_failure_retryable"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionTransportFailurePatcher()
    controller, events_path = _invoke_biased_online_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-transport-failure-retryable",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_outcome"] == "transport_failed"
    assert attempt_meta["adaptive_online_updated"] is False
    assert attempt_meta["adaptive_online_update_reason"] == "helper_transport_no_update"
    assert attempt_meta["adaptive_online_event_logged"] is True
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_online_policy_suppresses_update_for_helper_compat_reasoning_only(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_compat_reasoning_only"
    context_dir = tmp_path / "context_adaptive_online_compat_reasoning_only"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionCompatReasoningOnlyPatcher()
    controller, events_path = _invoke_biased_online_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-compat-reasoning-only",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_outcome"] == "compat_reasoning_only"
    assert attempt_meta["adaptive_online_updated"] is False
    assert attempt_meta["adaptive_online_update_reason"] == (
        "helper_compat_reasoning_only_no_update"
    )
    assert attempt_meta["adaptive_online_event_logged"] is True
    assert records[0]["patch_meta"]["stage_a_reason"] == "helper_compat_reasoning_only"
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_online_policy_updates_for_helper_request_budget_exceeded(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_request_budget_exceeded"
    context_dir = tmp_path / "context_adaptive_online_request_budget_exceeded"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionRequestBudgetPatcher()
    controller, events_path = _invoke_biased_online_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-request-budget-exceeded",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_helper_outcome"] == "request_budget_exceeded"
    assert attempt_meta["adaptive_helper_failure_kind"] == "request_budget_exceeded"
    assert attempt_meta["adaptive_helper_calls_used_after_attempt"] == 1
    assert attempt_meta["adaptive_online_updated"] is True
    assert attempt_meta["adaptive_online_update_reason"] == "on_policy_update"
    assert attempt_meta["adaptive_online_event_logged"] is True
    assert records[0]["patch_meta"]["stage_a_reason"] == "helper_request_budget_exceeded"
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_stagnation_override_forces_helper_after_policy_skip(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_stagnation_override"
    context_dir = tmp_path / "context_adaptive_stagnation_override"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher(name="adaptive_stagnation_override")
    controller, _events_path = _skip_biased_online_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=2,
            output_format="json_compact",
            run_id="run-adaptive-stagnation-override",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 2
    assert patcher.request_calls == 1
    assert records[0]["attempt_metadata"]["adaptive_stagnation_override_applied"] is False
    attempt_meta = records[1]["attempt_metadata"]
    assert attempt_meta["adaptive_stagnation_override_applied"] is True
    assert (
        attempt_meta["adaptive_stagnation_override_reason"]
        == "repeated_no_progress_after_stagnation_note"
    )
    assert attempt_meta["adaptive_helper_invoked"] is True
    assert attempt_meta["adaptive_online_action_applied"] == ACTION_SKIP_HELPER
    assert attempt_meta["adaptive_online_behavior_action"] == ACTION_INVOKE_HELPER
    assert attempt_meta["adaptive_online_behavior_propensity"] == 1.0

    events = [
        json.loads(line) for line in _events_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(events) == 2
    assert events[1]["applied_action"] == ACTION_SKIP_HELPER
    assert events[1]["behavior_action"] == ACTION_INVOKE_HELPER
    assert events[1]["behavior_propensity"] == 1.0


_CASE_LABEL_FEATURES = {
    "case_difficulty_ord",
    "case_snapshot_dependency_ord",
    "case_bug_source_imported",
}


def _run_online_feature_vector_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **run_kwargs: Any
) -> list[dict[str, Any]]:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_features"
    context_dir = tmp_path / "context_adaptive_online_features"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveDiscretionInvalidPatcher(name="adaptive_online_feature_keys")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    return run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-adaptive-online-feature-keys",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_informativeness_pregate=True,
        **run_kwargs,
    )


def test_adaptive_online_feature_vector_includes_extended_and_lagged_signals(
    tmp_path, monkeypatch
) -> None:
    records = _run_online_feature_vector_case(tmp_path, monkeypatch)

    assert len(records) == 2
    feature_vector_1 = records[0]["attempt_metadata"]["adaptive_online_feature_vector"]
    feature_vector_2 = records[1]["attempt_metadata"]["adaptive_online_feature_vector"]
    expected_keys = {
        "helper_budget_remaining_frac",
        "helper_topup_used_frac",
        "helper_topup_remaining_frac",
        "context_compaction_enabled",
        "supports_context_compaction",
        "context_compaction_calls_used_frac",
        "context_compaction_calls_remaining_frac",
        "last_context_compaction_requested",
        "last_context_compaction_invoked",
        "last_context_compaction_success",
        "last_context_compaction_ratio",
        "last_helper_need_more_context",
        "last_helper_invalid_response",
        "last_helper_declined",
        "pregate_exception_is_assertion",
        "pregate_exception_is_timeout",
        "pregate_exception_is_syntax",
    }
    assert expected_keys.issubset(feature_vector_1.keys())
    assert expected_keys.issubset(feature_vector_2.keys())
    raw_feature_vector_1 = records[0]["attempt_metadata"]["adaptive_online_feature_vector_raw"]
    assert expected_keys.issubset(raw_feature_vector_1.keys())
    # Oracle case labels (case.json) are masked by default, in every logged copy.
    assert _CASE_LABEL_FEATURES.isdisjoint(feature_vector_1.keys())
    assert _CASE_LABEL_FEATURES.isdisjoint(feature_vector_2.keys())
    assert _CASE_LABEL_FEATURES.isdisjoint(raw_feature_vector_1.keys())
    # attempts_remaining_frac was dropped as 1 - attempt_frac (POL-06).
    assert "attempts_remaining_frac" not in feature_vector_1
    assert feature_vector_1["attempt_frac"] == 0.5
    assert feature_vector_2["attempt_frac"] == 1.0
    assert feature_vector_1["pregate_exception_is_assertion"] == 1.0
    assert feature_vector_1["last_helper_invalid_response"] == 0.0
    assert feature_vector_2["last_helper_invalid_response"] == 1.0


def test_adaptive_online_feature_vector_case_labels_are_opt_in(tmp_path, monkeypatch) -> None:
    records = _run_online_feature_vector_case(
        tmp_path, monkeypatch, adaptive_online_include_case_labels=True
    )

    feature_vector_1 = records[0]["attempt_metadata"]["adaptive_online_feature_vector"]
    assert _CASE_LABEL_FEATURES.issubset(feature_vector_1.keys())
    assert feature_vector_1["case_difficulty_ord"] == 0.0
    assert feature_vector_1["case_snapshot_dependency_ord"] == 0.5
    assert feature_vector_1["case_bug_source_imported"] == 0.0


def test_adaptive_online_policy_logging_failure_does_not_abort_attempt(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_online_logging_failure"
    context_dir = tmp_path / "context_adaptive_online_logging_failure"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveGatedCapturePatcher(name="adaptive_online_logging_failure")
    controller, _events_path = _skip_biased_online_controller(tmp_path, mode="on")

    def _raise_record_outcome(**kwargs):
        del kwargs
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(controller, "record_outcome", _raise_record_outcome)
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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-online-logging-failure",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_online_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_online_updated"] is False
    assert attempt_meta["adaptive_online_event_logged"] is False
    assert attempt_meta["adaptive_online_update_reason"] == "event_logging_error:OSError"


@dataclass
class _FullTepNoDiffPatcher:
    """An openai-shaped patcher with the TEP generation hooks, proposing no diff."""

    name: str = "openai"
    max_propose_sec: float = 10.0
    max_tokens: int = 400
    response_mode: str = "edits"

    def propose_patch(self, *, prompt: str, context_dir, cwd, **_kwargs) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff=None, meta={"reason": "no_diff"})

    def generate_text(self, **_kwargs):
        raise AssertionError("the no-diff attempt must not reach the TEP text calls")

    def generate_json(self, **_kwargs):
        raise AssertionError("the no-diff attempt must not reach the TEP critic")


def test_full_tep_save_failure_reaches_the_streamed_record(tmp_path, monkeypatch) -> None:
    """ATT-17: a no-diff attempt skips the learning pass, so its own save must report.

    The failure used to be attached to the returned record after the row had already
    gone to `record_sink`, leaving the JSONL silent (codex 3-1#3).
    """
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_full_tep_save"
    context_dir = tmp_path / "context_full_tep_save"
    shutil.copytree(case.path, workdir)

    # The state path's parent is a file, so every save raises.
    blocker = tmp_path / "tep_blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    state_path = blocker / "state" / "tep.json"

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    streamed: list[str] = []
    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_FullTepNoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-full-tep-save-failure",
        include_files_context=False,
        force_patch_attempt=False,
        adaptive_tep_mode="full",
        adaptive_tep_state_path=str(state_path),
        record_sink=lambda record: streamed.append(json.dumps(record)),
    )

    assert len(records) == 1
    assert records[0]["note"] == "no_patch_proposed"
    assert len(streamed) == 1
    streamed_record = json.loads(streamed[0])
    save_error = streamed_record["attempt_metadata"]["adaptive_tep_save_error"]
    assert save_error
    assert streamed_record["patch_meta"]["adaptive_tep_save_error"] == save_error


def test_adaptive_budget_policy_shadow_keeps_baseline_budget(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_shadow"
    context_dir = tmp_path / "context_adaptive_budget_shadow"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher()
    controller, events_path = _increase_biased_budget_controller(tmp_path, mode="shadow")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-shadow",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    _next_tier_index, default_profile, next_profile = _default_first_budget_up_profile(
        is_openai_patcher=True
    )
    next_changes = run_eval._adaptive_budget_control_changes(
        current=default_profile,
        target=next_profile,
    )
    assert attempt_meta["adaptive_budget_policy_mode"] == "shadow"
    assert attempt_meta["adaptive_budget_action_selected"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_KEEP_BUDGET
    assert attempt_meta["adaptive_budget_behavior_action"] == ACTION_KEEP_BUDGET
    assert attempt_meta["adaptive_budget_behavior_propensity"] == 1.0
    assert attempt_meta["adaptive_budget_feature_scaling_profile"] == "v1"
    assert isinstance(attempt_meta["adaptive_budget_feature_vector_raw"], dict)
    assert attempt_meta["adaptive_budget_updated"] is False
    assert attempt_meta["adaptive_budget_update_reason"] == "shadow_no_update"
    assert attempt_meta["adaptive_budget_event_logged"] is True
    assert (
        attempt_meta["adaptive_budget_default_control_values"]["helper_max_calls"]
        == default_profile.helper_max_calls
    )
    assert (
        attempt_meta["adaptive_budget_selected_control_values"]["helper_max_calls"]
        == next_profile.helper_max_calls
    )
    assert (
        attempt_meta["adaptive_budget_applied_control_values"]["helper_max_calls"]
        == default_profile.helper_max_calls
    )
    assert (
        attempt_meta["adaptive_budget_selected_controls_changed"]["stage_a_requests"]
        == next_changes["stage_a_requests"]
    )
    assert attempt_meta["adaptive_budget_selected_controls_changed"]["propose_budget"] is True
    assert attempt_meta["adaptive_budget_applied_controls_changed"]["stage_a_requests"] is False
    assert attempt_meta["attempt_shared_propose_budget_base_sec"] == 10.0
    assert attempt_meta["attempt_shared_propose_budget_multiplier"] == 1.0
    assert attempt_meta["attempt_shared_propose_budget_sec"] == 10.0
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


def test_adaptive_budget_policy_on_applies_multiplier(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_on"
    context_dir = tmp_path / "context_adaptive_budget_on"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher()
    controller, events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-on",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    next_tier_index, default_profile, next_profile = _default_first_budget_up_profile(
        is_openai_patcher=True
    )
    next_changes = run_eval._adaptive_budget_control_changes(
        current=default_profile,
        target=next_profile,
    )
    assert attempt_meta["adaptive_budget_policy_mode"] == "on"
    assert attempt_meta["adaptive_budget_action_selected"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_behavior_action"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_behavior_propensity"] > 0.0
    assert attempt_meta["adaptive_budget_feature_scaling_profile"] == "v1"
    assert isinstance(attempt_meta["adaptive_budget_feature_vector_raw"], dict)
    assert isinstance(attempt_meta["adaptive_budget_feature_vector"], dict)
    assert attempt_meta["adaptive_budget_updated"] is True
    assert attempt_meta["adaptive_budget_update_reason"] == "on_policy_update"
    assert attempt_meta["adaptive_budget_event_logged"] is True
    assert attempt_meta["adaptive_budget_tier_index_before"] == 0
    assert attempt_meta["adaptive_budget_tier_index_after"] == next_tier_index
    assert attempt_meta["adaptive_budget_ups_used"] == 1
    assert (
        attempt_meta["adaptive_budget_default_control_values"]["openai_max_tokens"]
        == default_profile.openai_max_tokens
    )
    assert (
        attempt_meta["adaptive_budget_selected_control_values"]["openai_max_tokens"]
        == next_profile.openai_max_tokens
    )
    assert (
        attempt_meta["adaptive_budget_applied_control_values"]["stage_a_max_requests"]
        == next_profile.stage_a_max_requests
    )
    assert (
        attempt_meta["adaptive_budget_applied_controls_changed"]["retry_history_budget"]
        == next_changes["retry_history_budget"]
    )
    assert attempt_meta["attempt_shared_propose_budget_base_sec"] == 10.0
    assert attempt_meta["attempt_shared_propose_budget_multiplier"] == 1.2
    assert attempt_meta["attempt_shared_propose_budget_sec"] == 12.0
    assert attempt_meta["adaptive_budget_partial_override"] is False
    assert (
        attempt_meta["adaptive_budget_effective_openai_max_tokens"]
        == next_profile.openai_max_tokens
    )
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


@pytest.mark.parametrize(
    ("preferred_family", "expected_metric_key", "expected_metric_value"),
    [
        (ACTION_BUDGET_FAMILY_TOKENS, "adaptive_budget_effective_openai_max_tokens", 440),
        (ACTION_BUDGET_FAMILY_CONTEXT, "adaptive_budget_effective_stage_a_max_chars", 132000),
    ],
)
def test_adaptive_budget_policy_hierarchical_records_non_time_family_multiplier(
    tmp_path,
    monkeypatch,
    preferred_family: str,
    expected_metric_key: str,
    expected_metric_value: int,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / f"workdir_adaptive_budget_hierarchical_{preferred_family}"
    context_dir = tmp_path / f"context_adaptive_budget_hierarchical_{preferred_family}"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher()
    controller, events_path = _increase_biased_hierarchical_budget_controller(
        tmp_path,
        mode="on",
        preferred_family=preferred_family,
    )

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id=f"run-adaptive-budget-hierarchical-{preferred_family}",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
            adaptive_budget_policy_architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_budget_policy_architecture"] == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
    assert attempt_meta["adaptive_budget_action_selected"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_family_selected"] == preferred_family
    assert attempt_meta["adaptive_budget_family_applied"] == preferred_family
    assert attempt_meta["adaptive_budget_default_multiplier"] == 1.0
    assert attempt_meta["adaptive_budget_selected_multiplier"] == 1.1
    assert attempt_meta["adaptive_budget_applied_multiplier"] == 1.1
    assert attempt_meta["attempt_shared_propose_budget_multiplier"] == 1.0
    assert attempt_meta["attempt_shared_propose_budget_sec"] == 10.0
    assert attempt_meta["adaptive_budget_applied_controls_changed"]["propose_budget"] is False
    assert attempt_meta[expected_metric_key] == expected_metric_value
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 1


@pytest.mark.parametrize(
    ("mode", "expected_updated", "expected_reason"),
    [
        ("shadow", False, "shadow_no_update"),
        ("on", True, "on_policy_update"),
    ],
)
def test_adaptive_budget_policy_retry_gate_attempts_finalize_budget_bookkeeping(
    tmp_path,
    monkeypatch,
    mode: str,
    expected_updated: bool,
    expected_reason: str,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / f"workdir_adaptive_budget_retry_gate_{mode}"
    context_dir = tmp_path / f"context_adaptive_budget_retry_gate_{mode}"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetRetryGatePatcher()
    controller, events_path = _increase_biased_budget_controller(tmp_path, mode=mode)

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=2,
            output_format="json_compact",
            run_id=f"run-adaptive-budget-retry-gate-{mode}",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_gate_policy="aggressive",
            adaptive_max_helper_calls=2,
            adaptive_budget_policy_controller=controller,
            adaptive_budget_policy_mode=mode,
            adaptive_budget_policy_max_ups_per_case=2,
        )
    finally:
        controller.close()

    # ATT-05 / D9: attempt 1, the unscored gate block, then attempt 2 proposes.
    assert len(records) == 3
    retry_gate_record = records[1]
    attempt_meta = retry_gate_record["attempt_metadata"]
    assert retry_gate_record["note"] == "retry_gate_no_new_evidence"
    assert retry_gate_record["record_type"] == "infra_event"
    assert attempt_meta["adaptive_budget_policy_mode"] == mode
    assert attempt_meta["adaptive_budget_update_reason"] == expected_reason
    assert attempt_meta["adaptive_budget_event_logged"] is True
    assert attempt_meta["adaptive_budget_updated"] is expected_updated
    assert isinstance(attempt_meta["adaptive_budget_reward"], float)
    events = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # In shadow mode the budget policy stays eligible on the retried attempt; "on" spent its up.
    assert len(events) == (3 if mode == "shadow" else 2)


def test_adaptive_budget_policy_falls_back_when_budget_override_replace_fails(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_replace_fail"
    context_dir = tmp_path / "context_adaptive_budget_replace_fail"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffNonDataclassPatcher()
    controller, _events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-replace-fail",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    patch_meta = records[0]["patch_meta"]
    next_tier_index, _default_profile, next_profile = _default_first_budget_up_profile(
        is_openai_patcher=True
    )
    assert attempt_meta["adaptive_budget_action_selected"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_tier_index_before"] == 0
    assert attempt_meta["adaptive_budget_tier_index_after"] == next_tier_index
    assert attempt_meta["adaptive_budget_ups_used"] == 1
    assert attempt_meta["attempt_shared_propose_budget_base_sec"] == 10.0
    assert attempt_meta["attempt_shared_propose_budget_multiplier"] == 1.2
    assert attempt_meta["attempt_shared_propose_budget_sec"] == 10.0
    assert attempt_meta["adaptive_budget_partial_override"] is True
    assert (
        attempt_meta["adaptive_budget_effective_stage_a_max_requests"]
        == next_profile.stage_a_max_requests
    )
    assert attempt_meta["adaptive_budget_effective_openai_max_tokens"] == 0
    assert attempt_meta["adaptive_budget_partial_override_errors"]["max_propose_sec"].startswith(
        "TypeError:"
    )
    assert "TypeError" in str(patch_meta.get("adaptive_budget_propose_budget_override_error"))


def test_adaptive_budget_policy_penalizes_request_budget_exceeded(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_penalty"
    context_dir = tmp_path / "context_adaptive_budget_penalty"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetExceededPatcher()
    controller, _events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-penalty",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
            adaptive_reward_lambda_tokens=0.0,
            adaptive_reward_mu_latency=0.0,
            adaptive_reward_eta_progress=0.0,
            adaptive_reward_kappa_stagnation=0.0,
            adaptive_budget_reward_rho_budget_exceeded=0.5,
        )
    finally:
        controller.close()

    assert len(records) == 1
    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_budget_request_budget_exceeded"] is True
    assert attempt_meta["adaptive_budget_reward_budget_exceeded_penalty"] == 0.5
    assert attempt_meta["adaptive_budget_reward"] == -0.5
    assert attempt_meta["adaptive_budget_reward_progress_hint"] == 0.0
    assert attempt_meta["adaptive_budget_reward_stagnation_penalty"] == 0.0


def test_adaptive_budget_feature_vector_excludes_case_metadata(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_feature_keys"
    context_dir = tmp_path / "context_adaptive_budget_feature_keys"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher(name="openai", max_propose_sec=9.0)
    controller, _events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=2,
            output_format="json_compact",
            run_id="run-adaptive-budget-feature-keys",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
            adaptive_budget_policy_max_ups_per_case=2,
        )
    finally:
        controller.close()

    assert len(records) == 2
    feature_vector = records[0]["attempt_metadata"]["adaptive_budget_feature_vector"]
    expected_keys = {
        "attempt_frac",
        "helper_low_yield_ratio",
        "adaptive_no_progress_token_sum_k",
        "context_compaction_enabled",
        "current_budget_tier_frac",
        "can_increase_budget_tier",
        "budget_ups_used_frac",
        "budget_ups_remaining_frac",
        "base_propose_budget_sec",
        "budget_can_raise_propose_budget",
        "budget_can_raise_max_tokens",
        "budget_can_raise_helper_allowance",
        "budget_can_raise_stage_a_requests",
        "budget_can_raise_stage_a_chars",
        "budget_can_raise_retry_history_budget",
        "budget_can_raise_compaction_allowance",
        "budget_can_switch_response_mode",
        "budget_next_tier_multiplier_delta",
        "pregate_exception_is_assertion",
    }
    assert expected_keys.issubset(feature_vector.keys())
    assert "case_difficulty_ord" not in feature_vector
    assert "case_snapshot_dependency_ord" not in feature_vector
    assert "case_bug_source_imported" not in feature_vector


def test_adaptive_budget_policy_on_non_openai_patcher_still_applies_runner_controls(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_non_openai"
    context_dir = tmp_path / "context_adaptive_budget_non_openai"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher(name="command")
    controller, _events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-non-openai",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
        )
    finally:
        controller.close()

    attempt_meta = records[0]["attempt_metadata"]
    next_tier_index, _default_profile, next_profile = _default_first_budget_up_profile(
        is_openai_patcher=False
    )
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_tier_index_after"] == next_tier_index
    assert attempt_meta["attempt_shared_propose_budget_base_sec"] == 0.0
    assert attempt_meta["attempt_shared_propose_budget_sec"] == 0.0
    assert (
        attempt_meta["adaptive_budget_effective_stage_a_max_requests"]
        == next_profile.stage_a_max_requests
    )
    assert attempt_meta["adaptive_budget_effective_openai_max_tokens"] == 0
    assert attempt_meta["adaptive_budget_partial_override"] is False


def test_adaptive_budget_policy_keeps_unlimited_helper_budget_unlimited(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_unlimited_helper"
    context_dir = tmp_path / "context_adaptive_budget_unlimited_helper"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher()
    controller, _events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-unlimited-helper",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
            adaptive_max_helper_calls=0,
        )
    finally:
        controller.close()

    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_helper_budget_unlimited"] is True
    assert attempt_meta["adaptive_budget_effective_helper_max_calls"] == 0
    assert attempt_meta["adaptive_budget_applied_control_values"]["helper_max_calls"] == 0


def test_adaptive_budget_policy_highest_tier_switches_response_mode(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_diff_mode"
    context_dir = tmp_path / "context_adaptive_budget_diff_mode"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher()
    controller, _events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-diff-mode",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
            adaptive_budget_policy_tiers="1.0,2.0",
        )
    finally:
        controller.close()

    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_effective_response_mode"] == "diff"
    assert attempt_meta["adaptive_budget_applied_controls_changed"]["response_mode"] is True


def test_adaptive_budget_policy_does_not_autoswitch_full_file_mode(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_budget_full_file_mode"
    context_dir = tmp_path / "context_adaptive_budget_full_file_mode"
    shutil.copytree(case.path, workdir)
    patcher = _AdaptiveBudgetNoDiffPatcher(response_mode="full_file")
    controller, _events_path = _increase_biased_budget_controller(tmp_path, mode="on")

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
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-adaptive-budget-full-file-mode",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_budget_policy_controller=controller,
            adaptive_budget_policy_tiers="1.0,2.0",
        )
    finally:
        controller.close()

    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_budget_action_applied"] == ACTION_INCREASE_BUDGET
    assert attempt_meta["adaptive_budget_effective_response_mode"] == "full_file"
    assert attempt_meta["adaptive_budget_applied_controls_changed"]["response_mode"] is False


def test_parse_adaptive_budget_policy_tiers_validation() -> None:
    assert run_eval._parse_adaptive_budget_policy_tiers("1.0,1.5,2.0") == [1.0, 1.5, 2.0]
    invalid_values = (
        "",
        "1.5,2.0",
        "1.0,1.0,2.0",
        "2.0,1.0",
        "-1.0,1.0",
    )
    for raw in invalid_values:
        failed = False
        try:
            run_eval._parse_adaptive_budget_policy_tiers(raw)
        except ValueError:
            failed = True
        assert failed is True


def test_adaptive_online_local_reward_components_include_progress_and_stagnation_terms() -> None:
    (
        reward,
        success_hint,
        progress_hint,
        token_penalty,
        latency_penalty,
        stagnation_penalty,
        helper_materialization_bonus,
        _aux_total,
        _aux_pytest,
        _aux_novelty,
        _aux_compaction,
        _aux_informativeness,
    ) = run_eval._adaptive_online_local_reward_components(
        success=True,
        helper_materialized_context=True,
        local_tokens_total=1000,
        local_wall_sec=10.0,
        adaptive_progress_score=1.0,
        adaptive_no_progress_streak=0,
        lambda_tokens=0.1,
        mu_latency=0.2,
        token_scale=1000.0,
        latency_scale=10.0,
        eta_progress=0.15,
        kappa_stagnation=0.1,
    )
    assert success_hint == 1.0
    assert progress_hint == 0.15
    assert token_penalty == 0.1
    assert latency_penalty == 0.2
    assert helper_materialization_bonus == 0.075
    assert stagnation_penalty == 0.0
    assert reward == pytest.approx(0.925)


def test_adaptive_budget_local_reward_components_subtract_budget_penalty() -> None:
    (
        reward,
        success_hint,
        progress_hint,
        token_penalty,
        latency_penalty,
        stagnation_penalty,
        budget_exceeded_penalty,
        _aux_total,
        _aux_pytest,
        _aux_novelty,
    ) = run_eval._adaptive_budget_local_reward_components(
        success=False,
        local_tokens_total=0,
        local_wall_sec=0.0,
        adaptive_progress_score=0.0,
        adaptive_no_progress_streak=6,
        lambda_tokens=0.0,
        mu_latency=0.0,
        token_scale=1000.0,
        latency_scale=10.0,
        eta_progress=0.15,
        kappa_stagnation=0.1,
        rho_budget_exceeded=0.5,
        request_budget_exceeded=True,
    )
    assert success_hint == 0.0
    assert progress_hint == 0.0
    assert token_penalty == 0.0
    assert latency_penalty == 0.0
    assert stagnation_penalty == pytest.approx(0.1)
    assert budget_exceeded_penalty == 0.5
    assert reward == pytest.approx(-0.6)


def test_pytest_failed_total_from_text_counts_failed_and_errors() -> None:
    summary = "=== 2 failed, 1 error, 5 passed in 0.12s ==="
    assert run_eval._pytest_failed_total_from_text(summary) == 3


def test_adaptive_progress_signal_uses_weighted_components_and_reason_priority() -> None:
    before = run_eval.RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="=== 4 failed, 2 passed in 0.10s ===",
        stderr="",
    )
    after = run_eval.RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="=== 2 failed, 4 passed in 0.10s ===",
        stderr="",
    )
    progress = run_eval._adaptive_progress_signal(
        note="tests_still_failing",
        success=False,
        signature_repeated=False,
        new_evidence_count=1,
        before_run=before,
        after_run=after,
        previous_failure_context={"note": "patch_verifier_rejected", "patch_verifier_score": 1},
        patch_meta={
            "patch_verifier_score": 3,
            "patch_verifier_source_scope_matched": True,
        },
    )

    assert progress.made is True
    assert progress.reason == "failure_signature_changed"
    assert progress.pytest_before_failed_total == 4
    assert progress.pytest_after_failed_total == 2
    assert progress.stage_before == "verifier_rejected"
    assert progress.stage_after == "tests_still_failing"
    assert progress.score == pytest.approx(0.55)


def test_adaptive_online_delayed_rescue_bonus_uses_horizon_and_context_multiplier() -> None:
    # Only failure_signature_repeat triggers the context multiplier, and the helper's
    # evidence must reach the successful prompt (POL-11).
    assert run_eval._adaptive_online_delayed_rescue_bonus(
        origin_attempt=1,
        resolved_attempt=2,
        failure_signature_repeat=False,
        helper_evidence_ids=["file:helper.py"],
        prompt_evidence_ids=["file:helper.py"],
    ) == pytest.approx(0.5)
    assert run_eval._adaptive_online_delayed_rescue_bonus(
        origin_attempt=1,
        resolved_attempt=3,
        failure_signature_repeat=False,
        helper_evidence_ids=["file:helper.py"],
        prompt_evidence_ids=["file:helper.py", "frame:0"],
    ) == pytest.approx(0.25)


def test_adaptive_llm_discretion_resolves_delayed_helper_credit_on_next_success(
    tmp_path, monkeypatch
) -> None:
    records, patcher, events_path = _run_adaptive_delayed_credit_case(
        tmp_path,
        monkeypatch,
        patch_outcomes=("no_diff", "fix"),
        adaptive_max_helper_calls=2,
        k=2,
        run_id="run-adaptive-delayed-credit",
    )

    assert [bool(record["success"]) for record in records] == [False, True]
    assert patcher.request_calls == 2
    first_meta = records[0]["attempt_metadata"]
    # The first helper's evidence is sent again in the successful prompt (POL-11).
    assert set(records[0]["patch_meta"]["evidence_ids_new"]) & set(
        records[1]["patch_meta"]["evidence_ids_reused"]
    )
    assert first_meta["adaptive_online_reward_scope"] == "helper_credit_v2"
    assert first_meta["adaptive_online_reward_delayed_rescue_bonus"] == pytest.approx(0.5)
    assert first_meta["adaptive_online_reward_credit_horizon_attempts"] == 1
    assert first_meta["adaptive_online_reward_credit_resolution_reason"] == "future_success_attempt"

    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(event_lines) == 2


def test_adaptive_llm_discretion_withholds_rescue_bonus_when_evidence_not_resent(
    tmp_path, monkeypatch
) -> None:
    """POL-11: with one helper call, the success at attempt 2 never sees the helper's
    evidence. This scenario used to pay the 0.5 bonus on timing alone."""
    records, patcher, events_path = _run_adaptive_delayed_credit_case(
        tmp_path,
        monkeypatch,
        patch_outcomes=("no_diff", "fix"),
        adaptive_max_helper_calls=1,
        k=2,
        run_id="run-adaptive-delayed-credit-unused",
    )

    assert [bool(record["success"]) for record in records] == [False, True]
    assert patcher.request_calls == 1
    first_meta = records[0]["attempt_metadata"]
    assert first_meta["adaptive_online_reward_delayed_rescue_bonus"] == 0.0
    assert (
        first_meta["adaptive_online_reward_credit_resolution_reason"]
        == "local_only_evidence_not_in_success_prompt"
    )

    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(event_lines) == 1


def test_adaptive_llm_discretion_withholds_rescue_bonus_when_evidence_is_budgeted_out(
    tmp_path, monkeypatch
) -> None:
    """POL-11: the helper's evidence is selected for the successful prompt but the prompt
    budget trims the block out, so the rescue bonus is not earned (codex 3-1#2)."""
    records, patcher, events_path = _run_adaptive_delayed_credit_case(
        tmp_path,
        monkeypatch,
        patch_outcomes=("no_diff", "fix"),
        adaptive_max_helper_calls=2,
        k=2,
        run_id="run-adaptive-delayed-credit-trimmed",
        # Room for a few characters of the first block, so no `[id]` header survives.
        extra_run_kwargs={"prompt_budget_evidence_chars": 12},
    )

    assert [bool(record["success"]) for record in records] == [False, True]
    assert patcher.request_calls == 2
    # The helper's evidence was still selected for the successful attempt's prompt.
    assert set(records[0]["patch_meta"]["evidence_ids_new"]) & set(
        records[1]["patch_meta"]["evidence_ids_reused"]
    )
    final_prompt = patcher.prompts[-1]
    assert not any(
        f"[{evidence_id}]" in final_prompt
        for evidence_id in records[0]["patch_meta"]["evidence_ids_new"]
    )
    first_meta = records[0]["attempt_metadata"]
    assert first_meta["adaptive_online_reward_delayed_rescue_bonus"] == 0.0
    assert (
        first_meta["adaptive_online_reward_credit_resolution_reason"]
        == "local_only_evidence_not_in_success_prompt"
    )

    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(event_lines) == 2


def test_adaptive_llm_discretion_delayed_credit_sink_emits_resolved_case_end_metadata(
    tmp_path, monkeypatch
) -> None:
    streamed_records: list[dict[str, object]] = []

    def _record_sink(record: dict[str, object]) -> None:
        streamed_records.append(json.loads(json.dumps(record, ensure_ascii=False)))

    records, patcher, events_path = _run_adaptive_delayed_credit_case(
        tmp_path,
        monkeypatch,
        patch_outcomes=("no_diff",),
        adaptive_max_helper_calls=1,
        k=1,
        run_id="run-adaptive-delayed-credit-sink-case-end",
        extra_run_kwargs={"record_sink": _record_sink},
    )

    assert [bool(record["success"]) for record in records] == [False]
    assert patcher.request_calls == 1
    assert len(streamed_records) == 1
    attempt_meta = streamed_records[0]["attempt_metadata"]
    assert attempt_meta["adaptive_online_event_logged"] is True
    assert attempt_meta["adaptive_online_update_reason"] == "on_policy_update"
    assert (
        attempt_meta["adaptive_online_reward_credit_resolution_reason"] == "local_only_case_ended"
    )
    assert attempt_meta["adaptive_online_reward_delayed_rescue_bonus"] == pytest.approx(0.0)

    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(event_lines) == 1


def test_adaptive_llm_discretion_horizon_two_success_without_resent_evidence_earns_no_bonus(
    tmp_path, monkeypatch
) -> None:
    records, patcher, events_path = _run_adaptive_delayed_credit_case(
        tmp_path,
        monkeypatch,
        patch_outcomes=("no_diff", "no_diff", "fix"),
        adaptive_max_helper_calls=1,
        adaptive_budget_topup_max_calls=0,
        k=3,
        run_id="run-adaptive-delayed-credit-h2",
    )

    assert [bool(record["success"]) for record in records] == [False, False, True]
    assert patcher.request_calls == 1
    first_meta = records[0]["attempt_metadata"]
    # Two attempts later, without another helper call, the evidence isn't in the
    # successful prompt, so no bonus (POL-11; this used to pay 0.25 on timing alone).
    assert first_meta["adaptive_online_reward_delayed_rescue_bonus"] == 0.0
    assert (
        first_meta["adaptive_online_reward_credit_resolution_reason"]
        == "local_only_evidence_not_in_success_prompt"
    )

    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(event_lines) == 1


def test_adaptive_llm_discretion_credits_only_newest_helper_before_success(
    tmp_path, monkeypatch
) -> None:
    records, patcher, events_path = _run_adaptive_delayed_credit_case(
        tmp_path,
        monkeypatch,
        patch_outcomes=("no_diff", "no_diff", "fix"),
        adaptive_max_helper_calls=3,
        k=3,
        run_id="run-adaptive-delayed-credit-supersede",
    )

    assert [bool(record["success"]) for record in records] == [False, False, True]
    assert patcher.request_calls == 3
    first_meta = records[0]["attempt_metadata"]
    second_meta = records[1]["attempt_metadata"]
    assert first_meta["adaptive_online_reward_delayed_rescue_bonus"] == 0.0
    assert (
        first_meta["adaptive_online_reward_credit_resolution_reason"]
        == "local_only_superseded_by_newer_helper"
    )
    assert second_meta["adaptive_online_reward_delayed_rescue_bonus"] == pytest.approx(0.625)
    assert second_meta["adaptive_online_reward_credit_horizon_attempts"] == 1
    assert (
        second_meta["adaptive_online_reward_credit_resolution_reason"] == "future_success_attempt"
    )

    event_lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(event_lines) == 3


def test_run_one_condition_strips_eval_only_artifacts_from_workdir_and_prompt(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    (case.path / "case.json").write_text(
        json.dumps({"expected_exception": "AssertionError"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (case.path / "meta.json").write_text("{}", encoding="utf-8")
    (case.path / "oracle.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    workdir = tmp_path / "workdir_eval_artifacts"
    context_dir = tmp_path / "context_eval_artifacts"
    shutil.copytree(case.path, workdir)

    @dataclass
    class _CapturePromptPatcher:
        name: str = "capture_prompt"
        prompts: list[str] = field(default_factory=list)
        saw_meta_json: bool = False
        saw_oracle_patch: bool = False

        def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
            del context_dir
            self.prompts.append(prompt)
            self.saw_meta_json = (cwd / "case.json").exists() or (cwd / "meta.json").exists()
            self.saw_oracle_patch = (cwd / "oracle.patch").exists()
            return PatchResult(diff=None, meta={"reason": "no_diff"})

    patcher = _CapturePromptPatcher()

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

    run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-eval-artifact-strip",
        include_files_context=True,
        force_patch_attempt=False,
    )

    assert patcher.prompts
    prompt = patcher.prompts[0]
    assert patcher.saw_meta_json is False
    assert patcher.saw_oracle_patch is False
    assert "meta.json" not in prompt
    assert "oracle.patch" not in prompt
    assert "expected_exception" not in prompt
    assert not (workdir / "case.json").exists()
    assert not (workdir / "meta.json").exists()
    assert not (workdir / "oracle.patch").exists()


def test_finalize_adaptive_policy_uses_recomputed_progress_metadata() -> None:
    attempt_metadata: dict[str, object] = {
        "tokens_total": 0,
        "adaptive_progress_made": False,
        "adaptive_progress_score": 0.0,
        "adaptive_no_progress_streak": 3,
    }
    patch_meta: dict[str, object] = {}
    online_finalize_kwargs: dict[str, object] = {
        "is_adaptive_llm_discretion": True,
        "online_policy_decision": None,
        "adaptive_online_policy_controller": None,
        "run_id": "run-reward-sync",
        "case_id": "case-reward-sync",
        "condition_value": "layered_adaptive_llm",
        "attempt": 1,
        "attempt_metadata": attempt_metadata,
        "patch_meta": patch_meta,
        "online_policy_mode_for_attempt": "on",
        "online_policy_algo_for_attempt": "linucb",
        "online_policy_eligible": False,
        "online_policy_override_reason": "",
        "online_policy_feature_scaling_profile": "v1",
        "online_policy_features_raw": {},
        "online_policy_features_scaled": {},
        "helper_decision_reason": "llm_discretion_pending",
        "normalized_adaptive_online_policy_mode": "on",
        # Deliberately stale values: should be overwritten from attempt_metadata.
        "adaptive_progress_made": True,
        "adaptive_progress_score": 1.0,
        "adaptive_no_progress_streak_current": 0,
        "reward_lambda_tokens": 0.0,
        "reward_mu_latency": 0.0,
        "reward_token_scale": 1000.0,
        "reward_latency_scale": 10.0,
        "reward_eta_progress": 0.15,
        "reward_kappa_stagnation": 0.2,
    }
    budget_finalize_kwargs: dict[str, object] = {
        "is_adaptive_condition": False,
        "budget_policy_decision": None,
        "adaptive_budget_policy_controller": None,
        "run_id": "run-reward-sync",
        "case_id": "case-reward-sync",
        "condition_value": "layered_adaptive_llm",
        "attempt": 1,
        "attempt_metadata": attempt_metadata,
        "patch_meta": patch_meta,
        "budget_policy_mode_for_attempt": "on",
        "budget_policy_algo_for_attempt": "linucb",
        "budget_policy_eligible": False,
        "budget_policy_override_reason": "",
        "budget_policy_feature_scaling_profile": "v1",
        "budget_policy_features_raw": {},
        "budget_policy_features_scaled": {},
        "normalized_adaptive_budget_policy_mode": "on",
        # Deliberately stale values: should be overwritten from attempt_metadata.
        "adaptive_progress_made": True,
        "adaptive_progress_score": 1.0,
        "adaptive_no_progress_streak_current": 0,
        "reward_lambda_tokens": 0.0,
        "reward_mu_latency": 0.0,
        "reward_token_scale": 1000.0,
        "reward_latency_scale": 10.0,
        "reward_eta_progress": 0.15,
        "reward_kappa_stagnation": 0.2,
        "reward_rho_budget_exceeded": 0.5,
        "tiers": [1.0, 1.5],
        "tier_index_before": 0,
        "tier_index_after": 0,
        "selected_multiplier": 1.0,
        "applied_multiplier": 1.0,
        "default_multiplier": 1.0,
        "max_ups_per_case": 1,
        "ups_used": 0,
    }

    run_eval._finalize_adaptive_policy_attempt(
        note="tests_still_failing",
        success=False,
        attempt_wall_sec=0.0,
        online_finalize_kwargs=online_finalize_kwargs,
        budget_finalize_kwargs=budget_finalize_kwargs,
    )

    assert attempt_metadata["adaptive_online_reward_progress_hint"] == 0.0
    assert attempt_metadata["adaptive_online_reward_stagnation_penalty"] == pytest.approx(0.2)
    assert attempt_metadata["adaptive_budget_reward_progress_hint"] == 0.0
    assert attempt_metadata["adaptive_budget_reward_stagnation_penalty"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 2026-10-04 review, Phase 1C acceptance: evidence enforcement per action.
# ---------------------------------------------------------------------------

_ASK_FOR_EVERYTHING = {
    "need_more_context": True,
    "need_snapshot": True,
    "snapshot_justification": "need the local value of x in f() to see the wrong helper value",
    "requests": [
        {"type": "file", "path": "buggy.py"},
        {"type": "file", "path": "helper.py"},
        {"type": "search", "path": "helper.py", "pattern": "return"},
        {"type": "frame", "index": 0},
        {"type": "local_var", "frame": 0, "name": "x"},
    ],
    "sufficiency_rationale": "need everything",
}

_RESTRICTED_EVIDENCE = {
    "frame_only": {"file", "search", "seed", "snapshot"},
    "file_only": {"search", "frame", "local_var", "seed", "snapshot"},
    "search_only": {"file", "frame", "local_var", "seed", "snapshot"},
    "skip_helper": {"file", "search", "frame", "local_var", "seed", "snapshot"},
    "snapshot_lite": {"file", "search", "frame", "local_var", "seed", "snapshot"},
    "frame_locals_lite": {"file", "search", "frame", "local_var", "seed", "snapshot"},
    "snapshot_full_lite": {"file", "search", "frame", "local_var", "seed", "snapshot"},
}


def _evidence_for_action(tmp_path, monkeypatch, action: str) -> tuple[str, dict[str, int]]:
    patcher = _AdaptiveActionCapturePatcher(request_payload=dict(_ASK_FOR_EVERYTHING))
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path / action,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=action,
        run_id=f"run-evidence-{action}",
        extra_run_kwargs={"adaptive_reveal_mode": "full"},
    )
    meta = records[0]["attempt_metadata"]
    assert isinstance(meta, dict)
    return str(meta["adaptive_helper_action_executed"]), dict(meta["evidence_chars_by_type"])


@pytest.mark.parametrize("action", sorted(_RESTRICTED_EVIDENCE))
def test_restricted_actions_deliver_only_their_evidence_types(
    tmp_path, monkeypatch, action: str
) -> None:
    """ATT-03/04/06: denied evidence types never reach the Stage-B prompt."""
    (tmp_path / action).mkdir()
    executed, chars = _evidence_for_action(tmp_path, monkeypatch, action)
    assert executed == action
    leaked = {kind for kind in _RESTRICTED_EVIDENCE[action] if chars.get(kind, 0) > 0}
    assert not leaked, f"{action} leaked {sorted(leaked)}: {chars}"
    granted = {
        "frame_only": ("frame",),
        "file_only": ("file",),
        "search_only": ("search",),
        "snapshot_lite": ("snapshot_lite",),
        "frame_locals_lite": ("snapshot_lite",),
        "snapshot_full_lite": ("snapshot_lite",),
    }.get(action, ())
    assert all(chars.get(kind, 0) > 0 for kind in granted), f"{action} lost its own evidence"


def test_no_restricted_action_delivers_more_evidence_than_invoke_helper(
    tmp_path, monkeypatch
) -> None:
    totals: dict[str, int] = {}
    for action in ["invoke_helper", "frame_only", "file_only", "search_only", "skip_helper"]:
        (tmp_path / action).mkdir()
        executed, chars = _evidence_for_action(tmp_path, monkeypatch, action)
        assert executed == action
        totals[action] = sum(chars.values())
    assert totals["invoke_helper"] > 0
    assert all(totals[action] <= totals["invoke_helper"] for action in totals), totals


@pytest.mark.parametrize(
    ("action", "denied_request"),
    [
        (ACTION_SEARCH_ONLY, {"type": "file", "path": "buggy.py"}),
        (ACTION_FILE_ONLY, {"type": "search", "path": "helper.py", "pattern": "return"}),
        (ACTION_FRAME_ONLY, {"type": "file", "path": "helper.py"}),
    ],
)
def test_att03_fully_filtered_restricted_action_gets_no_monolithic_file_context(
    tmp_path, monkeypatch, action: str, denied_request: dict[str, object]
) -> None:
    """ATT-03: an emptied restricted request must not restore whole-project file context.

    With ``include_files_context=True`` the Stage-B prompt used to fall back to the
    monolithic ``Project files:`` block once layer-3 filtering removed every request,
    handing back exactly the evidence the policy denied.
    """
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "requests": [dict(denied_request)],
            "sufficiency_rationale": "need the denied evidence type",
        }
    )
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=action,
        run_id=f"run-att03-{action}",
        include_snapshot=False,
        include_files_context=True,
    )

    patch_meta = records[0]["patch_meta"]
    assert isinstance(patch_meta, dict)
    assert patch_meta["stage_a_requests_count"] == 0
    assert patch_meta["stage_a_reason"] == "restricted_action_no_valid_requests"
    assert patch_meta["context_protocol"] == "staged"

    prompt = patcher.patch_prompts[0]
    assert "Project files:" not in prompt
    assert "def value() -> int:" not in prompt


def test_att03_unrestricted_action_keeps_monolithic_file_context(tmp_path, monkeypatch) -> None:
    """The monolithic fallback stays for actions that never restricted the evidence types."""
    patcher = _AdaptiveActionCapturePatcher(request_payload=None)
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SKIP_HELPER,
        run_id="run-att03-skip",
        include_snapshot=False,
        include_files_context=True,
    )
    patch_meta = records[0]["patch_meta"]
    assert isinstance(patch_meta, dict)
    assert patch_meta["context_protocol"] == "monolithic"
    assert "Project files:" in patcher.patch_prompts[0]


def test_att06_delta_retry_does_not_claim_a_snapshot_it_never_delivered(
    tmp_path, monkeypatch
) -> None:
    """ATT-06: the snapshot fields describe the delivered prompt, not the Stage-A grant.

    Delta retry prompts omit the snapshot section entirely, so a granted snapshot on
    attempt 2 must not be recorded as included.
    """
    patcher = _AdaptiveActionCapturePatcher(request_payload=dict(_ASK_FOR_EVERYTHING))
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_INVOKE_HELPER,
        run_id="run-att06-delta",
        k=2,
        adaptive_helper_max_calls_per_case=4,
        extra_run_kwargs={"adaptive_reveal_mode": "full", "retry_prompt_mode": "delta"},
    )

    first, second = records[0], records[1]
    assert first["patch_meta"]["retry_delta_prompt"] is False
    assert first["attempt_metadata"]["stage_b_snapshot_in_prompt"] is True
    assert first["attempt_metadata"]["adaptive_snapshot_included"] is True
    assert "Snapshot" in patcher.patch_prompts[0]

    assert second["patch_meta"]["retry_delta_prompt"] is True
    assert second["attempt_metadata"]["adaptive_snapshot_requested"] is True
    assert second["attempt_metadata"]["stage_b_snapshot_in_prompt"] is False
    assert second["attempt_metadata"]["adaptive_snapshot_included"] is False
    assert second["attempt_metadata"]["evidence_chars_by_type"]["snapshot"] == 0


def test_att06_compaction_replacing_the_snapshot_clears_the_inclusion_fields(
    tmp_path, monkeypatch
) -> None:
    """ATT-06: successful compaction replaces the snapshot section, so it is not included."""
    patcher = _AdaptiveActionCapturePatcher(
        compacted_text="crash in buggy.py line 4; helper.value() returns 1 but test expects 2",
        compact_meta={"llm_call_count": 1},
    )
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_COMPACT_THEN_PATCH,
        run_id="run-att06-compaction",
        extra_run_kwargs={"adaptive_reveal_mode": "full"},
    )

    attempt_meta = records[0]["attempt_metadata"]
    assert attempt_meta["context_compaction_invoked"] is True
    assert records[0]["patch_meta"]["context_compaction_success"] is True
    assert "snapshot" in records[0]["patch_meta"]["context_compaction_replaced_sections"]
    assert attempt_meta["stage_b_snapshot_in_prompt"] is False
    assert attempt_meta["adaptive_snapshot_included"] is False
    assert attempt_meta["evidence_chars_by_type"]["snapshot"] == 0


def test_pol12_no_decision_logs_no_scaled_vector_under_percentile_scaling(
    tmp_path, monkeypatch
) -> None:
    """POL-12: only decide() knows percentile caps, so no decision means no scaled vector."""
    from dataclasses import replace as dc_replace

    original_factory = _preferred_action_online_controller

    def _percentile_controller(
        tmp_path_arg, *, mode: str, preferred_action: str, fallback_action: str | None = None
    ):
        controller, events = original_factory(
            tmp_path_arg,
            mode=mode,
            preferred_action=preferred_action,
            fallback_action=fallback_action,
        )
        controller.config = dc_replace(
            controller.config, mode="off", feature_scaling_profile="percentile_v1"
        )
        return controller, events

    monkeypatch.setitem(globals(), "_preferred_action_online_controller", _percentile_controller)
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=_AdaptiveActionCapturePatcher(),
        preferred_action=ACTION_INVOKE_HELPER,
        run_id="run-pol12",
    )
    meta = records[0]["attempt_metadata"]
    assert meta["adaptive_online_feature_scaling_profile"] == "percentile_v1"
    assert meta["adaptive_online_policy_mode"] == "off"
    assert meta["adaptive_online_feature_vector"] == {}
    assert meta["adaptive_online_feature_vector_raw"]


def test_bas07_native_attempt_records_a_case_information_manifest(tmp_path, monkeypatch) -> None:
    """BAS-07 (codex 1a#3): native arms record which case files reached their prompt."""
    patcher = _AdaptiveActionCapturePatcher(request_payload=None)
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SKIP_HELPER,
        run_id="run-bas07-files",
        include_snapshot=False,
        include_files_context=True,
    )
    manifest = records[0]["patch_meta"]["case_information"]
    assert isinstance(manifest, dict)
    assert manifest["case_info_policy"] == "neutral_v1"
    assert manifest["test_source_exposure"] == "full"
    assert manifest["prompt_files"] == ["buggy.py", "helper.py", "test_buggy.py"]
    assert "Reference tests" not in patcher.patch_prompts[0]
    assert "def test_f() -> None:" in patcher.patch_prompts[0]


def test_bas07_native_attempt_without_file_context_records_no_test_source(
    tmp_path, monkeypatch
) -> None:
    """The omitted-test-source half: no project files and no staged file evidence."""
    patcher = _AdaptiveActionCapturePatcher(request_payload=None)
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_SKIP_HELPER,
        run_id="run-bas07-nofiles",
        include_snapshot=False,
        include_files_context=False,
    )
    manifest = records[0]["patch_meta"]["case_information"]
    assert manifest["test_source_exposure"] == "none"
    assert manifest["prompt_files"] == []
    assert "def test_f() -> None:" not in patcher.patch_prompts[0]


def test_bas07_native_manifest_lists_only_staged_file_evidence(tmp_path, monkeypatch) -> None:
    """A restricted file-only action reports just the file it was granted."""
    patcher = _AdaptiveActionCapturePatcher(
        request_payload={
            "need_more_context": True,
            "requests": [{"type": "file", "path": "test_buggy.py"}],
            "sufficiency_rationale": "need the failing test",
        }
    )
    records, _events_path = _run_adaptive_action_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        patcher=patcher,
        preferred_action=ACTION_FILE_ONLY,
        run_id="run-bas07-staged",
        include_snapshot=False,
        include_files_context=True,
    )
    manifest = records[0]["patch_meta"]["case_information"]
    assert manifest["prompt_files"] == ["test_buggy.py"]
    assert manifest["test_source_exposure"] == "full"
