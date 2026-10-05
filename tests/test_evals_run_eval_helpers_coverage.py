"""Coverage tests for evals.run_eval helper layers.

Targets the pure helper branches that are not exercised by the higher-level
flow tests: record/result helpers, run-condition resolution, OpenAI transport
recovery, run-state repair, pytest/container wrappers, budget helpers, and the
patch-relevance heuristics.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from evals import run_eval
from evals.eval_types import (
    ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
    ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
    AdaptiveRuntimeConfig,
)
from evals.patchers.base import (
    ContextCompactionResult,
    ContextRequestResult,
    PatchPlanResult,
    PatchResult,
)
from evals.run_condition_builder import build_run_condition_inputs


@dataclass(frozen=True)
class _ProbePatcher:
    name: str = "openai"
    max_propose_sec: float = 0.0
    model: str = "demo-model"
    base_url: str = "http://example.invalid/v1"
    response_mode: str = "edits"
    probe_results: tuple[object, ...] = ()
    probe_calls: int = 0

    def probe_server_ready(self):
        if self.probe_calls >= len(self.probe_results):
            return True, {"server_ready_probe_provider": "stub"}
        value = self.probe_results[self.probe_calls]
        object.__setattr__(self, "probe_calls", self.probe_calls + 1)
        if isinstance(value, Exception):
            raise value
        return value

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff="diff --git a/a b/a\n", meta={"note": "stub"})

    def plan_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchPlanResult:
        del prompt, context_dir, cwd
        return PatchPlanResult(plan_text="plan", raw_response="plan", meta={"note": "stub"})


@dataclass(frozen=True)
class _BudgetPatcher:
    name: str = "openai"
    max_propose_sec: float = 0.0
    model: str = "m"
    base_url: str = "http://example.invalid/v1"
    response_mode: str = "edits"

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff="diff --git a/a b/a\n", meta={"note": "stub"})

    def plan_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchPlanResult:
        del prompt, context_dir, cwd
        return PatchPlanResult(plan_text="plan", raw_response="plan", meta={"note": "stub"})


class _FakeWaitHandle:
    def __init__(self, *, result: object = 0, logs: tuple[bytes, bytes] = (b"OUT", b"ERR")):
        self.result = result
        self.logs = logs
        self.kill_called = False

    def wait(self, timeout: int):
        del timeout
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def kill(self) -> None:
        self.kill_called = True


class _FakeDockerContainer:
    created: list[_FakeDockerContainer] = []

    def __init__(self, image: str) -> None:
        self.image = image
        self.volume_mappings: list[tuple[str, str, str]] = []
        self.command: list[str] | None = None
        self.kwargs: dict[str, object] = {}
        self.env: dict[str, str] = {}
        self.started = False
        self.stopped = False
        self.wrapped = _FakeWaitHandle()
        self.fail_on_wait = False
        self.raise_on_wait = False
        self.return_none_handle = False
        _FakeDockerContainer.created.append(self)

    def with_volume_mapping(self, src: str, dst: str, mode: str = "rw") -> _FakeDockerContainer:
        self.volume_mappings.append((src, dst, mode))
        return self

    def with_command(self, command: list[str]) -> _FakeDockerContainer:
        self.command = list(command)
        return self

    def with_kwargs(self, **kwargs: object) -> _FakeDockerContainer:
        self.kwargs.update(kwargs)
        return self

    def with_env(self, key: str, value: str) -> _FakeDockerContainer:
        self.env[key] = value
        return self

    def start(self) -> None:
        self.started = True

    def get_wrapped_container(self):
        if self.return_none_handle:
            return None
        return self.wrapped

    def get_logs(self) -> tuple[bytes, bytes]:
        return self.wrapped.logs

    def stop(self) -> None:
        self.stopped = True


def _make_case(tmp_path: Path, case_id: str = "case_1") -> run_eval.CaseSpec:
    case_dir = tmp_path / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="logic_error",
        difficulty="easy",
        snapshot_dependency="required",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def _run_condition_inputs() -> run_eval.RunConditionInputs:
    return run_eval.RunConditionInputs(
        patcher=SimpleNamespace(name="stub"),
        k=2,
        output_format="json",
        run_id="run-1",
        include_files_context=False,
        force_patch_attempt=False,
    )


def _builder_args() -> SimpleNamespace:
    return SimpleNamespace(
        context_protocol="staged",
        context_stage_a_max_requests=2,
        context_stage_a_max_chars=4000,
        retry_prompt_mode="delta",
        retry_context_mode="history",
        retry_history_max_chars=8000,
        prompt_retry_reflection=True,
        prompt_adaptive_stage_b_cot=False,
        prompt_repair_method_scaffold=True,
        prompt_retry_plan_then_patch=True,
        pytest_output_tail_lines=80,
        snapshot_artifact_max_string_chars=2000,
        prompt_budget_total_chars=32000,
        prompt_budget_retry_delta_chars=4000,
        prompt_budget_stdout_chars=8000,
        prompt_budget_snapshot_chars=12000,
        prompt_budget_evidence_chars=12000,
        prompt_budget_retry_history_chars=6000,
        context_compaction_mode="adaptive",
        context_compaction_scope="stage_b",
        context_compaction_max_input_chars=16000,
        context_compaction_target_chars=8000,
        context_compaction_max_calls=1,
        adaptive_gate_policy="balanced",
        adaptive_max_helper_calls=3,
        adaptive_budget_topup_max_calls=1,
        adaptive_stagnation_window=2,
        adaptive_saturation_threshold=0.8,
        adaptive_no_progress_token_cap=1000,
        adaptive_helper_budget_profile="default",
        adaptive_openai_helper_response_mode="responses",
        adaptive_controller_lambda_rlm=True,
        adaptive_controller_lambda_rlm_repo_dir="/tmp/lambda-rlm",
        adaptive_controller_lambda_rlm_python="/tmp/lambda-python",
        adaptive_controller_lambda_rlm_context_window_chars=4000,
        adaptive_controller_lambda_rlm_timeout_sec=30,
        adaptive_informativeness_pregate=True,
        adaptive_informativeness_pregate_threshold=0.3,
        adaptive_online_policy_mode="on",
        adaptive_online_policy_algo="linucb",
        adaptive_online_feature_scaling_profile="v1",
        adaptive_budget_policy_mode="on",
        adaptive_budget_policy_algo="linucb",
        adaptive_budget_policy_architecture="tier_binary",
        adaptive_budget_feature_scaling_profile="v1",
        adaptive_budget_policy_tiers="1.0,1.5",
        adaptive_budget_policy_max_ups_per_case=2,
        adaptive_budget_reward_rho_budget_exceeded=0.5,
        adaptive_reward_lambda_tokens=0.1,
        adaptive_reward_mu_latency=0.2,
        adaptive_reward_token_scale=1000.0,
        adaptive_reward_latency_scale=60.0,
        adaptive_reward_eta_progress=0.3,
        adaptive_reward_kappa_stagnation=0.4,
        adaptive_reward_success_hint=ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
        adaptive_reward_progress_scale=ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
        adaptive_reward_stagnation_scale=ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
        adaptive_reward_materialization_scale=ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
        adaptive_toon_encoding=False,
        syntax_preflight_enabled=True,
        syntax_preflight_max_cycles=2,
        syntax_preflight_fallthrough=False,
        patch_sanitize_enabled=True,
        patch_verifier_scope="adaptive_only",
        patch_verifier_min_score=75,
        patch_verifier_infer_test_source_scope=True,
        openai_response_mode_autoswitch="retry_text",
        openai_transport_recovery=True,
        record_prompt="hash",
        record_evidence="full",
        deterministic=True,
        seed=7,
        k=3,
        output_format="jsonl",
        no_files_context=False,
        force_patch_attempt=True,
        max_context_chars=12000,
        case_fail_fast_on_request_budget_exceeded=True,
    )


class TestBasicHelpers:
    def test_string_and_record_and_result_helpers(self) -> None:
        assert run_eval._single_line_error("alpha\nbeta") == "alpha | beta"
        assert run_eval._single_line_error("   \n  ") == "unknown runtime error"
        assert run_eval._single_line_error("x" * 50, max_chars=10) == "xxxxxxx..."

        assert run_eval._is_transport_runtime_error("Connection refused") is True
        assert run_eval._is_transport_runtime_error("brokenpipeerror during write") is True
        assert run_eval._is_transport_runtime_error("ok") is False

        attempt = {"record_type": run_eval.EVAL_RECORD_TYPE_ATTEMPT}
        infra = {"record_type": run_eval.EVAL_RECORD_TYPE_INFRA_EVENT}
        default_attempt = {"note": "fallback"}
        assert run_eval._record_type(attempt) == run_eval.EVAL_RECORD_TYPE_ATTEMPT
        assert run_eval._record_type(infra) == run_eval.EVAL_RECORD_TYPE_INFRA_EVENT
        assert run_eval._record_type(default_attempt) == run_eval.EVAL_RECORD_TYPE_ATTEMPT
        assert run_eval._is_attempt_record(attempt) is True
        assert run_eval._is_attempt_record(infra) is False
        assert run_eval._filter_attempt_records([attempt, infra, default_attempt]) == [
            attempt,
            default_attempt,
        ]

        class _Obj:
            def __init__(self, meta: object) -> None:
                self.meta = meta

        assert run_eval._result_meta_dict(_Obj({"a": 1})) == {"a": 1}
        assert run_eval._result_meta_dict(_Obj(None)) == {}
        assert run_eval._result_meta_dict(object()) == {}

        patch_result = PatchResult(diff="d", meta={"old": True})
        plan_result = PatchPlanResult(plan_text="plan", raw_response="raw", meta={"old": True})
        request_result = ContextRequestResult(
            request_json={"x": 1},
            raw_response="raw",
            meta={"old": True},
        )
        compact_result = ContextCompactionResult(
            compacted_text="abc",
            raw_response="raw",
            meta={"old": True},
        )
        assert run_eval._result_with_meta(patch_result, {"new": 1}) == PatchResult(
            diff="d",
            meta={"new": 1},
        )
        assert run_eval._result_with_meta(plan_result, {"new": 1}) == PatchPlanResult(
            plan_text="plan",
            raw_response="raw",
            meta={"new": 1},
        )
        assert run_eval._result_with_meta(request_result, {"new": 1}) == ContextRequestResult(
            request_json={"x": 1},
            raw_response="raw",
            meta={"new": 1},
        )
        assert run_eval._result_with_meta(compact_result, {"new": 1}) == ContextCompactionResult(
            compacted_text="abc",
            raw_response="raw",
            meta={"new": 1},
        )
        sentinel = object()
        assert run_eval._result_with_meta(sentinel, {"new": 1}) is sentinel

    def test_duration_and_verifier_scope_helpers(self) -> None:
        assert run_eval._format_duration(None) == "unknown"
        assert run_eval._format_duration(59.2) == "59s"
        assert run_eval._format_duration(65.0) == "1m05s"
        assert run_eval._format_duration(3661.0) == "1h01m01s"

        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope=run_eval.PATCH_VERIFIER_SCOPE_DISABLED,
                is_adaptive_condition=True,
            )
            == run_eval.PATCH_VERIFIER_SCOPE_DISABLED
        )
        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope=run_eval.PATCH_VERIFIER_SCOPE_ALL,
                is_adaptive_condition=False,
            )
            == run_eval.PATCH_VERIFIER_SCOPE_ALL
        )
        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope=run_eval.PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
                is_adaptive_condition=True,
            )
            == run_eval.PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY
        )
        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope=run_eval.PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
                is_adaptive_condition=False,
            )
            == run_eval.PATCH_VERIFIER_SCOPE_DISABLED
        )


class TestRunConditionInputs:
    def test_build_run_condition_inputs_groups_related_settings(self) -> None:
        args = _builder_args()
        patcher = SimpleNamespace(name="stub")
        online_policy = object()
        budget_policy = object()
        transport_gate = object()
        record_sink = object()
        run_metadata = {"run_id": "run-1"}

        inputs = build_run_condition_inputs(
            args=args,
            run_id="run-1",
            run_metadata=run_metadata,
            patcher=patcher,
            online_policy_controller=online_policy,
            budget_policy_controller=budget_policy,
            openai_context_window_tokens_effective=8192,
            openai_context_window_source="provider_probe",
            openai_context_window_probe_provider="stub",
            openai_context_window_probe_error="",
            transport_recovery_gate=transport_gate,
            record_sink=record_sink,
        )

        assert inputs.prompt_assembly.context_protocol == "staged"
        assert inputs.prompt_assembly.prompt_retry_plan_then_patch is True
        assert inputs.adaptive_runtime.adaptive_online_policy_controller is online_policy
        assert inputs.adaptive_runtime.adaptive_budget_policy_controller is budget_policy
        assert inputs.adaptive_runtime.adaptive_controller_lambda_rlm is True
        assert inputs.verification.patch_verifier_min_score == 75
        assert inputs.transport.openai_context_window_tokens == 8192
        assert inputs.transport.transport_recovery_gate is transport_gate
        assert inputs.recording.run_metadata == run_metadata
        assert inputs.recording.record_sink is record_sink
        assert inputs.patcher is patcher

    def test_build_run_condition_inputs_keeps_aux_reward_default_in_sync(self) -> None:
        args = _builder_args()
        inputs = build_run_condition_inputs(
            args=args,
            run_id="run-1",
            run_metadata={},
            patcher=SimpleNamespace(name="stub"),
            online_policy_controller=None,
            budget_policy_controller=None,
            openai_context_window_tokens_effective=8192,
            openai_context_window_source="provider_probe",
            openai_context_window_probe_provider="stub",
            openai_context_window_probe_error="",
            transport_recovery_gate=None,
            record_sink=None,
        )

        assert inputs.adaptive_runtime.enable_auxiliary_rewards is True
        assert inputs.enable_auxiliary_rewards is AdaptiveRuntimeConfig().enable_auxiliary_rewards

    def test_build_run_condition_inputs_maps_legacy_and_full_tep_modes(self) -> None:
        legacy_args = _builder_args()
        legacy_args.adaptive_tep = True
        legacy_args.adaptive_tep_equilibrium = True
        legacy_args.adaptive_tep_mode = "off"
        legacy_args.adaptive_tep_state_path = ""
        legacy_args.adaptive_tep_init_state_path = ""
        legacy_args.adaptive_tep_beta_init = 1.0
        legacy_args.adaptive_tep_beta_decay = 0.9
        legacy_args.adaptive_tep_full_equilibrium_max_iters = 20
        legacy_args.adaptive_tep_full_equilibrium_window = 3
        legacy_args.adaptive_tep_full_equilibrium_variance_threshold = 0.5
        legacy_args.adaptive_tep_validation_buffer_size = 4
        legacy_inputs = build_run_condition_inputs(
            args=legacy_args,
            run_id="run-1",
            run_metadata={},
            patcher=SimpleNamespace(name="stub"),
            online_policy_controller=None,
            budget_policy_controller=None,
            openai_context_window_tokens_effective=8192,
            openai_context_window_source="provider_probe",
            openai_context_window_probe_provider="stub",
            openai_context_window_probe_error="",
            transport_recovery_gate=None,
            record_sink=None,
        )
        assert legacy_inputs.adaptive_runtime.adaptive_tep_mode == "legacy"

        full_args = _builder_args()
        full_args.adaptive_tep = False
        full_args.adaptive_tep_equilibrium = False
        full_args.adaptive_tep_mode = "full"
        full_args.adaptive_tep_state_path = "/tmp/full_tep_state.json"
        full_args.adaptive_tep_init_state_path = "/tmp/init_tep_state.json"
        full_args.adaptive_tep_beta_init = 0.8
        full_args.adaptive_tep_beta_decay = 0.7
        full_args.adaptive_tep_full_equilibrium_max_iters = 9
        full_args.adaptive_tep_full_equilibrium_window = 5
        full_args.adaptive_tep_full_equilibrium_variance_threshold = 0.25
        full_args.adaptive_tep_validation_buffer_size = 6
        full_inputs = build_run_condition_inputs(
            args=full_args,
            run_id="run-1",
            run_metadata={},
            patcher=SimpleNamespace(name="stub"),
            online_policy_controller=None,
            budget_policy_controller=None,
            openai_context_window_tokens_effective=8192,
            openai_context_window_source="provider_probe",
            openai_context_window_probe_provider="stub",
            openai_context_window_probe_error="",
            transport_recovery_gate=None,
            record_sink=None,
        )
        assert full_inputs.adaptive_runtime.adaptive_tep_mode == "full"
        assert full_inputs.adaptive_runtime.adaptive_tep_state_path == "/tmp/full_tep_state.json"
        assert (
            full_inputs.adaptive_runtime.adaptive_tep_init_state_path == "/tmp/init_tep_state.json"
        )
        assert full_inputs.adaptive_runtime.adaptive_tep_beta_init == pytest.approx(0.8)
        assert full_inputs.adaptive_runtime.adaptive_tep_beta_decay == pytest.approx(0.7)
        assert full_inputs.adaptive_runtime.adaptive_tep_equilibrium_max_iters == 9
        assert full_inputs.adaptive_runtime.adaptive_tep_equilibrium_window == 5
        assert (
            full_inputs.adaptive_runtime.adaptive_tep_equilibrium_variance_threshold
            == pytest.approx(0.25)
        )
        assert full_inputs.adaptive_runtime.adaptive_tep_validation_buffer_size == 6

    def test_build_and_resolve_legacy_kwargs(self) -> None:
        inputs = _run_condition_inputs()
        legacy = dict(inputs.__dict__)
        resolved = run_eval._build_run_condition_inputs_from_legacy_kwargs(legacy_kwargs=legacy)
        assert resolved == inputs
        assert run_eval._resolve_run_condition_inputs(inputs=inputs, legacy_kwargs={}) == inputs

        with pytest.raises(TypeError, match="missing required input fields"):
            run_eval._build_run_condition_inputs_from_legacy_kwargs(
                legacy_kwargs={"patcher": SimpleNamespace(name="x")}
            )

        with pytest.raises(TypeError, match="received unknown input fields"):
            run_eval._build_run_condition_inputs_from_legacy_kwargs(
                legacy_kwargs={**legacy, "unknown": 1}
            )

        with pytest.raises(TypeError, match="received both inputs"):
            run_eval._resolve_run_condition_inputs(inputs=inputs, legacy_kwargs={"k": 1})

        without_patcher = run_eval.RunConditionInputs(
            patcher=None,
            k=1,
            output_format="json",
            run_id="run-1",
            include_files_context=False,
            force_patch_attempt=False,
        )
        with pytest.raises(TypeError, match="requires a patcher"):
            run_eval._resolve_run_condition_inputs(inputs=without_patcher, legacy_kwargs={})


class TestOpenAIContextWindowResolution:
    def test_manual_probe_and_disabled_branches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        args = SimpleNamespace(
            openai_context_window_tokens=4096,
            patcher="openai",
            openai_auto_context_window=True,
        )
        patcher = SimpleNamespace(name="openai")
        tokens, info = run_eval._resolve_openai_context_window_for_run(args=args, patcher=patcher)
        assert tokens == 4096
        assert info["openai_context_window_source"] == "manual_flag"

        args.openai_context_window_tokens = 0
        args.openai_auto_context_window = False
        tokens, info = run_eval._resolve_openai_context_window_for_run(args=args, patcher=patcher)
        assert tokens == 0
        assert info["openai_context_window_source"] == "disabled"

        args.openai_auto_context_window = True
        args.patcher = "null"
        tokens, info = run_eval._resolve_openai_context_window_for_run(args=args, patcher=patcher)
        assert tokens == 0
        assert info["openai_context_window_source"] == "disabled"

        args.patcher = "openai"
        patcher = SimpleNamespace(name="openai")
        tokens, info = run_eval._resolve_openai_context_window_for_run(args=args, patcher=patcher)
        assert tokens == 0
        assert info["openai_context_window_source"] == "auto_probe_unavailable"

        def _probe() -> tuple[int, dict[str, object]]:
            return 8192, {"context_window_probe_provider": "stub", "context_window_source": "auto"}

        patcher = SimpleNamespace(name="openai", probe_context_window_tokens=_probe)
        tokens, info = run_eval._resolve_openai_context_window_for_run(args=args, patcher=patcher)
        assert tokens == 8192
        assert info["openai_context_window_probe_provider"] == "stub"
        assert info["openai_context_window_source"] == "auto"

        def _probe_bad() -> tuple[str, dict[str, object]]:
            return "bad", {"error": "probe_failed"}

        patcher = SimpleNamespace(name="openai", probe_context_window_tokens=_probe_bad)
        tokens, info = run_eval._resolve_openai_context_window_for_run(args=args, patcher=patcher)
        assert tokens == 0
        assert info["openai_context_window_source"] == "auto_probe_failed"
        assert info["openai_context_window_probe_error"] == "probe_failed"

        def _probe_raises() -> tuple[int, dict[str, object]]:
            raise RuntimeError("boom")

        patcher = SimpleNamespace(name="openai", probe_context_window_tokens=_probe_raises)
        tokens, info = run_eval._resolve_openai_context_window_for_run(args=args, patcher=patcher)
        assert tokens == 0
        assert info["openai_context_window_source"] == "auto_probe_failed"
        assert "RuntimeError" in info["openai_context_window_probe_error"]


class TestOpenAITransportRecoveryGate:
    def test_disabled_and_recovery_branches(self) -> None:
        disabled = run_eval._OpenAITransportRecoveryGate(
            enabled=False,
            max_wait_sec=0.0,
            poll_sec=0.01,
        )
        assert disabled.recover(patcher=SimpleNamespace(), operation_name="probe") == {
            "attempted": False,
            "recovered": False,
            "timed_out": True,
            "wait_sec": 0.0,
            "probe_attempts": [],
            "probe_meta": {},
            "operation": "probe",
            "leader": False,
            "generation": 0,
        }

        probe_results = (
            RuntimeError("temporary outage"),
            (True, {"server_ready_probe_provider": "stub", "status": 200}),
        )
        gate = run_eval._OpenAITransportRecoveryGate(
            enabled=True,
            max_wait_sec=0.2,
            poll_sec=0.01,
        )
        outcome = gate.recover(
            patcher=_ProbePatcher(probe_results=probe_results),
            operation_name="request_context",
        )
        assert outcome["attempted"] is True
        assert outcome["recovered"] is True
        assert outcome["leader"] is True
        assert outcome["probe_attempts"][0]["error"] == "server_ready_probe_failed"
        assert outcome["probe_attempts"][1]["provider"] == "stub"

        gate._outage_active = True
        gate._outage_generation = 7
        gate._last_outcome = {
            "attempted": True,
            "recovered": True,
            "timed_out": False,
            "wait_sec": 0.05,
            "probe_attempts": [],
            "probe_meta": {},
            "operation": "request_context",
            "leader": True,
            "generation": 7,
        }

        def _clear_outage() -> None:
            time.sleep(0.01)
            with gate._condition:
                gate._outage_active = False
                gate._condition.notify_all()

        thread = threading.Thread(target=_clear_outage, daemon=True)
        thread.start()
        follower = gate.recover(
            patcher=_ProbePatcher(probe_results=((True, {"server_ready_probe_provider": "stub"}),)),
            operation_name="request_context",
        )
        thread.join(timeout=1.0)
        assert follower["leader"] is False
        assert follower["operation"] == "request_context"
        assert follower["recovered"] is True


class TestRunStateRepair:
    def test_load_existing_run_state_and_repair_trailing_tail(self, tmp_path: Path) -> None:
        case = _make_case(tmp_path, "case_a")
        run_id = "run-123"
        results_path = tmp_path / "results.jsonl"
        run_metadata = {"run_id": run_id, "run_settings_fingerprint": "fp1"}
        valid_attempt = {
            "run_id": run_id,
            "case_id": case.case_id,
            "condition": run_eval.Condition.BASE_REPAIR.value,
            "attempt": 1,
            "success": True,
            "run_metadata": run_metadata,
        }
        infra_event = {
            "record_type": run_eval.EVAL_RECORD_TYPE_INFRA_EVENT,
            "run_id": run_id,
            "case_id": case.case_id,
            "condition": run_eval.Condition.BASE_REPAIR.value,
            "attempt": 2,
            "success": False,
            "run_metadata": run_metadata,
        }
        results_path.write_text(
            "\n".join(
                [
                    run_eval.json.dumps(valid_attempt),
                    run_eval.json.dumps(infra_event),
                    "{bad json",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        completed, first_run_metadata, fingerprint, records_count, issue = (
            run_eval._load_existing_run_state(
                results_path=results_path,
                run_id=run_id,
                max_attempts_per_unit=1,
            )
        )
        assert completed == {(case.case_id, run_eval.Condition.BASE_REPAIR.value)}
        assert first_run_metadata == run_metadata
        assert fingerprint == "fp1"
        assert records_count == 2
        assert issue is not None

        sidecar_path, removed_bytes = run_eval._repair_trailing_results_tail(
            results_path=results_path,
            run_id=run_id,
            issue=issue,
        )
        assert sidecar_path.exists()
        assert removed_bytes > 0
        assert results_path.read_text(encoding="utf-8").strip().endswith("}")

    def test_build_run_settings_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        args = SimpleNamespace(
            cases_dir=tmp_path / "cases",
            patcher="openai",
            record_prompt="hash",
            record_evidence="full",
            record_env=["B", " A ", ""],
            record_deps="minimal",
            retry_context_mode="history",
            retry_history_max_chars=123,
            prompt_budget_retry_history_chars=456,
            seed=7,
            deterministic=False,
        )
        cases = [_make_case(tmp_path, "case_a"), _make_case(tmp_path, "case_b")]
        conditions = [run_eval.Condition.BASE_REPAIR, run_eval.Condition.COT_REPAIR]
        monkeypatch.setattr(run_eval, "_build_patcher_config", lambda **_kw: {"cfg": 1})
        fingerprint = run_eval._build_run_settings_fingerprint(
            args=args,
            cases=cases,
            conditions=conditions,
        )
        assert isinstance(fingerprint, str)
        assert len(fingerprint) > 0
        # ORC-05: the seed drives bandit exploration even without --deterministic.
        args.seed = 8
        reseeded = run_eval._build_run_settings_fingerprint(
            args=args, cases=cases, conditions=conditions
        )
        assert reseeded != fingerprint
        # CAP-20: a snapshot-format change is settings drift for resume.
        monkeypatch.setattr(run_eval, "_resolve_llmdebug_snapshot_schema_version", lambda: "9.9")
        assert (
            run_eval._build_run_settings_fingerprint(args=args, cases=cases, conditions=conditions)
            != reseeded
        )


class TestPytestAndContainerHelpers:
    def test_build_pytest_args_and_base_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert run_eval._build_pytest_args(
            python_cmd="python",
            python_args=["-m", "pytest", "-q"],
            condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        ) == ["python", "-m", "pytest", "-p", "llmdebug.pytest_plugin", "-q"]
        assert run_eval._build_pytest_args(
            python_cmd="python",
            python_args=["-q"],
            condition=run_eval.Condition.BASE_REPAIR,
        ) == ["python", "-q"]

        monkeypatch.setattr(run_eval._ACTIVE_RUN_SETTINGS, "deterministic", True, raising=False)
        monkeypatch.setattr(run_eval._ACTIVE_RUN_SETTINGS, "seed", 7, raising=False)
        env = run_eval._base_pytest_env(output_format="json", snapshot_condition=True)
        assert env["PYTHONHASHSEED"] == "7"
        assert env["TZ"] == "UTC"
        assert env["LLMDEBUG_CATEGORIZE_ERRORS"] == "0"
        monkeypatch.setattr(run_eval._ACTIVE_RUN_SETTINGS, "deterministic", False, raising=False)
        monkeypatch.setattr(run_eval._ACTIVE_RUN_SETTINGS, "seed", None, raising=False)

    def test_run_pytest_local_and_pooled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monitored = SimpleNamespace(
            timed_out=False,
            duration_sec=1.5,
            stdout="ok",
            stderr="",
            returncode=0,
        )
        monkeypatch.setattr(run_eval, "run_monitored_subprocess", lambda **_kw: monitored)
        result = run_eval._run_pytest_local(
            cwd=tmp_path,
            python_args=["-m", "pytest", "-q"],
            condition=run_eval.Condition.BASE_REPAIR,
            timeout_sec=30,
            output_format="json",
        )
        assert result.returncode == 0
        assert result.timed_out is False

        timed_out = SimpleNamespace(
            timed_out=True,
            duration_sec=2.5,
            stdout="out",
            stderr="err",
            returncode=1,
        )
        monkeypatch.setattr(run_eval, "run_monitored_subprocess", lambda **_kw: timed_out)
        result = run_eval._run_pytest_local(
            cwd=tmp_path,
            python_args=["-m", "pytest", "-q"],
            condition=run_eval.Condition.BASE_REPAIR,
            timeout_sec=30,
            output_format="json",
        )
        assert result.timed_out is True
        assert "pytest timed out" in result.stderr

        class _Container:
            def __init__(self) -> None:
                self.exec_calls: list[dict[str, object]] = []

            def exec_pytest(self, **kwargs):
                self.exec_calls.append(dict(kwargs))
                return run_eval.RunResult(
                    returncode=0,
                    duration_sec=0.1,
                    stdout="stdout",
                    stderr="stderr",
                    timed_out=False,
                    timeout_sec=None,
                )

        container = _Container()

        class _Pool:
            def __init__(self) -> None:
                self.released = False

            def acquire(self):
                return container

            def release(self, acquired) -> None:
                assert acquired is container
                self.released = True

        pool = _Pool()
        pooled_result = run_eval._run_pytest_pooled(
            cwd=tmp_path,
            python_args=["-m", "pytest", "-q"],
            condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            timeout_sec=20,
            output_format="json",
            execution=run_eval.ExecutionConfig(executor="testcontainers"),
            pool=pool,
        )
        assert pooled_result.returncode == 0
        assert pool.released is True
        assert container.exec_calls

    def test_run_pytest_wrapper_and_testcontainers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            run_eval,
            "_run_pytest_local",
            lambda **_kw: run_eval.RunResult(
                returncode=11,
                duration_sec=0.2,
                stdout="local",
                stderr="",
                timed_out=False,
                timeout_sec=None,
            ),
        )
        monkeypatch.setattr(
            run_eval,
            "get_active_execution_config",
            lambda: run_eval.ExecutionConfig(executor="local"),
        )
        result = run_eval.run_pytest(
            tmp_path,
            ["-m", "pytest", "-q"],
            run_eval.Condition.BASE_REPAIR,
            10,
            "json",
        )
        assert result.returncode == 11

        monkeypatch.setattr(
            run_eval,
            "get_active_execution_config",
            lambda: run_eval.ExecutionConfig(executor="testcontainers"),
        )
        pool = SimpleNamespace(
            acquire=lambda: SimpleNamespace(
                exec_pytest=lambda **_kw: run_eval.RunResult(
                    returncode=0,
                    duration_sec=0.1,
                    stdout="pooled",
                    stderr="",
                    timed_out=False,
                    timeout_sec=None,
                )
            ),
            release=lambda _container: None,
        )
        monkeypatch.setattr(run_eval, "_ACTIVE_CONTAINER_POOL", pool, raising=False)
        result = run_eval.run_pytest(
            tmp_path,
            ["-m", "pytest", "-q"],
            run_eval.Condition.BASE_REPAIR,
            10,
            "json",
        )
        assert result.stdout == "pooled"
        monkeypatch.setattr(run_eval, "_ACTIVE_CONTAINER_POOL", None, raising=False)

        with pytest.raises(RuntimeError, match="unsupported executor"):
            monkeypatch.setattr(
                run_eval,
                "get_active_execution_config",
                lambda: run_eval.ExecutionConfig(executor="bogus"),
            )
            run_eval.run_pytest(
                tmp_path,
                ["-m", "pytest", "-q"],
                run_eval.Condition.BASE_REPAIR,
                10,
                "json",
            )

        fake_module = ModuleType("testcontainers")
        fake_core = ModuleType("testcontainers.core")
        fake_container = ModuleType("testcontainers.core.container")
        fake_container.DockerContainer = _FakeDockerContainer
        sys.modules["testcontainers"] = fake_module
        sys.modules["testcontainers.core"] = fake_core
        sys.modules["testcontainers.core.container"] = fake_container

        container_result = run_eval._run_pytest_testcontainers(
            cwd=tmp_path,
            python_args=["-m", "pytest", "-q"],
            condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            timeout_sec=10,
            output_format="json",
            execution=run_eval.ExecutionConfig(executor="testcontainers"),
        )
        assert container_result.returncode == 0
        assert _FakeDockerContainer.created[-1].started is True
        assert _FakeDockerContainer.created[-1].stopped is True

        _FakeDockerContainer.created.clear()

        class _TimeoutError(Exception):
            pass

        timed_out_container = _FakeDockerContainer("image")
        timed_out_container.wrapped = _FakeWaitHandle(result=_TimeoutError("read timed out"))
        timed_out_container.wrapped.logs = (b"OUT", b"ERR")
        timed_out_container.return_none_handle = False

        class _TimeoutDockerContainer(_FakeDockerContainer):
            def __init__(self, image: str) -> None:
                super().__init__(image)
                self.wrapped = timed_out_container.wrapped

        fake_container.DockerContainer = _TimeoutDockerContainer
        timeout_result = run_eval._run_pytest_testcontainers(
            cwd=tmp_path,
            python_args=["-m", "pytest", "-q"],
            condition=run_eval.Condition.BASE_REPAIR,
            timeout_sec=10,
            output_format="json",
            execution=run_eval.ExecutionConfig(executor="testcontainers"),
        )
        assert timeout_result.timed_out is True

    def test_status_and_timeout_helpers(self) -> None:
        assert run_eval._status_code_from_wait_result(5) == 5
        assert run_eval._status_code_from_wait_result({"StatusCode": "3"}) == 3
        assert run_eval._status_code_from_wait_result({"StatusCode": object()}) == 1
        assert run_eval._status_code_from_wait_result("bad") == 1

        class ReadTimeoutError(Exception):
            pass

        assert run_eval._is_container_wait_timeout(ReadTimeoutError("read timed out")) is True
        assert run_eval._is_container_wait_timeout(RuntimeError("timed out")) is True
        assert run_eval._is_container_wait_timeout(RuntimeError("ok")) is False


class TestTruncationAndOutputHelpers:
    def test_truncate_persisted_output_and_snapshot_helpers(self, tmp_path: Path) -> None:
        text = "\n".join(f"line-{i}" for i in range(6)) + "\n"
        truncated = run_eval._truncate_persisted_pytest_output(text, tail_lines=2, hard_char_cap=80)
        assert "truncated" in truncated
        assert truncated.endswith("line-4\nline-5\n")

        char_truncated = run_eval._truncate_persisted_pytest_output(
            "x" * 200,
            tail_lines=0,
            hard_char_cap=25,
        )
        assert "earlier character" in char_truncated

        snapshot = {
            "text": "x" * 200,
            "nested": {"items": ["y" * 100]},
        }
        short = run_eval._truncate_snapshot_string_for_artifact("abc", max_chars=10)
        assert short == "abc"
        long = run_eval._truncate_snapshot_string_for_artifact("x" * 80, max_chars=20)
        assert len(long) == 20
        assert long.startswith("...[llmdebug: snap")
        rewritten = run_eval._truncate_snapshot_for_artifact(snapshot, max_string_chars=20)
        assert isinstance(rewritten, dict)
        assert len(rewritten["text"]) <= 20

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        run_eval._write_run_outputs(
            out_dir,
            run_eval.RunResult(
                returncode=1,
                duration_sec=1.0,
                stdout="stdout\nline2",
                stderr="stderr\nline2",
                timed_out=False,
                timeout_sec=None,
            ),
            prefix="pytest_before",
            tail_lines=5,
        )
        assert (out_dir / "pytest_before_stdout.txt").exists()
        assert (out_dir / "pytest_before_stderr.txt").exists()
        assert (out_dir / "pytest_before_combined.txt").exists()


class TestBudgetAndSnapshotHelpers:
    def test_budget_helpers_and_snapshot_context(self, tmp_path: Path) -> None:
        patcher = _BudgetPatcher(max_propose_sec=5.0)
        exhausted_patch = run_eval._build_attempt_budget_exhausted_patch_result(
            patcher=patcher,
            remaining_budget_sec=0.0,
        )
        exhausted_plan = run_eval._build_attempt_budget_exhausted_plan_result(
            patcher=patcher,
            remaining_budget_sec=0.0,
        )
        assert exhausted_patch.meta["error"] == "request_budget_exceeded"
        assert exhausted_plan.meta["error"] == "request_budget_exceeded"

        new_patcher, remaining, call_budget, clamped = run_eval._patcher_with_shared_attempt_budget(
            patcher=patcher,
            attempt_deadline_monotonic=time.monotonic() + 0.5,
        )
        assert remaining is not None
        assert call_budget is not None
        assert clamped in {True, False}
        assert new_patcher is not None

        result = run_eval._propose_patch_with_snapshot_context(
            patcher=patcher,
            prompt="fix it",
            context_dir=tmp_path,
            cwd=tmp_path,
            snapshot_full_text="full snapshot",
            snapshot_artifact_text="artifact snapshot",
            attempt_deadline_monotonic=time.monotonic() + 0.5,
        )
        assert isinstance(result, PatchResult)
        assert result.meta["attempt_shared_propose_deadline_enabled"] is True

        plan_result = run_eval._plan_patch_with_snapshot_context(
            patcher=patcher,
            prompt="plan it",
            context_dir=tmp_path,
            cwd=tmp_path,
            snapshot_full_text="full snapshot",
            snapshot_artifact_text="artifact snapshot",
            attempt_deadline_monotonic=time.monotonic() + 0.5,
        )
        assert isinstance(plan_result, PatchPlanResult)
        assert plan_result.meta["attempt_shared_propose_deadline_enabled"] is True

        assert run_eval._remaining_attempt_propose_budget_sec(None) is None
        assert run_eval._remaining_attempt_propose_budget_sec(time.monotonic() + 1.0) > 0

        assert run_eval._coerce_subprocess_output(None) == ""
        assert run_eval._coerce_subprocess_output(b"abc") == "abc"
        assert run_eval._timeout_fields(
            run_eval.RunResult(
                returncode=1,
                duration_sec=1.0,
                stdout="",
                stderr="",
                timed_out=True,
                timeout_sec=30,
            )
        ) == {"timed_out": True, "timeout_sec": 30}
        assert (
            run_eval._timeout_fields(
                run_eval.RunResult(
                    returncode=1,
                    duration_sec=1.0,
                    stdout="",
                    stderr="",
                    timed_out=False,
                    timeout_sec=None,
                )
            )
            == {}
        )

        base = {
            "note": "base",
            "patch_verifier_enabled": True,
            "patch_verifier_passed": False,
            "patch_verifier_reasons": ["scope_mismatch"],
            "patch_verifier_scope": "adaptive_only",
            "patch_verifier_score": 2,
            "patch_verifier_source_scope_matched": True,
        }
        enriched = run_eval._enrich_failure_context(
            base=base,
            note="retry_gate_no_new_evidence",
            patch_meta={
                "patch_verifier_enabled": True,
                "patch_verifier_passed": False,
                "patch_verifier_reasons": ["scope_mismatch"],
                "patch_verifier_scope": "adaptive_only",
                "patch_verifier_score": 2,
                "patch_verifier_source_scope_matched": True,
            },
            run=run_eval.RunResult(
                returncode=1,
                duration_sec=1.0,
                stdout="",
                stderr="",
                timed_out=True,
                timeout_sec=30,
            ),
        )
        assert enriched["timed_out"] is True
        assert enriched["patch_verifier_rejected"] is True

    def test_read_latest_snapshot_and_failure_signature_helpers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_snapshot = {"frames": []}
        monkeypatch.setattr(
            "llmdebug.output.get_latest_snapshot",
            lambda _path: fake_snapshot,
        )
        snap, err = run_eval.read_latest_snapshot(tmp_path)
        assert snap == fake_snapshot
        assert err is None

        def _raise(_path):
            raise RuntimeError("boom")

        monkeypatch.setattr("llmdebug.output.get_latest_snapshot", _raise)
        snap, err = run_eval.read_latest_snapshot(tmp_path)
        assert snap is None
        assert "RuntimeError" in err

        seen_kwargs: dict[str, Any] = {}

        def _fake_signature(snapshot: Any, **kwargs: Any) -> str:
            seen_kwargs.update(kwargs)
            return "sig-123"

        monkeypatch.setattr("llmdebug.rca_state.build_failure_signature", _fake_signature)
        signature = run_eval._build_failure_signature_for_attempt(
            run=run_eval.RunResult(
                returncode=1,
                duration_sec=1.0,
                stdout="out",
                stderr="err",
                timed_out=False,
                timeout_sec=None,
            ),
            snapshot=fake_snapshot,
        )
        assert signature == "sig-123"
        # RCA-12: the harness pins raw signatures for comparability with earlier runs.
        assert seen_kwargs["normalize"] is False

        def _raise_signature(snapshot, **_kwargs):
            raise RuntimeError("oops")

        monkeypatch.setattr("llmdebug.rca_state.build_failure_signature", _raise_signature)
        fallback = run_eval._build_failure_signature_for_attempt(
            run=run_eval.RunResult(
                returncode=1,
                duration_sec=1.0,
                stdout="out",
                stderr="err",
                timed_out=False,
                timeout_sec=None,
            ),
            snapshot=fake_snapshot,
        )
        assert fallback.startswith("TextFailure:")

        monkeypatch.setattr(
            "llmdebug.snapshot_diff.diff_snapshots",
            lambda prev, curr: {"changed": True},
        )
        monkeypatch.setattr(
            "llmdebug.snapshot_diff.structured_diff_summary",
            lambda diff: {"summary": "ok"},
        )
        structured = run_eval._build_structured_previous_attempt_diff(
            {"attempt": 1, "failure_signature": "sig-a", "snapshot": {"x": 1}},
            current_failure_context={
                "attempt": 2,
                "failure_signature": "sig-b",
                "snapshot": {"x": 2},
            },
        )
        assert structured is not None
        assert structured["attempt_prev"] == "attempt_01"

        fallback_diff = run_eval._build_structured_previous_attempt_diff(
            {"attempt": 1, "failure_signature": "sig-a", "stdout": "A", "stderr": "B"},
            current_failure_context={
                "attempt": 2,
                "failure_signature": "sig-b",
                "stdout": "C",
                "stderr": "D",
            },
        )
        assert fallback_diff["failure_signature_same"] is False

    def test_infer_source_scope_and_score_relevance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(run_eval, "_import_neighbors", lambda **_kw: ["helper.py", "helper.py"])
        monkeypatch.setattr(
            run_eval,
            "_extract_subprocess_script_neighbors",
            lambda **_kw: ["script.py"],
        )
        assert (
            run_eval._infer_test_source_scope(
                cwd=tmp_path,
                crash_file="test_buggy.py",
                failing_test_file="test_buggy.py",
                enabled=False,
            )
            == []
        )
        inferred = run_eval._infer_test_source_scope(
            cwd=tmp_path,
            crash_file="test_buggy.py",
            failing_test_file="test_buggy.py",
            enabled=True,
        )
        assert inferred == ["helper.py", "script.py"]

        buggy = tmp_path / "buggy.py"
        buggy.write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
        test_file = tmp_path / "test_buggy.py"
        test_file.write_text(
            "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
            encoding="utf-8",
        )
        passed, details = run_eval._score_patch_relevance(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            cwd=tmp_path,
            snapshot={
                "exception": {"type": "AssertionError", "message": "1 != 2"},
                "frames": [{"file_rel": "buggy.py"}],
            },
            stdout="",
            stderr="",
            min_score=1,
            infer_test_source_scope=True,
            file_index_lines=["buggy.py", "test_buggy.py"],
        )
        assert passed is True
        assert details["patch_verifier_crash_file_matched"] is True

        passed, details = run_eval._score_patch_relevance(
            diff=(
                "diff --git a/test_buggy.py b/test_buggy.py\n"
                "--- a/test_buggy.py\n"
                "+++ b/test_buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " assert True\n"
                "+# no-op\n"
            ),
            cwd=tmp_path,
            snapshot=None,
            stdout="",
            stderr="",
            min_score=10,
            infer_test_source_scope=False,
            file_index_lines=["test_buggy.py"],
        )
        assert passed is False
        assert (
            "touches_test_file" in details["patch_verifier_reasons"]
            or "touches_failing_test_file" in details["patch_verifier_reasons"]
        )
