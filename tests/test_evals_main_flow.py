"""Coverage-targeted tests for evals.eval_main_flow.

Focuses on high-impact uncovered paths:
- MainFlowCliError early exit
- remaining_units <= 0 (nothing-to-do)
- Patcher creation raising an exception
- Budget/online policy initialization and cleanup
- Early-stop warning paths
- BaseException cleanup in finally block
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import evals.eval_main_policies as policies_mod
from evals import run_eval as run_eval_mod
from evals.eval_types import (
    ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
    ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
)

# ---------------------------------------------------------------------------
# Minimal stub types to avoid pulling the full eval machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeCaseSpec:
    case_id: str = "fake_case"
    path: Path = field(default_factory=lambda: Path("/tmp/fake"))
    category: str = "hidden_state"
    difficulty: str = "easy"
    snapshot_dependency: str = "none"
    expected_exception: str = "TypeError"
    tags: list[str] = field(default_factory=list)
    python_args: list[str] = field(default_factory=list)
    timeout_sec: int = 60
    description: str | None = None
    bug_source: str | None = None


@dataclass(frozen=True)
class _FakeCondition:
    value: str = "base_repair"


@dataclass
class _FakeConfigState:
    config_paths: list[str] = field(default_factory=list)
    value_sources: dict[str, str] = field(default_factory=dict)


@dataclass
class _FakeSetup:
    config_state: Any = field(default_factory=_FakeConfigState)
    cases: list[Any] = field(default_factory=list)
    execution_config: Any = None
    conditions: list[Any] = field(default_factory=list)
    has_adaptive_condition: bool = False
    startup_warnings: list[str] = field(default_factory=list)
    run_id: str = "test-run-id"
    artifacts_root: Path = field(default_factory=lambda: Path("/tmp/artifacts"))
    results_path: Path = field(default_factory=lambda: Path("/tmp/results.jsonl"))


@dataclass
class _FakeResume:
    execution_units: list[Any] = field(default_factory=list)
    total_units: int = 0
    run_settings_fingerprint: str = "fp"
    resolved_config_lock_path: Path | None = None
    run_results_lock: Any = None
    existing_run_metadata: dict[str, Any] | None = field(
        default_factory=lambda: {"run_id": "test", "run_settings_fingerprint": "fp"}
    )
    resumed_completed_units: set[tuple[str, str]] = field(default_factory=set)
    skipped_units: int = 0
    remaining_units: int = 0
    decision_model_serving: dict[str, Any] = field(default_factory=dict)


def _make_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def _make_args(**overrides: Any) -> argparse.Namespace:
    defaults = {
        "progress_verbosity": "normal",
        "openai_transport_recovery": True,
        "openai_transport_recovery_max_wait_sec": 60.0,
        "openai_transport_recovery_poll_sec": 2.0,
        "jobs": 1,
        "patcher": "openai",
        "deterministic": False,
        "seed": None,
        "transport_fail_fast_streak": 0,
        "k": 3,
        "output_format": "unified_diff",
        "no_files_context": False,
        "max_context_chars": 50000,
        "case_fail_fast_on_request_budget_exceeded": False,
        "force_patch_attempt": False,
        "context_protocol": "auto",
        "context_stage_a_max_requests": 1,
        "context_stage_a_max_chars": 8000,
        "retry_prompt_mode": "default",
        "retry_context_mode": "default",
        "retry_history_max_chars": 4000,
        "prompt_retry_reflection": False,
        "prompt_adaptive_stage_b_cot": False,
        "prompt_repair_method_scaffold": False,
        "prompt_retry_plan_then_patch": False,
        "pytest_output_tail_lines": 50,
        "snapshot_artifact_max_string_chars": 1000,
        "prompt_budget_total_chars": 40000,
        "prompt_budget_retry_delta_chars": 2000,
        "prompt_budget_stdout_chars": 10000,
        "prompt_budget_snapshot_chars": 5000,
        "prompt_budget_evidence_chars": 5000,
        "prompt_budget_retry_history_chars": 5000,
        "context_compaction_mode": "off",
        "context_compaction_scope": "all",
        "context_compaction_max_input_chars": 10000,
        "context_compaction_target_chars": 5000,
        "context_compaction_max_calls": 3,
        "openai_auto_context_window": False,
        "openai_response_mode_autoswitch": "off",
        "openai_timeout_sec": 0.0,
        "baseline_name": "",
        "baseline_repo_dir": "",
        "baseline_python": "",
        "baseline_process_timeout_sec": None,
        "adaptive_gate_policy": "default",
        "adaptive_max_helper_calls": 3,
        "adaptive_budget_topup_max_calls": 1,
        "adaptive_stagnation_window": 3,
        "adaptive_saturation_threshold": 0.8,
        "adaptive_no_progress_token_cap": 0,
        "adaptive_helper_budget_profile": "default",
        "adaptive_openai_helper_response_mode": "auto",
        "adaptive_controller_lambda_rlm": False,
        "adaptive_controller_lambda_rlm_repo_dir": "",
        "adaptive_controller_lambda_rlm_python": "",
        "adaptive_controller_lambda_rlm_context_window_chars": 0,
        "adaptive_controller_lambda_rlm_timeout_sec": 30,
        "adaptive_informativeness_pregate": False,
        "adaptive_informativeness_pregate_threshold": 0.5,
        "syntax_preflight_enabled": False,
        "syntax_preflight_max_cycles": 3,
        "syntax_preflight_fallthrough": False,
        "patch_sanitize_enabled": False,
        "patch_verifier_scope": "disabled",
        "patch_verifier_min_score": 50,
        "patch_verifier_infer_test_source_scope": False,
        "adaptive_online_policy_mode": "off",
        "adaptive_online_policy_algo": "linucb",
        "adaptive_online_alpha": 0.5,
        "adaptive_online_epsilon": 0.1,
        "adaptive_online_l2": 0.01,
        "adaptive_online_warmup_decisions": 10,
        "adaptive_online_save_every": 5,
        "adaptive_online_decay": 0.0,
        "adaptive_online_feature_scaling_profile": "default",
        "adaptive_online_state_path": "/tmp/online_state.json",
        "adaptive_online_events_path": "/tmp/online_events.jsonl",
        "adaptive_budget_policy_mode": "off",
        "adaptive_budget_policy_algo": "linucb",
        "adaptive_budget_policy_architecture": "default",
        "adaptive_budget_online_alpha": 0.5,
        "adaptive_budget_online_epsilon": 0.1,
        "adaptive_budget_online_l2": 0.01,
        "adaptive_budget_online_warmup_decisions": 10,
        "adaptive_budget_online_save_every": 5,
        "adaptive_budget_online_decay": 0.0,
        "adaptive_budget_feature_scaling_profile": "default",
        "adaptive_budget_online_state_path": "/tmp/budget_state.json",
        "adaptive_budget_online_events_path": "/tmp/budget_events.jsonl",
        "adaptive_budget_policy_tiers": "default",
        "adaptive_budget_policy_max_ups_per_case": 3,
        "adaptive_budget_reward_rho_budget_exceeded": 0.0,
        "adaptive_reward_lambda_tokens": 0.0,
        "adaptive_reward_mu_latency": 0.0,
        "adaptive_reward_token_scale": 1.0,
        "adaptive_reward_latency_scale": 1.0,
        "adaptive_reward_eta_progress": 0.0,
        "adaptive_reward_kappa_stagnation": 0.0,
        "adaptive_reward_success_hint": ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
        "adaptive_reward_progress_scale": ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
        "adaptive_reward_stagnation_scale": ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
        "adaptive_reward_materialization_scale": ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
        "adaptive_toon_encoding": False,
        "keep_workdirs": False,
        "progress_heartbeat_sec": 30.0,
        "record_prompt": "off",
        "record_evidence": "off",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Test: MainFlowCliError early exit (lines 71-73)
# ---------------------------------------------------------------------------


class TestMainFlowCliError:
    def test_cli_error_returns_exit_code(self, monkeypatch) -> None:
        from evals.eval_main_setup import MainFlowCliError

        def raise_cli_error(**kwargs):
            raise MainFlowCliError("bad config", exit_code=3)

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", raise_cli_error)

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 3


# ---------------------------------------------------------------------------
# Test: remaining_units <= 0 (nothing-to-do path, line 127-133)
# ---------------------------------------------------------------------------


class TestNothingToDo:
    def test_remaining_units_zero_returns_zero(self, monkeypatch) -> None:
        setup = _FakeSetup()
        resume = _FakeResume(total_units=5, remaining_units=0, skipped_units=5)

        monkeypatch.setattr(
            "evals.eval_main_flow.prepare_main_setup",
            lambda **kw: setup,
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.prepare_main_resume",
            lambda **kw: resume,
        )
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr(
            "evals.eval_main_flow.create_run_display",
            lambda **kw: mock_display,
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.build_startup_sections",
            lambda **kw: [],
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0
        mock_display.render_final_lines.assert_called_once()
        msg = mock_display.render_final_lines.call_args[0][0][0]
        assert "Nothing to do" in msg


# ---------------------------------------------------------------------------
# Test: Patcher creation raising an exception (lines 151-154)
# ---------------------------------------------------------------------------


class TestPatcherCreationError:
    def test_patcher_error_closes_resources_and_reraises(self, monkeypatch) -> None:
        setup = _FakeSetup()
        resume = _FakeResume(total_units=1, remaining_units=1)

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
        # SynchronizedJsonlWriter mock — must have close()
        mock_writer = MagicMock()
        monkeypatch.setattr(
            "evals.eval_main_flow._SynchronizedJsonlWriter",
            lambda path: mock_writer,
        )
        monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

        # make_patcher raises
        import evals.eval_main_flow as flow_mod

        def _bad_patcher(args):
            raise RuntimeError("patcher init failed")

        monkeypatch.setattr("evals.patcher_factory.make_patcher", _bad_patcher)

        from evals.eval_main_flow import run_main_flow

        with pytest.raises(RuntimeError, match="patcher init failed"):
            run_main_flow(
                parser=_make_parser(),
                args=_make_args(),
                argv_tokens=[],
            )
        mock_display.close.assert_called()


# ---------------------------------------------------------------------------
# Test: _resolve_openai_context_window_for_run error (lines 166-169)
# ---------------------------------------------------------------------------


class TestContextWindowResolutionError:
    def test_context_window_error_closes_and_reraises(self, monkeypatch) -> None:
        setup = _FakeSetup()
        resume = _FakeResume(total_units=1, remaining_units=1)

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
        monkeypatch.setattr(
            "evals.eval_main_flow._SynchronizedJsonlWriter",
            lambda path: MagicMock(),
        )
        monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

        import evals.eval_main_flow as flow_mod

        mock_patcher = MagicMock()
        mock_patcher.name = "openai"
        monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)

        def bad_resolve(**kw):
            raise ValueError("context window probe failed")

        monkeypatch.setattr(
            "evals.eval_main_flow._resolve_openai_context_window_for_run",
            bad_resolve,
        )

        from evals.eval_main_flow import run_main_flow

        with pytest.raises(ValueError, match="context window probe failed"):
            run_main_flow(
                parser=_make_parser(),
                args=_make_args(),
                argv_tokens=[],
            )
        mock_display.close.assert_called()


# ---------------------------------------------------------------------------
# Test: OnlinePolicyConfig error path (line 296-299)
# ---------------------------------------------------------------------------


class TestOnlinePolicyConfigError:
    def test_policy_normalize_error_closes_and_reraises(self, monkeypatch) -> None:
        setup = _FakeSetup()
        resume = _FakeResume(total_units=1, remaining_units=1)

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
        monkeypatch.setattr(
            "evals.eval_main_flow._SynchronizedJsonlWriter",
            lambda path: MagicMock(),
        )
        monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

        import evals.eval_main_flow as flow_mod

        mock_patcher = MagicMock()
        mock_patcher.name = "openai"
        monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)
        monkeypatch.setattr(
            "evals.eval_main_flow._resolve_openai_context_window_for_run",
            lambda **kw: (128000, {}),
        )

        @staticmethod
        def bad_normalize(**kw):
            raise ValueError("bad online policy config")

        monkeypatch.setattr(policies_mod.OnlinePolicyConfig, "normalize", bad_normalize)

        from evals.eval_main_flow import run_main_flow

        with pytest.raises(ValueError, match="bad online policy config"):
            run_main_flow(
                parser=_make_parser(),
                args=_make_args(),
                argv_tokens=[],
            )
        mock_display.close.assert_called()


# ---------------------------------------------------------------------------
# Test: BudgetPolicyConfig error path (line 336-339)
# ---------------------------------------------------------------------------


class TestBudgetPolicyConfigError:
    def test_budget_policy_normalize_error_closes_and_reraises(self, monkeypatch) -> None:
        setup = _FakeSetup()
        resume = _FakeResume(total_units=1, remaining_units=1)

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
        monkeypatch.setattr(
            "evals.eval_main_flow._SynchronizedJsonlWriter",
            lambda path: MagicMock(),
        )
        monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

        import evals.eval_main_flow as flow_mod

        mock_patcher = MagicMock()
        mock_patcher.name = "openai"
        monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)
        monkeypatch.setattr(
            "evals.eval_main_flow._resolve_openai_context_window_for_run",
            lambda **kw: (128000, {}),
        )

        # Online policy normalize succeeds with mode=off
        mock_online_config = MagicMock()
        mock_online_config.mode = "off"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online_config),
        )

        @staticmethod
        def bad_budget_normalize(**kw):
            raise ValueError("bad budget config")

        monkeypatch.setattr(policies_mod.BudgetPolicyConfig, "normalize", bad_budget_normalize)

        from evals.eval_main_flow import run_main_flow

        with pytest.raises(ValueError, match="bad budget config"):
            run_main_flow(
                parser=_make_parser(),
                args=_make_args(),
                argv_tokens=[],
            )
        mock_display.close.assert_called()


# ---------------------------------------------------------------------------
# Test: Cleanup on BaseException in finally block (lines 1002-1029)
# ---------------------------------------------------------------------------


class TestFinallyCleanup:
    """The run's ``finally`` block closes each controller and reports close errors."""

    def test_online_policy_close_warning_on_error(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")
        mock_display, flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )
        mock_online_config = MagicMock()
        mock_online_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online_config),
        )
        mock_ctrl = MagicMock(load_error=None)
        mock_ctrl.close.side_effect = RuntimeError("close failed")
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveOnlinePolicyController", lambda cfg: mock_ctrl
        )

        assert flow_mod.run_main_flow(parser=_make_parser(), args=_make_args(), argv_tokens=[]) == 0

        mock_ctrl.close.assert_called_once()
        warnings = [
            c[1]["message"]
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning"
        ]
        assert "Online adaptive policy close warning: RuntimeError: close failed" in warnings

    def test_budget_policy_close_warning_on_error(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")
        mock_display, flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )
        mock_budget_config = MagicMock()
        mock_budget_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget_config),
        )
        mock_ctrl = MagicMock(load_error=None)
        mock_ctrl.close.side_effect = RuntimeError("budget close boom")
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveBudgetPolicyController", lambda cfg: mock_ctrl
        )

        assert flow_mod.run_main_flow(parser=_make_parser(), args=_make_args(), argv_tokens=[]) == 0

        warnings = [
            c[1]["message"]
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning"
        ]
        assert any("budget close boom" in message for message in warnings)


# ---------------------------------------------------------------------------
# Test: startup warnings displayed (line 125-126)
# ---------------------------------------------------------------------------


class TestStartupWarnings:
    def test_startup_warnings_passed_to_display(self, monkeypatch) -> None:
        setup = _FakeSetup(startup_warnings=["warn1", "warn2"])
        resume = _FakeResume(total_units=2, remaining_units=0, skipped_units=2)

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])

        from evals.eval_main_flow import run_main_flow

        run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )

        warning_calls = [
            c for c in mock_display.add_event.call_args_list if c[1].get("style") == "warning"
        ]
        assert len(warning_calls) >= 2


# ---------------------------------------------------------------------------
# Test: Full run loop with empty pending_units (all resumed)
# ---------------------------------------------------------------------------


class TestFullRunLoopAllResumed:
    """Drive through the complete run loop when all units are already resumed.

    This covers lines 300-370 (policy setup), 379-870 (run condition/unit
    input construction), and the finally cleanup (1002-1074).
    """

    def _setup_full_flow(self, monkeypatch):
        """Wire up mocks to drive past policy init into the run loop."""
        import evals.eval_main_flow as flow_mod

        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")
        setup = _FakeSetup(
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="local"),
        )
        resume = _FakeResume(
            execution_units=[(case, cond)],
            total_units=1,
            remaining_units=1,
            resumed_completed_units={("c1", "base_repair")},
            skipped_units=0,
        )

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
        monkeypatch.setattr(
            "evals.eval_main_flow._SynchronizedJsonlWriter",
            lambda path: MagicMock(),
        )
        monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

        mock_patcher = MagicMock()
        mock_patcher.name = "test_patcher"
        monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)
        monkeypatch.setattr(
            "evals.eval_main_flow._resolve_openai_context_window_for_run",
            lambda **kw: (128000, {}),
        )

        # Online and budget policies both return "off"
        mock_online = MagicMock()
        mock_online.mode = "off"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online),
        )
        mock_budget = MagicMock()
        mock_budget.mode = "off"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget),
        )

        # Mock set_active_execution_config to return a value
        monkeypatch.setattr(
            "evals.eval_main_flow.set_active_execution_config",
            lambda cfg: None,
        )

        # Mock RunConditionInputs and EvalUnitInputs to accept any kwargs
        monkeypatch.setattr(
            "evals.eval_main_flow.RunConditionInputs",
            lambda **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.EvalUnitInputs",
            lambda **kw: MagicMock(),
        )

        return mock_display, flow_mod

    def test_all_resumed_units_complete_successfully(self, monkeypatch) -> None:
        mock_display, _flow_mod = self._setup_full_flow(monkeypatch)

        # Save originals for cleanup

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0
        # Display should have been closed
        mock_display.close.assert_called()

    def test_run_with_runtime_errors_returns_2(self, monkeypatch) -> None:
        """Drive through with a unit that yields a runtime error."""
        import evals.eval_main_flow as flow_mod

        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")
        setup = _FakeSetup(
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="local"),
        )
        resume = _FakeResume(
            execution_units=[(case, cond)],
            total_units=1,
            remaining_units=1,
            resumed_completed_units=set(),
            skipped_units=0,
        )

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
        monkeypatch.setattr(
            "evals.eval_main_flow._SynchronizedJsonlWriter",
            lambda path: MagicMock(),
        )
        monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

        mock_patcher = MagicMock()
        mock_patcher.name = "test_patcher"
        monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)
        monkeypatch.setattr(
            "evals.eval_main_flow._resolve_openai_context_window_for_run",
            lambda **kw: (128000, {}),
        )

        mock_online = MagicMock()
        mock_online.mode = "off"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online),
        )
        mock_budget = MagicMock()
        mock_budget.mode = "off"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget),
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.set_active_execution_config",
            lambda cfg: None,
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.RunConditionInputs",
            lambda **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.EvalUnitInputs",
            lambda **kw: MagicMock(),
        )

        # Mock dataclasses.replace used for patcher_override injection
        monkeypatch.setattr(
            "evals.eval_main_flow.replace",
            lambda obj, **kw: obj,
        )

        # Mock _execute_eval_unit to return an outcome with runtime_error
        from evals.run_eval import UnitExecutionOutcome

        mock_outcome = MagicMock(spec=UnitExecutionOutcome)
        mock_outcome.case = case
        mock_outcome.condition = cond
        mock_outcome.case_records = None
        mock_outcome.records_streamed = False
        mock_outcome.unit_duration_sec = 0.5
        mock_outcome.runtime_error = "TestError: boom"

        monkeypatch.setattr(
            "evals.eval_main_flow._execute_eval_unit",
            lambda **kw: mock_outcome,
        )
        monkeypatch.setattr(
            "evals.eval_main_flow._filter_attempt_records",
            lambda recs: recs or [],
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 2  # runtime errors -> exit 2

    def test_run_with_successful_unit(self, monkeypatch) -> None:
        """Drive through with a successful unit execution."""
        import evals.eval_main_flow as flow_mod

        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")
        setup = _FakeSetup(
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="local"),
        )
        resume = _FakeResume(
            execution_units=[(case, cond)],
            total_units=1,
            remaining_units=1,
            resumed_completed_units=set(),
            skipped_units=0,
        )

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
        monkeypatch.setattr(
            "evals.eval_main_flow._SynchronizedJsonlWriter",
            lambda path: MagicMock(),
        )
        monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

        mock_patcher = MagicMock()
        mock_patcher.name = "test_patcher"
        monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)
        monkeypatch.setattr(
            "evals.eval_main_flow._resolve_openai_context_window_for_run",
            lambda **kw: (128000, {}),
        )

        mock_online = MagicMock()
        mock_online.mode = "off"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online),
        )
        mock_budget = MagicMock()
        mock_budget.mode = "off"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget),
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.set_active_execution_config",
            lambda cfg: None,
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.RunConditionInputs",
            lambda **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.EvalUnitInputs",
            lambda **kw: MagicMock(),
        )

        # Mock dataclasses.replace used for patcher_override injection
        monkeypatch.setattr(
            "evals.eval_main_flow.replace",
            lambda obj, **kw: obj,
        )

        # Mock _execute_eval_unit to return a successful outcome
        from evals.run_eval import UnitExecutionOutcome

        success_record = {
            "success": True,
            "attempt": 1,
            "note": "success",
            "record_type": "attempt",
            "before_duration_sec": 1.0,
            "after_duration_sec": 2.0,
            "patch_meta": {"propose_duration_sec": 0.5},
            "attempt_metadata": {
                "max_attempts": 3,
                "prompt_chars_total": 100,
                "completion_chars_total": 50,
                "tokens_prompt": 10,
                "tokens_completion": 5,
                "tokens_total": 15,
            },
        }
        mock_outcome = MagicMock(spec=UnitExecutionOutcome)
        mock_outcome.case = case
        mock_outcome.condition = cond
        mock_outcome.case_records = [success_record]
        mock_outcome.records_streamed = True
        mock_outcome.unit_duration_sec = 3.5
        mock_outcome.runtime_error = None

        monkeypatch.setattr(
            "evals.eval_main_flow._execute_eval_unit",
            lambda **kw: mock_outcome,
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("none", "none"),
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Helper to build a fully-mocked flow environment
# ---------------------------------------------------------------------------


def _wire_full_flow(
    monkeypatch,
    *,
    cases_and_conditions,
    resumed=None,
    remaining=None,
    run_results_lock=None,
    extra_args=None,
):
    """Set up all mocks needed to drive run_main_flow past policy init."""
    import evals.eval_main_flow as flow_mod

    execution_units = list(cases_and_conditions)
    total = len(execution_units)
    if resumed is None:
        resumed = set()
    if remaining is None:
        remaining = total - len(resumed)

    setup = _FakeSetup(
        cases=[c for c, _ in cases_and_conditions],
        conditions=[cnd for _, cnd in cases_and_conditions],
        execution_config=MagicMock(executor="local"),
    )
    resume = _FakeResume(
        execution_units=execution_units,
        total_units=total,
        remaining_units=remaining,
        resumed_completed_units=resumed,
        skipped_units=len(resumed),
        run_results_lock=run_results_lock,
    )

    monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
    monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
    mock_display = MagicMock()
    mock_display.suppress_inline_progress = False
    monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
    monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])
    monkeypatch.setattr(
        "evals.eval_main_flow._SynchronizedJsonlWriter",
        lambda path: MagicMock(),
    )
    monkeypatch.setattr("evals.eval_main_flow._OpenAITransportRecoveryGate", MagicMock)

    mock_patcher = MagicMock()
    mock_patcher.name = "test_patcher"
    monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)
    monkeypatch.setattr(
        "evals.eval_main_flow._resolve_openai_context_window_for_run",
        lambda **kw: (128000, {}),
    )

    mock_online = MagicMock()
    mock_online.mode = "off"
    monkeypatch.setattr(
        policies_mod.OnlinePolicyConfig,
        "normalize",
        staticmethod(lambda **kw: mock_online),
    )
    mock_budget = MagicMock()
    mock_budget.mode = "off"
    monkeypatch.setattr(
        policies_mod.BudgetPolicyConfig,
        "normalize",
        staticmethod(lambda **kw: mock_budget),
    )
    monkeypatch.setattr(
        "evals.eval_main_flow.set_active_execution_config",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        "evals.eval_main_flow.RunConditionInputs",
        lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "evals.eval_main_flow.EvalUnitInputs",
        lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "evals.eval_main_flow.replace",
        lambda obj, **kw: obj,
    )
    # eval_main_policies derives per-condition budget configs with dataclasses.replace,
    # which the MagicMock configs above are not.
    monkeypatch.setattr(
        "evals.eval_main_policies.replace",
        lambda obj, **kw: obj,
    )

    merged = dict(**(extra_args or {}))
    return mock_display, flow_mod, merged


# ---------------------------------------------------------------------------
# Test: Transport fail-fast trigger (lines 597-618)
# ---------------------------------------------------------------------------


class TestTransportFailFast:
    def test_transport_fail_fast_triggers_on_consecutive_errors(self, monkeypatch) -> None:
        case1 = _FakeCaseSpec(case_id="c1")
        case2 = _FakeCaseSpec(case_id="c2")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case1, cond), (case2, cond)],
        )

        call_count = 0

        def _mock_exec(**kw):
            nonlocal call_count
            call_count += 1
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=None,
                records_streamed=False,
                unit_duration_sec=0.1,
                runtime_error="openai_transport_error: connection reset",
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow._is_transport_runtime_error",
            lambda reason: True,
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(transport_fail_fast_streak=1),
            argv_tokens=[],
        )
        # Should abort with exit code 2
        assert result == 2


# ---------------------------------------------------------------------------
# Test: Early-stop warning (lines 680-701)
# ---------------------------------------------------------------------------


class TestEarlyStopWarning:
    def test_early_stop_warning_displayed(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        # Return a failed outcome with fewer attempts than max
        scored_record = {
            "success": False,
            "attempt": 1,
            "note": "tests_still_failing",
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 1.0,
            "patch_meta": {
                "propose_duration_sec": 0.5,
                "case_fail_fast_triggered": True,
                "case_fail_fast_reason": "budget_exceeded",
            },
            "attempt_metadata": {
                "max_attempts": 3,
                "prompt_chars_total": 100,
                "completion_chars_total": 50,
            },
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[scored_record],
                records_streamed=True,
                unit_duration_sec=1.5,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("budget", "budget_exceeded"),
        )

        from evals.eval_main_flow import run_main_flow

        run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        # Should still complete but with exit code 0 (test failures
        # aren't runtime errors)

        # Check early-stop warning was displayed
        warning_calls = [
            c
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning" and "early stop" in str(c[1].get("message", ""))
        ]
        assert len(warning_calls) >= 1


# ---------------------------------------------------------------------------
# Test: Cleanup with active controllers (lines 1003-1029)
# ---------------------------------------------------------------------------


class TestCleanupWithActiveControllers:
    def test_cleanup_closes_active_controllers(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        # remaining=1 so we don't exit early, but unit is resumed so
        # pending_units ends up empty after the loop
        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        # Override policy configs to return mode="on" so controllers get created
        mock_online_config = MagicMock()
        mock_online_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online_config),
        )

        mock_online_ctrl = MagicMock()
        mock_online_ctrl.load_error = None
        mock_online_ctrl.close = MagicMock()
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveOnlinePolicyController",
            lambda cfg: mock_online_ctrl,
        )

        mock_budget_config = MagicMock()
        mock_budget_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget_config),
        )

        mock_budget_ctrl = MagicMock()
        mock_budget_ctrl.load_error = None
        mock_budget_ctrl.close = MagicMock()
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveBudgetPolicyController",
            lambda cfg: mock_budget_ctrl,
        )

        from evals.eval_main_flow import run_main_flow

        run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )

        # Controllers should have been closed during cleanup
        mock_online_ctrl.close.assert_called_once()
        mock_budget_ctrl.close.assert_called_once()

    def test_cleanup_handles_close_exceptions(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        # Online controller that fails on close
        mock_online_config = MagicMock()
        mock_online_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online_config),
        )

        mock_online_ctrl = MagicMock()
        mock_online_ctrl.load_error = None
        mock_online_ctrl.close.side_effect = RuntimeError("close boom")
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveOnlinePolicyController",
            lambda cfg: mock_online_ctrl,
        )

        # Budget controller that fails on close
        mock_budget_config = MagicMock()
        mock_budget_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget_config),
        )

        mock_budget_ctrl = MagicMock()
        mock_budget_ctrl.load_error = None
        mock_budget_ctrl.close.side_effect = RuntimeError("budget close boom")
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveBudgetPolicyController",
            lambda cfg: mock_budget_ctrl,
        )

        from evals.eval_main_flow import run_main_flow

        # Should not raise despite close() errors
        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0

        # Check warning events
        warning_msgs = [
            c[1]["message"]
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning"
        ]
        assert any("close boom" in m for m in warning_msgs)
        assert any("budget close boom" in m for m in warning_msgs)


# ---------------------------------------------------------------------------
# Test: run_results_lock cleanup (lines 101-103)
# ---------------------------------------------------------------------------


class TestRunResultsLockCleanup:
    def test_lock_released_on_nothing_to_do(self, monkeypatch) -> None:
        setup = _FakeSetup()
        mock_lock = MagicMock()
        resume = _FakeResume(
            total_units=1,
            remaining_units=0,
            skipped_units=1,
            run_results_lock=mock_lock,
        )

        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)
        mock_display = MagicMock()
        mock_display.suppress_inline_progress = False
        monkeypatch.setattr("evals.eval_main_flow.create_run_display", lambda **kw: mock_display)
        monkeypatch.setattr("evals.eval_main_flow.build_startup_sections", lambda **kw: [])

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0
        mock_lock.release.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Baseline python resolution (lines 249-257)
# ---------------------------------------------------------------------------


class TestBaselinePythonResolution:
    def test_baseline_patcher_config_path(self, monkeypatch, tmp_path) -> None:
        """Exercise the baseline patcher_config section (lines 238-279)."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
        )

        # Override resume to have metadata with patcher_config
        setup = _FakeSetup(
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="local"),
        )
        resume = _FakeResume(
            execution_units=[(case, cond)],
            total_units=1,
            remaining_units=1,
            resumed_completed_units={("c1", "base_repair")},
            skipped_units=0,
            existing_run_metadata={
                "run_id": "test",
                "patcher_config": {},
            },
        )
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(patcher="baseline", baseline_name="test_bl"),
            argv_tokens=[],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Test: Transport fail-fast on successful outcome path (lines 703-729)
# ---------------------------------------------------------------------------


class TestTransportFailFastOnRecordFailure:
    def test_transport_record_failure_triggers_abort(self, monkeypatch) -> None:
        """A successful outcome but with transport request failure record."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        transport_failure_record = {
            "success": False,
            "attempt": 1,
            "note": "patcher_request_failed",
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 0.5,
            "patch_meta": {"error": "transport_timeout"},
            "attempt_metadata": {"max_attempts": 3},
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[transport_failure_record],
                records_streamed=True,
                unit_duration_sec=0.5,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow._record_is_transport_request_failure",
            lambda rec: True,
        )
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("transport", "transport_failure"),
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(transport_fail_fast_streak=1),
            argv_tokens=[],
        )
        assert result == 2  # fail-fast triggered


class TestInterruptedInfraOutcome:
    def test_infra_only_outcome_is_interrupted_not_failed(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="infra_case")
        cond = _FakeCondition(value="intervenor")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        infra_record = {
            "success": False,
            "attempt": 1,
            "record_type": "infra_event",
            "note": "infra_transport_failure",
            "patch_meta": {
                "infra_resume_eligible": True,
                "infra_failure_kind": "backend_unavailable",
            },
            "attempt_metadata": {"max_attempts": 2},
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[infra_record],
                records_streamed=True,
                unit_duration_sec=0.5,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0

        mock_display.finish_unit.assert_called_with(
            case_id="infra_case",
            condition="intervenor",
            status="interrupted",
            failure_reason="infra interruption",
        )

    def test_resume_eligible_attempt_outcome_is_interrupted_not_failed(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="infra_attempt_case")
        cond = _FakeCondition(value="ldb")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        interrupted_attempt_record = {
            "success": False,
            "attempt": 1,
            "record_type": "attempt",
            "note": "patcher_request_failed",
            "patch_meta": {
                "infra_resume_eligible": True,
                "infra_failure_kind": "transport",
            },
            "attempt_metadata": {"max_attempts": 2},
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[interrupted_attempt_record],
                records_streamed=True,
                unit_duration_sec=0.5,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0

        mock_display.finish_unit.assert_called_with(
            case_id="infra_attempt_case",
            condition="ldb",
            status="interrupted",
            failure_reason="infra interruption",
        )


# ---------------------------------------------------------------------------
# Test: Parallel execution path (lines 925-1001)
# ---------------------------------------------------------------------------


class TestParallelExecution:
    def test_parallel_jobs_gt_1(self, monkeypatch) -> None:
        """When jobs > 1, the ThreadPoolExecutor path is used."""
        case1 = _FakeCaseSpec(case_id="c1")
        case2 = _FakeCaseSpec(case_id="c2")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case1, cond), (case2, cond)],
        )

        success_record = {
            "success": True,
            "attempt": 1,
            "note": "success",
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 1.0,
            "patch_meta": {"propose_duration_sec": 0.5},
            "attempt_metadata": {
                "max_attempts": 3,
                "prompt_chars_total": 100,
                "completion_chars_total": 50,
            },
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[success_record],
                records_streamed=True,
                unit_duration_sec=1.0,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("none", "none"),
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(jobs=2),
            argv_tokens=[],
        )
        assert result == 0

    def test_parallel_execution_with_exception(self, monkeypatch) -> None:
        """When a future raises an exception, it becomes a runtime error."""
        case1 = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case1, cond)],
        )

        def _mock_exec(**kw):
            raise RuntimeError("unit exploded")

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(jobs=2),
            argv_tokens=[],
        )
        assert result == 2


# ---------------------------------------------------------------------------
# Test: Container pool setup/failure (lines 872-895)
# ---------------------------------------------------------------------------


class TestContainerPoolSetup:
    def test_container_pool_failure_falls_back(self, monkeypatch) -> None:
        """Container pool creation failure produces a warning."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        # Override execution_config to use testcontainers
        setup = _FakeSetup(
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="testcontainers"),
        )
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)

        # _EvalContainerPool raises
        monkeypatch.setattr(
            "evals.eval_main_flow._EvalContainerPool",
            MagicMock(side_effect=RuntimeError("docker unavailable")),
        )

        success_record = {
            "success": True,
            "attempt": 1,
            "note": "success",
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 1.0,
            "patch_meta": {"propose_duration_sec": 0.5},
            "attempt_metadata": {"max_attempts": 3},
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[success_record],
                records_streamed=True,
                unit_duration_sec=1.0,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("none", "none"),
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0

        # Should see warning about container pool failure
        warning_msgs = [
            c[1]["message"]
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning"
        ]
        assert any("docker unavailable" in m for m in warning_msgs)


# ---------------------------------------------------------------------------
# Test: Policy controller creation with load_error (lines 303-316, 342-356)
# ---------------------------------------------------------------------------


class TestPolicyControllerLoadError:
    def test_online_policy_load_error_warning(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        mock_online_config = MagicMock()
        mock_online_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online_config),
        )

        mock_ctrl = MagicMock()
        mock_ctrl.load_error = "corrupt state file"
        mock_ctrl.close = MagicMock()
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveOnlinePolicyController",
            lambda cfg: mock_ctrl,
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0

        warning_msgs = [
            c[1]["message"]
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning"
        ]
        assert any("corrupt state file" in m for m in warning_msgs)

    def test_budget_policy_load_error_warning(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        mock_budget_config = MagicMock()
        mock_budget_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget_config),
        )

        mock_ctrl = MagicMock()
        mock_ctrl.load_error = "bad budget state"
        mock_ctrl.close = MagicMock()
        monkeypatch.setattr(
            "evals.eval_main_policies.AdaptiveBudgetPolicyController",
            lambda cfg: mock_ctrl,
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0

        warning_msgs = [
            c[1]["message"]
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning"
        ]
        assert any("bad budget state" in m for m in warning_msgs)


# ---------------------------------------------------------------------------
# Test: Container pool success path (lines 879-886)
# ---------------------------------------------------------------------------


class TestContainerPoolSuccess:
    def test_container_pool_creation_success(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        setup = _FakeSetup(
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="testcontainers"),
        )
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)

        mock_pool = MagicMock()
        monkeypatch.setattr(
            "evals.eval_main_flow._EvalContainerPool",
            lambda **kw: mock_pool,
        )

        success_record = {
            "success": True,
            "attempt": 1,
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 1.0,
            "patch_meta": {"propose_duration_sec": 0.5},
            "attempt_metadata": {"max_attempts": 3},
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[success_record],
                records_streamed=True,
                unit_duration_sec=1.0,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("none", "none"),
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0

        # Container pool cleanup should have been called
        mock_pool.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Non-transport error resets streak (line 601)
# ---------------------------------------------------------------------------


class TestNonTransportErrorResetsStreak:
    def test_non_transport_error_resets_transport_streak(self, monkeypatch) -> None:
        """A non-transport runtime error should reset transport_failure_streak."""
        case1 = _FakeCaseSpec(case_id="c1")
        case2 = _FakeCaseSpec(case_id="c2")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case1, cond), (case2, cond)],
        )

        call_count = 0

        def _mock_exec(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First: a non-transport error
                return MagicMock(
                    case=kw["case"],
                    condition=kw["condition"],
                    case_records=None,
                    records_streamed=False,
                    unit_duration_sec=0.1,
                    runtime_error="ValueError: bad data",
                )
            # Second: a transport error
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=None,
                records_streamed=False,
                unit_duration_sec=0.1,
                runtime_error="openai_transport: connection reset",
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow._is_transport_runtime_error",
            lambda reason: "transport" in reason,
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(transport_fail_fast_streak=2),
            argv_tokens=[],
        )
        # Should NOT trigger fail-fast (only 1 consecutive transport err)
        assert result == 2  # still errors, but not fail-fast


# ---------------------------------------------------------------------------
# Test: _build_run_metadata path (lines 207-221)
# ---------------------------------------------------------------------------


class TestBuildRunMetadata:
    def test_fresh_run_builds_metadata(self, monkeypatch) -> None:
        """When existing_run_metadata is None, _build_run_metadata is called."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        # Use _wire_full_flow then override resume
        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        # Override resume to have None existing_run_metadata
        config_state = _FakeConfigState(config_paths=["config.toml"])
        setup = _FakeSetup(
            config_state=config_state,
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="local"),
        )
        resume = _FakeResume(
            execution_units=[(case, cond)],
            total_units=1,
            remaining_units=1,
            resumed_completed_units={("c1", "base_repair")},
            skipped_units=0,
            existing_run_metadata=None,
            resolved_config_lock_path=Path("/tmp/lock.json"),
        )
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)

        # Mock _build_run_metadata
        monkeypatch.setattr(
            "evals.eval_main_metadata._build_run_metadata",
            lambda **kw: {"run_id": "test", "patcher_config": {}},
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Test: suppress_inline_progress branch (line 860)
# ---------------------------------------------------------------------------


class TestSuppressInlineProgress:
    def test_suppress_inline_progress(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        # Set suppress_inline_progress to True
        mock_display.suppress_inline_progress = True

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Test: Parallel fail-fast with pending futures (lines 979-995)
# ---------------------------------------------------------------------------


class TestParallelFailFastAbort:
    def test_parallel_fail_fast_cancels_pending(self, monkeypatch) -> None:
        """Parallel execution abort should cancel pending futures."""
        cases = [_FakeCaseSpec(case_id=f"c{i}") for i in range(4)]
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(c, cond) for c in cases],
        )

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=None,
                records_streamed=False,
                unit_duration_sec=0.1,
                runtime_error="openai_transport: reset",
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow._is_transport_runtime_error",
            lambda reason: True,
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(jobs=2, transport_fail_fast_streak=1),
            argv_tokens=[],
        )
        assert result == 2


# ---------------------------------------------------------------------------
# Test: Online policy controller creation exception (lines 303-308)
# ---------------------------------------------------------------------------


class TestOnlinePolicyControllerCreationError:
    def test_controller_creation_error_closes_and_reraises(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        mock_online_config = MagicMock()
        mock_online_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online_config),
        )

        def _bad_ctrl(cfg):
            raise RuntimeError("controller init failed")

        monkeypatch.setattr("evals.eval_main_policies.AdaptiveOnlinePolicyController", _bad_ctrl)

        from evals.eval_main_flow import run_main_flow

        with pytest.raises(RuntimeError, match="controller init failed"):
            run_main_flow(
                parser=_make_parser(),
                args=_make_args(),
                argv_tokens=[],
            )
        mock_display.close.assert_called()


# ---------------------------------------------------------------------------
# Test: Budget policy controller creation exception (lines 343-348)
# ---------------------------------------------------------------------------


class TestBudgetPolicyControllerCreationError:
    def test_controller_creation_error_closes_and_reraises(self, monkeypatch) -> None:
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )

        mock_budget_config = MagicMock()
        mock_budget_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.BudgetPolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_budget_config),
        )

        def _bad_ctrl(cfg):
            raise RuntimeError("budget ctrl init failed")

        monkeypatch.setattr("evals.eval_main_policies.AdaptiveBudgetPolicyController", _bad_ctrl)

        from evals.eval_main_flow import run_main_flow

        with pytest.raises(RuntimeError, match="budget ctrl init failed"):
            run_main_flow(
                parser=_make_parser(),
                args=_make_args(),
                argv_tokens=[],
            )
        mock_display.close.assert_called()


# ---------------------------------------------------------------------------
# Test: Non-streamed records + _append_records_safely (lines 535-540, 622)
# ---------------------------------------------------------------------------


class TestAppendRecordsSafely:
    def test_non_streamed_records_appended(self, monkeypatch) -> None:
        """When records_streamed=False, records go through _append_records_safely."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        success_record = {
            "success": True,
            "attempt": 1,
            "note": "success",
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 1.0,
            "patch_meta": {"propose_duration_sec": 0.5},
            "attempt_metadata": {"max_attempts": 3},
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[success_record],
                records_streamed=False,  # Not pre-streamed
                unit_duration_sec=1.0,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("none", "none"),
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Test: Adaptive progress stop early-stop reason (line 688)
# ---------------------------------------------------------------------------


class TestAdaptiveProgressStopEarlyReason:
    def test_adaptive_progress_stop_warning(self, monkeypatch) -> None:
        """Early stop via adaptive_progress_stop_triggered hits line 688."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        scored_record = {
            "success": False,
            "attempt": 1,
            "note": "tests_still_failing",
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 1.0,
            "patch_meta": {
                "propose_duration_sec": 0.5,
                "adaptive_progress_stop_triggered": True,
                "adaptive_progress_stop_note": "no progress after 3 attempts",
            },
            "attempt_metadata": {
                "max_attempts": 5,
                "prompt_chars_total": 100,
                "completion_chars_total": 50,
            },
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[scored_record],
                records_streamed=True,
                unit_duration_sec=1.5,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("stagnation", "adaptive_stop"),
        )

        from evals.eval_main_flow import run_main_flow

        run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )

        # Verify the early stop warning mentions the adaptive progress note
        warning_calls = [
            c
            for c in mock_display.add_event.call_args_list
            if c[1].get("style") == "warning" and "early stop" in str(c[1].get("message", ""))
        ]
        assert len(warning_calls) >= 1
        msg = str(warning_calls[0][1]["message"])
        assert "no progress after 3 attempts" in msg


# ---------------------------------------------------------------------------
# Test: Baseline patcher config with repo_commit and compatibility_patches
# (lines 255, 267)
# ---------------------------------------------------------------------------


class TestBaselineConfigExtended:
    def test_baseline_with_repo_commit_and_patches(self, monkeypatch, tmp_path) -> None:
        """Exercise baseline paths: repo_commit and compatibility_patches (lines 255, 267)."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
        )

        setup = _FakeSetup(
            cases=[case],
            conditions=[cond],
            execution_config=MagicMock(executor="local"),
        )
        resume = _FakeResume(
            execution_units=[(case, cond)],
            total_units=1,
            remaining_units=1,
            resumed_completed_units={("c1", "base_repair")},
            skipped_units=0,
            existing_run_metadata={
                "run_id": "test",
                "patcher_config": {},
            },
        )
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)
        monkeypatch.setattr("evals.eval_main_flow.prepare_main_resume", lambda **kw: resume)

        # Mock patcher with repo_commit attribute
        mock_patcher = MagicMock()
        mock_patcher.name = "baseline"
        mock_patcher.repo_commit = "abc123"
        monkeypatch.setattr("evals.patcher_factory.make_patcher", lambda args: mock_patcher)

        # Mock _baseline_patch_inventory to return patches
        monkeypatch.setattr(
            "evals.eval_main_metadata._baseline_patch_inventory",
            lambda name: ["patch1.diff", "patch2.diff"],
        )
        monkeypatch.setattr(
            "evals.eval_main_metadata._baseline_extra_from_args",
            lambda args, baseline_name: {"extra_key": "extra_val"},
        )

        # Create a fake baseline python binary to exercise line 255
        bl_dir = tmp_path / "baselines" / "envs" / "test_bl" / "bin"
        bl_dir.mkdir(parents=True)
        bl_python = bl_dir / "python"
        bl_python.write_text("#!/bin/sh\n")
        monkeypatch.setattr("evals.eval_seams.REPO_ROOT", tmp_path)

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(
                patcher="baseline",
                baseline_name="test_bl",
                baseline_process_timeout_sec=30.0,
            ),
            argv_tokens=[],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Test: Token usage "unavailable" state (line 518)
# ---------------------------------------------------------------------------


class TestTokenUsageUnavailable:
    def test_token_usage_unavailable_counted(self, monkeypatch) -> None:
        """A record where _record_token_usage_state returns 'unavailable' hits line 518."""
        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")

        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
        )

        # Record with no token info
        record_no_tokens = {
            "success": True,
            "attempt": 1,
            "note": "success",
            "record_type": "attempt",
            "before_duration_sec": 0.0,
            "after_duration_sec": 1.0,
            "patch_meta": {"propose_duration_sec": 0.5},
            "attempt_metadata": {
                "max_attempts": 3,
                "prompt_chars_total": 100,
                "completion_chars_total": 50,
            },
        }

        def _mock_exec(**kw):
            return MagicMock(
                case=kw["case"],
                condition=kw["condition"],
                case_records=[record_no_tokens],
                records_streamed=True,
                unit_duration_sec=1.0,
                runtime_error=None,
            )

        monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _mock_exec)
        monkeypatch.setattr(
            "evals.eval_main_flow.classify_failure_reason",
            lambda case_records: ("none", "none"),
        )
        # Force token usage state to "unavailable"
        monkeypatch.setattr(
            "evals.eval_main_outcome._record_token_usage_state",
            lambda **kw: "unavailable",
        )

        from evals.eval_main_flow import run_main_flow

        result = run_main_flow(
            parser=_make_parser(),
            args=_make_args(),
            argv_tokens=[],
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Test: decision-model binding is attached to the helper-policy controller
# ---------------------------------------------------------------------------


class TestDecisionModelBinding:
    def test_binding_is_passed_only_when_a_role_is_enabled(self, monkeypatch) -> None:
        from evals.decision_model.policy import DecisionModelBinding

        case = _FakeCaseSpec(case_id="c1")
        cond = _FakeCondition(value="base_repair")
        _mock_display, _flow_mod, _ = _wire_full_flow(
            monkeypatch,
            cases_and_conditions=[(case, cond)],
            resumed={("c1", "base_repair")},
            remaining=1,
        )
        mock_online_config = MagicMock()
        mock_online_config.mode = "on"
        monkeypatch.setattr(
            policies_mod.OnlinePolicyConfig,
            "normalize",
            staticmethod(lambda **kw: mock_online_config),
        )
        captured: dict[str, object] = {}

        def _controller(cfg, decision_model=None):
            captured["decision_model"] = decision_model
            controller = MagicMock()
            controller.load_error = None
            return controller

        monkeypatch.setattr("evals.eval_main_policies.AdaptiveOnlinePolicyController", _controller)
        args = _make_args()
        args.adaptive_decision_model_role = "shadow"

        from evals.eval_main_flow import run_main_flow

        run_main_flow(parser=_make_parser(), args=args, argv_tokens=[])

        binding = captured["decision_model"]
        assert isinstance(binding, DecisionModelBinding)
        assert binding.role == "shadow"
        assert binding.advisor is not None
        binding.advisor.close()


# ---------------------------------------------------------------------------
# ORC-01 / ORC-02 (2026-10-04 review): run settings and the container pool
# must reach the pytest executor through run_main_flow.
# ---------------------------------------------------------------------------


def test_run_main_flow_settings_and_pool_reach_run_pytest(monkeypatch, tmp_path) -> None:
    """ORC-01/ORC-02, X-02: the run's settings and pool travel in the unit's RunContext."""
    from types import SimpleNamespace

    from evals.eval_types import Condition, ExecutionConfig, RunResult, UnitExecutionOutcome
    from evals.pytest_runner import run_pytest

    case = _FakeCaseSpec(case_id="c1")
    cond = _FakeCondition(value="base_repair")
    _mock_display, _flow_mod, _ = _wire_full_flow(monkeypatch, cases_and_conditions=[(case, cond)])

    def _replace(obj, **changes):
        # The real replace for RunContext; in place for the stubbed unit inputs.
        if dataclasses.is_dataclass(obj) and type(obj).__dataclass_params__.frozen:
            return dataclasses.replace(obj, **changes)
        for name, value in changes.items():
            setattr(obj, name, value)
        return obj

    monkeypatch.setattr("evals.eval_main_flow.replace", _replace)
    monkeypatch.setattr("evals.eval_main_flow.EvalUnitInputs", lambda **kw: SimpleNamespace(**kw))
    setup = _FakeSetup(
        cases=[case],
        conditions=[cond],
        execution_config=ExecutionConfig(executor="testcontainers", container_pool=True),
    )
    monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)

    seen: dict[str, object] = {}

    class _FakeContainer:
        def exec_pytest(self, *, host_cwd, command, env, timeout_sec):
            seen["env"] = dict(env)
            return RunResult(returncode=0, duration_sec=0.0, stdout="1 passed", stderr="")

    class _FakePool:
        def __init__(self, **kwargs):
            self.acquired = 0

        def acquire(self):
            self.acquired += 1
            seen["pool"] = self
            return _FakeContainer()

        def release(self, container):
            seen["released"] = True

        def shutdown(self):
            seen["shutdown"] = True

    monkeypatch.setattr("evals.eval_main_flow._EvalContainerPool", _FakePool)

    def _execute_unit(**kwargs):
        # Inside a unit, exactly what the attempt flow does: the real run_pytest with the
        # unit's run context.
        run_pytest(
            tmp_path,
            ["-m", "pytest", "-q"],
            Condition.BASE_REPAIR,
            5,
            "json",
            run_context=kwargs["inputs"].run_condition.run_context,
        )
        outcome = MagicMock(spec=UnitExecutionOutcome)
        outcome.case = case
        outcome.condition = cond
        outcome.case_records = []
        outcome.records_streamed = True
        outcome.unit_duration_sec = 0.1
        outcome.runtime_error = None
        return outcome

    monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", _execute_unit)
    monkeypatch.setattr(
        "evals.eval_main_flow.classify_failure_reason", lambda case_records: ("none", "none")
    )

    from evals.eval_main_flow import run_main_flow

    run_main_flow(
        parser=_make_parser(),
        args=_make_args(deterministic=True, seed=1234),
        argv_tokens=[],
    )

    env = seen["env"]
    assert isinstance(env, dict)
    assert env["PYTHONHASHSEED"] == "1234"  # ORC-01
    assert seen["pool"].acquired == 1  # ORC-02: the pool is used, not a one-shot container
    assert seen["shutdown"] is True


def test_run_main_flow_starts_no_pool_without_opt_in(monkeypatch) -> None:
    """ORC-02: pooling is opt-in; one-shot containers stay the default."""
    from evals.eval_types import ExecutionConfig, UnitExecutionOutcome

    case = _FakeCaseSpec(case_id="c1")
    cond = _FakeCondition(value="base_repair")
    _mock_display, _flow_mod, _ = _wire_full_flow(monkeypatch, cases_and_conditions=[(case, cond)])
    setup = _FakeSetup(
        cases=[case], conditions=[cond], execution_config=ExecutionConfig(executor="testcontainers")
    )
    monkeypatch.setattr("evals.eval_main_flow.prepare_main_setup", lambda **kw: setup)

    def _no_pool(**kwargs):
        raise AssertionError("container pool must not start without --container-pool")

    monkeypatch.setattr("evals.eval_main_flow._EvalContainerPool", _no_pool)
    outcome = MagicMock(spec=UnitExecutionOutcome)
    outcome.case = case
    outcome.condition = cond
    outcome.case_records = []
    outcome.records_streamed = True
    outcome.unit_duration_sec = 0.1
    outcome.runtime_error = None
    monkeypatch.setattr("evals.eval_main_flow._execute_eval_unit", lambda **kw: outcome)
    monkeypatch.setattr(
        "evals.eval_main_flow.classify_failure_reason", lambda case_records: ("none", "none")
    )
    from evals.eval_main_flow import run_main_flow

    run_main_flow(parser=_make_parser(), args=_make_args(), argv_tokens=[])


def test_attempt_flow_forwards_the_run_context_to_both_pytest_runs(tmp_path, monkeypatch) -> None:
    """ORC-01/ORC-02, X-02: the real attempt flow hands its RunContext to run_pytest.

    The test above stubs ``_execute_eval_unit``, so it only proves that the main flow
    builds the context and passes it into the unit. This one drives the real attempt
    loop and the real ``run_pytest``, stubbing nothing but the container executor, so
    dropping ``run_context=`` at either pytest call site (initial verification or
    post-patch verification) fails here.
    """
    import shutil

    from evals import run_eval
    from evals.eval_types import ExecutionConfig, RunResult, _ActiveRunSettings
    from evals.patchers.base import PatchResult
    from evals.run_context import RunContext

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    case = run_eval.CaseSpec(
        case_id="ctx_handoff",
        path=case_dir,
        category="data_shape_runtime",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=10,
    )

    # What the unit executor materializes before the attempt loop starts.
    workdir = tmp_path / "workdir"
    shutil.copytree(case_dir, workdir)

    # run_pytest's no-context fallback, so a dropped run_context runs locally instead of
    # failing on the missing testcontainers import: the assertions below then report the
    # lost seeding and pooling rather than an unrelated dependency error.
    monkeypatch.setattr(
        "evals.pytest_runner.get_active_execution_config",
        lambda: ExecutionConfig(executor="local"),
    )

    envs: list[dict[str, str]] = []

    class _FakeContainer:
        def exec_pytest(self, *, host_cwd, command, env, timeout_sec) -> RunResult:
            envs.append(dict(env))
            failing = len(envs) == 1
            return RunResult(
                returncode=1 if failing else 0,
                duration_sec=0.0,
                stdout="1 failed" if failing else "1 passed",
                stderr="E   AssertionError: expected 2 got 1" if failing else "",
            )

    class _FakePool:
        def __init__(self) -> None:
            self.acquired = 0
            self.released = 0

        def acquire(self) -> _FakeContainer:
            self.acquired += 1
            return _FakeContainer()

        def release(self, container: _FakeContainer) -> None:
            self.released += 1

    pool = _FakePool()

    class _FixedDiffPatcher:
        name = "fixed_diff"

        def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            return PatchResult(
                diff=(
                    "--- a/buggy.py\n"
                    "+++ b/buggy.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def f() -> int:\n"
                    "-    return 1\n"
                    "+    return 2\n"
                ),
                meta={},
            )

    run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=tmp_path / "context",
        patcher=_FixedDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="ctx-handoff",
        include_files_context=False,
        force_patch_attempt=False,
        run_context=RunContext(
            execution=ExecutionConfig(executor="testcontainers", container_pool=True),
            settings=_ActiveRunSettings(deterministic=True, seed=1234),
            container_pool=pool,
        ),
    )

    assert len(envs) == 2, "initial and post-patch verification must both run pytest"
    # ORC-01: both runs are seeded from the run's settings.
    assert [env.get("PYTHONHASHSEED") for env in envs] == ["1234", "1234"]
    assert [env.get("TZ") for env in envs] == ["UTC", "UTC"]
    # ORC-02: both runs come from the run's pool, not from one-shot containers.
    assert (pool.acquired, pool.released) == (2, 2)
