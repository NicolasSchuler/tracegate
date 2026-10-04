"""Tests for decision-model flag handling during run setup (force-off, checks, TOML)."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import eval_config, eval_main_setup
from evals.decision_model.config import add_decision_model_arguments, add_policy_input_arguments
from evals.eval_main_setup import MainFlowCliError
from evals.eval_types import Condition


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-online-policy-mode", default="on")
    add_decision_model_arguments(parser)
    add_policy_input_arguments(parser)
    return parser.parse_args(list(argv))


def _config_state() -> SimpleNamespace:
    return SimpleNamespace(value_sources={})


def test_non_adaptive_runs_force_the_role_off() -> None:
    args = _args("--adaptive-decision-model-role", "policy")
    state = _config_state()

    has_adaptive, warnings = eval_main_setup._force_adaptive_policies_off_when_unused(
        args=args, config_state=state, conditions=[Condition.BASE_REPAIR]
    )

    assert has_adaptive is False
    assert args.adaptive_decision_model_role == "off"
    assert state.value_sources["adaptive_decision_model_role"] == (
        "forced_off:no_adaptive_condition"
    )
    assert "adaptive_decision_model_role" in warnings[0]


def test_checks_accept_a_valid_policy_run() -> None:
    args = _args("--adaptive-decision-model-role", "policy")
    warnings: list[str] = []

    eval_main_setup._apply_decision_model_checks(
        args=args,
        config_state=_config_state(),
        conditions=[Condition.LAYERED_ADAPTIVE_LLM],
        startup_warnings=warnings,
    )

    assert args.adaptive_decision_model_role == "policy"
    assert warnings == []


def test_checks_reject_roles_that_need_policy_mode_on() -> None:
    args = _args(
        "--adaptive-decision-model-role", "features", "--adaptive-online-policy-mode", "shadow"
    )

    with pytest.raises(MainFlowCliError, match="requires --adaptive-online-policy-mode on"):
        eval_main_setup._apply_decision_model_checks(
            args=args,
            config_state=_config_state(),
            conditions=[Condition.LAYERED_ADAPTIVE_LLM],
            startup_warnings=[],
        )


def test_checks_force_off_for_gated_only_runs() -> None:
    args = _args("--adaptive-decision-model-role", "shadow")
    state = _config_state()
    warnings: list[str] = []

    eval_main_setup._apply_decision_model_checks(
        args=args,
        config_state=state,
        conditions=[Condition.LAYERED_ADAPTIVE_GATED],
        startup_warnings=warnings,
    )

    assert args.adaptive_decision_model_role == "off"
    assert state.value_sources["adaptive_decision_model_role"] == (
        "forced_off:no_llm_discretion_condition"
    )
    assert "layered_adaptive_llm" in warnings[0]


def test_toml_adaptive_table_sets_decision_model_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "dm.toml"
    config_path.write_text(
        "[run_eval.adaptive]\n"
        'decision_model_role = "shadow"\n'
        'decision_model_profile = "evidence_atomic_v1"\n'
        "online_include_case_labels = true\n"
        'lite_snapshot_mode = "supplement"\n',
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[])
    add_decision_model_arguments(parser)
    add_policy_input_arguments(parser)
    argv = ["--config", str(config_path)]
    args = parser.parse_args(argv)

    state = eval_config._apply_toml_configs_to_args(parser=parser, args=args, argv_tokens=argv)

    assert args.adaptive_decision_model_role == "shadow"
    assert args.adaptive_decision_model_profile == "evidence_atomic_v1"
    assert args.adaptive_online_include_case_labels is True
    assert args.adaptive_lite_snapshot_mode == "supplement"
    assert state.value_sources["adaptive_decision_model_role"] != "default"


def test_settings_ignore_magicmock_attribute_leakage() -> None:
    from unittest.mock import MagicMock

    from evals.decision_model.config import decision_model_settings_from_args

    settings = decision_model_settings_from_args(MagicMock())
    assert not settings.enabled


def test_resume_preflight_fails_fast_when_the_sidecar_is_unhealthy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from unittest.mock import MagicMock

    from evals import eval_main_resume, run_eval
    from evals.eval_main_setup import MainSetupState
    from evals.run_eval import CaseSpec, ExecutionConfig

    case = CaseSpec(
        case_id="c1",
        path=tmp_path,
        category="logic_error",
        difficulty="easy",
        expected_exception="ValueError",
        python_args=["python", "test.py"],
        timeout_sec=30,
        snapshot_dependency="required",
        tags=[],
    )
    setup = MainSetupState(
        config_state=MagicMock(),
        execution_config=ExecutionConfig(executor="local"),
        cases=[case],
        conditions=[Condition.LAYERED_ADAPTIVE_LLM],
        run_id="test-run",
        artifacts_root=tmp_path / "artifacts",
        results_path=tmp_path / "results",
        has_adaptive_condition=True,
        startup_warnings=[],
    )
    monkeypatch.setattr(run_eval, "_build_run_settings_fingerprint", lambda **_kw: "fp")
    monkeypatch.setattr(
        run_eval, "_write_resolved_run_config_lockfile", lambda **_kw: tmp_path / "lock"
    )
    monkeypatch.setattr(
        run_eval,
        "_RunResultsLock",
        MagicMock(acquire=MagicMock(return_value=(MagicMock(), None, None))),
    )
    monkeypatch.setattr(
        run_eval, "_load_existing_run_state", lambda **_kw: (set(), None, "", 0, None)
    )
    monkeypatch.setattr(
        eval_main_resume, "check_decision_model_health", lambda _settings: "sidecar down"
    )
    args = MagicMock()
    args.k = 1
    args.adaptive_decision_model_role = "shadow"

    with pytest.raises(MainFlowCliError, match="Environment check failed: sidecar down"):
        eval_main_resume.prepare_main_resume(setup=setup, args=args, argv_tokens=[])
