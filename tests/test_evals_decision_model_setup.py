"""Tests for decision-model flag handling during run setup (force-off, checks, TOML)."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import eval_config, eval_main_setup
from evals.decision_model.config import add_decision_model_arguments, add_policy_input_arguments
from evals.decision_model.runtime import DecisionModelPreflight
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
        eval_main_resume,
        "decision_model_preflight",
        lambda _settings: DecisionModelPreflight(error="sidecar down"),
    )
    args = MagicMock()
    args.k = 1
    args.adaptive_decision_model_role = "shadow"

    with pytest.raises(MainFlowCliError, match="Environment check failed: sidecar down"):
        eval_main_resume.prepare_main_resume(setup=setup, args=args, argv_tokens=[])


def _health_transport(body: dict[str, object]) -> object:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_preflight_resolves_a_sanitized_serving_identity() -> None:
    from evals.decision_model.client import SystemOneClient
    from evals.decision_model.config import (
        DecisionModelClientConfig,
        DecisionModelPolicyConfig,
        DecisionModelSettings,
    )
    from evals.decision_model.runtime import decision_model_preflight

    settings = DecisionModelSettings(
        policy=DecisionModelPolicyConfig(role="shadow"),
        client=DecisionModelClientConfig(url="http://sidecar.test"),
    )
    client = SystemOneClient(
        settings.client,
        transport=_health_transport(
            {
                "status": "ok",
                "model": "Cloudflare/clef-flash",
                "revision": "a" * 40,
                "device": "cuda",
                "dtype": "bfloat16",
                "quantization": "nf4",
                "max_length": 8192.0,
                "uptime_sec": 12.5,  # not part of the identity
            }
        ),
    )
    try:
        preflight = decision_model_preflight(settings, client=client)
    finally:
        client.close()

    assert preflight.error == ""
    assert preflight.identity == {
        "model": "Cloudflare/clef-flash",
        "revision": "a" * 40,
        "device": "cuda",
        "dtype": "bfloat16",
        "quantization": "nf4",
        "max_length": 8192,
    }

    # An endpoint that reports nothing about the release contributes no configuration.
    quiet = SystemOneClient(settings.client, transport=_health_transport({"status": "ok"}))
    try:
        assert decision_model_preflight(settings, client=quiet).identity == {}
    finally:
        quiet.close()


def _metadata_inputs(serving: dict[str, object]) -> object:
    from evals.eval_main_metadata import RunMetadataInputs

    return RunMetadataInputs(
        args=SimpleNamespace(
            native_baseline_mode="",
            openai_auto_context_window=False,
            openai_response_mode_autoswitch="off",
            patcher="openai",
        ),
        conditions=[SimpleNamespace(value="layered_adaptive_llm")],
        config_state=SimpleNamespace(config_paths=[]),
        decision_model_serving=serving,
        existing_run_metadata={
            "run_id": "r1",
            "run_settings_fingerprint": "fp",
            "patcher_config": {},
        },
        openai_context_window_probe_error=None,
        openai_context_window_probe_provider="",
        openai_context_window_source="",
        openai_context_window_tokens_effective=0,
        patcher=object(),
        patcher_name="openai",
        resolved_config_lock_path=None,
        run_id="r1",
        run_settings_fingerprint="fp",
        seams=SimpleNamespace(repo_root=Path(".")),
        setup=SimpleNamespace(cases=[]),
    )


def test_run_metadata_records_the_resolved_serving_identity() -> None:
    from evals.decision_model.config import DECISION_MODEL_SERVING_CONFIG_KEY
    from evals.eval_main_metadata import build_run_metadata_for_run

    identity = {"model": "Cloudflare/clef-flash", "revision": "abc123", "quantization": "nf4"}
    assembly = build_run_metadata_for_run(_metadata_inputs(dict(identity)))
    assert assembly.patcher_config[DECISION_MODEL_SERVING_CONFIG_KEY] == identity

    # Nothing resolved (no decision model, or an endpoint that reports no identity).
    plain = build_run_metadata_for_run(_metadata_inputs({}))
    assert DECISION_MODEL_SERVING_CONFIG_KEY not in plain.patcher_config


def _resume_with_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    existing_identity: dict[str, object] | None,
    resolved_identity: dict[str, object],
    allow_drift: bool,
) -> object:
    """Run ``prepare_main_resume`` over existing records with a given serving identity."""
    from dataclasses import replace
    from unittest.mock import MagicMock

    from evals import eval_main_resume
    from evals.decision_model.config import DECISION_MODEL_SERVING_CONFIG_KEY
    from evals.eval_main_setup import MainSetupState
    from evals.eval_seams import default_eval_seams
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
        results_path=tmp_path / "results.jsonl",
        has_adaptive_condition=True,
        startup_warnings=[],
    )
    patcher_config: dict[str, object] = {}
    if existing_identity is not None:
        patcher_config[DECISION_MODEL_SERVING_CONFIG_KEY] = dict(existing_identity)
    existing_metadata = {"run_id": "test-run", "patcher_config": patcher_config}
    seams = replace(
        default_eval_seams(),
        build_run_settings_fingerprint=lambda **_kw: "fp",
        write_resolved_run_config_lockfile=lambda **_kw: tmp_path / "lock",
        load_existing_run_state=lambda **_kw: (set(), existing_metadata, "fp", 3, None),
        run_results_lock=SimpleNamespace(acquire=lambda **_kw: (None, None, None)),
    )
    monkeypatch.setattr(
        eval_main_resume,
        "decision_model_preflight",
        lambda _settings: DecisionModelPreflight(identity=dict(resolved_identity)),
    )
    args = MagicMock()
    args.k = 1
    args.adaptive_decision_model_role = "shadow"
    args.allow_settings_drift = allow_drift
    return eval_main_resume.prepare_main_resume(setup=setup, args=args, argv_tokens=[], seams=seams)


def test_resume_rejects_a_changed_serving_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A restarted sidecar on another revision/quantization must not extend an existing run."""
    with pytest.raises(MainFlowCliError, match="serves a different release"):
        _resume_with_serving(
            monkeypatch,
            tmp_path,
            existing_identity={"model": "clef", "revision": "old", "quantization": "none"},
            resolved_identity={"model": "clef", "revision": "old", "quantization": "nf4"},
            allow_drift=False,
        )


def test_resume_allows_a_changed_serving_identity_with_the_drift_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resume = _resume_with_serving(
        monkeypatch,
        tmp_path,
        existing_identity={"model": "clef", "revision": "old"},
        resolved_identity={"model": "clef", "revision": "new"},
        allow_drift=True,
    )
    assert resume.decision_model_serving == {"model": "clef", "revision": "new"}
    assert "serving identity changed" in capsys.readouterr().out


def test_resume_accepts_records_written_before_the_serving_identity_existed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resume = _resume_with_serving(
        monkeypatch,
        tmp_path,
        existing_identity=None,
        resolved_identity={"model": "clef", "revision": "new"},
        allow_drift=False,
    )
    assert resume.decision_model_serving == {"model": "clef", "revision": "new"}
