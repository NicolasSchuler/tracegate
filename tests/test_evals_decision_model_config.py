"""Tests for decision-model settings, CLI flags and cross-flag validation."""

from __future__ import annotations

import argparse

import pytest

from evals.decision_model import config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    config.add_decision_model_arguments(parser)
    config.add_policy_input_arguments(parser)
    return parser


def test_flags_default_to_off_and_safe_policy_inputs() -> None:
    args = _parser().parse_args([])

    assert args.adaptive_decision_model_role == "off"
    assert args.adaptive_decision_model_profile == "evidence_v1"
    assert args.adaptive_decision_model_log_state == "full"
    assert args.adaptive_online_include_case_labels is False
    assert args.adaptive_lite_snapshot_mode == "replace"


def test_case_label_flag_pair_round_trips() -> None:
    parser = _parser()
    assert parser.parse_args(
        ["--adaptive-online-include-case-labels"]
    ).adaptive_online_include_case_labels
    assert not parser.parse_args(
        ["--adaptive-online-include-case-labels", "--no-adaptive-online-include-case-labels"]
    ).adaptive_online_include_case_labels


def test_settings_from_args_parses_every_flag() -> None:
    args = _parser().parse_args(
        [
            "--adaptive-decision-model-role",
            "prior",
            "--adaptive-decision-model-url",
            "http://gpu-box:9000/",
            "--adaptive-decision-model-name",
            "clef",
            "--adaptive-decision-model-profile",
            "evidence_atomic_v1",
            "--adaptive-decision-model-timeout-sec",
            "3.5",
            "--adaptive-decision-model-temperature",
            "0.7",
            "--adaptive-decision-model-prior-beta",
            "0.25",
            "--adaptive-decision-model-prior-anneal-updates",
            "20",
            "--adaptive-decision-model-warmup-mix",
            "0.5",
            "--adaptive-decision-model-log-state",
            "hash",
        ]
    )

    settings = config.decision_model_settings_from_args(args)

    assert settings.enabled
    assert settings.policy == config.DecisionModelPolicyConfig(
        role="prior",
        profile="evidence_atomic_v1",
        temperature=0.7,
        prior_beta=0.25,
        prior_anneal_updates=20,
        warmup_mix=0.5,
    )
    assert settings.client == config.DecisionModelClientConfig(
        url="http://gpu-box:9000", model="clef", timeout_sec=3.5
    )
    assert settings.log_state == "hash"


def test_settings_tolerate_namespaces_without_the_flags() -> None:
    settings = config.decision_model_settings_from_args(argparse.Namespace())
    assert not settings.enabled
    assert settings.policy == config.DecisionModelPolicyConfig()


def test_url_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DECISION_MODEL_URL_ENV, raising=False)
    assert config.resolve_decision_model_url("") == config.DEFAULT_DECISION_MODEL_URL
    assert (
        config.resolve_decision_model_url("", environ={config.DECISION_MODEL_URL_ENV: "http://e/"})
        == "http://e"
    )
    assert (
        config.resolve_decision_model_url(
            "http://flag", environ={config.DECISION_MODEL_URL_ENV: "http://e"}
        )
        == "http://flag"
    )


def _settings(**policy: object) -> config.DecisionModelSettings:
    return config.DecisionModelSettings(policy=config.DecisionModelPolicyConfig(**policy))


def test_check_passes_for_off_and_valid_combinations() -> None:
    assert (
        config.check_decision_model_settings(
            _settings(), llm_discretion_condition=False, online_policy_mode="off"
        )
        == config.DecisionModelSetupCheck()
    )
    for role, mode in (("shadow", "shadow"), ("shadow", "on"), ("policy", "on")):
        check = config.check_decision_model_settings(
            _settings(role=role), llm_discretion_condition=True, online_policy_mode=mode
        )
        assert check.errors == ()
        assert check.force_off is False


@pytest.mark.parametrize(
    ("policy", "mode", "message"),
    [
        ({"role": "policy"}, "shadow", "requires --adaptive-online-policy-mode on"),
        ({"role": "features"}, "off", "requires --adaptive-online-policy-mode on"),
        ({"role": "shadow"}, "off", "shadow or on"),
        ({"role": "features", "profile": "evidence_v1_noaux"}, "on", "no diagnostic questions"),
        ({"role": "prior", "profile": "evidence_atomic_v1"}, "on", "no action question"),
        ({"role": "policy", "temperature": -1.0}, "on", "temperature must be >= 0"),
        ({"role": "prior", "warmup_mix": 1.5}, "on", "warmup-mix must be within"),
        ({"role": "prior", "prior_anneal_updates": 0}, "on", "anneal-updates must be >= 1"),
        ({"role": "policy", "profile": "nope"}, "on", "unknown decision-model profile"),
    ],
)
def test_check_reports_invalid_combinations(
    policy: dict[str, object], mode: str, message: str
) -> None:
    check = config.check_decision_model_settings(
        _settings(**policy), llm_discretion_condition=True, online_policy_mode=mode
    )
    assert any(message in error for error in check.errors), check.errors


def test_check_forces_role_off_without_llm_discretion_condition() -> None:
    check = config.check_decision_model_settings(
        _settings(role="policy"), llm_discretion_condition=False, online_policy_mode="on"
    )
    assert check.force_off is True
    assert check.errors == ()
    assert "layered_adaptive_llm" in check.warnings[0]


def test_run_config_excludes_the_url_and_records_policy_inputs() -> None:
    args = _parser().parse_args(
        [
            "--adaptive-decision-model-role",
            "shadow",
            "--adaptive-decision-model-url",
            "http://secret-host:1",
            "--adaptive-online-include-case-labels",
            "--adaptive-lite-snapshot-mode",
            "supplement",
        ]
    )

    run_config = config.decision_model_run_config(args)

    assert run_config["adaptive_decision_model_role"] == "shadow"
    assert run_config["adaptive_online_include_case_labels"] is True
    assert run_config["adaptive_lite_snapshot_mode"] == "supplement"
    assert all("secret-host" not in str(value) for value in run_config.values())


def test_api_key_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DECISION_MODEL_API_KEY_ENV, " token-123 ")
    settings = config.decision_model_settings_from_args(_parser().parse_args([]))
    assert settings.client.api_key == "token-123"
    assert "token-123" not in str(config.decision_model_run_config(_parser().parse_args([])))


def test_api_key_flag_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.DECISION_MODEL_API_KEY_ENV, "env-token")
    args = _parser().parse_args(["--adaptive-decision-model-api-key", " cli-token "])
    settings = config.decision_model_settings_from_args(args)
    assert settings.client.api_key == "cli-token"
    assert "cli-token" not in repr(settings)
    assert "cli-token" not in str(config.decision_model_run_config(args))
