"""Tests for shipped adaptive probe config presets."""

from __future__ import annotations

from pathlib import Path

from evals import eval_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _REPO_ROOT / "evals" / "configs"


def _flatten_config(config_name: str) -> dict[str, object]:
    root = eval_config._load_toml_document(_CONFIG_ROOT / config_name)
    payload = eval_config._extract_run_eval_config_payload(_CONFIG_ROOT / config_name, root)
    return eval_config._flatten_config_mapping(payload)


def test_decay_probe_presets_pin_jobs_one_and_decay_values() -> None:
    decay_098 = _flatten_config("openai_adaptive_probe_decay_098.template.toml")
    decay_0995 = _flatten_config("openai_adaptive_probe_decay_0995.template.toml")

    assert decay_098["jobs"] == 1
    assert decay_0995["jobs"] == 1
    assert decay_098["adaptive_online_decay"] == 0.98
    assert decay_098["adaptive_budget_online_decay"] == 0.98
    assert decay_0995["adaptive_online_decay"] == 0.995
    assert decay_0995["adaptive_budget_online_decay"] == 0.995


def test_prompt_order_probe_presets_select_expected_modes() -> None:
    crash_first = _flatten_config("openai_adaptive_probe_prompt_order_crash_first.template.toml")
    evidence_last = _flatten_config(
        "openai_adaptive_probe_prompt_order_evidence_last.template.toml"
    )

    assert crash_first["jobs"] == 1
    assert evidence_last["jobs"] == 1
    assert crash_first["prompt_order_mode"] == "crash_first"
    assert evidence_last["prompt_order_mode"] == "evidence_last"


def test_current_knobs_probe_preset_uses_existing_compaction_and_budget_knobs() -> None:
    compact = _flatten_config("openai_adaptive_probe_current_knobs_compact.template.toml")

    assert compact["jobs"] == 1
    assert compact["context_stage_a_max_requests"] == 4
    assert compact["context_stage_a_max_chars"] == 80000
    assert compact["prompt_budget_total_chars"] == 140000
    assert compact["prompt_budget_snapshot_chars"] == 24000
    assert compact["prompt_budget_evidence_chars"] == 48000
    assert compact["prompt_budget_retry_history_chars"] == 8000
    assert compact["adaptive_helper_budget_profile"] == "compact"
    assert compact["context_compaction_max_calls"] == 2
    assert compact["context_compaction_target_chars"] == 12000


def test_warm_start_probe_preset_sets_distinct_init_and_state_paths() -> None:
    warm = _flatten_config("openai_adaptive_probe_warm_start.template.toml")

    assert warm["jobs"] == 1
    assert warm["adaptive_online_init_state_path"]
    assert warm["adaptive_budget_init_state_path"]
    assert warm["adaptive_online_init_state_path"] != warm["adaptive_online_state_path"]
    assert warm["adaptive_budget_init_state_path"] != warm["adaptive_budget_online_state_path"]


def test_bwk_probe_preset_selects_bwk_architecture() -> None:
    bwk = _flatten_config("openai_adaptive_probe_bwk_v1.template.toml")

    assert bwk["jobs"] == 1
    assert bwk["adaptive_budget_policy_architecture"] == "bwk_v1"
    assert bwk["adaptive_budget_bwk_run_token_budget"] == 50000


def test_research_probe_presets_select_expected_nondefault_knobs() -> None:
    hierarchical = _flatten_config("openai_adaptive_probe_hierarchical_v1.template.toml")
    thompson = _flatten_config("openai_adaptive_probe_thompson.template.toml")
    interactions = _flatten_config(
        "openai_adaptive_probe_feature_scaling_interactions.template.toml"
    )
    percentile = _flatten_config("openai_adaptive_probe_feature_scaling_percentile.template.toml")
    traceback_first = _flatten_config("openai_adaptive_probe_traceback_first.template.toml")
    pruning = _flatten_config("openai_adaptive_probe_snapshot_pruning_heuristic.template.toml")
    tep_full = _flatten_config("openai_adaptive_probe_tep_full.template.toml")

    assert hierarchical["jobs"] == 1
    assert hierarchical["adaptive_budget_policy_architecture"] == "hierarchical_v1"

    assert thompson["jobs"] == 1
    assert thompson["adaptive_online_policy_algo"] == "thompson"
    assert thompson["adaptive_budget_policy_algo"] == "thompson"

    assert interactions["jobs"] == 1
    assert interactions["adaptive_online_feature_scaling_profile"] == "v1_interactions"
    assert interactions["adaptive_budget_feature_scaling_profile"] == "v1"

    assert percentile["jobs"] == 1
    assert percentile["adaptive_online_feature_scaling_profile"] == "percentile_v1"
    assert percentile["adaptive_budget_feature_scaling_profile"] == "percentile_v1"

    assert traceback_first["jobs"] == 1
    assert traceback_first["adaptive_reveal_mode"] == "traceback_first"

    assert pruning["jobs"] == 1
    assert pruning["snapshot_frame_pruning_mode"] == "heuristic"

    assert tep_full["jobs"] == 1
    assert tep_full["adaptive_tep_mode"] == "full"
    assert tep_full["adaptive_tep_state_path"]


def test_decision_model_probe_presets_select_roles_and_profiles() -> None:
    presets = {
        slug: _flatten_config(f"openai_adaptive_probe_decision_model_{slug}.template.toml")
        for slug in ("shadow", "policy", "policy_atomic", "features", "prior")
    }

    for slug, config in presets.items():
        assert config["jobs"] == 1, slug
        assert config["conditions"] == "layered_adaptive_llm", slug
        # Every live role (and the shadow logging path) needs the helper policy running.
        assert config["adaptive_online_policy_mode"] == "on", slug

    assert presets["shadow"]["adaptive_decision_model_role"] == "shadow"
    assert presets["policy"]["adaptive_decision_model_role"] == "policy"
    assert presets["policy_atomic"]["adaptive_decision_model_role"] == "policy"
    assert presets["features"]["adaptive_decision_model_role"] == "features"
    assert presets["prior"]["adaptive_decision_model_role"] == "prior"

    # Only the atomic preset overrides the default question profile.
    assert presets["policy_atomic"]["adaptive_decision_model_profile"] == "evidence_atomic_v1"
    for slug in ("shadow", "policy", "features", "prior"):
        assert "adaptive_decision_model_profile" not in presets[slug], slug


def test_gated_policy_probe_presets_select_expected_gate_policies() -> None:
    aggressive = _flatten_config("openai_adaptive_gated_aggressive.template.toml")
    conservative = _flatten_config("openai_adaptive_gated_conservative.template.toml")

    assert aggressive["conditions"] == "layered_adaptive_gated"
    assert conservative["conditions"] == "layered_adaptive_gated"
    assert aggressive["adaptive_gate_policy"] == "aggressive"
    assert conservative["adaptive_gate_policy"] == "conservative"
    assert aggressive["adaptive_online_policy_mode"] == "off"
    assert conservative["adaptive_budget_policy_mode"] == "off"


def test_tuned_template_only_changes_policy_knobs_not_reward_scales() -> None:
    base = _flatten_config("openai_adaptive.template.toml")
    tuned = _flatten_config("openai_adaptive_tuned.template.toml")

    assert base["adaptive_online_action_score_biases"] == "search_only=-1.00,frame_only=-0.50"
    assert tuned["adaptive_online_action_score_biases"] == "search_only=-1.00,frame_only=-0.50"
    assert base["adaptive_budget_policy_tiers"] == "1.0,1.1,1.2,1.35,1.5"
    assert tuned["adaptive_budget_policy_tiers"] == "1.0,1.1,1.2,1.35,1.5"
    assert base["adaptive_budget_policy_max_ups_per_case"] == 1
    assert tuned["adaptive_budget_policy_max_ups_per_case"] == 1
    assert base["adaptive_online_warmup_decisions"] == 50
    assert tuned["adaptive_online_epsilon"] == 0.15
    assert tuned["adaptive_online_warmup_decisions"] == 50
    assert tuned["adaptive_online_decay"] == 0.95
    assert tuned["adaptive_budget_online_epsilon"] == 0.08
    assert tuned["adaptive_budget_online_decay"] == 0.95
    assert tuned["adaptive_reward_lambda_tokens"] == base["adaptive_reward_lambda_tokens"]
    assert tuned["adaptive_reward_mu_latency"] == base["adaptive_reward_mu_latency"]
    for key in (
        "adaptive_reward_success_hint",
        "adaptive_reward_progress_scale",
        "adaptive_reward_stagnation_scale",
        "adaptive_reward_materialization_scale",
    ):
        assert key not in tuned
