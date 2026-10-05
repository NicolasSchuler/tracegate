"""Run-level setup and validation helpers for ``evals.run_eval``."""

from __future__ import annotations

import argparse
import difflib
import re
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evals.adaptive_budget_online_policy import BUDGET_POLICY_ARCHITECTURES
from evals.baseline_runtime import (
    _normalize_adaptive_controller_lambda_rlm_settings,
    _normalize_baseline_selection,
    _resolve_conditions_for_run,
    _resolve_external_baseline_runtime_requirements,
)
from evals.decision_model.config import (
    check_decision_model_settings,
    decision_model_settings_from_args,
)
from evals.eval_config import (
    _apply_toml_configs_to_args,
    _parse_adaptive_budget_policy_tiers,
    _parse_adaptive_online_action_score_biases,
    build_execution_config_from_args,
    resolve_execution_runtime_metadata,
)
from evals.eval_seams import EvalSeams, default_eval_seams
from evals.eval_types import (
    ADAPTIVE_BUDGET_POLICY_ALGOS,
    ADAPTIVE_BUDGET_POLICY_MODES,
    ADAPTIVE_FEATURE_SCALING_PROFILES,
    ADAPTIVE_GATE_POLICIES,
    ADAPTIVE_HELPER_BUDGET_PROFILES,
    ADAPTIVE_ONLINE_POLICY_ALGOS,
    ADAPTIVE_ONLINE_POLICY_MODES,
    ADAPTIVE_OPENAI_HELPER_RESPONSE_MODES,
    ADAPTIVE_REVEAL_MODES,
    CONTEXT_COMPACTION_MODES,
    CONTEXT_COMPACTION_SCOPE_TOKENS,
    CONTEXT_PROTOCOLS,
    DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
    ONLINE_ACTION_SCORE_BIAS_LIMIT,
    OPENAI_RESPONSE_MODE_AUTOSWITCH_MODES,
    PROGRESS_VERBOSITIES,
    PROMPT_ORDER_MODES,
    RECORD_MODES,
    RETRY_CONTEXT_MODES,
    RETRY_PROMPT_MODES,
    SNAPSHOT_FRAME_PRUNING_MODES,
    CaseSpec,
    Condition,
)


class MainFlowCliError(Exception):
    """User-facing configuration error for the run-eval main flow."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(slots=True)
class MainSetupState:
    config_state: Any
    execution_config: Any
    cases: list[CaseSpec]
    conditions: list[Condition]
    has_adaptive_condition: bool
    startup_warnings: list[str]
    run_id: str
    artifacts_root: Path
    results_path: Path


def _fail(message: str) -> None:
    raise MainFlowCliError(message)


def _apply_decision_model_checks(
    *,
    args: argparse.Namespace,
    config_state: Any,
    conditions: list[Condition],
    startup_warnings: list[str],
) -> None:
    """Validate decision-model flags once conditions are known (cross-flag rules)."""
    check = check_decision_model_settings(
        decision_model_settings_from_args(args),
        llm_discretion_condition=Condition.LAYERED_ADAPTIVE_LLM in conditions,
        online_policy_mode=str(getattr(args, "adaptive_online_policy_mode", "off")),
    )
    if check.errors:
        _fail("Invalid decision-model settings: " + "; ".join(check.errors))
    startup_warnings.extend(check.warnings)
    if check.force_off:
        args.adaptive_decision_model_role = "off"
        config_state.value_sources["adaptive_decision_model_role"] = (
            "forced_off:no_llm_discretion_condition"
        )


def _require_choice(*, value: Any, allowed: set[str] | tuple[str, ...], option: str) -> None:
    if str(value) not in allowed:
        _fail(f"Invalid {option}: {value}")


def _namespace_arg(args: Any, name: str, default: Any) -> Any:
    """Read argparse-style fields without MagicMock auto-attribute leakage."""
    try:
        mapping = vars(args)
    except TypeError:
        mapping = None
    if isinstance(mapping, dict) and name in mapping:
        return mapping[name]
    return default


def _require_int_ge(*, value: Any, minimum: int, option: str) -> None:
    if int(value) < minimum:
        _fail(f"Invalid {option}: must be >= {minimum}")


def _require_float_ge(*, value: Any, minimum: float, option: str) -> None:
    if float(value) < minimum:
        _fail(f"Invalid {option}: must be >= {minimum:g}")


def _require_float_gt(*, value: Any, minimum: float, option: str) -> None:
    if float(value) <= minimum:
        _fail(f"Invalid {option}: must be > {minimum:g}")


def _require_unit_interval(*, value: Any, option: str, left_closed: bool = True) -> None:
    numeric = float(value)
    if left_closed:
        ok = 0.0 <= numeric <= 1.0
        range_text = "between 0.0 and 1.0"
    else:
        ok = 0.0 < numeric <= 1.0
        range_text = "in the range (0.0, 1.0]"
    if not ok:
        _fail(f"Invalid {option}: must be {range_text}")


def _validate_context_compaction_scope(scope: str) -> None:
    tokens = {token.strip().lower() for token in str(scope or "").split(",") if token.strip()}
    invalid_tokens = sorted(
        token for token in tokens if token not in CONTEXT_COMPACTION_SCOPE_TOKENS
    )
    if invalid_tokens:
        _fail(
            "Invalid --context-compaction-scope: unknown token(s): "
            + ", ".join(invalid_tokens)
            + " (allowed: snapshot,pytest,files)"
        )


def _load_filtered_cases(
    args: argparse.Namespace, seams: EvalSeams | None = None
) -> list[CaseSpec]:
    seams = seams if seams is not None else default_eval_seams()
    cases_dir = seams.repo_root / args.cases_dir
    try:
        cases = seams.load_cases(cases_dir)
    except ValueError as exc:
        _fail(f"Invalid case metadata: {exc}")

    wanted_case_ids = set(args.case)
    case_list_ids: list[str] = []
    for case_list_file in args.case_list_file:
        path = Path(str(case_list_file)).expanduser()
        if not path.exists():
            _fail(f"Case-list file not found: {path}")
        if not path.is_file():
            _fail(f"Case-list path is not a file: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            _fail(f"Failed to read case-list file {path}: {exc}")
        for line in content.splitlines():
            normalized = line.strip()
            if normalized and not normalized.startswith("#"):
                case_list_ids.append(normalized)

    wanted_case_ids.update(case_list_ids)
    if wanted_case_ids:
        available_case_ids = {case.case_id for case in cases}
        unknown_case_ids = sorted(wanted_case_ids.difference(available_case_ids))
        if unknown_case_ids:
            _fail(
                "Unknown case id(s) requested via --case/--case-list-file: "
                + ", ".join(unknown_case_ids)
            )
        cases = [case for case in cases if case.case_id in wanted_case_ids]
    if args.category:
        wanted_categories = {value.strip() for value in args.category if value.strip()}
        cases = [case for case in cases if case.category in wanted_categories]
    if args.difficulty:
        wanted_difficulties = {value.strip() for value in args.difficulty if value.strip()}
        cases = [case for case in cases if case.difficulty in wanted_difficulties]
    if args.snapshot_dependency:
        wanted_snapshot_dependency = {
            value.strip() for value in args.snapshot_dependency if value.strip()
        }
        cases = [case for case in cases if case.snapshot_dependency in wanted_snapshot_dependency]
    if args.bug_source:
        wanted_bug_sources = {value.strip() for value in args.bug_source if value.strip()}
        cases = [
            case for case in cases if (case.bug_source or "hand_crafted") in wanted_bug_sources
        ]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        _fail("No cases found.")
    return cases


def _force_adaptive_policies_off_when_unused(
    *,
    args: argparse.Namespace,
    config_state: Any,
    conditions: list[Condition],
) -> tuple[bool, list[str]]:
    adaptive_conditions = {Condition.LAYERED_ADAPTIVE_GATED, Condition.LAYERED_ADAPTIVE_LLM}
    has_adaptive_condition = any(condition in adaptive_conditions for condition in conditions)
    startup_warnings: list[str] = []
    if has_adaptive_condition:
        return True, startup_warnings

    forced_labels: list[str] = []

    def _force_value(*, attr: str, forced_value: Any, label: str) -> None:
        current_value = getattr(args, attr, None)
        if current_value != forced_value:
            forced_labels.append(label)
        setattr(args, attr, forced_value)
        config_state.value_sources[attr] = "forced_off:no_adaptive_condition"

    _force_value(
        attr="adaptive_online_policy_mode",
        forced_value="off",
        label="adaptive_online_policy_mode",
    )
    _force_value(
        attr="adaptive_budget_policy_mode",
        forced_value="off",
        label="adaptive_budget_policy_mode",
    )
    _force_value(
        attr="adaptive_decision_model_role",
        forced_value="off",
        label="adaptive_decision_model_role",
    )
    _force_value(
        attr="context_protocol",
        forced_value="monolithic",
        label="context_protocol",
    )
    _force_value(
        attr="adaptive_controller_lambda_rlm",
        forced_value=False,
        label="adaptive_controller_lambda_rlm",
    )
    _force_value(
        attr="adaptive_tep_mode",
        forced_value="off",
        label="adaptive_tep_mode",
    )
    _force_value(
        attr="adaptive_tep",
        forced_value=False,
        label="adaptive_tep",
    )
    _force_value(
        attr="adaptive_tep_equilibrium",
        forced_value=False,
        label="adaptive_tep_equilibrium",
    )

    if forced_labels:
        startup_warnings.append(
            "Non-adaptive run: forced adaptive-only features off "
            + f"({', '.join(forced_labels)})."
        )

    return False, startup_warnings


_RUN_ID_PATTERN = re.compile(r"(?!\.+$)[A-Za-z0-9._-]{1,128}")


def _derive_run_paths(
    *, args: argparse.Namespace, config_state: Any, repo_root: Path
) -> tuple[str, Path, Path]:
    # UTC, like every timestamp in the run metadata (ORC-24); the Z says so.
    run_id = str(args.run_id).strip() or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    )
    if not str(args.adaptive_online_state_path).strip():
        args.adaptive_online_state_path = str(
            repo_root / "evals" / "artifacts" / "policies" / f"{run_id}.json"
        )
        if config_state.value_sources.get("adaptive_online_state_path") == "default":
            config_state.value_sources["adaptive_online_state_path"] = "derived_from_run_id"
    if not str(args.adaptive_online_events_path).strip():
        args.adaptive_online_events_path = str(
            repo_root / "evals" / "results" / f"{run_id}.adaptive_policy.jsonl"
        )
        if config_state.value_sources.get("adaptive_online_events_path") == "default":
            config_state.value_sources["adaptive_online_events_path"] = "derived_from_run_id"
    if not str(args.adaptive_budget_online_state_path).strip():
        args.adaptive_budget_online_state_path = str(
            repo_root / "evals" / "artifacts" / "policies" / f"{run_id}.adaptive_budget_policy.json"
        )
        if config_state.value_sources.get("adaptive_budget_online_state_path") == "default":
            config_state.value_sources["adaptive_budget_online_state_path"] = "derived_from_run_id"
    if not str(args.adaptive_budget_online_events_path).strip():
        args.adaptive_budget_online_events_path = str(
            repo_root / "evals" / "results" / f"{run_id}.adaptive_budget_policy.jsonl"
        )
        if config_state.value_sources.get("adaptive_budget_online_events_path") == "default":
            config_state.value_sources["adaptive_budget_online_events_path"] = "derived_from_run_id"
    if not str(getattr(args, "adaptive_tep_state_path", "")).strip():
        args.adaptive_tep_state_path = str(
            repo_root / "evals" / "artifacts" / "policies" / f"{run_id}.adaptive_tep_state.json"
        )
        if config_state.value_sources.get("adaptive_tep_state_path") == "default":
            config_state.value_sources["adaptive_tep_state_path"] = "derived_from_run_id"

    artifacts_root = repo_root / "evals" / "artifacts" / run_id
    results_root = repo_root / "evals" / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    results_path = results_root / f"{run_id}.jsonl"
    return run_id, artifacts_root, results_path


def _find_similar_existing_run_ids(*, run_id: str, results_root: Path) -> list[str]:
    candidates: set[str] = set()
    try:
        for path in results_root.glob("*.jsonl"):
            name = path.name
            if name.endswith(".adaptive_policy.jsonl") or name.endswith(
                ".adaptive_budget_policy.jsonl"
            ):
                continue
            candidate_run_id = path.stem
            if candidate_run_id and candidate_run_id != run_id:
                candidates.add(candidate_run_id)
    except OSError:
        return []
    sorted_candidates = sorted(candidates)
    prefix_matches = [
        candidate
        for candidate in sorted_candidates
        if candidate.startswith(run_id) or run_id.startswith(candidate)
    ]
    if prefix_matches:
        return prefix_matches
    return difflib.get_close_matches(run_id, sorted_candidates, n=3, cutoff=0.9)


def _append_similar_run_id_warning(
    *,
    run_id: str,
    results_path: Path,
    startup_warnings: list[str],
) -> None:
    if results_path.exists():
        return
    similar_run_ids = _find_similar_existing_run_ids(
        run_id=run_id, results_root=results_path.parent
    )
    if not similar_run_ids:
        return
    if len(similar_run_ids) == 1:
        startup_warnings.append(
            "No existing results file found for "
            f"--run-id {run_id!r}. Found similar existing run-id {similar_run_ids[0]!r}; "
            "if you intended to resume it, use the exact run-id."
        )
        return
    preview = ", ".join(repr(candidate) for candidate in similar_run_ids[:3])
    if len(similar_run_ids) > 3:
        preview += ", ..."
    startup_warnings.append(
        "No existing results file found for "
        f"--run-id {run_id!r}. Similar existing run-ids: {preview}. "
        "If you intended to resume one of them, use the exact run-id."
    )


def prepare_main_setup(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv_tokens: list[str],
    seams: EvalSeams | None = None,
) -> MainSetupState:
    seams = seams if seams is not None else default_eval_seams()
    try:
        config_state = _apply_toml_configs_to_args(
            parser=parser,
            args=args,
            argv_tokens=argv_tokens,
        )
    except ValueError as exc:
        _fail(f"Invalid --config: {exc}")

    try:
        _normalize_baseline_selection(args=args, config_state=config_state)
        _normalize_adaptive_controller_lambda_rlm_settings(args=args, config_state=config_state)
        if str(args.patcher) == "baseline":
            (
                args.baseline_repo_dir,
                args.baseline_python,
            ) = _resolve_external_baseline_runtime_requirements(
                baseline_name=str(getattr(args, "baseline_name", "") or ""),
                baseline_repo_dir=str(getattr(args, "baseline_repo_dir", "") or ""),
                baseline_python=str(getattr(args, "baseline_python", "") or ""),
            )
    except ValueError as exc:
        _fail(f"Invalid arguments: {exc}")

    if args.deterministic and args.seed is None:
        _fail("Invalid arguments: --seed is required when --deterministic is enabled.")
    _require_choice(value=args.record_prompt, allowed=RECORD_MODES, option="--record-prompt mode")
    _require_choice(
        value=args.record_evidence, allowed=RECORD_MODES, option="--record-evidence mode"
    )
    _require_choice(
        value=args.context_protocol, allowed=CONTEXT_PROTOCOLS, option="--context-protocol"
    )
    _require_int_ge(
        value=args.context_stage_a_max_requests,
        minimum=1,
        option="--context-stage-a-max-requests",
    )
    _require_int_ge(
        value=args.context_stage_a_max_chars,
        minimum=1,
        option="--context-stage-a-max-chars",
    )
    _require_choice(
        value=args.retry_prompt_mode, allowed=RETRY_PROMPT_MODES, option="--retry-prompt-mode"
    )
    _require_choice(
        value=args.retry_context_mode, allowed=RETRY_CONTEXT_MODES, option="--retry-context-mode"
    )
    if bool(args.prompt_retry_plan_then_patch) and str(args.patcher) != "openai":
        _fail(
            "Invalid arguments: --prompt-retry-plan-then-patch currently requires --patcher openai."
        )
    _require_int_ge(
        value=args.retry_history_max_chars, minimum=0, option="--retry-history-max-chars"
    )
    _require_int_ge(
        value=args.pytest_output_tail_lines, minimum=0, option="--pytest-output-tail-lines"
    )
    _require_int_ge(
        value=args.snapshot_artifact_max_string_chars,
        minimum=0,
        option="--snapshot-artifact-max-string-chars",
    )
    _require_int_ge(
        value=args.prompt_budget_total_chars, minimum=0, option="--prompt-budget-total-chars"
    )
    _require_int_ge(
        value=args.prompt_budget_retry_delta_chars,
        minimum=0,
        option="--prompt-budget-retry-delta-chars",
    )
    _require_int_ge(
        value=args.prompt_budget_stdout_chars, minimum=0, option="--prompt-budget-stdout-chars"
    )
    _require_int_ge(
        value=args.prompt_budget_snapshot_chars,
        minimum=0,
        option="--prompt-budget-snapshot-chars",
    )
    _require_int_ge(
        value=args.prompt_budget_evidence_chars,
        minimum=0,
        option="--prompt-budget-evidence-chars",
    )
    _require_int_ge(
        value=args.prompt_budget_retry_history_chars,
        minimum=0,
        option="--prompt-budget-retry-history-chars",
    )
    _require_choice(
        value=args.context_compaction_mode,
        allowed=CONTEXT_COMPACTION_MODES,
        option="--context-compaction-mode",
    )
    _require_choice(
        value=_namespace_arg(args, "prompt_order_mode", "current"),
        allowed=PROMPT_ORDER_MODES,
        option="--prompt-order-mode",
    )
    _require_choice(
        value=_namespace_arg(args, "adaptive_reveal_mode", "full"),
        allowed=ADAPTIVE_REVEAL_MODES,
        option="--adaptive-reveal-mode",
    )
    _require_choice(
        value=_namespace_arg(args, "snapshot_frame_pruning_mode", "off"),
        allowed=SNAPSHOT_FRAME_PRUNING_MODES,
        option="--snapshot-frame-pruning-mode",
    )
    _validate_context_compaction_scope(str(args.context_compaction_scope))
    _require_int_ge(
        value=args.context_compaction_max_input_chars,
        minimum=0,
        option="--context-compaction-max-input-chars",
    )
    _require_int_ge(
        value=args.context_compaction_target_chars,
        minimum=1,
        option="--context-compaction-target-chars",
    )
    _require_int_ge(
        value=args.context_compaction_max_calls,
        minimum=0,
        option="--context-compaction-max-calls",
    )
    _require_int_ge(
        value=args.openai_context_window_tokens, minimum=0, option="--openai-context-window-tokens"
    )
    _require_int_ge(
        value=args.openai_transport_max_retries,
        minimum=0,
        option="--openai-transport-max-retries",
    )
    _require_float_ge(
        value=args.openai_transport_retry_backoff_base_sec,
        minimum=0.0,
        option="--openai-transport-retry-backoff-base-sec",
    )
    _require_float_ge(
        value=args.openai_transport_retry_backoff_max_sec,
        minimum=0.0,
        option="--openai-transport-retry-backoff-max-sec",
    )
    if float(args.openai_transport_retry_backoff_max_sec) < float(
        args.openai_transport_retry_backoff_base_sec
    ):
        _fail(
            "Invalid --openai-transport-retry-backoff-max-sec: must be >= "
            "--openai-transport-retry-backoff-base-sec"
        )
    _require_float_ge(
        value=args.openai_transport_recovery_max_wait_sec,
        minimum=0.0,
        option="--openai-transport-recovery-max-wait-sec",
    )
    _require_float_gt(
        value=args.openai_transport_recovery_poll_sec,
        minimum=0.0,
        option="--openai-transport-recovery-poll-sec",
    )
    _require_choice(
        value=args.openai_response_mode_autoswitch,
        allowed=OPENAI_RESPONSE_MODE_AUTOSWITCH_MODES,
        option="--openai-response-mode-autoswitch",
    )
    _require_choice(
        value=args.adaptive_gate_policy,
        allowed=ADAPTIVE_GATE_POLICIES,
        option="--adaptive-gate-policy",
    )
    _require_int_ge(
        value=args.adaptive_max_helper_calls, minimum=0, option="--adaptive-max-helper-calls"
    )
    _require_int_ge(
        value=args.adaptive_budget_topup_max_calls,
        minimum=0,
        option="--adaptive-budget-topup-max-calls",
    )
    _require_int_ge(
        value=args.adaptive_stagnation_window, minimum=1, option="--adaptive-stagnation-window"
    )
    _require_unit_interval(
        value=args.adaptive_saturation_threshold, option="--adaptive-saturation-threshold"
    )
    _require_int_ge(
        value=args.adaptive_no_progress_token_cap,
        minimum=0,
        option="--adaptive-no-progress-token-cap",
    )
    _require_int_ge(
        value=args.patch_verifier_min_score, minimum=0, option="--patch-verifier-min-score"
    )
    _require_choice(
        value=args.adaptive_helper_budget_profile,
        allowed=ADAPTIVE_HELPER_BUDGET_PROFILES,
        option="--adaptive-helper-budget-profile",
    )
    _require_choice(
        value=args.adaptive_openai_helper_response_mode,
        allowed=ADAPTIVE_OPENAI_HELPER_RESPONSE_MODES,
        option="--adaptive-openai-helper-response-mode",
    )
    _require_unit_interval(
        value=args.adaptive_informativeness_pregate_threshold,
        option="--adaptive-informativeness-pregate-threshold",
    )
    _require_choice(
        value=args.adaptive_online_policy_mode,
        allowed=ADAPTIVE_ONLINE_POLICY_MODES,
        option="--adaptive-online-policy-mode",
    )
    _require_choice(
        value=args.adaptive_online_policy_algo,
        allowed=ADAPTIVE_ONLINE_POLICY_ALGOS,
        option="--adaptive-online-policy-algo",
    )
    _require_float_ge(
        value=args.adaptive_online_alpha, minimum=0.0, option="--adaptive-online-alpha"
    )
    _require_unit_interval(value=args.adaptive_online_epsilon, option="--adaptive-online-epsilon")
    _require_float_gt(value=args.adaptive_online_l2, minimum=0.0, option="--adaptive-online-l2")
    _require_int_ge(
        value=args.adaptive_online_warmup_decisions,
        minimum=0,
        option="--adaptive-online-warmup-decisions",
    )
    _require_int_ge(
        value=args.adaptive_online_save_every, minimum=1, option="--adaptive-online-save-every"
    )
    _require_unit_interval(
        value=args.adaptive_online_decay,
        option="--adaptive-online-decay",
        left_closed=False,
    )
    _require_choice(
        value=args.adaptive_online_feature_scaling_profile,
        allowed=ADAPTIVE_FEATURE_SCALING_PROFILES,
        option="--adaptive-online-feature-scaling-profile",
    )
    try:
        _parse_adaptive_online_action_score_biases(args.adaptive_online_action_score_biases)
    except (TypeError, ValueError):
        _fail(
            "Invalid --adaptive-online-action-score-biases: expected comma-separated "
            "action=bias pairs using known helper actions, each bias within "
            f"+/-{ONLINE_ACTION_SCORE_BIAS_LIMIT:g} of the success reward "
            f"(for example: {DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES})."
        )
    _require_choice(
        value=args.adaptive_budget_policy_mode,
        allowed=ADAPTIVE_BUDGET_POLICY_MODES,
        option="--adaptive-budget-policy-mode",
    )
    _require_choice(
        value=args.adaptive_budget_policy_algo,
        allowed=ADAPTIVE_BUDGET_POLICY_ALGOS,
        option="--adaptive-budget-policy-algo",
    )
    _require_choice(
        value=args.adaptive_budget_policy_architecture,
        allowed=BUDGET_POLICY_ARCHITECTURES,
        option="--adaptive-budget-policy-architecture",
    )
    if str(args.adaptive_budget_policy_architecture) == "bwk_v1":
        _require_int_ge(
            value=_namespace_arg(args, "adaptive_budget_bwk_run_token_budget", 0),
            minimum=1,
            option="--adaptive-budget-bwk-run-token-budget",
        )
    try:
        adaptive_budget_policy_tiers = _parse_adaptive_budget_policy_tiers(
            args.adaptive_budget_policy_tiers
        )
    except (TypeError, ValueError):
        _fail(
            "Invalid --adaptive-budget-policy-tiers: expected strictly increasing "
            "comma-separated positive floats including 1.0 "
            "(for example: 1.0,1.1,1.2,1.35,1.5)."
        )
    _require_int_ge(
        value=args.adaptive_budget_policy_max_ups_per_case,
        minimum=0,
        option="--adaptive-budget-policy-max-ups-per-case",
    )
    _require_float_ge(
        value=args.adaptive_budget_online_alpha,
        minimum=0.0,
        option="--adaptive-budget-online-alpha",
    )
    _require_unit_interval(
        value=args.adaptive_budget_online_epsilon,
        option="--adaptive-budget-online-epsilon",
    )
    _require_float_gt(
        value=args.adaptive_budget_online_l2, minimum=0.0, option="--adaptive-budget-online-l2"
    )
    _require_int_ge(
        value=args.adaptive_budget_online_warmup_decisions,
        minimum=0,
        option="--adaptive-budget-online-warmup-decisions",
    )
    _require_int_ge(
        value=args.adaptive_budget_online_save_every,
        minimum=1,
        option="--adaptive-budget-online-save-every",
    )
    _require_unit_interval(
        value=args.adaptive_budget_online_decay,
        option="--adaptive-budget-online-decay",
        left_closed=False,
    )
    _require_choice(
        value=args.adaptive_budget_feature_scaling_profile,
        allowed=ADAPTIVE_FEATURE_SCALING_PROFILES,
        option="--adaptive-budget-feature-scaling-profile",
    )
    _require_float_ge(
        value=args.adaptive_budget_reward_rho_budget_exceeded,
        minimum=0.0,
        option="--adaptive-budget-reward-rho-budget-exceeded",
    )
    _require_float_ge(
        value=args.adaptive_reward_lambda_tokens,
        minimum=0.0,
        option="--adaptive-reward-lambda-tokens",
    )
    _require_float_ge(
        value=args.adaptive_reward_mu_latency, minimum=0.0, option="--adaptive-reward-mu-latency"
    )
    _require_float_gt(
        value=args.adaptive_reward_token_scale, minimum=0.0, option="--adaptive-reward-token-scale"
    )
    _require_float_gt(
        value=args.adaptive_reward_latency_scale,
        minimum=0.0,
        option="--adaptive-reward-latency-scale",
    )
    _require_float_ge(
        value=args.adaptive_reward_eta_progress,
        minimum=0.0,
        option="--adaptive-reward-eta-progress",
    )
    _require_float_ge(
        value=args.adaptive_reward_kappa_stagnation,
        minimum=0.0,
        option="--adaptive-reward-kappa-stagnation",
    )
    _require_int_ge(
        value=args.syntax_preflight_max_cycles, minimum=1, option="--syntax-preflight-max-cycles"
    )
    _require_choice(
        value=args.progress_verbosity, allowed=PROGRESS_VERBOSITIES, option="--progress-verbosity"
    )
    _require_float_ge(
        value=args.progress_heartbeat_sec, minimum=0.0, option="--progress-heartbeat-sec"
    )
    _require_int_ge(value=args.jobs, minimum=1, option="--jobs")
    _require_int_ge(
        value=args.transport_fail_fast_streak,
        minimum=0,
        option="--transport-fail-fast-streak",
    )

    args.adaptive_budget_policy_tiers = ",".join(
        f"{value:g}" for value in adaptive_budget_policy_tiers
    )
    try:
        execution_config = build_execution_config_from_args(args)
    except ValueError as exc:
        _fail(f"Invalid arguments: {exc}")
    config_state = replace(
        config_state,
        runtime_resolution=resolve_execution_runtime_metadata(args),
    )
    for warning in execution_config.warnings:
        print(f"Warning: {warning}")

    if args.deterministic and args.patcher == "openai" and float(args.openai_temperature) != 0.0:
        print("Deterministic mode enabled: forcing --openai-temperature=0.0")
        args.openai_temperature = 0.0
    if (
        str(args.openai_response_mode_autoswitch) == "retry_to_diff"
        and int(args.syntax_preflight_max_cycles) < 2
    ):
        print(
            "Warning: --openai-response-mode-autoswitch=retry_to_diff is ineffective when "
            "--syntax-preflight-max-cycles < 2."
        )

    cases = _load_filtered_cases(args, seams)
    try:
        conditions = _resolve_conditions_for_run(args=args, config_state=config_state)
    except ValueError as exc:
        _fail(f"Invalid --conditions value: {exc}")

    has_adaptive_condition, startup_warnings = _force_adaptive_policies_off_when_unused(
        args=args,
        config_state=config_state,
        conditions=conditions,
    )
    _apply_decision_model_checks(
        args=args,
        config_state=config_state,
        conditions=conditions,
        startup_warnings=startup_warnings,
    )
    explicit_run_id = bool(str(args.run_id).strip())
    if explicit_run_id and not _RUN_ID_PATTERN.fullmatch(str(args.run_id).strip()):
        # ORC-21: the run id becomes part of results, artifact and policy-state paths.
        _fail(
            f"Invalid --run-id {str(args.run_id).strip()!r}: use 1-128 characters from "
            "A-Z, a-z, 0-9, '.', '_' and '-'."
        )
    run_id, artifacts_root, results_path = _derive_run_paths(
        args=args, config_state=config_state, repo_root=seams.repo_root
    )
    if explicit_run_id:
        _append_similar_run_id_warning(
            run_id=run_id,
            results_path=results_path,
            startup_warnings=startup_warnings,
        )
    return MainSetupState(
        config_state=config_state,
        execution_config=execution_config,
        cases=cases,
        conditions=conditions,
        has_adaptive_condition=has_adaptive_condition,
        startup_warnings=startup_warnings,
        run_id=run_id,
        artifacts_root=artifacts_root,
        results_path=results_path,
    )
