"""Multi-condition repair benchmark harness for llmdebug experiments.

Orchestrates case discovery, condition execution across native llmdebug
conditions and external baselines, patch proposal via pluggable patchers,
pytest verification, and JSONL result recording. Supports local and
Docker-based (testcontainers) execution, TOML config files, resume from
partial runs, adaptive gating policies, online contextual-bandit policy
controllers, and budget-aware resource management.
"""
# ruff: noqa: F401

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if __package__ in {None, ""}:
    _BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
    _BOOTSTRAP_SRC_ROOT = _BOOTSTRAP_REPO_ROOT / "src"
    for _candidate in (str(_BOOTSTRAP_REPO_ROOT), str(_BOOTSTRAP_SRC_ROOT)):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    from evals import _json as json
else:
    from . import _json as json

try:
    import tomllib as _toml_loader
except ModuleNotFoundError:
    _toml_loader = None

from baselines.openai_compat import COMPAT_PROFILE_CHOICES
from baselines.subprocess_monitor import run_monitored_subprocess
from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_CONTEXT,
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ACTIONS,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_ARCHITECTURES,
    BUDGET_POLICY_FAMILIES,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
    BudgetPolicyDecision,
)
from evals.adaptive_online_policy import (
    ACTION_COMPACT_THEN_PATCH,
    ACTION_INVOKE_HELPER,
    ACTION_RETRY_HELPER_STRICT_JSON,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_LITE,
    ONLINE_POLICY_ACTIONS,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
    PolicyDecision,
)
from evals.eval_patch_constants import PATCH_APPLY_TIMEOUT_SEC, TRUNCATED_RESPONSE_ERROR
from evals.eval_prompts import (
    _apply_prompt_budget,
    _build_retry_history_section,
    _build_retry_summary,
    _format_snapshot_index_section,
    _is_generic_snapshot_justification,
    _render_retry_history_turn,
    _render_retry_packet,
    _should_invoke_context_compaction,
    render_stage_a_request_prompt,
)
from evals.policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    normalize_feature_scaling_profile,
    scale_feature_mapping,
)
from evals.progress_output import RunProgressSummary, build_progress_palette

if TYPE_CHECKING:
    from evals.patchers.payload_types import ContextRequestPayload
else:
    ContextRequestPayload = dict[str, Any]  # type: ignore[misc,assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if __package__ in {None, ""} and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evals.decision_model.config import (  # noqa: E402
    add_decision_model_arguments,
    add_policy_input_arguments,
)
from evals.eval_config import (  # noqa: E402
    _apply_toml_configs_to_args,
    _default_openai_api_key,
    _parse_adaptive_budget_policy_tiers,
    _parse_adaptive_online_action_score_biases,
    _write_resolved_run_config_lockfile,
    add_execution_arguments,
    build_execution_config_from_args,
    get_active_execution_config,
    resolve_openai_api_key_default,
    set_active_execution_config,
)
from evals.eval_containers import (  # noqa: E402
    _docker_client_from_env as _docker_client_from_env,
)
from evals.eval_containers import (  # noqa: E402  # noqa: E402
    _EvalContainerPool,
    check_testcontainers_runtime_available,
    cleanup_stale_eval_containers,
)
from evals.eval_policy_budget import (  # noqa: E402
    _adaptive_budget_behavior_fields,
    _finalize_adaptive_budget_policy_attempt,
    _finalize_adaptive_policy_attempt,
    _sync_adaptive_budget_policy_metadata,
    _sync_finalize_progress_kwargs_from_attempt_metadata,
)
from evals.eval_policy_features import (  # noqa: E402
    _adaptive_budget_control_changes,
    _adaptive_budget_control_profile_to_dict,
    _adaptive_budget_controls_changed,
    _adaptive_budget_family_availability,
    _adaptive_budget_feature_vector,
    _adaptive_budget_level_multiplier,
    _adaptive_budget_level_step_delta,
    _adaptive_budget_local_reward_components,
    _adaptive_budget_profile_for_family_step,
    _adaptive_budget_recorded_multiplier,
    _adaptive_budget_union_change_flags,
    _adaptive_online_feature_vector,
    _adaptive_online_local_reward_components,
    _adaptive_policy_local_accounting,
    _apply_adaptive_budget_profile_to_patcher,
    _build_adaptive_budget_control_profile,
    _build_adaptive_budget_family_profile,
)
from evals.eval_policy_online import (  # noqa: E402
    _adaptive_online_behavior_fields,
    _adaptive_online_delayed_rescue_bonus,
    _adaptive_online_reward_breakdown,
    _adaptive_online_should_defer_credit,
    _adaptive_online_update_suppression_reason,
    _finalize_adaptive_online_attempt,
    _record_adaptive_online_policy_outcome,
    _resolve_pending_adaptive_online_credit,
    _sync_adaptive_online_metadata,
    _sync_record_metadata_copies,
)
from evals.eval_recording import (  # noqa: E402
    DEFAULT_REPO_OPENAI_COMPAT_PROFILE,
    _baseline_openai_compat_settings_from_args,
    _build_base_record,
    _build_model_metadata,
    _build_patcher_config,
    _build_run_metadata,
    _default_run_metadata,
    _empty_attempt_metadata,
    _materialize_attempt_metadata,
    _native_openai_compat_settings_from_args,
    _resolve_llmdebug_snapshot_schema_version,
    _RunResultsLock,
    _SynchronizedJsonlWriter,
)
from evals.eval_types import (  # noqa: E402
    _ASSERTION_VALUE_PATTERN,
    _DIAGNOSTIC_MESSAGE_PATTERNS,
    _EXCEPTION_TYPE_SPECIFICITY,
    _EXCEPTION_TYPE_SPECIFICITY_DEFAULT,
    _PROGRESS_LEVEL_ORDER,
    _PROGRESS_PRINT_LOCK,
    _PYTEST_SUMMARY_COUNT_RE,
    _STDOUT_PYTEST_ERROR_RE,
    _STDOUT_TRACEBACK_ERROR_RE,
    ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS,
    ADAPTIVE_BUDGET_POLICY_ALGOS,
    ADAPTIVE_BUDGET_POLICY_MODES,
    ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
    ADAPTIVE_BUDGET_REWARD_SCOPE,
    ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS,
    ADAPTIVE_FEATURE_SCALING_PROFILES,
    ADAPTIVE_GATE_POLICIES,
    ADAPTIVE_HELPER_BUDGET_PROFILES,
    ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY,
    ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD,
    ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
    ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED,
    ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT,
    ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL,
    ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY,
    ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED,
    ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED,
    ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_CONTEXT_MULTIPLIER,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_HORIZON_TWO_FACTOR,
    ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE,
    ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    ADAPTIVE_ONLINE_POLICY_ALGOS,
    ADAPTIVE_ONLINE_POLICY_MODES,
    ADAPTIVE_ONLINE_REWARD_SCOPE,
    ADAPTIVE_OPENAI_HELPER_RESPONSE_MODES,
    ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
    ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
    ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
    ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
    ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
    ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
    ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
    ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
    CONTEXT_COMPACTION_MODES,
    CONTEXT_COMPACTION_SCOPE_TOKENS,
    CONTEXT_PROTOCOLS,
    CONTEXT_RECOVERY_FAILURE_KINDS,
    CONTEXT_RECOVERY_PROMPT_BUDGET,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
    DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
    DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
    DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
    DEFAULT_ADAPTIVE_ONLINE_ALPHA,
    DEFAULT_ADAPTIVE_ONLINE_DECAY,
    DEFAULT_ADAPTIVE_ONLINE_EPSILON,
    DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS,
    DEFAULT_ADAPTIVE_POLICY_L2,
    DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
    DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    DEFAULT_EXECUTOR,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_MAX_WAIT_SEC,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_POLL_SEC,
    DEFAULT_PROGRESS_HEARTBEAT_SEC,
    DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
    DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
    DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
    DEFAULT_TRANSPORT_FAIL_FAST_STREAK,
    EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    EVAL_CONTAINER_LABEL_KEY,
    EVAL_CONTAINER_LABEL_VALUE,
    EVAL_RECORD_SCHEMA_VERSION,
    EVAL_RECORD_TYPE_ATTEMPT,
    EVAL_RECORD_TYPE_INFRA_EVENT,
    HELPER_CONTEXT_MATERIALIZING_ACTIONS,
    HELPER_STAGE_A_ACTIONS,
    OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
    OPENAI_RESPONSE_MODE_AUTOSWITCH_MODES,
    PATCH_VERIFIER_MIN_SCORE,
    PATCH_VERIFIER_RECOVERY_RESULT_NO_DIFF,
    PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE,
    PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER,
    PATCH_VERIFIER_RECOVERY_RESULT_REQUEST_FAILED,
    PATCH_VERIFIER_RECOVERY_RESULT_SKIPPED_DEADLINE_EXHAUSTED,
    PATCH_VERIFIER_RECOVERY_RESULT_STILL_REJECTED,
    PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
    PATCH_VERIFIER_SCOPE_ALL,
    PATCH_VERIFIER_SCOPE_DISABLED,
    PATCHER_CONTEXT_LIMIT_ERROR_MARKERS,
    PERSISTED_PYTEST_OUTPUT_HARD_CHAR_CAP,
    PROGRESS_VERBOSITIES,
    PYTEST_TIMEOUT_RETURNCODE,
    RECORD_MODES,
    RETRY_CONTEXT_MODES,
    RETRY_PROMPT_MODES,
    RETRYABLE_HTTP_STATUS_CODES,
    RUN_EVAL_LOCKFILE_REDACTED,
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
    VALID_SNAPSHOT_DEPENDENCIES,
    AdaptiveBudgetControlProfile,
    AdaptiveOnlineRewardBreakdown,
    AdaptivePolicyLocalAccounting,
    AdaptiveProgressState,
    CaseSpec,
    Condition,
    ConfigResolutionState,
    EvalUnitInputs,
    ExecutionConfig,
    PendingAdaptiveOnlineCredit,
    PreGateResult,
    RunConditionInputs,
    RunResult,
    StageAContextData,
    UnitExecutionOutcome,
    _ActiveRunSettings,
    _default_helper_action,
    _empty_stage_a_context,
    _filter_context_request_payload_for_action,
    _helper_action_availability,
    _TrailingResultsTailIssue,
    baseline_condition_for_name,
    is_baseline_condition,
)
from evals.oracle_integrity import (  # noqa: E402
    clear_junit_reports,
    consume_junit_counts,
    junit_report_name,
)
from llmdebug._hashing import xxh64_hexdigest_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for the eval harness.

    Parses arguments, discovers cases, resumes partial runs, and
    dispatches (case, condition) units to a thread-pool executor.

    Returns:
        Exit code: 0 on success, 1 on partial failure, 2 on fatal error.
    """
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run the llmdebug multi-condition repair benchmark harness.",
        # ORC-06: config-file values must not override abbreviated CLI flags.
        allow_abbrev=False,
    )
    parser.add_argument("--cases-dir", default="evals/cases", help="Cases directory.")
    parser.add_argument("--case", action="append", default=[], help="Case id(s) to run.")
    parser.add_argument(
        "--case-list-file",
        action="append",
        default=[],
        help=(
            "Path to newline-delimited case ids to include (repeatable). "
            "Blank lines and '#' comments are ignored."
        ),
    )
    parser.add_argument(
        "--category", action="append", default=[], help="Category filter (repeatable)."
    )
    parser.add_argument(
        "--difficulty", action="append", default=[], help="Difficulty filter (repeatable)."
    )
    parser.add_argument(
        "--snapshot-dependency",
        action="append",
        default=[],
        help="Snapshot dependency filter (repeatable): required|helpful|none.",
    )
    parser.add_argument(
        "--bug-source", action="append", default=[], help="Bug source filter (repeatable)."
    )
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Max number of cases to run (0 = all)."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Number of case/condition units to execute concurrently.",
    )
    parser.add_argument(
        "--progress-verbosity",
        default="basic",
        help="Progress verbosity level for runtime status output (basic|detailed|trace).",
    )
    parser.add_argument(
        "--progress-heartbeat-sec",
        type=float,
        default=DEFAULT_PROGRESS_HEARTBEAT_SEC,
        help="Heartbeat interval in seconds for in-flight unit progress (0 disables).",
    )
    parser.add_argument(
        "--transport-fail-fast-streak",
        type=int,
        default=DEFAULT_TRANSPORT_FAIL_FAST_STREAK,
        help=(
            "Abort early after this many consecutive terminal transport request failures "
            "(0 disables; default: 5)."
        ),
    )
    parser.add_argument(
        "--case-fail-fast-on-request-budget-exceeded",
        dest="case_fail_fast_on_request_budget_exceeded",
        action="store_true",
        default=True,
        help=(
            "Stop remaining attempts for the current case/condition when an attempt reports "
            "`request_budget_exceeded` (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-case-fail-fast-on-request-budget-exceeded",
        dest="case_fail_fast_on_request_budget_exceeded",
        action="store_false",
        help=(
            "Disable per-case fail-fast on `request_budget_exceeded` and continue retry attempts."
        ),
    )
    parser.add_argument(
        "--conditions",
        default="base_repair",
        help=(
            "Comma-separated list: "
            "base_repair,cot_repair,"
            "layered_adaptive_gated,layered_adaptive_llm,"
            "ldb,intervenor,tracecoder,lambda_rlm,rlm"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier. If the matching results file exists, unfinished units are resumed.",
    )
    parser.add_argument(
        "--allow-settings-drift",
        action="store_true",
        help=(
            "Resume even when the run settings fingerprint differs from the existing results; "
            "new records then carry their own run metadata (ORC-04)."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Path to TOML config file with run settings. "
            "Top-level keys or [run_eval] keys map to CLI destinations."
        ),
    )
    parser.add_argument("--k", type=int, default=2, help="Max patch attempts per condition.")
    parser.add_argument(
        "--patcher",
        default="command",
        choices=["command", "heuristic", "null", "openai", "baseline"],
        help="Patch strategy.",
    )
    parser.add_argument(
        "--baseline",
        default="",
        choices=_BASELINE_SHORTHAND_CHOICES,
        help=(
            "One-switch baseline selector. External baselines resolve to "
            "--patcher baseline --baseline-name NAME and auto-fill the standard "
            "baselines/repos/<repo> and baselines/envs/<name>/bin/python paths when present. "
            "Native shorthand 'cot_repair' resolves onto the native OpenAI path instead."
        ),
    )
    parser.add_argument(
        "--patch-cmd",
        default="",
        help="Command to run for command patcher (overrides LLMDEBUG_EVAL_PATCH_CMD).",
    )
    parser.add_argument(
        "--patch-cmd-timeout-sec",
        type=float,
        default=30.0,
        help="Timeout in seconds for command patcher subprocess execution.",
    )
    parser.add_argument(
        "--patch-cmd-shell",
        action="store_true",
        help="Run command patcher with shell=True (default: disabled).",
    )
    parser.add_argument(
        "--baseline-name",
        default="",
        help=(
            "External baseline adapter name (for --patcher baseline). "
            "E.g. ldb, intervenor, tracecoder, lambda_rlm, rlm."
        ),
    )
    parser.add_argument(
        "--baseline-repo-dir",
        default="",
        help="Path to the cloned external baseline repository (for --patcher baseline).",
    )
    parser.add_argument(
        "--baseline-python",
        default="",
        help=(
            "Optional Python executable for the selected baseline runtime env. "
            "If omitted, run_eval requires baselines/envs/<baseline>/bin/python."
        ),
    )
    parser.add_argument(
        "--baseline-process-timeout-sec",
        type=float,
        default=None,
        help=(
            "Optional timeout for one baseline-owned subprocess attempt. Defaults to "
            "--openai-timeout-sec when unset."
        ),
    )
    parser.add_argument(
        "--baseline-max-iterations",
        type=int,
        default=None,
        help=(
            "Max internal iterations per baseline invocation. Controls the adapter's "
            "debug/repair loop count independently of --k (which sets the harness-level "
            "retry count). Defaults to 2 when omitted."
        ),
    )
    parser.add_argument(
        "--baseline-openai-compat-profile",
        default=DEFAULT_REPO_OPENAI_COMPAT_PROFILE,
        choices=["auto", "qwen_openai_raw", "qwen_dashscope", "lmstudio_model_yaml", "off"],
        help=(
            "Shared OpenAI-compatible request-shaping profile for external baselines. "
            "Use qwen_openai_raw for vLLM/SGLang-style Qwen endpoints, "
            "qwen_dashscope for DashScope-compatible Qwen endpoints, "
            "lmstudio_model_yaml for LM Studio / llama.cpp model-template setups, "
            "or off to disable request shaping."
        ),
    )
    parser.add_argument(
        "--baseline-openai-allow-reasoning-fallback",
        dest="baseline_openai_allow_reasoning_fallback",
        action="store_true",
        default=True,
        help=(
            "Allow external baselines to consume reasoning-only response fields when "
            "message.content is empty (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-baseline-openai-allow-reasoning-fallback",
        dest="baseline_openai_allow_reasoning_fallback",
        action="store_false",
        help="Disable reasoning-only response fallback for external baselines.",
    )
    parser.add_argument(
        "--baseline-openai-enable-thinking",
        dest="baseline_openai_enable_thinking",
        action="store_true",
        default=True,
        help=(
            "Enable thinking/reasoning mode for external baselines "
            "(e.g. Qwen enable_thinking=True). Also auto-enables "
            "--baseline-openai-allow-reasoning-fallback (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-baseline-openai-enable-thinking",
        dest="baseline_openai_enable_thinking",
        action="store_false",
        help="Disable thinking/reasoning mode for external baselines.",
    )
    parser.add_argument(
        "--baseline-hybrid-guidance-mode",
        default="tuned_v1",
        choices=["tuned_v1", "fair_v1", "fixed_v1"],
        help=(
            "How llmdebug controller guidance is exposed to hybrid adaptive+baseline runs: "
            "'tuned_v1' preserves baseline-specific prompt integration, 'fair_v1' "
            "uses one shared canonical guidance block across all baselines, and "
            "'fixed_v1' uses a deterministic evidence block with full-snapshot "
            "injection intended for INTERVENOR fixed-evidence studies."
        ),
    )
    parser.add_argument(
        "--baseline-intervenor-teacher-max-tokens",
        type=int,
        default=1024,
        help="Teacher-stage max_tokens for the INTERVENOR adapter (default: 1024).",
    )
    parser.add_argument(
        "--baseline-intervenor-learner-max-tokens",
        type=int,
        default=2048,
        help="Learner-stage max_tokens for the INTERVENOR adapter (default: 2048).",
    )
    parser.add_argument(
        "--baseline-intervenor-teacher-guidance-max-chars",
        type=int,
        default=4000,
        help="Teacher-stage controller-guidance budget for INTERVENOR (default: 4000).",
    )
    parser.add_argument(
        "--baseline-intervenor-learner-guidance-max-chars",
        type=int,
        default=6000,
        help="Learner-stage controller-guidance budget for INTERVENOR (default: 6000).",
    )
    parser.add_argument(
        "--baseline-intervenor-append-output-contract",
        dest="baseline_intervenor_append_output_contract",
        action="store_true",
        default=True,
        help="Append the explicit single-file output suffix to INTERVENOR learner prompts (default: enabled).",
    )
    parser.add_argument(
        "--no-baseline-intervenor-append-output-contract",
        dest="baseline_intervenor_append_output_contract",
        action="store_false",
        help="Disable the explicit output-contract suffix on INTERVENOR learner prompts.",
    )
    parser.add_argument(
        "--baseline-tracecoder-controller-guidance-max-chars",
        type=int,
        default=2000,
        help="Controller-guidance budget appended to TraceCoder prompts (default: 2000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-max-no-improvement-streak",
        type=int,
        default=2,
        help="TraceCoder --max-no-improvement-streak value (default: 2).",
    )
    parser.add_argument(
        "--baseline-tracecoder-analysis-max-tokens",
        type=int,
        default=4096,
        help="TraceCoder max_tokens for instrumentation/analysis-stage calls (default: 4096).",
    )
    parser.add_argument(
        "--baseline-tracecoder-implementation-max-tokens",
        type=int,
        default=4096,
        help="TraceCoder max_tokens for direct/code-implementation calls (default: 4096).",
    )
    parser.add_argument(
        "--baseline-tracecoder-problem-summary-max-chars",
        type=int,
        default=4000,
        help="Max chars kept in the compact TraceCoder task/problem summary (default: 4000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-exact-test-context-max-chars",
        type=int,
        default=3000,
        help="Max chars of verbatim failing-test context injected into TraceCoder analysis prompts (default: 3000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-failure-feedback-max-chars",
        type=int,
        default=6000,
        help="Max chars retained in TraceCoder failure feedback summaries (default: 6000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-failure-feedback-tail-max-chars",
        type=int,
        default=2000,
        help="Max chars reserved for the bounded tail excerpt in TraceCoder failure feedback (default: 2000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-instrumentation-output-max-lines",
        type=int,
        default=80,
        help="Max lines retained from llmdebug instrumentation output before TraceCoder prompts (default: 80).",
    )
    parser.add_argument(
        "--baseline-tracecoder-instrumentation-output-max-chars",
        type=int,
        default=12000,
        help="Max chars retained from llmdebug instrumentation output before TraceCoder prompts (default: 12000).",
    )
    parser.add_argument(
        "--baseline-lambda-context-window-chars",
        type=int,
        default=DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
        help=(
            "Context window budget in chars passed to the lambda_rlm baseline "
            f"(default: {DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS})."
        ),
    )
    parser.add_argument(
        "--baseline-lambda-accuracy-target",
        type=float,
        default=0.80,
        help="Target accuracy passed to the lambda_rlm planner (default: 0.80).",
    )
    parser.add_argument(
        "--baseline-lambda-a-leaf",
        type=float,
        default=0.95,
        help="Leaf-call accuracy prior for the lambda_rlm planner (default: 0.95).",
    )
    parser.add_argument(
        "--baseline-lambda-a-compose",
        type=float,
        default=0.90,
        help="Composition accuracy prior for the lambda_rlm planner (default: 0.90).",
    )
    parser.add_argument(
        "--baseline-lambda-verbose",
        dest="baseline_lambda_verbose",
        action="store_true",
        default=False,
        help="Enable verbose native lambda_rlm plan logging inside the isolated env.",
    )
    parser.add_argument(
        "--baseline-lambda-temperature",
        type=float,
        default=None,
        help="Optional temperature override passed to lambda_rlm's OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-top-p",
        type=float,
        default=None,
        help="Optional top-p override passed to lambda_rlm's OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override passed to lambda_rlm's OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-stream",
        dest="baseline_lambda_stream",
        action="store_true",
        default=False,
        help="Enable streaming responses inside the lambda_rlm OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-timeout-sec",
        type=int,
        default=None,
        help=(
            "Optional per-attempt timeout cap inside the lambda_rlm isolated env. "
            "Defaults to the effective baseline process timeout when unset."
        ),
    )
    parser.add_argument(
        "--baseline-lambda-force-task-type",
        default="",
        help=(
            "Optional lambda-RLM task-type override forwarded into the isolated env "
            '(for example "qa") when small models misclassify repair prompts.'
        ),
    )
    parser.add_argument(
        "--baseline-lambda-prompt-contract",
        default="",
        help=(
            "Optional lambda-RLM prompt contract override: 'canonical' or "
            "'repair_fair_v1'. Defaults to canonical when unset."
        ),
    )
    parser.add_argument(
        "--baseline-lambda-controller-guidance-max-chars",
        type=int,
        default=2500,
        help="Controller-guidance budget appended to lambda-RLM prompts (default: 2500).",
    )
    parser.add_argument(
        "--baseline-lambda-failure-feedback-max-chars",
        type=int,
        default=3500,
        help="Failure-feedback budget injected into lambda-RLM prompts (default: 3500).",
    )
    parser.add_argument(
        "--baseline-lambda-failure-feedback-meta-max-chars",
        type=int,
        default=2000,
        help="Failure-feedback truncation budget stored in lambda-RLM attempt metadata (default: 2000).",
    )
    parser.add_argument(
        "--baseline-rlm-max-depth",
        type=int,
        default=DEFAULT_BASELINE_RLM_MAX_DEPTH,
        help=(
            "Maximum recursive depth for the standalone rlm baseline "
            f"(default: {DEFAULT_BASELINE_RLM_MAX_DEPTH})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-max-iterations",
        type=int,
        default=DEFAULT_BASELINE_RLM_MAX_ITERATIONS,
        help=(
            "Maximum native RLM iterations per outer attempt "
            f"(default: {DEFAULT_BASELINE_RLM_MAX_ITERATIONS})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-timeout-sec",
        type=int,
        default=None,
        help=(
            "Optional per-attempt timeout cap inside the standalone rlm env. "
            "Defaults to the effective baseline process timeout when unset."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-controller-guidance-max-chars",
        type=int,
        default=2500,
        help="Controller-guidance budget appended to RLM root prompts (default: 2500).",
    )
    parser.add_argument(
        "--baseline-rlm-prompt-contract",
        default="",
        help=(
            "Optional RLM root prompt contract override: 'canonical' or "
            "'repair_fair_v1'. Defaults to canonical when unset."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-failure-feedback-max-chars",
        type=int,
        default=3500,
        help="Failure-feedback budget injected into RLM root prompts (default: 3500).",
    )
    parser.add_argument(
        "--baseline-rlm-failure-feedback-meta-max-chars",
        type=int,
        default=2000,
        help="Failure-feedback truncation budget stored in RLM attempt metadata (default: 2000).",
    )
    parser.add_argument(
        "--baseline-rlm-max-errors",
        type=int,
        default=DEFAULT_BASELINE_RLM_MAX_ERRORS,
        help=(
            "Maximum consecutive REPL errors before aborting an rlm attempt "
            f"(default: {DEFAULT_BASELINE_RLM_MAX_ERRORS})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-max-tokens",
        type=int,
        default=None,
        help="Optional OpenAI-compatible max_tokens override passed to the standalone rlm backend.",
    )
    parser.add_argument(
        "--baseline-rlm-max-total-tokens",
        type=int,
        default=None,
        help="Optional total token ceiling enforced by the standalone rlm loop.",
    )
    parser.add_argument(
        "--baseline-rlm-compaction",
        dest="baseline_rlm_compaction",
        action="store_true",
        default=False,
        help="Enable native trajectory compaction inside the standalone rlm loop.",
    )
    parser.add_argument(
        "--baseline-rlm-compaction-threshold-pct",
        type=float,
        default=DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT,
        help=(
            "Compaction trigger threshold for the standalone rlm loop "
            f"(default: {DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-temperature",
        type=float,
        default=None,
        help="Optional temperature override passed to the standalone rlm OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-rlm-top-p",
        type=float,
        default=None,
        help="Optional top-p override passed to the standalone rlm OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-rlm-verbose",
        dest="baseline_rlm_verbose",
        action="store_true",
        default=False,
        help="Enable verbose native rlm logging inside the isolated env.",
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm",
        dest="adaptive_controller_lambda_rlm",
        action="store_true",
        default=False,
        help=(
            "Enable lambda-RLM as a controller-owned decomposition backend for adaptive "
            "hybrid baseline attempts (default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-adaptive-controller-lambda-rlm",
        dest="adaptive_controller_lambda_rlm",
        action="store_false",
        help="Disable the lambda-RLM controller backend.",
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-repo-dir",
        default="",
        help=(
            "Path to the cloned lambda-RLM repository. Defaults to "
            "baselines/repos/lambda-RLM when --adaptive-controller-lambda-rlm is enabled."
        ),
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-python",
        default="",
        help=(
            "Optional Python executable for the lambda-RLM env. Defaults to "
            "baselines/envs/lambda_rlm/bin/python when available, else the current interpreter."
        ),
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-context-window-chars",
        type=int,
        default=DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
        help=(
            "Context window budget in chars passed to lambda-RLM "
            f"(default: {DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS})."
        ),
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-timeout-sec",
        type=int,
        default=DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
        help=(
            "Per-attempt timeout in seconds for lambda-RLM controller calls "
            f"(default: {DEFAULT_LAMBDA_RLM_TIMEOUT_SEC})."
        ),
    )
    parser.add_argument(
        "--output-format", default="json_compact", choices=["json", "json_compact", "toon"]
    )
    parser.add_argument("--keep-workdirs", action="store_true", help="Keep copied case workdirs.")
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:12000/v1"),
    )
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument(
        "--openai-compat-profile",
        default=os.getenv("OPENAI_COMPAT_PROFILE", DEFAULT_REPO_OPENAI_COMPAT_PROFILE),
        choices=COMPAT_PROFILE_CHOICES,
        help=(
            "OpenAI-compatible request-shaping profile for the internal OpenAI patcher. "
            "Defaults to lmstudio_model_yaml for local llama.cpp / model-template "
            "servers. Use qwen_openai_raw for raw Qwen endpoints that need "
            "chat_template_kwargs.enable_thinking=false."
        ),
    )
    parser.add_argument(
        "--openai-allow-reasoning-fallback",
        dest="openai_allow_reasoning_fallback",
        action="store_true",
        default=True,
        help=(
            "Allow the internal OpenAI patcher to consume reasoning-only response fields "
            "when message.content is empty (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-openai-allow-reasoning-fallback",
        dest="openai_allow_reasoning_fallback",
        action="store_false",
        help="Disable reasoning-only response fallback for the internal OpenAI patcher.",
    )
    parser.add_argument(
        "--openai-enable-thinking",
        dest="openai_enable_thinking",
        action="store_true",
        default=True,
        help=(
            "Enable thinking/reasoning mode for the internal OpenAI patcher "
            "(e.g. Qwen enable_thinking=True). Also auto-enables "
            "--openai-allow-reasoning-fallback (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-openai-enable-thinking",
        dest="openai_enable_thinking",
        action="store_false",
        help="Disable thinking/reasoning mode for the internal OpenAI patcher.",
    )
    resolved_openai_api_key = resolve_openai_api_key_default()
    parser.set_defaults(
        _openai_api_key_default_source=resolved_openai_api_key.source,
        _openai_api_key_default_env_key=resolved_openai_api_key.env_key,
    )
    parser.add_argument("--openai-api-key", default=resolved_openai_api_key.value)
    parser.add_argument(
        "--openai-http2",
        dest="openai_http2",
        action="store_true",
        default=True,
        help="Enable HTTP/2 for OpenAI-compatible transport (default: enabled).",
    )
    parser.add_argument(
        "--no-openai-http2",
        dest="openai_http2",
        action="store_false",
        help="Disable HTTP/2 and force HTTP/1.1 for OpenAI-compatible transport.",
    )
    parser.add_argument("--openai-timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--openai-transport-max-retries",
        type=int,
        default=6,
        help=(
            "Max immediate retries for retryable OpenAI transport failures inside one request "
            "(default: 6)."
        ),
    )
    parser.add_argument(
        "--openai-transport-retry-backoff-base-sec",
        type=float,
        default=1.0,
        help="Base exponential backoff seconds for OpenAI transport retries (default: 1.0).",
    )
    parser.add_argument(
        "--openai-transport-retry-backoff-max-sec",
        type=float,
        default=8.0,
        help="Max backoff seconds for OpenAI transport retries (default: 8.0).",
    )
    parser.add_argument(
        "--openai-transport-recovery",
        dest="openai_transport_recovery",
        action="store_true",
        default=DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED,
        help=(
            "Wait for OpenAI-compatible backends to recover after terminal transport failures "
            "and replay the same high-level operation (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-openai-transport-recovery",
        dest="openai_transport_recovery",
        action="store_false",
        help="Disable runner-managed OpenAI transport recovery.",
    )
    parser.add_argument(
        "--openai-transport-recovery-max-wait-sec",
        type=float,
        default=DEFAULT_OPENAI_TRANSPORT_RECOVERY_MAX_WAIT_SEC,
        help="Max seconds to wait for OpenAI backend recovery after an outage (default: 60).",
    )
    parser.add_argument(
        "--openai-transport-recovery-poll-sec",
        type=float,
        default=DEFAULT_OPENAI_TRANSPORT_RECOVERY_POLL_SEC,
        help="Polling interval in seconds for OpenAI backend recovery probes (default: 2.0).",
    )
    parser.add_argument(
        "--openai-max-propose-sec",
        type=float,
        default=300.0,
        help=(
            "Maximum wall-clock budget per patch attempt for OpenAI propose calls "
            "(initial + recovery proposals), including per-call retries/fallback calls "
            "(0 disables cap)."
        ),
    )
    parser.add_argument("--openai-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--openai-plan-patch-max-tokens",
        type=int,
        default=768,
        help="Max tokens for native cot_repair planning calls before patch generation (default: 768).",
    )
    parser.add_argument(
        "--openai-context-window-tokens",
        type=int,
        default=0,
        help=(
            "Optional OpenAI-compatible context window for Stage-B prompt preflight capping "
            "(0 disables guard)."
        ),
    )
    parser.add_argument(
        "--openai-auto-context-window",
        action="store_true",
        default=False,
        help=(
            "Probe OpenAI-compatible model metadata for context window tokens before run start "
            "(manual --openai-context-window-tokens overrides)."
        ),
    )
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument(
        "--openai-response-mode",
        choices=["edits", "diff", "full_file"],
        default="edits",
    )
    parser.add_argument(
        "--openai-response-mode-autoswitch",
        choices=["off", "retry_to_diff"],
        default="off",
        help=(
            "OpenAI retry policy for internal syntax/sanitize retries: off|retry_to_diff "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--openai-force-patch-attempt",
        action="store_true",
        help="Force a minimal patch attempt when model output cannot be parsed into a diff.",
    )
    parser.add_argument(
        "--force-patch-attempt",
        dest="force_patch_attempt",
        action="store_true",
        default=False,
        help=(
            "Try a runner-side marker patch when the model's diff is missing or fails to apply "
            "(default: off). The attempt is still recorded as patch_apply_failed (ATT-07)."
        ),
    )
    parser.add_argument(
        "--no-force-patch-attempt",
        dest="force_patch_attempt",
        action="store_false",
        help="Disable runner-side forced patch attempts when no diff is proposed.",
    )
    parser.add_argument(
        "--no-files-context", action="store_true", help="Do not include full file text in prompt."
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=200_000,
        help="Max characters for project file context in prompt (default: 200000).",
    )
    parser.add_argument(
        "--context-protocol",
        choices=["monolithic", "staged"],
        default="monolithic",
        help="Context protocol for prompting: monolithic|staged (default: monolithic).",
    )
    parser.add_argument(
        "--context-stage-a-max-requests",
        type=int,
        default=6,
        help="Max Stage-A context requests when --context-protocol=staged (default: 6).",
    )
    parser.add_argument(
        "--context-stage-a-max-chars",
        type=int,
        default=120_000,
        help="Max characters returned from Stage-A context retrieval (default: 120000).",
    )
    parser.add_argument(
        "--retry-prompt-mode",
        choices=["full", "delta", "auto"],
        default="auto",
        help="Retry prompt protocol: full|delta|auto (default: auto).",
    )
    parser.add_argument(
        "--retry-context-mode",
        choices=["fresh", "history"],
        default="history",
        help="Retry context mode: fresh|history (default: history).",
    )
    parser.add_argument(
        "--retry-history-max-chars",
        type=int,
        default=48_000,
        help="Max characters of retry history transcript to include when --retry-context-mode=history (default: 48000).",
    )
    parser.add_argument(
        "--prompt-retry-reflection",
        dest="prompt_retry_reflection",
        action="store_true",
        default=False,
        help=(
            "Enable retry-only reflection-on-failure guidance in Stage-B prompts "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-retry-reflection",
        dest="prompt_retry_reflection",
        action="store_false",
        help="Disable retry-only reflection guidance in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--prompt-adaptive-stage-b-cot",
        dest="prompt_adaptive_stage_b_cot",
        action="store_true",
        default=False,
        help=(
            "Enable CoT-style Stage-B reasoning scaffolding for adaptive conditions "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-adaptive-stage-b-cot",
        dest="prompt_adaptive_stage_b_cot",
        action="store_false",
        help="Disable CoT-style Stage-B scaffolding in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--prompt-repair-method-scaffold",
        dest="prompt_repair_method_scaffold",
        action="store_true",
        default=False,
        help=(
            "Enable teacher-style modification-method scaffolding in Stage-B prompts "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-repair-method-scaffold",
        dest="prompt_repair_method_scaffold",
        action="store_false",
        help="Disable repair-method scaffolding in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--prompt-retry-plan-then-patch",
        dest="prompt_retry_plan_then_patch",
        action="store_true",
        default=False,
        help=(
            "Enable retry-only plan-then-patch orchestration before the final Stage-B patch call "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-retry-plan-then-patch",
        dest="prompt_retry_plan_then_patch",
        action="store_false",
        help="Disable retry-only plan-then-patch orchestration in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--pytest-output-tail-lines",
        type=int,
        default=DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
        help=(
            "Persist at most the latest N lines in pytest stdout/stderr artifact files "
            f"(default: {DEFAULT_PYTEST_OUTPUT_TAIL_LINES}, 0 disables line truncation)."
        ),
    )
    parser.add_argument(
        "--snapshot-artifact-max-string-chars",
        type=int,
        default=DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
        help=(
            "Per-string cap for persisted snapshot.json artifacts "
            f"(default: {DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS}, 0 disables)."
        ),
    )
    parser.add_argument(
        "--prompt-budget-total-chars",
        type=int,
        default=180_000,
        help="Stage-B prompt total character budget (0 disables total budget cap).",
    )
    parser.add_argument(
        "--prompt-budget-retry-delta-chars",
        type=int,
        default=16_000,
        help="Per-section character cap for retry delta packet in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-stdout-chars",
        type=int,
        default=32_000,
        help="Per-section character cap for pytest stdout/stderr in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-snapshot-chars",
        type=int,
        default=32_000,
        help="Per-section character cap for snapshot summary in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-evidence-chars",
        type=int,
        default=64_000,
        help="Per-section character cap for Stage-A evidence blocks in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-retry-history-chars",
        type=int,
        default=12_000,
        help="Per-section character cap for retry-history transcript in Stage-B prompts.",
    )
    parser.add_argument(
        "--context-compaction-mode",
        choices=["off", "auto_overflow", "always"],
        default="auto_overflow",
        help=(
            "Delegated context compaction policy for Stage-B prompts: "
            "off|auto_overflow|always (default: auto_overflow)."
        ),
    )
    parser.add_argument(
        "--context-compaction-scope",
        default="snapshot,pytest,files",
        help=(
            "Comma-separated raw context sources eligible for delegated compaction "
            "(snapshot,pytest,files)."
        ),
    )
    parser.add_argument(
        "--context-compaction-max-input-chars",
        type=int,
        default=120_000,
        help="Max characters sent to delegated context compaction calls (default: 120000).",
    )
    parser.add_argument(
        "--context-compaction-target-chars",
        type=int,
        default=18_000,
        help="Target compacted context size in characters (default: 18000).",
    )
    parser.add_argument(
        "--context-compaction-max-calls",
        type=int,
        default=1,
        help="Max delegated context compaction calls per case/condition (default: 1, 0 disables).",
    )
    parser.add_argument(
        "--prompt-order-mode",
        choices=["current", "crash_first", "evidence_last"],
        default="current",
        help="Stable Stage-B section ordering mode (default: current).",
    )
    parser.add_argument(
        "--adaptive-reveal-mode",
        choices=["full", "traceback_first"],
        default="full",
        help="Adaptive Stage-B reveal policy for adaptive conditions (default: full).",
    )
    parser.add_argument(
        "--snapshot-frame-pruning-mode",
        choices=["off", "heuristic"],
        default="off",
        help="Snapshot-frame pruning mode applied before Stage-B/TOON assembly (default: off).",
    )
    parser.add_argument(
        "--adaptive-gate-policy",
        choices=["aggressive", "balanced", "conservative"],
        default="balanced",
        help="Gate policy for layered_adaptive_gated helper invocation.",
    )
    parser.add_argument(
        "--adaptive-max-helper-calls",
        type=int,
        default=1,
        help="Max helper invocations per case/condition for adaptive modes (default: 1, 0 = unlimited).",
    )
    parser.add_argument(
        "--adaptive-budget-topup-max-calls",
        type=int,
        default=DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
        help=(
            "Extra one-time helper budget grants per case/condition when adaptive policies are "
            "stagnating (default: 1, 0 disables)."
        ),
    )
    parser.add_argument(
        "--adaptive-stagnation-window",
        type=int,
        default=1,
        help="Consecutive repeated-failure window for adaptive gate decisions.",
    )
    parser.add_argument(
        "--adaptive-saturation-threshold",
        type=float,
        default=0.75,
        help="Disable further helper calls when low-yield helper ratio reaches this threshold.",
    )
    parser.add_argument(
        "--adaptive-no-progress-token-cap",
        type=int,
        default=DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
        help=(
            "Stop adaptive loops when repeated no-progress attempts accumulate at least this many "
            "tokens (default: 150000, 0 disables)."
        ),
    )
    parser.add_argument(
        "--adaptive-helper-budget-profile",
        choices=["default", "compact"],
        default="default",
        help="Prompt-budget profile applied when adaptive helper is invoked.",
    )
    parser.add_argument(
        "--adaptive-openai-helper-response-mode",
        choices=["inherit", "edits", "diff", "full_file"],
        default="inherit",
        help="OpenAI response-mode override for adaptive helper-invoked attempts (default: inherit).",
    )
    parser.add_argument(
        "--adaptive-informativeness-pregate",
        action="store_true",
        default=False,
        help="Enable rule-based traceback informativeness pre-gate for adaptive conditions.",
    )
    parser.add_argument(
        "--adaptive-informativeness-pregate-threshold",
        type=float,
        default=0.7,
        help="Score threshold to skip Stage-A (0.0-1.0, default: 0.7).",
    )
    parser.add_argument(
        "--adaptive-online-policy-mode",
        choices=["off", "shadow", "on"],
        default="off",
        help="Online adaptive policy mode for layered_adaptive_llm (default: off).",
    )
    parser.add_argument(
        "--adaptive-online-policy-algo",
        choices=["linucb", "thompson"],
        default="linucb",
        help="Online adaptive policy algorithm (default: linucb).",
    )
    parser.add_argument(
        "--adaptive-online-alpha",
        type=float,
        default=DEFAULT_ADAPTIVE_ONLINE_ALPHA,
        help=(
            "LinUCB exploration coefficient alpha for the helper policy "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_ALPHA:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-epsilon",
        type=float,
        default=DEFAULT_ADAPTIVE_ONLINE_EPSILON,
        help=(
            "Epsilon-greedy exploration probability for the helper policy "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_EPSILON:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-l2",
        type=float,
        default=DEFAULT_ADAPTIVE_POLICY_L2,
        help=f"L2 regularization for LinUCB matrices (default: {DEFAULT_ADAPTIVE_POLICY_L2}).",
    )
    parser.add_argument(
        "--adaptive-online-warmup-decisions",
        type=int,
        default=DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS,
        help=(
            "Uniform warmup decisions before LinUCB greedy/explore policy "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-save-every",
        type=int,
        default=DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
        help=(
            "Persist online policy state every N updates "
            f"(default: {DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-decay",
        type=float,
        default=DEFAULT_ADAPTIVE_ONLINE_DECAY,
        help=(
            "Optional LinUCB recency decay in (0,1], applied to the updated arm only; "
            f"1.0 disables forgetting (default: {DEFAULT_ADAPTIVE_ONLINE_DECAY:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-feature-scaling-profile",
        choices=sorted(ADAPTIVE_FEATURE_SCALING_PROFILES),
        default=DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        help="Feature scaling profile for adaptive online policy (default: v1).",
    )
    parser.add_argument(
        "--adaptive-online-action-score-biases",
        default=DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
        help=(
            "Comma-separated additive helper-action score priors as action=bias pairs, "
            "each within +/-0.1 of the success reward "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-state-path",
        default="",
        help="Optional path for adaptive online policy state JSON.",
    )
    parser.add_argument(
        "--adaptive-online-init-state-path",
        default="",
        help="Optional warm-start path to load adaptive online policy state without reusing save path.",
    )
    parser.add_argument(
        "--adaptive-online-events-path",
        default="",
        help="Optional path for adaptive online policy JSONL event log.",
    )
    parser.add_argument(
        "--adaptive-budget-policy-mode",
        choices=["off", "shadow", "on"],
        default="on",
        help="Online adaptive budget policy mode for adaptive conditions (default: on).",
    )
    parser.add_argument(
        "--no-adaptive-budget-policy",
        dest="adaptive_budget_policy_mode",
        action="store_const",
        const="off",
        help="Disable adaptive budget policy.",
    )
    parser.add_argument(
        "--adaptive-budget-policy-algo",
        choices=["linucb", "thompson"],
        default="linucb",
        help="Online adaptive budget policy algorithm (default: linucb).",
    )
    parser.add_argument(
        "--adaptive-budget-policy-architecture",
        choices=sorted(BUDGET_POLICY_ARCHITECTURES),
        default=BUDGET_POLICY_ARCH_TIER_BINARY,
        help=(
            "Adaptive budget policy architecture: legacy bundled tiers or hierarchical "
            "family escalation (default: tier_binary)."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-policy-tiers",
        default=DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
        help=(
            "Comma-separated budget multipliers for attempt propose budget tiers "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS}). Must include 1.0."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-policy-max-ups-per-case",
        type=int,
        default=DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
        help=(
            "Max applied budget increases per case/condition "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE}, 0 disables)."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-alpha",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA,
        help=(
            "LinUCB exploration coefficient alpha for adaptive budget policy "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-epsilon",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON,
        help=(
            "Epsilon-greedy exploration probability for budget policy "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-l2",
        type=float,
        default=DEFAULT_ADAPTIVE_POLICY_L2,
        help=(
            "L2 regularization for adaptive budget LinUCB matrices "
            f"(default: {DEFAULT_ADAPTIVE_POLICY_L2})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-warmup-decisions",
        type=int,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS,
        help=(
            "Uniform warmup decisions before budget policy exploitation "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-save-every",
        type=int,
        default=DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
        help=(
            "Persist adaptive budget policy state every N updates "
            f"(default: {DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-decay",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
        help=(
            "Optional LinUCB recency decay in (0,1] for the budget policy, applied to the "
            "updated arm only; 1.0 disables forgetting "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-feature-scaling-profile",
        choices=sorted(ADAPTIVE_FEATURE_SCALING_PROFILES),
        default=DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        help="Feature scaling profile for adaptive budget policy (default: v1).",
    )
    parser.add_argument(
        "--adaptive-budget-online-state-path",
        default="",
        help="Optional path for adaptive budget policy state JSON.",
    )
    parser.add_argument(
        "--adaptive-budget-init-state-path",
        default="",
        help="Optional warm-start path to load adaptive budget policy state without reusing save path.",
    )
    parser.add_argument(
        "--adaptive-budget-online-events-path",
        default="",
        help="Optional path for adaptive budget policy JSONL event log.",
    )
    parser.add_argument(
        "--adaptive-budget-bwk-run-token-budget",
        type=int,
        default=0,
        help="Explicit run-level token budget required by adaptive budget architecture bwk_v1.",
    )
    parser.add_argument(
        "--adaptive-budget-reward-rho-budget-exceeded",
        type=float,
        default=ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
        help="Reward penalty coefficient when attempt ends with request_budget_exceeded.",
    )
    parser.add_argument(
        "--adaptive-reward-lambda-tokens",
        type=float,
        default=ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
        help=(
            "Reward penalty coefficient for token usage: "
            "reward = success - lambda*(tokens/token_scale) - mu*(latency/latency_scale)."
        ),
    )
    parser.add_argument(
        "--adaptive-reward-mu-latency",
        type=float,
        default=ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
        help="Reward penalty coefficient for latency term (default: 0.05).",
    )
    parser.add_argument(
        "--adaptive-reward-token-scale",
        type=float,
        default=ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
        help="Token normalization scale in adaptive reward (default: 1000).",
    )
    parser.add_argument(
        "--adaptive-reward-latency-scale",
        type=float,
        default=ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
        help="Latency normalization scale in adaptive reward seconds (default: 10).",
    )
    parser.add_argument(
        "--adaptive-reward-eta-progress",
        type=float,
        default=ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
        help="Reward bonus coefficient for adaptive progress signal (default: 0.15).",
    )
    parser.add_argument(
        "--adaptive-reward-kappa-stagnation",
        type=float,
        default=ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
        help="Reward penalty coefficient for adaptive no-progress streak (default: 0.10).",
    )
    parser.add_argument(
        "--adaptive-reward-success-hint",
        type=float,
        default=ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
        help="Reward bonus for a successful attempt (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-reward-progress-scale",
        type=float,
        default=ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
        help="Scale multiplier for the progress-hint reward component (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-reward-stagnation-scale",
        type=float,
        default=ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
        help="Scale multiplier for the stagnation penalty component (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-reward-materialization-scale",
        type=float,
        default=ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
        help="Scale multiplier for the helper-context materialization bonus (default: 0.5).",
    )
    parser.add_argument(
        "--adaptive-toon-encoding",
        action="store_true",
        default=False,
        help=(
            "Use TOON encoding for LLM-bound snapshot/evidence payloads in adaptive conditions "
            "(layered_adaptive_gated, layered_adaptive_llm). Stage-B snapshot payloads stay "
            "within the legacy prompt snapshot surface and are only substituted when shorter "
            "than the standard snapshot section. Silently ignored for non-adaptive conditions. "
            "Requires 'pip install llmdebug[toon]' for full TOON; falls back to compact JSON "
            "when toons is unavailable."
        ),
    )
    parser.add_argument(
        "--adaptive-aux-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable auxiliary reward signals (pytest reduction, evidence novelty, compaction, informativeness). On by default; use --no-adaptive-aux-reward to disable.",
    )
    parser.add_argument(
        "--adaptive-aux-pytest-reduction-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
        help=f"Weight for pytest fail reduction auxiliary reward (default: {ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT}).",
    )
    parser.add_argument(
        "--adaptive-aux-evidence-novelty-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
        help=f"Weight for evidence novelty auxiliary reward (default: {ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT}).",
    )
    parser.add_argument(
        "--adaptive-aux-compaction-efficiency-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
        help=f"Weight for compaction efficiency auxiliary reward (default: {ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT}).",
    )
    parser.add_argument(
        "--adaptive-aux-informativeness-alignment-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
        help=f"Weight for informativeness alignment auxiliary reward (default: {ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT}).",
    )
    # TEP configuration.
    parser.add_argument(
        "--adaptive-tep",
        action="store_true",
        default=False,
        help="Enable TEP-inspired stagnation nudges in the retry loop.",
    )
    parser.add_argument(
        "--adaptive-tep-equilibrium",
        action="store_true",
        default=False,
        help="Enable TEP-inspired multi-iteration retry-plan stability checks (implies --adaptive-tep).",
    )
    parser.add_argument(
        "--adaptive-tep-equilibrium-max-iterations",
        type=int,
        default=2,
        help="Max free-phase iterations for TEP equilibrium check (default: 2).",
    )
    parser.add_argument(
        "--adaptive-tep-equilibrium-similarity-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for TEP plan convergence (default: 0.85).",
    )
    parser.add_argument(
        "--adaptive-tep-mode",
        choices=("off", "legacy", "full"),
        default="off",
        help=(
            "Select the adaptive TEP implementation. "
            "'off' disables TEP, 'legacy' keeps the current nudge/stability loop, "
            "and 'full' enables the native-graph prompt-state optimizer."
        ),
    )
    parser.add_argument(
        "--adaptive-tep-state-path",
        default="",
        help=(
            "Path to the persisted full-mode TEP prompt state. "
            "Defaults to evals/artifacts/policies/<run_id>.adaptive_tep_state.json."
        ),
    )
    parser.add_argument(
        "--adaptive-tep-init-state-path",
        default="",
        help="Optional initial prompt-state artifact used to warm-start full-mode TEP.",
    )
    parser.add_argument(
        "--adaptive-tep-beta-init",
        type=float,
        default=1.0,
        help="Initial nudge strength for full-mode TEP (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-tep-beta-decay",
        type=float,
        default=0.9,
        help="Decay applied to beta after each accepted full-mode TEP update cycle (default: 0.9).",
    )
    parser.add_argument(
        "--adaptive-tep-full-equilibrium-max-iters",
        type=int,
        default=20,
        help="Maximum free-phase iterations per full-mode TEP node (default: 20).",
    )
    parser.add_argument(
        "--adaptive-tep-full-equilibrium-window",
        type=int,
        default=3,
        help="Rolling score window used for full-mode TEP equilibrium checks (default: 3).",
    )
    parser.add_argument(
        "--adaptive-tep-full-equilibrium-variance-threshold",
        type=float,
        default=0.5,
        help="Variance threshold for full-mode TEP score stabilization (default: 0.5).",
    )
    parser.add_argument(
        "--adaptive-tep-validation-buffer-size",
        type=int,
        default=4,
        help="Rolling validation-buffer size for full-mode TEP prompt updates (default: 4).",
    )
    parser.add_argument(
        "--syntax-preflight-enabled",
        dest="syntax_preflight_enabled",
        action="store_true",
        default=True,
        help=(
            "Validate patched Python files with py_compile before test execution "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-syntax-preflight",
        dest="syntax_preflight_enabled",
        action="store_false",
        help="Disable py_compile syntax preflight before test execution.",
    )
    parser.add_argument(
        "--syntax-preflight-max-cycles",
        type=int,
        default=DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
        help=(
            "Maximum internal syntax-repair cycles per logical attempt "
            "(default: 2, does not consume additional retry attempts)."
        ),
    )
    parser.add_argument(
        "--syntax-preflight-fallthrough",
        dest="syntax_preflight_fallthrough",
        action="store_true",
        default=True,
        help=(
            "When syntax preflight exhausts all cycles, fall through to test execution "
            "instead of terminating the attempt. Normalizes comparison with baselines that "
            "self-validate internally (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-syntax-preflight-fallthrough",
        dest="syntax_preflight_fallthrough",
        action="store_false",
        help="Disable syntax preflight fallthrough (old behavior: exhaustion terminates attempt).",
    )
    parser.add_argument(
        "--patch-sanitize-enabled",
        dest="patch_sanitize_enabled",
        action="store_true",
        default=True,
        help=(
            "Reject malformed patch text patterns before patch apply/syntax preflight "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-patch-sanitize",
        dest="patch_sanitize_enabled",
        action="store_false",
        help="Disable patch text sanitation guard before syntax preflight.",
    )
    parser.add_argument(
        "--patch-verifier-scope",
        choices=[
            PATCH_VERIFIER_SCOPE_DISABLED,
            PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
            PATCH_VERIFIER_SCOPE_ALL,
        ],
        default=PATCH_VERIFIER_SCOPE_ALL,
        help=(
            "Patch verifier execution scope: disabled|adaptive_only|all (default: all, so every "
            "condition faces the same verifier; adaptive_only is an ablation)."
        ),
    )
    parser.add_argument(
        "--patch-verifier-min-score",
        type=int,
        default=PATCH_VERIFIER_MIN_SCORE,
        help=(
            "Minimum patch verifier relevance score required to pass "
            f"(default: {PATCH_VERIFIER_MIN_SCORE})."
        ),
    )
    parser.add_argument(
        "--patch-verifier-infer-test-source-scope",
        dest="patch_verifier_infer_test_source_scope",
        action="store_true",
        default=True,
        help=(
            "Allow patch verifier crash-scope matching via source imports inferred from "
            "failing/crash test files (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-patch-verifier-infer-test-source-scope",
        dest="patch_verifier_infer_test_source_scope",
        action="store_false",
        help="Disable patch verifier test-to-source crash-scope inference.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (requires --seed; fails fast if guarantees are missing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed used for patch generation and subprocess hash randomization.",
    )
    parser.add_argument(
        "--record-prompt",
        choices=["full", "hash", "none"],
        default="hash",
        help="How to persist prompt material into attempt metadata.",
    )
    parser.add_argument(
        "--record-evidence",
        choices=["full", "hash", "none"],
        default="hash",
        help="How to persist evidence material into attempt metadata.",
    )
    parser.add_argument(
        "--record-env",
        action="append",
        default=[],
        help="Environment variable key to include in run_metadata (repeatable).",
    )
    parser.add_argument(
        "--record-deps",
        choices=["none", "minimal", "runtime"],
        default="minimal",
        help="Dependency version capture mode for run_metadata.",
    )
    add_execution_arguments(parser, default_executor=DEFAULT_EXECUTOR)
    add_decision_model_arguments(parser)
    add_policy_input_arguments(parser)
    args = parser.parse_args(argv_tokens)
    return run_main_flow(parser=parser, args=args, argv_tokens=argv_tokens, seams=_current_seams())


def _current_seams() -> EvalSeams:
    """This module's collaborators as they are now, monkeypatches included (X-02)."""
    return EvalSeams(
        repo_root=_REPO_ROOT,
        load_cases=load_cases,
        make_patcher=make_patcher,
        run_one_condition=run_one_condition,
        build_run_settings_fingerprint=_build_run_settings_fingerprint,
        write_resolved_run_config_lockfile=_write_resolved_run_config_lockfile,
        load_existing_run_state=_load_existing_run_state,
        repair_trailing_results_tail=_repair_trailing_results_tail,
        check_testcontainers_runtime_available=check_testcontainers_runtime_available,
        cleanup_stale_eval_containers=cleanup_stale_eval_containers,
        run_results_lock=_RunResultsLock,
    )


def _execute_eval_unit(
    *,
    case: CaseSpec,
    condition: Condition,
    artifacts_root: Path,
    inputs: EvalUnitInputs,
) -> UnitExecutionOutcome:
    return execute_eval_unit_impl(
        case=case,
        condition=condition,
        artifacts_root=artifacts_root,
        inputs=inputs if inputs.seams is not None else replace(inputs, seams=_current_seams()),
    )


from evals.adaptive_controller_packet import remove_adaptive_controller_sidecars  # noqa: E402
from evals.adaptive_controller_runtime import (  # noqa: E402
    ControllerRuntimeInputs,
    prepare_controller_runtime,
    write_controller_sidecars_for_cycle,
)
from evals.adaptive_history import persist_adaptive_attempt_history  # noqa: E402
from evals.eval_attempt_flow import execute_eval_unit_impl, run_one_condition_impl  # noqa: E402
from evals.eval_evidence import (  # noqa: E402
    _coerce_context_request_payload,
    _collect_file_index_lines,
    _compute_evidence_fingerprint,
    _crash_file_from_snapshot,
    _detect_failing_test_file,
    _evidence_ids_from_items,
    _extract_subprocess_script_neighbors,
    _import_neighbors,
    _merge_evidence_items,
    _render_evidence_index,
    _render_evidence_items,
    _strip_eval_only_case_artifacts,
    _xxh3_128_jsonlike,
    _xxh3_128_text,
    build_fallback_context_bundle,
    collect_text_context,
    execute_context_requests,
)
from evals.eval_main_flow import run_main_flow  # noqa: E402
from evals.eval_seams import EvalSeams  # noqa: E402
from evals.lambda_rlm_backend import (  # noqa: E402
    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
    DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
    default_lambda_rlm_repo_dir,
    invoke_lambda_rlm_decomposition,
)
from evals.patchers.baseline_patcher import resolve_effective_baseline_model  # noqa: E402

# snapshot_chars set by the instrumentation in run_eval; keep if already set


def run_one_condition(
    *,
    case: CaseSpec,
    condition: Condition,
    workdir: Path,
    context_dir_base: Path,
    inputs: RunConditionInputs | None = None,
    **legacy_kwargs: Any,
) -> list[dict[str, Any]]:
    return run_one_condition_impl(
        case=case,
        condition=condition,
        workdir=workdir,
        context_dir_base=context_dir_base,
        inputs=inputs,
        **legacy_kwargs,
    )


from evals.adaptive_progress import (  # noqa: E402
    _adaptive_gate_should_invoke_helper,
    _adaptive_helper_outcome_for_attempt,
    _adaptive_optional_bool,
    _adaptive_optional_int,
    _adaptive_pipeline_stage_before,
    _adaptive_pipeline_stage_for_outcome,
    _adaptive_progress_signal,
    _adaptive_prompt_budget_for_attempt,
    _adaptive_should_stop_for_no_progress,
    _apply_no_progress_token_cap_candidate,
    _extract_exception_info,
    _mark_adaptive_progress_stop,
    _openai_preflight_total_char_cap,
    _pytest_failed_total,
    _pytest_failed_total_from_text,
    _recompute_adaptive_progress_for_attempt_outcome,
    _sync_adaptive_progress_metadata,
    _traceback_informativeness_pre_gate,
    _traceback_informativeness_score,
)
from evals.attempt_proposal import (  # noqa: E402
    _build_attempt_budget_exhausted_patch_result,
    _build_attempt_budget_exhausted_plan_result,
    _ensure_deterministic_seed_ack,
    _measure_attempt_wall_clock_sec,
    _patcher_with_shared_attempt_budget,
    _plan_patch_with_snapshot_context,
    _propose_patch_with_snapshot_context,
    _remaining_attempt_propose_budget_sec,
)
from evals.attempt_records import (  # noqa: E402
    _append_retry_history_turn_if_enabled,
    _build_attempt_outcome_record,
    _build_infra_event_record,
    _build_structured_previous_attempt_diff,
    _finalize_breakdowns,
)
from evals.attempt_usage import (  # noqa: E402
    _aggregate_attempt_llm_usage,
    _apply_usage_totals_to_patch_meta,
    _baseline_metric_float,
    _baseline_metrics_from_patch_meta,
    _clip_patcher_meta_text,
    _combine_reasoning_usage_segments,
    _completion_chars_from_response,
    _format_progress_token_usage,
    _llm_call_count_from_meta,
    _nonnegative_int,
    _optional_nonnegative_int,
    _patch_attempt_char_deltas,
    _prompt_chars_for_llm_calls,
    _prompt_chars_from_patch_meta,
    _reasoning_token_counts_from_patch_meta,
    _reasoning_token_counts_from_usage_payload,
    _reasoning_usage_segment_from_meta,
    _record_token_usage_state,
    _summarize_context_compaction_meta,
    _summarize_recovery_usage_meta,
    _sync_attempt_prompt_char_metadata,
    _sync_attempt_usage_metadata,
    _timeout_error_from_meta,
    _token_usage_available_from_meta,
    _token_usage_blocked_from_meta,
    _usage_available_flag_from_meta,
    _usage_payload_from_counts,
    _usage_token_counts_from_patch_meta,
    _usage_token_counts_from_usage_payload,
)
from evals.attempt_verification import (  # noqa: E402  # noqa: E402
    _sync_patch_diff_metadata,
    _sync_patch_sanitize_metadata,
    _sync_patch_verifier_metadata,
    _sync_patch_verifier_recovery_metadata,
    _sync_syntax_preflight_metadata,
    apply_diff_with_optional_recovery,
    evaluate_patch_sanitize,
    evaluate_syntax_preflight,
    make_syntax_preflight_state,
)
from evals.baseline_runtime import (  # noqa: E402
    _BASELINE_REPO_DIR_BY_NAME,
    _BASELINE_SHORTHAND_CHOICES,
    _NATIVE_BASELINE_MODES,
    _NATIVE_COT_BASELINE_ALLOWED_CONDITIONS,
    DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT,
    DEFAULT_BASELINE_RLM_MAX_DEPTH,
    DEFAULT_BASELINE_RLM_MAX_ERRORS,
    DEFAULT_BASELINE_RLM_MAX_ITERATIONS,
    DEFAULT_BASELINE_RLM_TIMEOUT_SEC,
    _absolute_symlink_portability_hint,
    _autodetect_baseline_python,
    _autodetect_lambda_rlm_python,
    _baseline_extra_from_args,
    _baseline_patch_inventory,
    _baseline_runtime_manifest_entry,
    _baseline_runtime_manifest_path,
    _default_baseline_repo_dir,
    _default_baseline_repo_relative_parts,
    _normalize_adaptive_controller_lambda_rlm_settings,
    _normalize_baseline_selection,
    _normalize_repo_local_path_to_current_root,
    _resolve_conditions_for_run,
    _resolve_external_baseline_runtime_requirements,
    _resolve_manifest_path_entry,
    _validate_external_baseline_runtime_requirements,
)
from evals.case_loading import (  # noqa: E402
    load_cases,
)
from evals.eval_patches import (  # noqa: E402
    _build_runner_forced_attempt_diff,
    _compute_patch_path_metrics,
    _extract_candidate_patch_files,
    _is_forced_noop_marker_diff,
    _is_test_file_path,
    _patch_contains_substantive_change,
    _patch_mentions_crash_file,
    _patch_mentions_failing_test_context,
    _patch_mentions_scope_files,
    _render_patch_verifier_recovery_feedback,
    apply_diff,  # re-exported for importers/validators
)
from evals.native_openai_attempt_flow import (  # noqa: E402
    NativeOpenAIAttemptPromptRequest,
    assemble_native_openai_attempt_prompt,
)
from evals.native_openai_attempt_flow import (  # noqa: E402
    maybe_autoswitch_openai_response_mode as _maybe_autoswitch_openai_response_mode,
)
from evals.native_openai_attempt_flow import (  # noqa: E402
    sync_openai_response_mode_autoswitch_metadata as _sync_openai_response_mode_autoswitch_metadata,
)
from evals.native_stage_b_flow import (  # noqa: E402
    StageBPromptInputs,
    build_stage_b_prompt_sections,
    build_stage_b_prompt_with_budget,
    maybe_run_retry_plan_then_patch,
    openai_preflight_total_char_cap,
)
from evals.patch_relevance import (  # noqa: E402
    _effective_patch_verifier_scope,
    _infer_test_source_scope,
    _score_patch_relevance,
)
from evals.patcher_factory import (  # noqa: E402
    _resolve_openai_context_window_for_run,
    make_patcher,
)
from evals.progress_display import (  # noqa: E402
    _format_duration,
    _format_duration_precise,
    _normalized_progress_verbosity,
    _progress_is_enabled,
    _progress_print,
)
from evals.pytest_runner import (  # noqa: E402
    _base_pytest_env,
    _build_failure_signature_for_attempt,
    _build_pytest_args,
    _coerce_subprocess_output,
    _enrich_failure_context,
    _is_container_wait_timeout,
    _is_snapshot_condition,
    _run_pytest_local,
    _run_pytest_pooled,
    _run_pytest_testcontainers,
    _status_code_from_wait_result,
    _timeout_fields,
    _truncate_persisted_pytest_output,
    _truncate_snapshot_for_artifact,
    _truncate_snapshot_string_for_artifact,
    _write_run_outputs,
    read_latest_snapshot,
    run_pytest,
)
from evals.run_condition_builder import (  # noqa: E402
    _RUN_CONDITION_INPUT_FIELD_NAMES,
    _RUN_CONDITION_INPUT_REQUIRED_KEYS,
    _build_run_condition_inputs_from_legacy_kwargs,
    _resolve_run_condition_inputs,
)
from evals.run_state import (  # noqa: E402
    _build_run_settings_fingerprint,
    _load_existing_run_state,
    _repair_trailing_results_tail,
)
from evals.transport_failures import (  # noqa: E402
    _INFRA_FAILURE_TEXT_MARKERS,
    _INFRA_HTTP_STATUS_RE,
    _classify_patcher_request_failure,
    _filter_attempt_records,
    _is_attempt_record,
    _is_context_limit_http_error_message,
    _is_retryable_transport_http_status,
    _is_transport_runtime_error,
    _OpenAITransportRecoveryGate,
    _patcher_infra_failure_kind_from_meta,
    _record_is_infra_resume_eligible,
    _record_is_transport_request_failure,
    _record_type,
    _request_context_failure_kind_from_meta,
    _request_context_recovery_failure_kind_from_meta,
    _request_context_retry_counts_from_meta,
    _result_meta_dict,
    _result_with_meta,
    _single_line_error,
    _sync_patcher_request_failure_metadata,
    _transport_recovery_failure_kind_from_meta,
)

if __name__ == "__main__":
    raise SystemExit(main())
