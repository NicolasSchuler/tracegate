"""Multi-condition repair benchmark harness for llmdebug experiments.

Orchestrates case discovery, condition execution across native llmdebug
conditions and external baselines, patch proposal via pluggable patchers,
pytest verification, and JSONL result recording. Supports local and
Docker-based (testcontainers) execution, TOML config files, resume from
partial runs, adaptive gating policies, online contextual-bandit policy
controllers, and budget-aware resource management.
"""
# ruff: noqa: F401, F811

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
from evals.eval_patch_constants import PATCH_APPLY_TIMEOUT_SEC
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
from evals.subprocess_monitor import run_monitored_subprocess

if TYPE_CHECKING:
    from evals.patchers.payload_types import ContextRequestPayload
else:
    ContextRequestPayload = dict[str, Any]  # type: ignore[misc,assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if __package__ in {None, ""} and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

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
    DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
    DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
    DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
    DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
    DEFAULT_ADAPTIVE_ONLINE_ALPHA,
    DEFAULT_ADAPTIVE_ONLINE_DECAY,
    DEFAULT_ADAPTIVE_ONLINE_EPSILON,
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

_ACTIVE_RUN_SETTINGS = _ActiveRunSettings()
_ACTIVE_ONLINE_POLICY_CONTROLLER: AdaptiveOnlinePolicyController | None = None
_ACTIVE_BUDGET_POLICY_CONTROLLER: AdaptiveBudgetPolicyController | None = None


import evals.eval_containers as _eval_containers_mod  # noqa: E402
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
from evals.eval_containers import (  # noqa: E402
    _EvalContainerPool,
    check_testcontainers_runtime_available,
)
from evals.eval_containers import (  # noqa: E402
    cleanup_stale_eval_containers as _cleanup_stale_eval_containers_impl,
)


def cleanup_stale_eval_containers(
    *,
    grace_sec: int = DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    now_utc: datetime | None = None,
) -> tuple[int, int, str | None]:
    """Remove stopped eval containers older than *grace_sec* seconds.

    Thin wrapper that patches ``eval_containers._docker_client_from_env``
    so that monkeypatching ``run_eval._docker_client_from_env`` in tests
    propagates correctly.
    """
    _eval_containers_mod._docker_client_from_env = _docker_client_from_env  # type: ignore[attr-defined]
    return _cleanup_stale_eval_containers_impl(
        grace_sec=grace_sec,
        now_utc=now_utc,
    )


def _baseline_patch_inventory(baseline_name: str) -> list[str]:
    patch_dir = _REPO_ROOT / "baselines" / str(baseline_name).strip().lower() / "patches"
    if not patch_dir.is_dir():
        return []
    return sorted(path.name for path in patch_dir.glob("*.patch") if path.is_file())


_BASELINE_REPO_DIR_BY_NAME = {
    "ldb": "LLMDebugger",
    "intervenor": "INTERVENOR",
    "tracecoder": "TraceCoder",
    "lambda_rlm": "lambda-RLM",
    "rlm": "rlm",
}
_NATIVE_BASELINE_MODES = frozenset({Condition.COT_REPAIR.value})
_BASELINE_SHORTHAND_CHOICES = sorted({*_BASELINE_REPO_DIR_BY_NAME, *_NATIVE_BASELINE_MODES})
_NATIVE_COT_BASELINE_ALLOWED_CONDITIONS = frozenset(
    {
        Condition.COT_REPAIR,
        Condition.LAYERED_ADAPTIVE_GATED,
        Condition.LAYERED_ADAPTIVE_LLM,
    }
)

DEFAULT_BASELINE_RLM_MAX_DEPTH = 2
DEFAULT_BASELINE_RLM_MAX_ITERATIONS = 12
DEFAULT_BASELINE_RLM_TIMEOUT_SEC = 120
DEFAULT_BASELINE_RLM_MAX_ERRORS = 5
DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT = 0.85


def _default_baseline_repo_relative_parts(baseline_name: str) -> tuple[str, ...]:
    normalized = str(baseline_name or "").strip().lower()
    baseline_condition_for_name(normalized)
    repo_name = _BASELINE_REPO_DIR_BY_NAME[normalized]
    return ("baselines", "repos", repo_name)


def _default_baseline_repo_dir(baseline_name: str) -> Path:
    return _REPO_ROOT.joinpath(*_default_baseline_repo_relative_parts(baseline_name))


def _normalize_repo_local_path_to_current_root(
    path_text: str,
    *,
    repo_relative_parts: Sequence[str],
) -> Path:
    stripped = str(path_text or "").strip()
    repo_relative_tuple = tuple(str(part) for part in repo_relative_parts)
    repo_local_path = _REPO_ROOT.joinpath(*repo_relative_tuple)
    if not stripped:
        return repo_local_path

    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        if tuple(candidate.parts[: len(repo_relative_tuple)]) == repo_relative_tuple:
            return repo_local_path
        return candidate

    if (
        len(candidate.parts) >= len(repo_relative_tuple)
        and tuple(candidate.parts[-len(repo_relative_tuple) :]) == repo_relative_tuple
        and not candidate.exists()
        and repo_local_path.exists()
    ):
        return repo_local_path
    return candidate


def _autodetect_baseline_python(baseline_name: str) -> str:
    normalized = str(baseline_name or "").strip().lower()
    if not normalized:
        return ""
    candidate = _REPO_ROOT / "baselines" / "envs" / normalized / "bin" / "python"
    return str(candidate) if candidate.is_file() else ""


def _baseline_runtime_manifest_entry(baseline_name: str) -> dict[str, Any] | None:
    normalized = str(baseline_name or "").strip().lower()
    if not normalized:
        return None
    manifest_path = _REPO_ROOT / "baselines" / "envs" / "runtime_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    entry = payload.get(normalized)
    return entry if isinstance(entry, dict) else None


def _resolve_manifest_path_entry(
    entry: Mapping[str, Any] | None,
    *,
    absolute_key: str,
    repo_relative_key: str,
) -> str:
    if entry is None:
        return ""
    absolute = str(entry.get(absolute_key, "") or "").strip()
    repo_relative = str(entry.get(repo_relative_key, "") or "").strip()
    if repo_relative:
        candidate = _REPO_ROOT / repo_relative
        if candidate.exists() or not absolute:
            return str(candidate)
    return absolute


def _baseline_runtime_manifest_path(
    baseline_name: str,
    *,
    absolute_key: str,
    repo_relative_key: str,
    repo_relative_parts: Sequence[str],
) -> Path | None:
    entry = _baseline_runtime_manifest_entry(baseline_name)
    manifest_path = _resolve_manifest_path_entry(
        entry,
        absolute_key=absolute_key,
        repo_relative_key=repo_relative_key,
    )
    if not manifest_path:
        return None
    return _normalize_repo_local_path_to_current_root(
        manifest_path,
        repo_relative_parts=repo_relative_parts,
    )


def _absolute_symlink_portability_hint(path: Path) -> str:
    if not path.is_symlink():
        return ""
    try:
        link_target = os.readlink(path)
    except OSError:
        return ""
    if not os.path.isabs(link_target):
        return ""
    return (
        " The interpreter symlink target "
        f"{link_target!r} is absolute, which is not portable across repo-root remaps. "
        "Recreate the env with baselines/setup_envs.sh from the current checkout."
    )


def _resolve_external_baseline_runtime_requirements(
    *,
    baseline_name: str,
    baseline_repo_dir: str,
    baseline_python: str,
) -> tuple[str, str]:
    normalized = str(baseline_name or "").strip().lower()
    baseline_condition_for_name(normalized)

    expected_repo_dir = _default_baseline_repo_dir(normalized)
    repo_dir_relative_parts = _default_baseline_repo_relative_parts(normalized)
    repo_dir_text = str(baseline_repo_dir or expected_repo_dir).strip()
    repo_dir_path = _normalize_repo_local_path_to_current_root(
        repo_dir_text,
        repo_relative_parts=repo_dir_relative_parts,
    )

    expected_python_path = _REPO_ROOT / "baselines" / "envs" / normalized / "bin" / "python"
    python_relative_parts = expected_python_path.relative_to(_REPO_ROOT).parts
    configured_python = str(baseline_python or "").strip()
    python_path = _normalize_repo_local_path_to_current_root(
        configured_python or str(expected_python_path),
        repo_relative_parts=python_relative_parts,
    )

    manifest_repo_path = _baseline_runtime_manifest_path(
        normalized,
        absolute_key="repo_dir",
        repo_relative_key="repo_dir_repo_relative",
        repo_relative_parts=repo_dir_relative_parts,
    )
    manifest_python_path = _baseline_runtime_manifest_path(
        normalized,
        absolute_key="python_executable",
        repo_relative_key="python_repo_relative",
        repo_relative_parts=python_relative_parts,
    )
    if manifest_repo_path is not None:
        configured_repo_text = str(baseline_repo_dir or "").strip()
        uses_default_repo_dir = not configured_repo_text or repo_dir_path == expected_repo_dir
        if uses_default_repo_dir and manifest_repo_path.is_dir():
            repo_dir_path = manifest_repo_path
    manifest_python = str(manifest_python_path) if manifest_python_path is not None else ""
    manifest_hint = ""
    if manifest_python and manifest_python != str(python_path):
        manifest_hint = (
            " runtime_manifest.json reports "
            f"{manifest_python!r}; pass --baseline-python if you intentionally use that env."
        )

    if not repo_dir_path.is_dir():
        raise ValueError(
            f"external baseline runtime for {normalized!r} is not ready: "
            f"baseline repo dir {str(repo_dir_path)!r} is missing or not a directory; "
            f"expected interpreter {str(python_path)!r}. "
            "Provision cloned baseline repos with baselines/setup_repos.sh."
        )

    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        python_option_hint = (
            " configured interpreter " if configured_python else " expected default interpreter "
        )
        portability_hint = _absolute_symlink_portability_hint(python_path)
        raise ValueError(
            f"external baseline runtime for {normalized!r} is not ready: "
            f"baseline repo dir {str(repo_dir_path)!r} exists, but{python_option_hint}"
            f"{str(python_path)!r} is missing or not executable. "
            "Provision isolated baseline envs with baselines/setup_envs.sh."
            f"{portability_hint}{manifest_hint}"
        )

    return str(repo_dir_path), str(python_path)


def _validate_external_baseline_runtime_requirements(
    *,
    baseline_name: str,
    baseline_repo_dir: str,
    baseline_python: str,
) -> str:
    _, python_path = _resolve_external_baseline_runtime_requirements(
        baseline_name=baseline_name,
        baseline_repo_dir=baseline_repo_dir,
        baseline_python=baseline_python,
    )
    return python_path


def _autodetect_lambda_rlm_python() -> str:
    from evals.lambda_rlm_backend import default_lambda_rlm_python

    candidate = default_lambda_rlm_python()
    return str(candidate) if candidate.is_file() else ""


def _normalize_adaptive_controller_lambda_rlm_settings(
    *,
    args: argparse.Namespace,
    config_state: ConfigResolutionState | None,
) -> None:
    if not bool(getattr(args, "adaptive_controller_lambda_rlm", False)):
        return

    from evals.lambda_rlm_backend import default_lambda_rlm_repo_dir

    repo_dir = (getattr(args, "adaptive_controller_lambda_rlm_repo_dir", "") or "").strip()
    if not repo_dir:
        manifest_repo_dir = _baseline_runtime_manifest_path(
            "lambda_rlm",
            absolute_key="repo_dir",
            repo_relative_key="repo_dir_repo_relative",
            repo_relative_parts=("baselines", "repos", "lambda-RLM"),
        )
        default_repo_dir = default_lambda_rlm_repo_dir()
        resolved_repo_dir = (
            manifest_repo_dir
            if manifest_repo_dir is not None
            and manifest_repo_dir.is_dir()
            and not default_repo_dir.is_dir()
            else default_repo_dir
        )
        args.adaptive_controller_lambda_rlm_repo_dir = str(resolved_repo_dir)
        if (
            config_state is not None
            and config_state.value_sources.get("adaptive_controller_lambda_rlm_repo_dir") != "cli"
        ):
            config_state.value_sources["adaptive_controller_lambda_rlm_repo_dir"] = (
                "derived_from_adaptive_controller_lambda_rlm"
            )
    else:
        normalized_repo_dir = _normalize_repo_local_path_to_current_root(
            repo_dir,
            repo_relative_parts=("baselines", "repos", "lambda-RLM"),
        )
        args.adaptive_controller_lambda_rlm_repo_dir = str(normalized_repo_dir)

    python_executable = (getattr(args, "adaptive_controller_lambda_rlm_python", "") or "").strip()
    if not python_executable:
        detected_python = _autodetect_lambda_rlm_python()
        if detected_python:
            args.adaptive_controller_lambda_rlm_python = detected_python
        else:
            manifest_python = _baseline_runtime_manifest_path(
                "lambda_rlm",
                absolute_key="python_executable",
                repo_relative_key="python_repo_relative",
                repo_relative_parts=("baselines", "envs", "lambda_rlm", "bin", "python"),
            )
            if manifest_python is not None and manifest_python.is_file():
                args.adaptive_controller_lambda_rlm_python = str(manifest_python)
        if (
            getattr(args, "adaptive_controller_lambda_rlm_python", "")
            and config_state is not None
            and config_state.value_sources.get("adaptive_controller_lambda_rlm_python") != "cli"
        ):
            config_state.value_sources["adaptive_controller_lambda_rlm_python"] = (
                "derived_from_adaptive_controller_lambda_rlm"
            )
    else:
        normalized_python = _normalize_repo_local_path_to_current_root(
            python_executable,
            repo_relative_parts=("baselines", "envs", "lambda_rlm", "bin", "python"),
        )
        args.adaptive_controller_lambda_rlm_python = str(normalized_python)


def _baseline_extra_from_args(
    *,
    args: argparse.Namespace,
    baseline_name: str,
) -> dict[str, Any]:
    normalized = str(baseline_name or "").strip().lower()
    shared_openai_extra: dict[str, Any] = {}
    if normalized in {"ldb", "intervenor", "tracecoder", "lambda_rlm", "rlm"}:
        shared_openai_extra = {
            **_baseline_openai_compat_settings_from_args(args=args),
            "baseline_hybrid_guidance_mode": str(
                getattr(args, "baseline_hybrid_guidance_mode", "tuned_v1") or "tuned_v1"
            ).strip()
            or "tuned_v1",
        }
    if normalized == "lambda_rlm":
        extra: dict[str, Any] = {
            **shared_openai_extra,
            "baseline_lambda_context_window_chars": int(
                getattr(
                    args,
                    "baseline_lambda_context_window_chars",
                    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
                )
            ),
            "baseline_lambda_accuracy_target": float(
                getattr(args, "baseline_lambda_accuracy_target", 0.80)
            ),
            "baseline_lambda_a_leaf": float(getattr(args, "baseline_lambda_a_leaf", 0.95)),
            "baseline_lambda_a_compose": float(getattr(args, "baseline_lambda_a_compose", 0.90)),
            "baseline_lambda_verbose": bool(getattr(args, "baseline_lambda_verbose", False)),
            "baseline_lambda_stream": bool(getattr(args, "baseline_lambda_stream", False)),
            "baseline_lambda_controller_guidance_max_chars": int(
                getattr(args, "baseline_lambda_controller_guidance_max_chars", 2500)
            ),
            "baseline_lambda_failure_feedback_max_chars": int(
                getattr(args, "baseline_lambda_failure_feedback_max_chars", 3500)
            ),
            "baseline_lambda_failure_feedback_meta_max_chars": int(
                getattr(args, "baseline_lambda_failure_feedback_meta_max_chars", 2000)
            ),
        }
        timeout_sec = getattr(args, "baseline_lambda_timeout_sec", None)
        if timeout_sec is not None:
            extra["baseline_lambda_timeout_sec"] = int(timeout_sec)
        force_task_type = str(getattr(args, "baseline_lambda_force_task_type", "") or "").strip()
        if force_task_type:
            extra["baseline_lambda_force_task_type"] = force_task_type
        prompt_contract = str(getattr(args, "baseline_lambda_prompt_contract", "") or "").strip()
        if prompt_contract:
            extra["baseline_lambda_prompt_contract"] = prompt_contract
        for source_key, target_key in (
            ("baseline_lambda_temperature", "baseline_lambda_temperature"),
            ("baseline_lambda_top_p", "baseline_lambda_top_p"),
            ("baseline_lambda_max_tokens", "baseline_lambda_max_tokens"),
        ):
            value = getattr(args, source_key, None)
            if value is not None:
                extra[target_key] = value
        return extra

    if normalized == "rlm":
        extra = {
            **shared_openai_extra,
            "baseline_rlm_max_depth": int(
                getattr(args, "baseline_rlm_max_depth", DEFAULT_BASELINE_RLM_MAX_DEPTH)
            ),
            "baseline_rlm_max_iterations": int(
                getattr(args, "baseline_rlm_max_iterations", DEFAULT_BASELINE_RLM_MAX_ITERATIONS)
            ),
            "baseline_rlm_max_errors": int(
                getattr(args, "baseline_rlm_max_errors", DEFAULT_BASELINE_RLM_MAX_ERRORS)
            ),
            "baseline_rlm_compaction": bool(getattr(args, "baseline_rlm_compaction", False)),
            "baseline_rlm_compaction_threshold_pct": float(
                getattr(
                    args,
                    "baseline_rlm_compaction_threshold_pct",
                    DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT,
                )
            ),
            "baseline_rlm_verbose": bool(getattr(args, "baseline_rlm_verbose", False)),
            "baseline_rlm_controller_guidance_max_chars": int(
                getattr(args, "baseline_rlm_controller_guidance_max_chars", 2500)
            ),
            "baseline_rlm_failure_feedback_max_chars": int(
                getattr(args, "baseline_rlm_failure_feedback_max_chars", 3500)
            ),
            "baseline_rlm_failure_feedback_meta_max_chars": int(
                getattr(args, "baseline_rlm_failure_feedback_meta_max_chars", 2000)
            ),
        }
        timeout_sec = getattr(args, "baseline_rlm_timeout_sec", None)
        if timeout_sec is not None:
            extra["baseline_rlm_timeout_sec"] = int(timeout_sec)
        prompt_contract = str(getattr(args, "baseline_rlm_prompt_contract", "") or "").strip()
        if prompt_contract:
            extra["baseline_rlm_prompt_contract"] = prompt_contract
        for source_key in (
            "baseline_rlm_temperature",
            "baseline_rlm_top_p",
            "baseline_rlm_max_tokens",
            "baseline_rlm_max_total_tokens",
        ):
            value = getattr(args, source_key, None)
            if value is not None:
                extra[source_key] = value
        return extra

    if normalized == "intervenor":
        return {
            **shared_openai_extra,
            "baseline_intervenor_teacher_max_tokens": int(
                getattr(args, "baseline_intervenor_teacher_max_tokens", 1024)
            ),
            "baseline_intervenor_learner_max_tokens": int(
                getattr(args, "baseline_intervenor_learner_max_tokens", 2048)
            ),
            "baseline_intervenor_teacher_guidance_max_chars": int(
                getattr(args, "baseline_intervenor_teacher_guidance_max_chars", 4000)
            ),
            "baseline_intervenor_learner_guidance_max_chars": int(
                getattr(args, "baseline_intervenor_learner_guidance_max_chars", 6000)
            ),
            "baseline_intervenor_append_output_contract": bool(
                getattr(args, "baseline_intervenor_append_output_contract", True)
            ),
        }

    if normalized == "tracecoder":
        return {
            **shared_openai_extra,
            "baseline_tracecoder_controller_guidance_max_chars": int(
                getattr(args, "baseline_tracecoder_controller_guidance_max_chars", 2000)
            ),
            "baseline_tracecoder_max_no_improvement_streak": int(
                getattr(args, "baseline_tracecoder_max_no_improvement_streak", 2)
            ),
            "baseline_tracecoder_analysis_max_tokens": int(
                getattr(args, "baseline_tracecoder_analysis_max_tokens", 4096)
            ),
            "baseline_tracecoder_implementation_max_tokens": int(
                getattr(args, "baseline_tracecoder_implementation_max_tokens", 4096)
            ),
            "baseline_tracecoder_problem_summary_max_chars": int(
                getattr(args, "baseline_tracecoder_problem_summary_max_chars", 4000)
            ),
            "baseline_tracecoder_exact_test_context_max_chars": int(
                getattr(args, "baseline_tracecoder_exact_test_context_max_chars", 3000)
            ),
            "baseline_tracecoder_failure_feedback_max_chars": int(
                getattr(args, "baseline_tracecoder_failure_feedback_max_chars", 6000)
            ),
            "baseline_tracecoder_failure_feedback_tail_max_chars": int(
                getattr(args, "baseline_tracecoder_failure_feedback_tail_max_chars", 2000)
            ),
            "baseline_tracecoder_instrumentation_output_max_lines": int(
                getattr(args, "baseline_tracecoder_instrumentation_output_max_lines", 80)
            ),
            "baseline_tracecoder_instrumentation_output_max_chars": int(
                getattr(args, "baseline_tracecoder_instrumentation_output_max_chars", 12000)
            ),
        }

    if shared_openai_extra:
        return shared_openai_extra

    return {}


def _normalize_baseline_selection(
    *,
    args: argparse.Namespace,
    config_state: ConfigResolutionState | None,
) -> None:
    baseline_alias = (getattr(args, "baseline", "") or "").strip().lower()
    patcher_name = str(getattr(args, "patcher", "") or "").strip()
    baseline_name = (getattr(args, "baseline_name", "") or "").strip().lower()
    native_baseline_mode = str(getattr(args, "native_baseline_mode", "") or "").strip().lower()

    if not hasattr(args, "native_baseline_mode"):
        args.native_baseline_mode = native_baseline_mode

    if baseline_alias:
        baseline_source = "default"
        if config_state is not None:
            baseline_source = config_state.value_sources.get("baseline", "default")
        if baseline_alias in _NATIVE_BASELINE_MODES:
            if patcher_name in {"baseline", "heuristic", "null"}:
                raise ValueError(
                    f"--baseline {baseline_alias!r} cannot be combined with --patcher "
                    f"{patcher_name!r}; omit --patcher or use --patcher openai."
                )
            if baseline_name:
                raise ValueError(
                    f"--baseline {baseline_alias!r} cannot be combined with --baseline-name "
                    f"{baseline_name!r}."
                )
            for option_name in ("baseline_repo_dir", "baseline_python"):
                option_value = str(getattr(args, option_name, "") or "").strip()
                if option_value:
                    raise ValueError(
                        f"--baseline {baseline_alias!r} cannot be combined with "
                        f"--{option_name.replace('_', '-')}."
                    )
            args.patcher = "openai"
            args.native_baseline_mode = baseline_alias
            args.baseline_name = ""
            args.baseline_repo_dir = ""
            args.baseline_python = ""
            if config_state is not None:
                if config_state.value_sources.get("patcher") != "cli":
                    config_state.value_sources["patcher"] = (
                        f"derived_from_baseline:{baseline_source}"
                    )
                if config_state.value_sources.get("native_baseline_mode") != "cli":
                    config_state.value_sources["native_baseline_mode"] = (
                        f"derived_from_baseline:{baseline_source}"
                    )
            return

        baseline_condition_for_name(baseline_alias)
        if patcher_name not in {"", "baseline", "command"}:
            raise ValueError(
                f"--baseline {baseline_alias!r} cannot be combined with --patcher "
                f"{patcher_name!r}; omit --patcher or use --patcher baseline."
            )
        if native_baseline_mode:
            raise ValueError(
                f"--baseline {baseline_alias!r} conflicts with native baseline mode "
                f"{native_baseline_mode!r}."
            )
        if baseline_name and baseline_name != baseline_alias:
            raise ValueError(
                f"--baseline {baseline_alias!r} conflicts with --baseline-name {baseline_name!r}."
            )
        args.patcher = "baseline"
        args.baseline_name = baseline_alias
        args.native_baseline_mode = ""
        if config_state is not None:
            if config_state.value_sources.get("patcher") != "cli":
                config_state.value_sources["patcher"] = f"derived_from_baseline:{baseline_source}"
            if config_state.value_sources.get("baseline_name") != "cli":
                config_state.value_sources["baseline_name"] = (
                    f"derived_from_baseline:{baseline_source}"
                )

    native_baseline_mode = str(getattr(args, "native_baseline_mode", "") or "").strip().lower()
    if native_baseline_mode:
        if native_baseline_mode not in _NATIVE_BASELINE_MODES:
            raise ValueError(
                f"unknown native baseline mode {native_baseline_mode!r}; expected one of: "
                + ", ".join(sorted(_NATIVE_BASELINE_MODES))
            )
        if str(getattr(args, "patcher", "") or "").strip() != "openai":
            raise ValueError(
                f"native baseline mode {native_baseline_mode!r} requires --patcher openai."
            )
        return

    if str(getattr(args, "patcher", "") or "").strip() != "baseline":
        return

    baseline_name = (getattr(args, "baseline_name", "") or "").strip().lower()
    if not baseline_name:
        raise ValueError(
            "--baseline-name is required for --patcher baseline. "
            "Available: ldb, intervenor, tracecoder, lambda_rlm, rlm."
        )
    baseline_condition_for_name(baseline_name)
    args.baseline_name = baseline_name

    repo_dir = (getattr(args, "baseline_repo_dir", "") or "").strip()
    if not repo_dir:
        args.baseline_repo_dir = str(_default_baseline_repo_dir(baseline_name))
        if (
            config_state is not None
            and config_state.value_sources.get("baseline_repo_dir") != "cli"
        ):
            baseline_name_source = config_state.value_sources.get("baseline_name", "default")
            config_state.value_sources["baseline_repo_dir"] = (
                f"derived_from_baseline_name:{baseline_name_source}"
            )

    baseline_python = (getattr(args, "baseline_python", "") or "").strip()
    if not baseline_python:
        detected_python = _autodetect_baseline_python(baseline_name)
        if detected_python:
            args.baseline_python = detected_python
            if (
                config_state is not None
                and config_state.value_sources.get("baseline_python") != "cli"
            ):
                baseline_name_source = config_state.value_sources.get("baseline_name", "default")
                config_state.value_sources["baseline_python"] = (
                    f"derived_from_baseline_name:{baseline_name_source}"
                )


def _resolve_conditions_for_run(
    *,
    args: argparse.Namespace,
    config_state: ConfigResolutionState,
) -> list[Condition]:
    def _finalize(resolved_conditions: list[Condition]) -> list[Condition]:
        args.conditions = ",".join(condition.value for condition in resolved_conditions)
        return resolved_conditions

    raw_conditions = [token.strip() for token in str(args.conditions).split(",") if token.strip()]
    native_baseline_mode = str(getattr(args, "native_baseline_mode", "") or "").strip().lower()
    if native_baseline_mode:
        conditions_source = config_state.value_sources.get("conditions", "default")
        if native_baseline_mode == Condition.COT_REPAIR.value and (
            not raw_conditions
            or (conditions_source == "default" and raw_conditions == [Condition.BASE_REPAIR.value])
        ):
            return _finalize([Condition.COT_REPAIR])

        parsed = [Condition(token) for token in raw_conditions]
        if len(parsed) != 1:
            raise ValueError(
                f"native baseline mode {native_baseline_mode!r} requires exactly one condition."
            )
        if parsed[0] not in _NATIVE_COT_BASELINE_ALLOWED_CONDITIONS:
            allowed = ", ".join(
                condition.value
                for condition in sorted(
                    _NATIVE_COT_BASELINE_ALLOWED_CONDITIONS, key=lambda item: item.value
                )
            )
            raise ValueError(
                f"native baseline mode {native_baseline_mode!r} requires one of {allowed}; "
                f"got {parsed[0].value!r}."
            )
        return _finalize(parsed)

    if str(args.patcher) != "baseline":
        return _finalize([Condition(token) for token in raw_conditions])

    baseline_name = (getattr(args, "baseline_name", "") or "").strip()
    if not baseline_name:
        raise ValueError(
            "--baseline-name is required for --patcher baseline. Available: ldb, intervenor, tracecoder, lambda_rlm, rlm."
        )
    expected = baseline_condition_for_name(baseline_name)
    conditions_source = config_state.value_sources.get("conditions", "default")
    if conditions_source == "default" and (
        not raw_conditions or raw_conditions == [Condition.BASE_REPAIR.value]
    ):
        return _finalize([expected])

    parsed = [Condition(token) for token in raw_conditions]
    if len(parsed) != 1:
        raise ValueError(
            "--patcher baseline requires exactly one condition: either the selected baseline "
            f"({expected.value}) or one adaptive condition."
        )
    allowed_conditions = {
        expected,
        Condition.LAYERED_ADAPTIVE_GATED,
        Condition.LAYERED_ADAPTIVE_LLM,
    }
    if parsed[0] not in allowed_conditions:
        allowed = ", ".join(
            condition.value
            for condition in (
                expected,
                *sorted(
                    {
                        Condition.LAYERED_ADAPTIVE_GATED,
                        Condition.LAYERED_ADAPTIVE_LLM,
                    },
                    key=lambda item: item.value,
                ),
            )
        )
        raise ValueError(
            "--patcher baseline requires a matching baseline condition or one adaptive condition. "
            f"Expected one of {allowed} for --baseline-name {baseline_name!r}, "
            f"got {parsed[0].value!r}."
        )
    return _finalize(parsed)


_ACTIVE_EXECUTION_CONFIG = ExecutionConfig()
_ACTIVE_CONTAINER_POOL: _EvalContainerPool | None = None


def _normalized_progress_verbosity(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    return text if text in PROGRESS_VERBOSITIES else "basic"


def _progress_is_enabled(verbosity: str, level: str) -> bool:
    normalized_verbosity = _normalized_progress_verbosity(verbosity)
    normalized_level = _normalized_progress_verbosity(level)
    return _PROGRESS_LEVEL_ORDER[normalized_verbosity] >= _PROGRESS_LEVEL_ORDER[normalized_level]


def _progress_print(
    *,
    verbosity: str,
    level: str,
    message: str,
    flush: bool = True,
) -> None:
    if not _progress_is_enabled(verbosity, level):
        return
    with _PROGRESS_PRINT_LOCK:
        print(message, flush=flush)


def _format_duration_precise(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    normalized = max(0.0, float(seconds))
    if normalized < 60.0:
        return f"{normalized:.2f}s"
    return _format_duration(normalized)


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
    _RunResultsLock,
    _SynchronizedJsonlWriter,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for the eval harness.

    Parses arguments, discovers cases, resumes partial runs, and
    dispatches (case, condition) units to a thread-pool executor.

    Returns:
        Exit code: 0 on success, 1 on partial failure, 2 on fatal error.
    """
    global _ACTIVE_RUN_SETTINGS, _ACTIVE_CONTAINER_POOL
    global _ACTIVE_ONLINE_POLICY_CONTROLLER, _ACTIVE_BUDGET_POLICY_CONTROLLER
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run the llmdebug multi-condition repair benchmark harness."
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
        default=1.0,
        help="L2 regularization for LinUCB matrices (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-online-warmup-decisions",
        type=int,
        default=50,
        help="Uniform warmup decisions before LinUCB greedy/explore policy (default: 50).",
    )
    parser.add_argument(
        "--adaptive-online-save-every",
        type=int,
        default=50,
        help="Persist online policy state every N updates (default: 50).",
    )
    parser.add_argument(
        "--adaptive-online-decay",
        type=float,
        default=DEFAULT_ADAPTIVE_ONLINE_DECAY,
        help=(
            "Optional LinUCB recency decay in (0,1]; 1.0 disables forgetting "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_DECAY:g})."
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
            "Comma-separated additive helper-action score priors as action=bias pairs "
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
        default=1.0,
        help="L2 regularization for adaptive budget LinUCB matrices (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-budget-online-warmup-decisions",
        type=int,
        default=20,
        help="Uniform warmup decisions before budget policy exploitation (default: 20).",
    )
    parser.add_argument(
        "--adaptive-budget-online-save-every",
        type=int,
        default=50,
        help="Persist adaptive budget policy state every N updates (default: 50).",
    )
    parser.add_argument(
        "--adaptive-budget-online-decay",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
        help=(
            "Optional LinUCB recency decay in (0,1] for budget policy; "
            f"{DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY:g} enables forgetting."
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
    from evals.eval_main_flow import run_main_flow

    return run_main_flow(parser=parser, args=args, argv_tokens=argv_tokens)


def _single_line_error(raw: str, *, max_chars: int = 240) -> str:
    text = " | ".join(segment.strip() for segment in str(raw).splitlines() if segment.strip())
    if not text:
        text = str(raw).strip() or "unknown runtime error"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _is_transport_runtime_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    markers = (
        "segmentation fault",
        "connection refused",
        "connection reset",
        "brokenpipeerror",
        "remotedisconnected",
        "temporarily unavailable",
        "server disconnected",
        "failed to establish a new connection",
    )
    return any(marker in text for marker in markers)


_INFRA_HTTP_STATUS_RE = re.compile(r"\b(?:status|code)\s*[:= -]?\s*(50[234])\b", re.IGNORECASE)
_INFRA_FAILURE_TEXT_MARKERS: tuple[tuple[str, str], ...] = (
    ("loading model", "backend_unavailable"),
    ("unavailable_error", "backend_unavailable"),
    ("temporarily unavailable", "backend_unavailable"),
    ("service unavailable", "backend_unavailable"),
    ("server disconnected", "transport"),
    ("connection refused", "transport"),
    ("connection reset", "transport"),
    ("brokenpipeerror", "transport"),
    ("remotedisconnected", "transport"),
    ("failed to establish a new connection", "transport"),
)


def _patcher_infra_failure_kind_from_meta(meta: Mapping[str, Any] | None) -> str:
    """Classify patcher metadata into a normalized infra failure kind."""
    if not isinstance(meta, Mapping):
        return ""
    meta_candidates = [meta]
    baseline_meta = meta.get("baseline_meta")
    if isinstance(baseline_meta, Mapping):
        meta_candidates.append(baseline_meta)
    for candidate in meta_candidates:
        error = str(candidate.get("error") or "").strip()
        if error in {"request_failed", "api_timeout"}:
            return "transport"
        if error == "http_error":
            status = _nonnegative_int(candidate.get("status"), default=0)
            if status in {502, 503, 504}:
                return "backend_unavailable"
            if _is_retryable_transport_http_status(status):
                return "transport"
        if error not in {"repair_failed", "api_error"}:
            continue

        status = _nonnegative_int(candidate.get("status"), default=0)
        if status in {502, 503, 504}:
            return "backend_unavailable"
        if _is_retryable_transport_http_status(status):
            return "transport"

        text_fields = [
            str(candidate.get("body") or ""),
            str(candidate.get("message") or ""),
            str(candidate.get("stderr") or ""),
            str(candidate.get("stdout") or ""),
            str(candidate.get("stderr_tail") or ""),
            str(candidate.get("stdout_tail") or ""),
            str(candidate.get("exc") or ""),
        ]
        combined = " ".join(part for part in text_fields if part).lower()
        if not combined:
            continue
        status_match = _INFRA_HTTP_STATUS_RE.search(combined)
        if status_match:
            status_code = int(status_match.group(1))
            if status_code in {502, 503, 504}:
                return "backend_unavailable"
            if _is_retryable_transport_http_status(status_code):
                return "transport"
        for marker, kind in _INFRA_FAILURE_TEXT_MARKERS:
            if marker in combined:
                return kind
        if error == "api_error":
            return "transport"
    return ""


def _record_is_transport_request_failure(record: dict[str, Any]) -> bool:
    if _record_type(record) != EVAL_RECORD_TYPE_ATTEMPT:
        return False
    note = str(record.get("note") or "")
    if note != "patcher_request_failed":
        return False
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return False
    if bool(patch_meta.get("patcher_request_failed")):
        return str(patch_meta.get("patcher_request_failure_kind") or "") == "transport"
    failed, kind = _classify_patcher_request_failure(patch_meta)
    return bool(failed and kind == "transport")


def _record_type(record: Mapping[str, Any]) -> str:
    raw = str(record.get("record_type") or "").strip()
    if raw == EVAL_RECORD_TYPE_INFRA_EVENT:
        return EVAL_RECORD_TYPE_INFRA_EVENT
    return EVAL_RECORD_TYPE_ATTEMPT


def _is_attempt_record(record: Mapping[str, Any]) -> bool:
    return _record_type(record) == EVAL_RECORD_TYPE_ATTEMPT


def _filter_attempt_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _is_attempt_record(record)]


def _result_meta_dict(result: Any) -> dict[str, Any]:
    meta = getattr(result, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _result_with_meta(result: Any, meta: dict[str, Any]) -> Any:
    from evals.patchers.base import (
        ContextCompactionResult,
        ContextRequestResult,
        PatchPlanResult,
        PatchResult,
    )

    if isinstance(result, PatchResult):
        return PatchResult(diff=result.diff, meta=meta)
    if isinstance(result, PatchPlanResult):
        return PatchPlanResult(
            plan_text=result.plan_text,
            raw_response=result.raw_response,
            meta=meta,
        )
    if isinstance(result, ContextRequestResult):
        return ContextRequestResult(
            request_json=result.request_json,
            raw_response=result.raw_response,
            meta=meta,
        )
    if isinstance(result, ContextCompactionResult):
        return ContextCompactionResult(
            compacted_text=result.compacted_text,
            raw_response=result.raw_response,
            meta=meta,
        )
    return result


class _OpenAITransportRecoveryGate:
    """Coordinate OpenAI transport outage recovery across worker threads."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_wait_sec: float,
        poll_sec: float,
    ) -> None:
        self._enabled = bool(enabled)
        self._max_wait_sec = max(0.0, float(max_wait_sec))
        self._poll_sec = max(0.1, float(poll_sec))
        self._condition = threading.Condition()
        self._outage_active = False
        self._outage_generation = 0
        self._leader_claimed_generation = 0
        self._last_outcome: dict[str, Any] | None = None

    def wait_if_open(self) -> None:
        if not self._enabled:
            return
        with self._condition:
            while self._outage_active:
                self._condition.wait(timeout=self._poll_sec)

    def recover(
        self,
        *,
        patcher: Any,
        operation_name: str,
    ) -> dict[str, Any]:
        if not self._enabled or self._max_wait_sec <= 0:
            return {
                "attempted": False,
                "recovered": False,
                "timed_out": True,
                "wait_sec": 0.0,
                "probe_attempts": [],
                "probe_meta": {},
                "operation": operation_name,
                "leader": False,
                "generation": 0,
            }

        started_at = time.monotonic()
        deadline = started_at + self._max_wait_sec
        with self._condition:
            if not self._outage_active:
                self._outage_active = True
                self._outage_generation += 1
                self._leader_claimed_generation = self._outage_generation
                generation = self._outage_generation
                leader = True
            else:
                generation = self._outage_generation
                leader = False

        if not leader:
            with self._condition:
                while self._outage_active and self._outage_generation == generation:
                    self._condition.wait(timeout=self._poll_sec)
                outcome = dict(self._last_outcome or {})
            outcome.setdefault("attempted", True)
            outcome.setdefault("operation", operation_name)
            outcome["leader"] = False
            return outcome

        probe_attempts: list[dict[str, Any]] = []
        recovered = False
        probe_meta: dict[str, Any] = {}
        while time.monotonic() < deadline:
            probe_started = time.monotonic()
            try:
                ready, raw_probe_meta = patcher.probe_server_ready()
            except Exception as exc:
                ready = False
                raw_probe_meta = {
                    "error": "server_ready_probe_failed",
                    "exc": f"{type(exc).__name__}: {exc}",
                    "server_ready": False,
                }
            probe_meta = dict(raw_probe_meta or {})
            probe_attempts.append(
                {
                    "provider": str(probe_meta.get("server_ready_probe_provider") or ""),
                    "error": str(probe_meta.get("error") or ""),
                    "status": probe_meta.get("status"),
                    "duration_sec": max(0.0, time.monotonic() - probe_started),
                }
            )
            if ready:
                recovered = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self._poll_sec, remaining))
        finished_at = time.monotonic()

        outcome = {
            "attempted": True,
            "recovered": recovered,
            "timed_out": not recovered,
            "wait_sec": max(0.0, min(self._max_wait_sec, finished_at - started_at)),
            "probe_attempts": probe_attempts,
            "probe_meta": probe_meta,
            "operation": operation_name,
            "leader": True,
            "generation": generation,
        }
        with self._condition:
            self._last_outcome = dict(outcome)
            if self._outage_generation == generation:
                self._outage_active = False
                self._condition.notify_all()
        return outcome


def _build_infra_event_record(
    *,
    case: CaseSpec,
    condition: Condition,
    run_id: str,
    attempt: int,
    patcher_name: str,
    run_metadata: dict[str, Any],
    attempt_metadata: dict[str, Any] | None,
    patcher: Any,
    patch_meta: dict[str, Any],
    context_protocol: str,
    before: RunResult,
    cumulative_runtime_sec: float,
    note: str = "infra_transport_failure",
) -> dict[str, Any]:
    materialized_attempt_metadata = (
        dict(attempt_metadata)
        if isinstance(attempt_metadata, dict)
        else _empty_attempt_metadata(attempt=attempt)
    )
    materialized_attempt_metadata["attempt"] = attempt
    record = _build_base_record(
        case=case,
        condition=condition,
        run_id=run_id,
        attempt=attempt,
        patcher_name=patcher_name,
        run_metadata=run_metadata,
        attempt_metadata=materialized_attempt_metadata,
        model_metadata=_build_model_metadata(patcher=patcher, patch_meta=patch_meta),
        context_protocol=context_protocol,
    )
    record["record_type"] = EVAL_RECORD_TYPE_INFRA_EVENT
    record.update(
        {
            "success": False,
            "before_returncode": before.returncode,
            "before_duration_sec": before.duration_sec,
            "cumulative_runtime_sec": cumulative_runtime_sec,
            "attempt_wall_clock_sec": 0.0,
            "note": note,
            "patch_meta": dict(patch_meta),
        }
    )
    record.update(_timeout_fields(before))
    return record


_RUN_CONDITION_INPUT_FIELD_NAMES = frozenset(RunConditionInputs.__dataclass_fields__)
_RUN_CONDITION_INPUT_REQUIRED_KEYS = frozenset(
    {"patcher", "k", "output_format", "run_id", "include_files_context", "force_patch_attempt"}
)


def _build_run_condition_inputs_from_legacy_kwargs(
    *,
    legacy_kwargs: Mapping[str, Any],
) -> RunConditionInputs:
    missing_keys = sorted(
        key for key in _RUN_CONDITION_INPUT_REQUIRED_KEYS if key not in legacy_kwargs
    )
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise TypeError(f"run_one_condition() missing required input fields: {joined}")
    unknown_keys = sorted(set(legacy_kwargs) - _RUN_CONDITION_INPUT_FIELD_NAMES)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise TypeError(f"run_one_condition() received unknown input fields: {joined}")
    return RunConditionInputs(**dict(legacy_kwargs))


def _resolve_run_condition_inputs(
    *,
    inputs: RunConditionInputs | None,
    legacy_kwargs: Mapping[str, Any],
) -> RunConditionInputs:
    if inputs is not None and legacy_kwargs:
        legacy_keys = ", ".join(sorted(legacy_kwargs))
        raise TypeError(
            "run_one_condition() received both inputs=... and legacy keyword arguments: "
            f"{legacy_keys}"
        )
    resolved = inputs or _build_run_condition_inputs_from_legacy_kwargs(legacy_kwargs=legacy_kwargs)
    if resolved.patcher is None:
        raise TypeError("run_one_condition() requires a patcher in inputs.patcher")
    return resolved


def _execute_eval_unit(
    *,
    case: CaseSpec,
    condition: Condition,
    artifacts_root: Path,
    inputs: EvalUnitInputs,
) -> UnitExecutionOutcome:
    from evals.eval_attempt_flow import execute_eval_unit_impl

    return execute_eval_unit_impl(
        case=case,
        condition=condition,
        artifacts_root=artifacts_root,
        inputs=inputs,
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole_seconds = max(0, round(seconds))
    hours, rem = divmod(whole_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _build_run_settings_fingerprint(
    *,
    args: argparse.Namespace,
    cases: list[CaseSpec],
    conditions: list[Condition],
) -> str:
    """Compute a stable xxh3_128 fingerprint of the run-level settings."""
    payload = {
        "cases_dir": str(args.cases_dir),
        "selected_case_ids": [case.case_id for case in cases],
        "conditions": [condition.value for condition in conditions],
        "patcher_config": _build_patcher_config(args=args, patcher_name=str(args.patcher)),
        "record_prompt": str(args.record_prompt),
        "record_evidence": str(args.record_evidence),
        "record_env": sorted(key.strip() for key in args.record_env if str(key).strip()),
        "record_deps": str(args.record_deps),
        "retry_context_mode": str(args.retry_context_mode),
        "retry_history_max_chars": int(args.retry_history_max_chars),
        "prompt_budget_retry_history_chars": int(args.prompt_budget_retry_history_chars),
    }
    return _xxh3_128_jsonlike(payload)


def _load_existing_run_state(
    *,
    results_path: Path,
    run_id: str,
    max_attempts_per_unit: int,
) -> tuple[
    set[tuple[str, str]],
    dict[str, Any] | None,
    str | None,
    int,
    _TrailingResultsTailIssue | None,
]:
    """Scan an existing results JSONL to determine completed units.

    Returns:
        Tuple of (completed_unit_keys, first_run_metadata,
        settings_fingerprint, records_count, trailing_tail_issue).
    """
    if not results_path.exists():
        return set(), None, None, 0, None

    per_unit_stats: dict[tuple[str, str], dict[str, Any]] = {}
    first_run_metadata: dict[str, Any] | None = None
    settings_fingerprint: str | None = None
    records_count = 0
    trailing_tail_issue: _TrailingResultsTailIssue | None = None

    def _consume_record(*, record: dict[str, Any], line_no: int) -> None:
        nonlocal first_run_metadata, settings_fingerprint, records_count

        record_run_id = record.get("run_id")
        if not isinstance(record_run_id, str) or not record_run_id:
            raise ValueError(f"{results_path}:{line_no}: run_id must be a non-empty string")
        if record_run_id != run_id:
            raise ValueError(
                f"{results_path}:{line_no}: run_id mismatch ({record_run_id!r} != {run_id!r})"
            )

        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{results_path}:{line_no}: case_id must be a non-empty string")
        condition = record.get("condition")
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"{results_path}:{line_no}: condition must be a non-empty string")
        record_type = _record_type(record)
        attempt = record.get("attempt")
        if not isinstance(attempt, int) or attempt < 0:
            raise ValueError(f"{results_path}:{line_no}: attempt must be a non-negative integer")
        success = record.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"{results_path}:{line_no}: success must be a boolean")

        run_metadata = record.get("run_metadata")
        if not isinstance(run_metadata, dict):
            raise ValueError(f"{results_path}:{line_no}: run_metadata must be an object")
        if run_metadata.get("run_id") != run_id:
            raise ValueError(f"{results_path}:{line_no}: run_metadata.run_id must match run_id")

        current_fingerprint = run_metadata.get("run_settings_fingerprint")
        if current_fingerprint is not None and (
            not isinstance(current_fingerprint, str) or not current_fingerprint
        ):
            raise ValueError(
                f"{results_path}:{line_no}: run_settings_fingerprint must be a non-empty string"
            )
        if isinstance(current_fingerprint, str):
            if settings_fingerprint is None:
                settings_fingerprint = current_fingerprint
            elif settings_fingerprint != current_fingerprint:
                raise ValueError(
                    f"{results_path}:{line_no}: inconsistent run_settings_fingerprint values"
                )

        if first_run_metadata is None:
            first_run_metadata = dict(run_metadata)

        unit_key = (case_id, condition)
        if record_type == EVAL_RECORD_TYPE_INFRA_EVENT:
            records_count += 1
            return
        if _record_is_infra_resume_eligible(record):
            records_count += 1
            return
        unit_stats = per_unit_stats.setdefault(
            unit_key,
            {"max_attempt": -1, "any_success": False},
        )
        unit_stats["max_attempt"] = max(int(unit_stats["max_attempt"]), attempt)
        if success:
            unit_stats["any_success"] = True
        records_count += 1

    def _parse_record_from_line_bytes(*, line_bytes: bytes, line_no: int) -> dict[str, Any]:
        try:
            line_text = line_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{results_path}:{line_no}: invalid UTF-8 ({type(exc).__name__}: {exc})"
            ) from exc
        try:
            record = json.loads(line_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{results_path}:{line_no}: invalid JSON ({type(exc).__name__}: {exc})"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"{results_path}:{line_no}: record must be a JSON object")
        return record

    pending_line_no: int | None = None
    pending_line_start_offset: int = 0
    pending_line_bytes: bytes | None = None
    offset_bytes = 0
    with results_path.open("rb") as handle:
        for line_no, line_bytes in enumerate(handle, start=1):
            line_start_offset = offset_bytes
            offset_bytes += len(line_bytes)
            if not line_bytes.strip():
                continue
            if pending_line_no is not None and pending_line_bytes is not None:
                record = _parse_record_from_line_bytes(
                    line_bytes=pending_line_bytes,
                    line_no=pending_line_no,
                )
                _consume_record(record=record, line_no=pending_line_no)
            pending_line_no = line_no
            pending_line_start_offset = line_start_offset
            pending_line_bytes = line_bytes

    if pending_line_no is not None and pending_line_bytes is not None:
        try:
            trailing_record = _parse_record_from_line_bytes(
                line_bytes=pending_line_bytes,
                line_no=pending_line_no,
            )
        except ValueError as exc:
            error_text = str(exc)
            is_parse_error = "invalid JSON" in error_text or "invalid UTF-8" in error_text
            if not is_parse_error:
                raise
            trailing_tail_issue = _TrailingResultsTailIssue(
                line_no=pending_line_no,
                truncate_offset_bytes=pending_line_start_offset,
                raw_line_bytes=pending_line_bytes,
                parse_error=error_text,
            )
        else:
            _consume_record(record=trailing_record, line_no=pending_line_no)

    required_attempts = max(1, int(max_attempts_per_unit))
    completed_units = {
        unit_key
        for unit_key, stats in per_unit_stats.items()
        if bool(stats.get("any_success")) or int(stats.get("max_attempt", -1)) >= required_attempts
    }
    return (
        completed_units,
        first_run_metadata,
        settings_fingerprint,
        records_count,
        trailing_tail_issue,
    )


def _repair_trailing_results_tail(
    *,
    results_path: Path,
    run_id: str,
    issue: _TrailingResultsTailIssue,
) -> tuple[Path, int]:
    """Truncate a corrupt trailing record and archive the removed bytes.

    Returns:
        Tuple of (sidecar_backup_path, removed_bytes_count).
    """
    original_size = int(results_path.stat().st_size)
    truncate_offset = int(issue.truncate_offset_bytes)
    if truncate_offset < 0 or truncate_offset > original_size:
        raise OSError(f"invalid truncate offset {truncate_offset} for file size {original_size}")
    with results_path.open("rb") as source:
        source.seek(truncate_offset)
        removed_tail_bytes = source.read()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sidecar_path = results_path.with_name(f"{results_path.name}.corrupt-tail-{timestamp}.log")
    suffix_index = 1
    while sidecar_path.exists():
        sidecar_path = results_path.with_name(
            f"{results_path.name}.corrupt-tail-{timestamp}.{suffix_index}.log"
        )
        suffix_index += 1

    header = (
        "# llmdebug eval trailing-tail repair backup\n"
        f"run_id={run_id}\n"
        f"results_path={results_path}\n"
        f"line_no={issue.line_no}\n"
        f"truncate_offset_bytes={truncate_offset}\n"
        f"original_size_bytes={original_size}\n"
        f"parse_error={issue.parse_error}\n\n"
    ).encode("utf-8", errors="strict")
    with sidecar_path.open("wb") as sidecar:
        sidecar.write(header)
        sidecar.write(removed_tail_bytes)

    with results_path.open("r+b") as handle:
        handle.truncate(truncate_offset)
        handle.flush()

    removed_bytes = original_size - truncate_offset
    return sidecar_path, removed_bytes


def load_cases(cases_dir: Path) -> list[CaseSpec]:
    """Load and validate all eval cases from *cases_dir*.

    Each subdirectory must contain a ``case.json`` with required fields
    (category, difficulty, snapshot_dependency, expected_exception).

    Raises:
        ValueError: If any case metadata is missing or invalid.
    """
    if not cases_dir.exists():
        return []

    out: list[CaseSpec] = []
    for child in sorted(cases_dir.iterdir()):
        if not child.is_dir():
            continue
        case_id = child.name
        meta_path = child / "case.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"{case_id}: invalid JSON in case.json: {e}") from e
            except OSError as e:
                raise ValueError(f"{case_id}: cannot read case.json: {e}") from e

        category = meta.get("category")
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            raise ValueError(f"{case_id}: category must be one of {sorted(VALID_CATEGORIES)}")

        difficulty = meta.get("difficulty")
        if not isinstance(difficulty, str) or difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"{case_id}: difficulty must be one of {sorted(VALID_DIFFICULTIES)}")

        snapshot_dependency = meta.get("snapshot_dependency")
        if (
            not isinstance(snapshot_dependency, str)
            or snapshot_dependency not in VALID_SNAPSHOT_DEPENDENCIES
        ):
            raise ValueError(
                f"{case_id}: snapshot_dependency must be one of {sorted(VALID_SNAPSHOT_DEPENDENCIES)}"
            )

        expected_exception = meta.get("expected_exception")
        if not isinstance(expected_exception, str) or not expected_exception:
            raise ValueError(f"{case_id}: expected_exception must be a non-empty string")

        tags_raw = meta.get("tags")
        if tags_raw is None:
            tags = []
        elif isinstance(tags_raw, list) and all(isinstance(x, str) and x for x in tags_raw):
            tags = list(tags_raw)
        else:
            raise ValueError(f"{case_id}: tags must be an array of non-empty strings")

        python_args_raw = meta.get("python_args")
        if (
            not isinstance(python_args_raw, list)
            or not python_args_raw
            or not all(isinstance(x, str) and x for x in python_args_raw)
        ):
            python_args = ["-m", "pytest", "-q"]
        else:
            python_args = list(python_args_raw)

        timeout_sec_raw = meta.get("timeout_sec")
        if not isinstance(timeout_sec_raw, int) or timeout_sec_raw <= 0:
            timeout_sec = 120
        else:
            timeout_sec = timeout_sec_raw

        description = meta.get("description") if isinstance(meta.get("description"), str) else None

        # Extended optional fields for imported cases.
        bug_source = meta.get("bug_source") if isinstance(meta.get("bug_source"), str) else None
        contamination_risk = (
            meta.get("contamination_risk")
            if isinstance(meta.get("contamination_risk"), str)
            else None
        )
        expected_fix_lines_raw = meta.get("expected_fix_lines")
        expected_fix_lines = (
            expected_fix_lines_raw
            if isinstance(expected_fix_lines_raw, int) and expected_fix_lines_raw > 0
            else None
        )
        source_id = meta.get("source_id") if isinstance(meta.get("source_id"), str) else None
        difficulty_rationale = (
            meta.get("difficulty_rationale")
            if isinstance(meta.get("difficulty_rationale"), str)
            else None
        )

        out.append(
            CaseSpec(
                case_id=case_id,
                path=child,
                category=category,
                difficulty=difficulty,
                snapshot_dependency=snapshot_dependency,
                expected_exception=expected_exception,
                tags=tags,
                python_args=python_args,
                timeout_sec=timeout_sec,
                description=description,
                bug_source=bug_source,
                contamination_risk=contamination_risk,
                expected_fix_lines=expected_fix_lines,
                source_id=source_id,
                difficulty_rationale=difficulty_rationale,
            )
        )
    return out


def make_patcher(args: argparse.Namespace):
    """Instantiate the patcher backend selected by ``--patcher``.

    Returns one of ``OpenAICompatPatcher``, ``HeuristicSnapshotPatcher``,
    ``NullPatcher``, ``CommandPatcher``, or ``BaselinePatcher`` depending
    on *args.patcher*.
    """
    from evals.patchers import CommandPatcher, HeuristicSnapshotPatcher, NullPatcher
    from evals.patchers.command_patcher import patcher_from_env

    try:
        _normalize_baseline_selection(args=args, config_state=None)
        _normalize_adaptive_controller_lambda_rlm_settings(args=args, config_state=None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.patcher == "baseline":
        from evals.patchers.baseline_patcher import build_baseline_patcher

        baseline_name = (getattr(args, "baseline_name", "") or "").strip()
        if not baseline_name:
            raise SystemExit(
                "--baseline-name is required for --patcher baseline. "
                "Available: ldb, intervenor, tracecoder, lambda_rlm, rlm."
            )
        repo_dir = (getattr(args, "baseline_repo_dir", "") or "").strip()
        if not repo_dir:
            raise SystemExit(
                "--baseline-repo-dir is required for --patcher baseline. "
                "Point to the cloned baseline repository."
            )
        baseline_python = (getattr(args, "baseline_python", "") or "").strip()
        try:
            repo_dir, baseline_python = _resolve_external_baseline_runtime_requirements(
                baseline_name=baseline_name,
                baseline_repo_dir=repo_dir,
                baseline_python=baseline_python,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        args.baseline_repo_dir = repo_dir
        args.baseline_python = baseline_python
        api_key = (args.openai_api_key or "").strip() or ""
        baseline_extra = _baseline_extra_from_args(args=args, baseline_name=baseline_name)
        baseline_process_timeout_raw = getattr(args, "baseline_process_timeout_sec", None)
        baseline_process_timeout = (
            float(baseline_process_timeout_raw)
            if isinstance(baseline_process_timeout_raw, (int, float))
            and float(baseline_process_timeout_raw) > 0.0
            else float(getattr(args, "openai_timeout_sec", 300.0))
        )
        baseline_max_iterations = getattr(args, "baseline_max_iterations", None)
        return build_baseline_patcher(
            baseline_name=baseline_name,
            repo_dir=repo_dir,
            python_executable=baseline_python,
            model=str(args.openai_model or ""),
            api_base_url=str(args.openai_base_url),
            api_key=api_key,
            max_attempts=(
                int(baseline_max_iterations) if baseline_max_iterations is not None else 2
            ),
            api_timeout_sec=float(getattr(args, "openai_timeout_sec", 300.0)),
            process_timeout_sec=baseline_process_timeout,
            timeout_sec=None,
            extra=baseline_extra,
            strict_extra_config=True,
        )

    if args.patcher == "openai":
        from evals.patchers import OpenAICompatPatcher

        model = (args.openai_model or "").strip()
        if not model:
            raise SystemExit(
                "--openai-model is required for --patcher openai (or set OPENAI_MODEL)."
            )
        api_key = (args.openai_api_key or "").strip() or None
        compat_settings = _native_openai_compat_settings_from_args(args=args)
        return OpenAICompatPatcher(
            base_url=str(args.openai_base_url),
            model=model,
            api_key=api_key,
            timeout_sec=float(args.openai_timeout_sec),
            http2=bool(getattr(args, "openai_http2", True)),
            transport_max_retries=int(args.openai_transport_max_retries),
            transport_retry_backoff_base_sec=float(args.openai_transport_retry_backoff_base_sec),
            transport_retry_backoff_max_sec=float(args.openai_transport_retry_backoff_max_sec),
            max_propose_sec=float(args.openai_max_propose_sec),
            max_tokens=int(args.openai_max_tokens),
            plan_patch_max_tokens=int(getattr(args, "openai_plan_patch_max_tokens", 768)),
            temperature=float(args.openai_temperature),
            compat_profile=str(compat_settings["openai_compat_profile"]),
            allow_reasoning_fallback=bool(compat_settings["openai_allow_reasoning_fallback"]),
            enable_thinking=bool(compat_settings["openai_enable_thinking"]),
            response_mode=str(args.openai_response_mode),
            force_patch_attempt=bool(args.openai_force_patch_attempt),
            seed=args.seed if bool(args.deterministic) else None,
        )

    if args.patcher == "heuristic":
        return HeuristicSnapshotPatcher()
    if args.patcher == "null":
        return NullPatcher()

    if args.patch_cmd:
        return CommandPatcher(
            cmd=str(args.patch_cmd),
            shell=bool(args.patch_cmd_shell),
            timeout_sec=float(args.patch_cmd_timeout_sec),
        )
    env_patcher = patcher_from_env()
    if env_patcher is None:
        raise SystemExit(
            "Command patcher selected but no command configured. "
            "Set --patch-cmd or LLMDEBUG_EVAL_PATCH_CMD."
        )
    return env_patcher


def _resolve_openai_context_window_for_run(
    *,
    args: argparse.Namespace,
    patcher: Any,
) -> tuple[int, dict[str, Any]]:
    """Resolve the effective OpenAI context-window size for the run.

    Combines manual ``--openai-context-window-tokens`` with optional
    automatic probing.  Returns (effective_tokens, info_dict).
    """
    manual_tokens = max(0, int(getattr(args, "openai_context_window_tokens", 0)))
    info: dict[str, Any] = {
        "openai_context_window_tokens_effective": manual_tokens,
        "openai_context_window_source": "manual_flag" if manual_tokens > 0 else "disabled",
        "openai_context_window_probe_provider": "",
        "openai_context_window_probe_error": "",
    }
    if manual_tokens > 0:
        return manual_tokens, info

    if str(getattr(args, "patcher", "")) != "openai":
        return 0, info
    if str(getattr(patcher, "name", "")) != "openai":
        return 0, info
    if not bool(getattr(args, "openai_auto_context_window", False)):
        return 0, info

    probe_context_window = getattr(patcher, "probe_context_window_tokens", None)
    if not callable(probe_context_window):
        info["openai_context_window_source"] = "auto_probe_unavailable"
        info["openai_context_window_probe_error"] = "patcher_missing_probe_context_window_tokens"
        return 0, info

    try:
        probe_tokens_raw, probe_meta_raw = probe_context_window()
    except Exception as exc:
        info["openai_context_window_source"] = "auto_probe_failed"
        info["openai_context_window_probe_error"] = f"{type(exc).__name__}: {exc}"
        return 0, info

    probe_meta = dict(probe_meta_raw) if isinstance(probe_meta_raw, dict) else {}
    provider = str(
        probe_meta.get("context_window_probe_provider") or probe_meta.get("provider") or ""
    )
    source = str(probe_meta.get("context_window_source") or "auto_probe")
    probe_error = str(probe_meta.get("context_window_probe_error") or probe_meta.get("error") or "")
    info["openai_context_window_probe_provider"] = provider

    try:
        probe_tokens = int(probe_tokens_raw)
    except (TypeError, ValueError):
        probe_tokens = 0
    if probe_tokens > 0:
        info["openai_context_window_tokens_effective"] = probe_tokens
        info["openai_context_window_source"] = source
        info["openai_context_window_probe_error"] = ""
        return probe_tokens, info

    info["openai_context_window_source"] = "auto_probe_failed"
    info["openai_context_window_probe_error"] = probe_error or "probe_returned_no_context_window"
    return 0, info


from evals.adaptive_controller_packet import remove_adaptive_controller_sidecars  # noqa: E402
from evals.adaptive_controller_runtime import (  # noqa: E402
    ControllerRuntimeInputs,
    prepare_controller_runtime,
    write_controller_sidecars_for_cycle,
)
from evals.adaptive_history import persist_adaptive_attempt_history  # noqa: E402
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
from evals.lambda_rlm_backend import (  # noqa: E402
    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
    DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
    default_lambda_rlm_repo_dir,
    invoke_lambda_rlm_decomposition,
)
from evals.patchers.baseline_patcher import resolve_effective_baseline_model  # noqa: E402


def _request_context_failure_kind_from_meta(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return ""
    explicit = str(meta.get("request_context_failure_kind") or "").strip()
    if explicit:
        return explicit
    error = str(meta.get("error") or "").strip()
    if error == "request_budget_exceeded":
        return "request_budget_exceeded"
    if error == "openai_compat_reasoning_only":
        return "compat_reasoning_only"
    if bool(meta.get("request_context_backend_unavailable")):
        return "backend_unavailable"
    if error == "invalid_request_context_payload":
        return "invalid_payload"
    if error == "request_failed":
        return "transport"
    if error == "http_error":
        status = _nonnegative_int(meta.get("status"), default=0)
        if status in {502, 503, 504}:
            return "backend_unavailable"
        return "transport"
    return ""


def _request_context_recovery_failure_kind_from_meta(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return ""
    error = str(meta.get("error") or "").strip()
    if error == "request_failed":
        return "transport"
    if error == "http_error":
        status = _nonnegative_int(meta.get("status"), default=0)
        if status in {502, 503, 504}:
            return "backend_unavailable"
        if _is_retryable_transport_http_status(status):
            return "transport"
    return ""


def _request_context_retry_counts_from_meta(meta: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(meta, dict):
        return 0, 0
    retry_count = _nonnegative_int(meta.get("request_context_retry_count"), default=0)
    transport_retry_count = _nonnegative_int(
        meta.get("request_context_transport_retry_count"),
        default=0,
    )
    return retry_count, transport_retry_count


def _adaptive_helper_outcome_for_attempt(
    *,
    helper_invoked: bool,
    helper_materialized_context: bool,
    helper_decision_reason: str,
    stage_a_meta: dict[str, Any] | None,
) -> tuple[str, str]:
    decision_reason = str(helper_decision_reason or "").strip()
    failure_kind = _request_context_failure_kind_from_meta(stage_a_meta)
    if helper_invoked:
        if helper_materialized_context:
            return ADAPTIVE_HELPER_OUTCOME_MATERIALIZED, failure_kind
        if failure_kind == "compat_reasoning_only":
            return ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY, failure_kind
        if failure_kind == "backend_unavailable":
            return ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND, failure_kind
        if failure_kind == "transport":
            return ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED, failure_kind
        if failure_kind == "request_budget_exceeded":
            return ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED, failure_kind
        if failure_kind == "invalid_payload" or decision_reason == "llm_invalid_response_no_helper":
            return ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD, failure_kind or "invalid_payload"
        return ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT, failure_kind

    if decision_reason == "request_context_unavailable":
        return ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED, ""
    if decision_reason in {
        "informativeness_pregate_skip",
        "helper_budget_exhausted",
        "helper_saturation_guard",
    }:
        return ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL, ""
    if decision_reason in {"not_adaptive_condition", ""}:
        return ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED, ""
    return ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY, ""


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, coerced)


def _optional_nonnegative_int(value: Any) -> int | None:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, coerced)


def _usage_token_counts_from_usage_payload(usage: Any) -> tuple[int, int, int]:
    if not isinstance(usage, dict):
        return 0, 0, 0

    prompt_tokens = _nonnegative_int(usage.get("prompt_tokens"), default=0)
    completion_tokens = _nonnegative_int(usage.get("completion_tokens"), default=0)
    # Anthropic-compatible payloads often use input/output token fields.
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _nonnegative_int(usage.get("input_tokens"), default=0)
        completion_tokens = _nonnegative_int(usage.get("output_tokens"), default=0)

    total_tokens_raw = usage.get("total_tokens")
    if total_tokens_raw is None:
        total_tokens = prompt_tokens + completion_tokens
    else:
        total_tokens = _nonnegative_int(total_tokens_raw, default=prompt_tokens + completion_tokens)
    return prompt_tokens, completion_tokens, total_tokens


def _reasoning_token_counts_from_usage_payload(usage: Any) -> tuple[int, bool]:
    if not isinstance(usage, dict):
        return 0, False
    candidates = [
        usage,
        usage.get("output_tokens_details"),
        usage.get("completion_tokens_details"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict) or "reasoning_tokens" not in candidate:
            continue
        parsed = _optional_nonnegative_int(candidate.get("reasoning_tokens"))
        if parsed is not None:
            return parsed, True
    return 0, False


def _usage_payload_from_counts(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    reasoning_tokens: int | None = None,
) -> dict[str, int]:
    out = {
        "prompt_tokens": _nonnegative_int(prompt_tokens, default=0),
        "completion_tokens": _nonnegative_int(completion_tokens, default=0),
        "total_tokens": _nonnegative_int(
            total_tokens,
            default=_nonnegative_int(prompt_tokens, default=0)
            + _nonnegative_int(completion_tokens, default=0),
        ),
    }
    if reasoning_tokens is not None:
        out["reasoning_tokens"] = _nonnegative_int(reasoning_tokens, default=0)
    return out


def _token_usage_blocked_from_meta(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    explicit = meta.get("token_usage_available")
    if isinstance(explicit, bool):
        return not explicit
    usage_available = meta.get("usage_available")
    if isinstance(usage_available, bool):
        return not usage_available
    baseline_metrics = meta.get("baseline_metrics")
    if not isinstance(baseline_metrics, dict):
        return False
    baseline_usage_available = baseline_metrics.get("usage_available")
    return isinstance(baseline_usage_available, bool) and not baseline_usage_available


def _usage_token_counts_from_patch_meta(patch_meta: dict[str, Any] | None) -> tuple[int, int, int]:
    if not isinstance(patch_meta, dict):
        return 0, 0, 0
    if _token_usage_blocked_from_meta(patch_meta):
        return 0, 0, 0

    usage_prompt, usage_completion, usage_total = _usage_token_counts_from_usage_payload(
        patch_meta.get("usage")
    )
    baseline_metrics = _baseline_metrics_from_patch_meta(patch_meta)
    reported_prompt = 0
    reported_completion = 0
    reported_total = 0
    baseline_usage_available = False
    if isinstance(baseline_metrics, dict):
        baseline_usage_available = bool(baseline_metrics.get("usage_available"))
        reported_prompt = _nonnegative_int(
            baseline_metrics.get("reported_tokens_prompt"),
            default=0,
        )
        reported_completion = _nonnegative_int(
            baseline_metrics.get("reported_tokens_completion"),
            default=0,
        )
        reported_total = _nonnegative_int(
            baseline_metrics.get("reported_tokens_total"),
            default=reported_prompt + reported_completion,
        )
    if (
        baseline_usage_available
        and usage_prompt == 0
        and usage_completion == 0
        and usage_total == 0
        and (reported_prompt > 0 or reported_completion > 0 or reported_total > 0)
    ):
        usage_prompt, usage_completion, usage_total = (
            reported_prompt,
            reported_completion,
            reported_total,
        )
    prompt_tokens = _nonnegative_int(patch_meta.get("tokens_prompt"), default=usage_prompt)
    completion_tokens = _nonnegative_int(
        patch_meta.get("tokens_completion"),
        default=usage_completion,
    )
    total_tokens = _nonnegative_int(
        patch_meta.get("tokens_total"),
        default=usage_total if usage_total > 0 else (prompt_tokens + completion_tokens),
    )
    return prompt_tokens, completion_tokens, total_tokens


def _reasoning_token_counts_from_patch_meta(
    patch_meta: dict[str, Any] | None,
) -> tuple[int, bool]:
    if not isinstance(patch_meta, dict):
        return 0, False
    if _token_usage_blocked_from_meta(patch_meta):
        return 0, False

    explicit = patch_meta.get("reasoning_token_usage_available")
    usage_reasoning, usage_available = _reasoning_token_counts_from_usage_payload(
        patch_meta.get("usage")
    )
    baseline_reasoning = 0
    baseline_available = False
    baseline_metrics = _baseline_metrics_from_patch_meta(patch_meta)
    if isinstance(baseline_metrics, dict) and bool(baseline_metrics.get("usage_available")):
        reported = _optional_nonnegative_int(baseline_metrics.get("reported_tokens_reasoning"))
        if reported is not None:
            baseline_reasoning = reported
            baseline_available = True

    top_level_raw = patch_meta.get("tokens_reasoning")
    top_level_present = "tokens_reasoning" in patch_meta
    top_level_reasoning = _nonnegative_int(
        top_level_raw,
        default=usage_reasoning if usage_available else baseline_reasoning,
    )
    if isinstance(explicit, bool):
        return (top_level_reasoning if explicit else 0), explicit
    if top_level_present:
        return top_level_reasoning, True
    if usage_available:
        return usage_reasoning, True
    if baseline_available:
        return baseline_reasoning, True
    return 0, False


def _reasoning_usage_segment_from_meta(
    meta: dict[str, Any] | None,
    *,
    default_llm_calls: int = 0,
) -> tuple[int, bool, bool]:
    if not isinstance(meta, dict):
        if _nonnegative_int(default_llm_calls, default=0) > 0:
            return 0, True, False
        return 0, False, False
    llm_calls = _llm_call_count_from_meta(meta, default=default_llm_calls)
    usage_available = _token_usage_available_from_meta(meta)
    prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(meta)
    applicable = bool(
        llm_calls > 0
        or usage_available
        or prompt_tokens > 0
        or completion_tokens > 0
        or total_tokens > 0
    )
    if not applicable:
        return 0, False, False
    reasoning_tokens, reasoning_available = _reasoning_token_counts_from_patch_meta(meta)
    return reasoning_tokens, True, reasoning_available


def _combine_reasoning_usage_segments(
    *segments: tuple[dict[str, Any] | None, int],
) -> tuple[int, bool]:
    total_reasoning_tokens = 0
    any_applicable = False
    full_coverage = True
    for meta, default_llm_calls in segments:
        reasoning_tokens, applicable, available = _reasoning_usage_segment_from_meta(
            meta,
            default_llm_calls=default_llm_calls,
        )
        if not applicable:
            continue
        any_applicable = True
        if available:
            total_reasoning_tokens += reasoning_tokens
        else:
            full_coverage = False
    return total_reasoning_tokens, bool(any_applicable and full_coverage)


def _completion_chars_from_response(
    *,
    diff: str | None = None,
    patch_meta: dict[str, Any] | None = None,
) -> int:
    if not isinstance(patch_meta, dict):
        return max(0, len(diff)) if isinstance(diff, str) else 0

    for key in (
        "completion_chars_total",
        "response_output_chars_total",
        "response_output_chars",
        "chars_output",
        "patch_diff_chars",
    ):
        value = patch_meta.get(key)
        if isinstance(value, int):
            return _nonnegative_int(value, default=0)

    if isinstance(diff, str):
        return max(0, len(diff))

    raw_response = patch_meta.get("raw_response")
    if isinstance(raw_response, str):
        return max(0, len(raw_response))
    response = patch_meta.get("response")
    if isinstance(response, str):
        return max(0, len(response))
    return 0


def _prompt_chars_from_patch_meta(patch_meta: dict[str, Any] | None) -> int:
    if not isinstance(patch_meta, dict):
        return 0
    for key in (
        "prompt_chars_total",
        "response_input_chars_total",
        "response_input_chars",
        "chars_input",
    ):
        value = patch_meta.get(key)
        if isinstance(value, int):
            return _nonnegative_int(value, default=0)
    return 0


def _patch_attempt_char_deltas(
    *,
    cycle_prompt: str,
    diff: str | None,
    patch_meta: dict[str, Any] | None,
    is_baseline_patcher: bool,
) -> tuple[int, int]:
    if is_baseline_patcher:
        return (
            _prompt_chars_from_patch_meta(patch_meta),
            _completion_chars_from_response(diff=None, patch_meta=patch_meta),
        )
    return len(cycle_prompt), _completion_chars_from_response(diff=diff, patch_meta=patch_meta)


def _baseline_metrics_from_patch_meta(patch_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(patch_meta, dict):
        return None
    baseline_metrics = patch_meta.get("baseline_metrics")
    return baseline_metrics if isinstance(baseline_metrics, dict) else None


def _baseline_metric_float(
    patch_meta: dict[str, Any] | None,
    key: str,
) -> float | None:
    baseline_metrics = _baseline_metrics_from_patch_meta(patch_meta)
    if baseline_metrics is None:
        return None
    value = baseline_metrics.get(key)
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, float(value))


def _llm_call_count_from_meta(meta: dict[str, Any] | None, *, default: int = 0) -> int:
    if not isinstance(meta, dict):
        return max(0, int(default))
    value = meta.get("llm_call_count")
    if isinstance(value, int) and value >= 0:
        return value
    return max(0, int(default))


def _usage_available_flag_from_meta(meta: dict[str, Any] | None) -> bool | None:
    if not isinstance(meta, dict):
        return None
    usage_available = meta.get("usage_available")
    if isinstance(usage_available, bool):
        return usage_available
    baseline_metrics = _baseline_metrics_from_patch_meta(meta)
    if baseline_metrics is None:
        return None
    baseline_usage_available = baseline_metrics.get("usage_available")
    return baseline_usage_available if isinstance(baseline_usage_available, bool) else None


def _token_usage_available_from_meta(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    explicit = meta.get("token_usage_available")
    prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(meta)
    has_counts = bool(prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0)
    baseline_metrics = _baseline_metrics_from_patch_meta(meta)
    reported_has_counts = False
    if isinstance(baseline_metrics, dict):
        reported_prompt = _nonnegative_int(
            baseline_metrics.get("reported_tokens_prompt"), default=0
        )
        reported_completion = _nonnegative_int(
            baseline_metrics.get("reported_tokens_completion"),
            default=0,
        )
        reported_total = _nonnegative_int(
            baseline_metrics.get("reported_tokens_total"),
            default=reported_prompt + reported_completion,
        )
        reported_has_counts = bool(
            reported_prompt > 0 or reported_completion > 0 or reported_total > 0
        )
    if isinstance(explicit, bool):
        if not explicit:
            return False
        return has_counts or reported_has_counts
    usage_available = _usage_available_flag_from_meta(meta)
    if isinstance(usage_available, bool):
        if not usage_available:
            return False
        return has_counts or reported_has_counts
    usage = meta.get("usage")
    if not isinstance(usage, dict):
        return False
    prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_usage_payload(usage)
    return bool(prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0)


def _timeout_error_from_meta(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    error = meta.get("error")
    if isinstance(error, str) and error.strip().lower() == "timeout":
        return True
    baseline_meta = meta.get("baseline_meta")
    return _timeout_error_from_meta(baseline_meta) if isinstance(baseline_meta, dict) else False


def _record_token_usage_state(
    *,
    attempt_meta: dict[str, Any] | None,
    patch_meta: dict[str, Any] | None,
    prompt_tokens: int,
    total_tokens: int,
) -> str:
    if prompt_tokens > 0 or total_tokens > 0:
        return "available"
    llm_call_count = max(
        _llm_call_count_from_meta(attempt_meta), _llm_call_count_from_meta(patch_meta)
    )
    timeout_without_usage = _timeout_error_from_meta(attempt_meta) or _timeout_error_from_meta(
        patch_meta
    )
    for meta in (attempt_meta, patch_meta):
        if _usage_available_flag_from_meta(meta) is True:
            return "available"
    for meta in (attempt_meta, patch_meta):
        if _usage_available_flag_from_meta(meta) is False:
            return (
                "unavailable" if llm_call_count > 0 or timeout_without_usage else "not_applicable"
            )
    if llm_call_count > 0 or timeout_without_usage:
        return "unavailable"
    return "not_applicable"


def _format_progress_token_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    available_records: int,
    unavailable_records: int,
) -> str:
    if available_records <= 0:
        return "tokens=unavailable" if unavailable_records > 0 else "tokens=n/a"
    rendered = f"tokens=prompt:{prompt_tokens},completion:{completion_tokens},total:{total_tokens}"
    if unavailable_records > 0:
        rendered += ",coverage=partial"
    return rendered


def _aggregate_attempt_llm_usage(
    *,
    stage_a_meta: dict[str, Any] | None,
    patch_meta: dict[str, Any] | None,
    stage_a_default_llm_calls: int,
    patch_default_llm_calls: int,
) -> tuple[int, int, int, int]:
    """Sum LLM call counts and token usage across Stage-A and patch steps."""
    stage_calls = _llm_call_count_from_meta(stage_a_meta, default=stage_a_default_llm_calls)
    patch_calls = _llm_call_count_from_meta(patch_meta, default=patch_default_llm_calls)
    stage_prompt, stage_completion, stage_total = _usage_token_counts_from_patch_meta(stage_a_meta)
    patch_prompt, patch_completion, patch_total = _usage_token_counts_from_patch_meta(patch_meta)
    return (
        stage_calls + patch_calls,
        stage_prompt + patch_prompt,
        stage_completion + patch_completion,
        stage_total + patch_total,
    )


def _apply_usage_totals_to_patch_meta(
    *,
    patch_meta: dict[str, Any],
    llm_call_count: int,
    tokens_prompt: int,
    tokens_completion: int,
    tokens_total: int,
    tokens_reasoning: int | None = None,
    reasoning_token_usage_available: bool | None = None,
) -> None:
    normalized_calls = _nonnegative_int(llm_call_count, default=0)
    token_usage_blocked = _token_usage_blocked_from_meta(patch_meta)
    if token_usage_blocked:
        normalized_prompt = 0
        normalized_completion = 0
        normalized_total = 0
        normalized_reasoning = 0
        normalized_reasoning_available = False
    else:
        normalized_prompt = _nonnegative_int(tokens_prompt, default=0)
        normalized_completion = _nonnegative_int(tokens_completion, default=0)
        normalized_total = _nonnegative_int(
            tokens_total,
            default=normalized_prompt + normalized_completion,
        )
        if reasoning_token_usage_available is None:
            normalized_reasoning, normalized_reasoning_available = (
                _reasoning_token_counts_from_patch_meta(patch_meta)
            )
        else:
            normalized_reasoning = _nonnegative_int(tokens_reasoning, default=0)
            normalized_reasoning_available = bool(reasoning_token_usage_available)
        if not normalized_reasoning_available:
            normalized_reasoning = 0
    patch_meta["llm_call_count"] = normalized_calls
    patch_meta["tokens_prompt"] = normalized_prompt
    patch_meta["tokens_completion"] = normalized_completion
    patch_meta["tokens_total"] = normalized_total
    patch_meta["tokens_reasoning"] = normalized_reasoning
    patch_meta["reasoning_token_usage_available"] = normalized_reasoning_available
    if token_usage_blocked:
        patch_meta.pop("usage", None)
        return
    patch_meta["usage"] = _usage_payload_from_counts(
        prompt_tokens=normalized_prompt,
        completion_tokens=normalized_completion,
        total_tokens=normalized_total,
        reasoning_tokens=normalized_reasoning if normalized_reasoning_available else None,
    )


def _sync_attempt_usage_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    llm_call_count: int,
) -> None:
    token_usage_available = _token_usage_available_from_meta(patch_meta)
    if token_usage_available:
        prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(
            patch_meta
        )
        reasoning_tokens, reasoning_token_usage_available = _reasoning_token_counts_from_patch_meta(
            patch_meta
        )
    else:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0
        reasoning_token_usage_available = False
    normalized_calls = _nonnegative_int(llm_call_count, default=0)
    attempt_metadata["llm_call_count"] = normalized_calls
    attempt_metadata["token_usage_available"] = token_usage_available
    attempt_metadata["tokens_prompt"] = prompt_tokens
    attempt_metadata["tokens_completion"] = completion_tokens
    attempt_metadata["tokens_total"] = total_tokens
    attempt_metadata["tokens_reasoning"] = reasoning_tokens
    attempt_metadata["reasoning_token_usage_available"] = reasoning_token_usage_available
    attempt_metadata["tokens_input"] = prompt_tokens
    attempt_metadata["tokens_output"] = completion_tokens
    patch_meta["llm_call_count"] = normalized_calls
    patch_meta["token_usage_available"] = token_usage_available
    patch_meta["tokens_prompt"] = prompt_tokens
    patch_meta["tokens_completion"] = completion_tokens
    patch_meta["tokens_total"] = total_tokens
    patch_meta["tokens_reasoning"] = reasoning_tokens
    patch_meta["reasoning_token_usage_available"] = reasoning_token_usage_available
    patch_meta["tokens_input"] = prompt_tokens
    patch_meta["tokens_output"] = completion_tokens


def _sync_attempt_prompt_char_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    prompt_chars_total: int,
    completion_chars_total: int = 0,
) -> None:
    normalized_prompt = _nonnegative_int(prompt_chars_total, default=0)
    normalized_completion = _nonnegative_int(completion_chars_total, default=0)
    normalized_total = normalized_prompt + normalized_completion
    attempt_metadata["prompt_chars_total"] = normalized_prompt
    attempt_metadata["completion_chars_total"] = normalized_completion
    attempt_metadata["chars_input"] = normalized_prompt
    attempt_metadata["chars_output"] = normalized_completion
    attempt_metadata["chars_total"] = normalized_total
    patch_meta["prompt_chars_total"] = normalized_prompt
    patch_meta["completion_chars_total"] = normalized_completion
    patch_meta["chars_input"] = normalized_prompt
    patch_meta["chars_output"] = normalized_completion
    patch_meta["chars_total"] = normalized_total


def _prompt_chars_for_llm_calls(prompt: str | None, *, llm_call_count: int) -> int:
    if not isinstance(prompt, str) or not prompt:
        return 0
    return len(prompt) * _nonnegative_int(llm_call_count, default=0)


def _clip_patcher_meta_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _summarize_context_compaction_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "llm_call_count",
        "compact_context_duration_sec",
        "compact_context_target_chars",
        "compact_context_input_chars",
        "compact_context_output_chars",
        "compact_context_output_truncated",
        "api_compat_seed_requested",
        "api_compat_seed_acknowledged",
        "api_compat_seed_response",
    ):
        if key in meta:
            out[key] = meta[key]
    status_value = meta.get("status")
    if isinstance(status_value, int):
        out["status"] = status_value
    usage = meta.get("usage")
    if isinstance(usage, dict):
        prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_usage_payload(
            usage
        )
        reasoning_tokens, reasoning_available = _reasoning_token_counts_from_usage_payload(usage)
        out["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if reasoning_available:
            out["usage"]["reasoning_tokens"] = reasoning_tokens
    for key in ("error", "exc", "reason", "fallback_reason"):
        text = _clip_patcher_meta_text(meta.get(key), max_chars=240)
        if text:
            out[key] = text
    body_text = _clip_patcher_meta_text(meta.get("body"), max_chars=320)
    if body_text:
        out["body"] = body_text
    return out


def _summarize_recovery_usage_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "llm_call_count",
        "api_compat_seed_requested",
        "api_compat_seed_acknowledged",
        "api_compat_seed_response",
    ):
        if key in meta:
            out[key] = meta[key]
    status_value = meta.get("status")
    if isinstance(status_value, int):
        out["status"] = status_value
    if any(
        key in meta
        for key in (
            "usage",
            "tokens_prompt",
            "tokens_completion",
            "tokens_total",
            "tokens_reasoning",
        )
    ):
        prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(meta)
        reasoning_tokens, reasoning_available = _reasoning_token_counts_from_patch_meta(meta)
        out["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if reasoning_available:
            out["usage"]["reasoning_tokens"] = reasoning_tokens
    for key in ("error", "exc", "reason", "fallback_reason"):
        text = _clip_patcher_meta_text(meta.get(key), max_chars=240)
        if text:
            out[key] = text
    body_text = _clip_patcher_meta_text(meta.get("body"), max_chars=320)
    if body_text:
        out["body"] = body_text
    return out


def _sync_syntax_preflight_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    enabled: bool,
    cycles_used: int,
    failures: int,
    last_error_type: str,
    last_error_msg: str,
    events: list[dict[str, Any]],
) -> None:
    patch_meta["syntax_preflight_enabled"] = bool(enabled)
    patch_meta["syntax_preflight_cycles_used"] = int(cycles_used)
    patch_meta["syntax_preflight_failures"] = int(failures)
    patch_meta["syntax_preflight_last_error_type"] = str(last_error_type or "")
    patch_meta["syntax_preflight_last_error_msg"] = str(last_error_msg or "")
    patch_meta["syntax_preflight_events"] = list(events)
    attempt_metadata["syntax_preflight_enabled"] = bool(enabled)
    attempt_metadata["syntax_preflight_cycles_used"] = int(cycles_used)
    attempt_metadata["syntax_preflight_failures"] = int(failures)
    attempt_metadata["syntax_preflight_last_error_type"] = str(last_error_type or "")
    attempt_metadata["syntax_preflight_last_error_msg"] = str(last_error_msg or "")


def _is_context_limit_http_error_message(body: Any) -> bool:
    text = str(body or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in PATCHER_CONTEXT_LIMIT_ERROR_MARKERS)


def _is_retryable_transport_http_status(status: Any) -> bool:
    if not isinstance(status, int):
        return False
    return status in RETRYABLE_HTTP_STATUS_CODES or (500 <= status < 600)


def _classify_patcher_request_failure(meta: dict[str, Any]) -> tuple[bool, str]:
    """Classify a patcher request error as (failed, kind).

    Returns ``(True, "transport")`` or ``(True, "context_limit")``
    when the error is a recognized failure mode, else ``(False, "")``.
    """
    seen_meta_ids: set[int] = set()
    candidate: dict[str, Any] | Any = meta
    while isinstance(candidate, dict) and id(candidate) not in seen_meta_ids:
        seen_meta_ids.add(id(candidate))
        error = str(candidate.get("error") or "").strip()
        if error == "openai_compat_reasoning_only":
            return True, "compat_reasoning_only"
        if error in {"request_failed", "request_budget_exceeded", "bad_request_url"}:
            return True, "transport"
        if error == "http_error" and _is_context_limit_http_error_message(candidate.get("body")):
            return True, "context_limit"
        if error == "http_error" and _is_retryable_transport_http_status(candidate.get("status")):
            return True, "transport"
        infra_kind = _patcher_infra_failure_kind_from_meta(candidate)
        if infra_kind:
            return True, infra_kind
        candidate = candidate.get("fallback_meta")
    return False, ""


def _transport_recovery_failure_kind_from_meta(meta: dict[str, Any] | None) -> str:
    return _patcher_infra_failure_kind_from_meta(meta)


def _sync_patcher_request_failure_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    failed: bool,
    kind: str = "",
    error: str = "",
    status: int | None = None,
    exc: str = "",
    runner_forced_patch_skipped: bool = False,
    runner_forced_patch_skip_reason: str = "",
) -> None:
    values: dict[str, Any] = {
        "patcher_request_failed": bool(failed),
        "patcher_request_failure_kind": str(kind or ""),
        "patcher_request_failure_error": str(error or ""),
        "patcher_request_failure_status": int(status) if isinstance(status, int) else None,
        "patcher_request_failure_exc": str(exc or ""),
        "runner_forced_patch_skipped": bool(runner_forced_patch_skipped),
        "runner_forced_patch_skip_reason": str(runner_forced_patch_skip_reason or ""),
    }
    patch_meta.update(values)
    attempt_metadata.update(values)


def _record_is_infra_resume_eligible(record: Mapping[str, Any]) -> bool:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, Mapping):
        return False
    return bool(patch_meta.get("infra_resume_eligible"))


def _sync_patch_sanitize_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    enabled: bool,
    passed: bool,
    rejected: bool,
    reasons: list[str],
    error_msg: str,
) -> None:
    values: dict[str, Any] = {
        "patch_sanitize_enabled": bool(enabled),
        "patch_sanitize_passed": bool(passed),
        "patch_sanitize_rejected": bool(rejected),
        "patch_sanitize_reasons": list(reasons),
        "patch_sanitize_error_msg": str(error_msg or ""),
    }
    patch_meta.update(values)
    attempt_metadata.update(values)


def _sync_patch_diff_metadata(*, patch_meta: dict[str, Any], diff: str) -> None:
    patch_meta["patch_diff_xxh3_128"] = _xxh3_128_text(diff)
    patch_meta["patch_diff_chars"] = len(diff)
    patch_meta.update(_compute_patch_path_metrics(diff))


def _sync_patch_verifier_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    enabled: bool,
    scope: str,
    passed: bool | None,
    score: int | None,
    reasons: list[str],
    min_score: int,
) -> None:
    values: dict[str, Any] = {
        "patch_verifier_enabled": bool(enabled),
        "patch_verifier_scope": str(scope or PATCH_VERIFIER_SCOPE_DISABLED),
        "patch_verifier_min_score": max(0, int(min_score)),
        "patch_verifier_passed": passed if isinstance(passed, bool) else None,
        "patch_verifier_score": int(score) if isinstance(score, int) else None,
        "patch_verifier_reasons": list(reasons),
    }
    patch_meta.update(values)
    attempt_metadata.update(values)


def _sync_patch_verifier_recovery_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    eligible: bool,
    invoked: bool,
    result: str,
    prompt_chars_before: int = 0,
    prompt_chars_after: int = 0,
    duration_sec: float = 0.0,
    terminal_note: str = "",
    score_before: int | None = None,
    reasons_before: Sequence[str] | None = None,
    patch_files_before: Sequence[str] | None = None,
    source_scope_matched_before: bool = False,
    raw_meta: Any = None,
) -> None:
    normalized_raw_meta = dict(raw_meta) if isinstance(raw_meta, dict) else raw_meta
    values: dict[str, Any] = {
        "patch_verifier_recovery_eligible": bool(eligible),
        "patch_verifier_recovery_invoked": bool(invoked),
        "patch_verifier_recovery_result": str(
            result or PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE
        ),
        "patch_verifier_recovery_prompt_chars_before": int(prompt_chars_before),
        "patch_verifier_recovery_prompt_chars_after": int(prompt_chars_after),
        "patch_verifier_recovery_duration_sec": max(0.0, float(duration_sec)),
        "patch_verifier_recovery_terminal_note": str(terminal_note or ""),
        "patch_verifier_recovery_score_before": (
            int(score_before) if isinstance(score_before, int) else None
        ),
        "patch_verifier_recovery_reasons_before": [str(item) for item in (reasons_before or [])],
        "patch_verifier_recovery_patch_files_before": [
            str(item) for item in (patch_files_before or [])
        ],
        "patch_verifier_recovery_source_scope_matched_before": bool(source_scope_matched_before),
        "patch_verifier_recovery_raw_meta": normalized_raw_meta,
    }
    patch_meta.update(values)
    attempt_metadata.update(values)


def _append_retry_history_turn_if_enabled(
    *,
    retry_context_mode: str,
    retry_history_turns: list[str],
    attempt: int,
    note: str,
    failure_signature: str,
    patch_meta: dict[str, Any],
) -> None:
    if retry_context_mode != "history":
        return
    retry_history_turns.append(
        _render_retry_history_turn(
            attempt=attempt,
            note=note,
            failure_signature=failure_signature,
            patch_meta=patch_meta,
        )
    )


def _build_attempt_outcome_record(
    *,
    case: CaseSpec,
    condition: Condition,
    run_id: str,
    attempt: int,
    patcher_name: str,
    run_metadata: dict[str, Any],
    attempt_metadata: dict[str, Any],
    patcher: Any,
    patch_meta: dict[str, Any] | None,
    context_protocol: str,
    success: bool,
    note: str,
    before: RunResult,
    cumulative_runtime_sec: float,
    after: RunResult | None = None,
    attempt_wall_clock_sec: float | None = None,
) -> dict[str, Any]:
    patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else None
    retry_summary = _build_retry_summary(
        attempt=attempt,
        note=note,
        failure_signature=(
            str(patch_meta_dict.get("adaptive_postpatch_failure_signature") or "")
            if patch_meta_dict is not None
            else ""
        ),
        patch_meta=patch_meta_dict,
    )
    if not str(retry_summary.get("failure_signature") or ""):
        retry_summary["failure_signature"] = _build_failure_signature_for_attempt(
            run=before,
            snapshot=None,
        )
    attempt_metadata["retry_summary"] = dict(retry_summary)
    if patch_meta_dict is not None:
        patch_meta_dict["retry_summary"] = dict(retry_summary)
    if (
        bool(attempt_metadata.get("patch_verifier_recovery_invoked"))
        or bool(attempt_metadata.get("patch_verifier_recovery_eligible"))
        or (
            patch_meta_dict is not None
            and (
                bool(patch_meta_dict.get("patch_verifier_recovery_invoked"))
                or bool(patch_meta_dict.get("patch_verifier_recovery_eligible"))
            )
        )
    ):
        attempt_metadata["patch_verifier_recovery_terminal_note"] = str(note or "")
        if patch_meta_dict is not None:
            patch_meta_dict["patch_verifier_recovery_terminal_note"] = str(note or "")
    record = _build_base_record(
        case=case,
        condition=condition,
        run_id=run_id,
        attempt=attempt,
        patcher_name=patcher_name,
        run_metadata=run_metadata,
        attempt_metadata=attempt_metadata,
        model_metadata=_build_model_metadata(patcher=patcher, patch_meta=patch_meta),
        context_protocol=context_protocol,
    )
    default_attempt_wall_clock_sec = float(before.duration_sec)
    if after is not None:
        default_attempt_wall_clock_sec += float(after.duration_sec)
    measured_attempt_wall_clock_sec = (
        float(attempt_wall_clock_sec)
        if isinstance(attempt_wall_clock_sec, (int, float))
        else default_attempt_wall_clock_sec
    )
    measured_attempt_wall_clock_sec = max(0.0, measured_attempt_wall_clock_sec)
    payload: dict[str, Any] = {
        "success": success,
        "before_returncode": before.returncode,
        "before_duration_sec": before.duration_sec,
        "cumulative_runtime_sec": cumulative_runtime_sec,
        "attempt_wall_clock_sec": measured_attempt_wall_clock_sec,
        "note": note,
    }
    if after is not None:
        payload["after_returncode"] = after.returncode
        payload["after_duration_sec"] = after.duration_sec
    if patch_meta_dict is not None:
        payload["patch_meta"] = patch_meta_dict
        payload["patch_touches_tests"] = bool(patch_meta_dict.get("patch_touches_tests"))
        payload["patch_test_file_count"] = _nonnegative_int(
            patch_meta_dict.get("patch_test_file_count")
        )
        payload["patch_file_count"] = _nonnegative_int(patch_meta_dict.get("patch_file_count"))
    record.update(payload)
    record.update(_timeout_fields(after if after is not None else before))
    return record


def _measure_attempt_wall_clock_sec(attempt_start_monotonic: float) -> float:
    return max(0.0, float(time.perf_counter() - attempt_start_monotonic))


def _finalize_breakdowns(record: dict[str, Any]) -> None:
    """Populate timing_breakdown and token_breakdown from the final record.

    Extracts per-phase timing from patch_meta and record fields, and
    per-phase token counts from stage-A meta and patch-level usage.
    Called just before the record is appended to the results list.
    """
    attempt_meta = record.get("attempt_metadata")
    if not isinstance(attempt_meta, dict):
        return
    timing = attempt_meta.get("timing_breakdown")
    tokens = attempt_meta.get("token_breakdown")
    if not isinstance(timing, dict) or not isinstance(tokens, dict):
        return

    patch_meta = record.get("patch_meta")
    pm = patch_meta if isinstance(patch_meta, dict) else {}
    # ── Timing breakdown ────────────────────────────────────────────────
    # Stage-A LLM: from patcher meta request_context_duration_sec
    stage_a_meta = pm.get("stage_a_meta")
    stage_a_llm_sec = 0.0
    if isinstance(stage_a_meta, dict):
        stage_a_llm_sec = max(0.0, float(stage_a_meta.get("request_context_duration_sec") or 0.0))
    timing["stage_a_llm_sec"] = stage_a_llm_sec
    timing["adaptive_controller_prepare_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_prepare_sec") or 0.0),
    )
    timing["adaptive_controller_summarizer_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_summarizer_sec") or 0.0),
    )
    timing["adaptive_controller_sidecar_build_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_sidecar_build_sec") or 0.0),
    )
    timing["adaptive_controller_sidecar_write_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_sidecar_write_sec") or 0.0),
    )
    timing["baseline_case_conversion_sec"] = max(
        0.0,
        float(pm.get("baseline_case_conversion_sec") or 0.0),
    )
    timing["baseline_sidecar_load_sec"] = max(
        0.0,
        float(pm.get("baseline_sidecar_load_sec") or 0.0),
    )

    baseline_llm_sec = _baseline_metric_float(pm, "llm_duration_sec")
    timing["stage_b_llm_sec"] = (
        baseline_llm_sec
        if baseline_llm_sec is not None
        else max(0.0, float(pm.get("propose_duration_sec") or 0.0))
    )
    timing["baseline_internal_total_sec"] = (
        _baseline_metric_float(pm, "internal_total_duration_sec") or 0.0
    )
    timing["baseline_eval_sec"] = _baseline_metric_float(pm, "eval_duration_sec") or 0.0
    timing["baseline_orchestration_sec"] = (
        _baseline_metric_float(pm, "orchestration_duration_sec") or 0.0
    )

    context_compaction_invoked = bool(
        attempt_meta.get("context_compaction_invoked") or pm.get("context_compaction_invoked")
    )
    context_compaction_meta = (
        pm.get("context_compaction_meta")
        if isinstance(pm.get("context_compaction_meta"), dict)
        else None
    )

    # Compaction LLM: prefer the persisted nested metadata, with top-level fallback for old records
    compaction_duration_raw: Any
    if isinstance(context_compaction_meta, dict) and (
        "compact_context_duration_sec" in context_compaction_meta
    ):
        compaction_duration_raw = context_compaction_meta.get("compact_context_duration_sec")
    else:
        compaction_duration_raw = pm.get("compact_context_duration_sec")
    timing["context_compaction_llm_sec"] = max(0.0, float(compaction_duration_raw or 0.0))

    # Patch apply: apply_duration_sec
    timing["patch_apply_sec"] = max(0.0, float(pm.get("apply_duration_sec") or 0.0))

    # Test durations (from top-level record fields)
    timing["test_before_sec"] = max(0.0, float(record.get("before_duration_sec") or 0.0))
    timing["test_after_sec"] = max(0.0, float(record.get("after_duration_sec") or 0.0))

    # Recovery: sum context_recovery + no_diff_recovery + verifier_recovery durations
    recovery_llm_sec = max(0.0, float(pm.get("context_recovery_duration_sec") or 0.0))
    recovery_llm_sec += max(0.0, float(pm.get("no_diff_recovery_duration_sec") or 0.0))
    recovery_llm_sec += max(0.0, float(pm.get("patch_verifier_recovery_duration_sec") or 0.0))
    timing["recovery_llm_sec"] = recovery_llm_sec
    timing["recovery_total_sec"] = recovery_llm_sec

    # llmdebug_total_sec: snapshot_read + context_fulfillment (tool calls for llmdebug data)
    timing["llmdebug_total_sec"] = max(
        0.0,
        timing.get("snapshot_read_sec", 0.0) + timing.get("stage_a_context_fulfillment_sec", 0.0),
    )

    # llm_total_sec: Stage-B propose_duration_sec already includes recovery retries.
    timing["llm_total_sec"] = max(
        0.0,
        stage_a_llm_sec
        + timing.get("stage_b_llm_sec", 0.0)
        + timing.get("context_compaction_llm_sec", 0.0),
    )

    # overhead_sec: wall_clock - test_before - test_after - llm_total
    attempt_wall = max(0.0, float(record.get("attempt_wall_clock_sec") or 0.0))
    timing["overhead_sec"] = max(
        0.0,
        attempt_wall
        - timing.get("test_before_sec", 0.0)
        - timing.get("test_after_sec", 0.0)
        - timing.get("llm_total_sec", 0.0)
        - timing.get("baseline_eval_sec", 0.0)
        - timing.get("baseline_orchestration_sec", 0.0),
    )

    # ── Token breakdown ─────────────────────────────────────────────────
    # Stage-A tokens
    sa_p, sa_c, sa_t = _usage_token_counts_from_patch_meta(
        stage_a_meta if isinstance(stage_a_meta, dict) else None
    )
    tokens["stage_a_tokens_prompt"] = sa_p
    tokens["stage_a_tokens_completion"] = sa_c
    tokens["stage_a_tokens_total"] = sa_t

    # Compaction tokens
    if isinstance(context_compaction_meta, dict):
        cc_meta = context_compaction_meta
    elif context_compaction_invoked:
        # Compaction meta may be stored at top level of patch_meta
        cc_meta = pm
    else:
        cc_meta = None
    cc_p = cc_c = cc_t = 0
    if cc_meta is not None and isinstance(cc_meta, dict) and context_compaction_invoked:
        cc_p, cc_c, cc_t = _usage_token_counts_from_patch_meta(cc_meta)
    tokens["compaction_tokens_prompt"] = cc_p
    tokens["compaction_tokens_completion"] = cc_c
    tokens["compaction_tokens_total"] = cc_t

    # Recovery tokens: request-failure, no-diff, and verifier-guided retries
    context_recovery_meta = pm.get("context_recovery_meta")
    if not isinstance(context_recovery_meta, dict):
        raw_context_recovery_meta = pm.get("context_recovery_raw_meta")
        context_recovery_meta = (
            raw_context_recovery_meta if isinstance(raw_context_recovery_meta, dict) else None
        )
    no_diff_recovery_meta = pm.get("no_diff_recovery_meta")
    if not isinstance(no_diff_recovery_meta, dict):
        raw_no_diff_recovery_meta = pm.get("no_diff_recovery_raw_meta")
        no_diff_recovery_meta = (
            raw_no_diff_recovery_meta if isinstance(raw_no_diff_recovery_meta, dict) else None
        )
    verifier_recovery_meta = (
        pm.get("patch_verifier_recovery_raw_meta")
        if isinstance(pm.get("patch_verifier_recovery_raw_meta"), dict)
        else None
    )
    recovery_prompt_tokens = recovery_completion_tokens = recovery_total_tokens = 0
    for recovery_meta in (
        context_recovery_meta,
        no_diff_recovery_meta,
        verifier_recovery_meta,
    ):
        rec_p, rec_c, rec_t = _usage_token_counts_from_patch_meta(
            recovery_meta if isinstance(recovery_meta, dict) else None
        )
        recovery_prompt_tokens += rec_p
        recovery_completion_tokens += rec_c
        recovery_total_tokens += rec_t
    tokens["recovery_tokens_prompt"] = recovery_prompt_tokens
    tokens["recovery_tokens_completion"] = recovery_completion_tokens
    tokens["recovery_tokens_total"] = recovery_total_tokens

    # Stage-B tokens: total attempt tokens minus stage-A minus compaction
    total_prompt = max(0, int(attempt_meta.get("tokens_prompt") or 0))
    total_completion = max(0, int(attempt_meta.get("tokens_completion") or 0))
    total_total = max(0, int(attempt_meta.get("tokens_total") or 0))
    stage_b_prompt = max(
        0,
        total_prompt
        - tokens.get("stage_a_tokens_prompt", 0)
        - tokens.get("compaction_tokens_prompt", 0)
        - tokens.get("recovery_tokens_prompt", 0),
    )
    stage_b_completion = max(
        0,
        total_completion
        - tokens.get("stage_a_tokens_completion", 0)
        - tokens.get("compaction_tokens_completion", 0)
        - tokens.get("recovery_tokens_completion", 0),
    )
    stage_b_total = max(
        0,
        total_total
        - tokens.get("stage_a_tokens_total", 0)
        - tokens.get("compaction_tokens_total", 0)
        - tokens.get("recovery_tokens_total", 0),
    )
    tokens["stage_b_tokens_prompt"] = stage_b_prompt
    tokens["stage_b_tokens_completion"] = stage_b_completion
    tokens["stage_b_tokens_total"] = stage_b_total

    # Tool response chars and snapshot chars
    tokens["tool_response_chars"] = max(0, int(attempt_meta.get("stage_a_chars_returned") or 0))
    # snapshot_chars set by the instrumentation in run_eval; keep if already set


def _ensure_deterministic_seed_ack(
    *,
    deterministic: bool,
    seed: int | None,
    patcher_name: str,
    patch_meta: dict[str, Any],
    case_id: str,
    condition: str,
    attempt: int,
) -> None:
    if not deterministic or patcher_name != "openai":
        return

    requested = patch_meta.get("api_compat_seed_requested")
    acknowledged = patch_meta.get("api_compat_seed_acknowledged")
    if requested != seed:
        raise RuntimeError(
            "deterministic seed mismatch for "
            f"{case_id}/{condition}/attempt={attempt}: requested={requested!r}, expected={seed!r}"
        )
    if acknowledged is not True:
        error = patch_meta.get("error")
        body = patch_meta.get("body")
        exc = patch_meta.get("exc")
        details = error or body or exc or "backend did not acknowledge seed support"
        raise RuntimeError(
            f"deterministic mode failed for {case_id}/{condition}/attempt={attempt}: {details}"
        )


def _adaptive_gate_should_invoke_helper(
    *,
    policy: str,
    attempt: int,
    timed_out: bool,
    signature_repeated: bool,
    stagnation_streak: int,
    stagnation_window: int,
) -> tuple[bool, str]:
    """Decide whether the adaptive gate should invoke the helper.

    Returns (should_invoke, reason_token) based on the gate policy,
    attempt number, timeout state, and stagnation streak.
    """
    if timed_out:
        return True, "gate_timeout"
    if policy == "aggressive":
        return True, "gate_aggressive"

    normalized_window = max(1, int(stagnation_window))
    if policy == "conservative":
        required_repeats = max(normalized_window + 1, 2)
        if signature_repeated and stagnation_streak >= required_repeats:
            return True, "gate_conservative_repeated_signature"
        return False, "gate_conservative_waiting"

    # balanced (default)
    if attempt >= 2 and signature_repeated and stagnation_streak >= normalized_window:
        return True, "gate_balanced_repeated_signature"
    return False, "gate_balanced_waiting"


def _adaptive_progress_signal(
    *,
    note: str = "",
    success: bool = False,
    signature_repeated: bool,
    new_evidence_count: int,
    before_run: RunResult | None = None,
    after_run: RunResult | None = None,
    previous_failure_context: dict[str, Any] | None = None,
    patch_meta: dict[str, Any] | None = None,
) -> AdaptiveProgressState:
    """Compute a multi-signal progress assessment for an attempt.

    Evaluates failure-signature repetition, new-evidence count,
    pytest-failure reduction, pipeline-stage advancement, verifier
    score, and scope alignment to produce an ``AdaptiveProgressState``.
    """
    patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
    previous_context = (
        previous_failure_context if isinstance(previous_failure_context, dict) else {}
    )

    component_signature = 0.0 if signature_repeated else 1.0
    if bool(success):
        component_signature = 1.0
    component_new_evidence = 1.0 if int(new_evidence_count) > 0 else 0.0

    pytest_before_failed_total = _pytest_failed_total(before_run)
    pytest_after_failed_total = _pytest_failed_total(after_run)
    if bool(success) and pytest_after_failed_total is None:
        pytest_after_failed_total = 0

    component_pytest_fail_reduction = 0.0
    if (
        pytest_before_failed_total is not None
        and pytest_after_failed_total is not None
        and pytest_before_failed_total > 0
        and pytest_after_failed_total < pytest_before_failed_total
    ):
        component_pytest_fail_reduction = min(
            1.0,
            max(
                0.0,
                (pytest_before_failed_total - pytest_after_failed_total)
                / float(pytest_before_failed_total),
            ),
        )

    stage_before, stage_before_ord = _adaptive_pipeline_stage_before(
        previous_failure_context=previous_failure_context,
        current_run=before_run,
    )
    stage_after, stage_after_ord = _adaptive_pipeline_stage_for_outcome(
        note=note,
        success=success,
    )
    component_stage_advance = min(
        1.0,
        max(0.0, float(stage_after_ord - stage_before_ord) / 4.0),
    )

    prev_verifier_score = _adaptive_optional_int(previous_context.get("patch_verifier_score"))
    curr_verifier_score = _adaptive_optional_int(patch_meta_dict.get("patch_verifier_score"))
    component_verifier_score = 0.0
    if (
        prev_verifier_score is not None
        and curr_verifier_score is not None
        and curr_verifier_score > prev_verifier_score
    ):
        component_verifier_score = min(
            1.0,
            max(0.0, float(curr_verifier_score - prev_verifier_score) / 4.0),
        )

    prev_scope_matched = _adaptive_optional_bool(
        previous_context.get("patch_verifier_source_scope_matched")
    )
    curr_scope_matched = _adaptive_optional_bool(
        patch_meta_dict.get("patch_verifier_source_scope_matched")
    )
    component_scope_alignment = (
        1.0 if prev_scope_matched is False and curr_scope_matched is True else 0.0
    )

    contributions = {
        "failure_signature_changed": 0.20 * component_signature,
        "new_evidence_added": 0.10 * component_new_evidence,
        "pytest_failure_count_reduction": 0.25 * component_pytest_fail_reduction,
        "pipeline_stage_advance": 0.20 * component_stage_advance,
        "patch_verifier_score_improvement": 0.15 * component_verifier_score,
        "source_scope_alignment_improvement": 0.10 * component_scope_alignment,
    }
    score = min(1.0, max(0.0, sum(contributions.values())))
    reason_order = (
        "failure_signature_changed",
        "pytest_failure_count_reduction",
        "pipeline_stage_advance",
        "patch_verifier_score_improvement",
        "source_scope_alignment_improvement",
        "new_evidence_added",
    )
    reason = "no_progress_v2"
    best_contribution = 0.0
    for reason_key in reason_order:
        contribution = contributions.get(reason_key, 0.0)
        if contribution > best_contribution:
            best_contribution = contribution
            reason = reason_key

    return AdaptiveProgressState(
        made=bool(score > 0.0),
        reason=reason if score > 0.0 else "no_progress_v2",
        score=float(score),
        component_signature=float(component_signature),
        component_new_evidence=float(component_new_evidence),
        component_pytest_fail_reduction=float(component_pytest_fail_reduction),
        component_stage_advance=float(component_stage_advance),
        component_verifier_score=float(component_verifier_score),
        component_scope_alignment=float(component_scope_alignment),
        pytest_before_failed_total=pytest_before_failed_total,
        pytest_after_failed_total=pytest_after_failed_total,
        stage_before=stage_before,
        stage_after=stage_after,
        stage_before_ord=stage_before_ord,
        stage_after_ord=stage_after_ord,
    )


def _adaptive_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _adaptive_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _pytest_failed_total(run: RunResult | None) -> int | None:
    if run is None:
        return None
    for text in (str(run.stdout or ""), str(run.stderr or "")):
        total = _pytest_failed_total_from_text(text)
        if total is not None:
            return total
    if not run.timed_out and int(run.returncode) == 0:
        return 0
    return None


def _pytest_failed_total_from_text(text: str) -> int | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        lower = line.lower()
        if "short test summary info" in lower:
            continue
        if "no tests ran" in lower:
            return 0
        if "=" not in line and " in " not in lower:
            continue
        if not any(token in lower for token in (" failed", " error", " errors", " passed")):
            continue
        total = 0
        matched = False
        for match in _PYTEST_SUMMARY_COUNT_RE.finditer(lower):
            matched = True
            total += int(match.group(1))
        if matched:
            return total
        if " passed" in lower:
            return 0
    return None


def _adaptive_pipeline_stage_for_outcome(*, note: str, success: bool) -> tuple[str, int]:
    if bool(success) or str(note or "") == "success":
        return "success", 4
    note_norm = str(note or "").strip()
    if note_norm in {"patcher_request_failed", "no_patch_proposed", "retry_gate_no_new_evidence"}:
        return "request_or_no_patch", 0
    if note_norm in {"forced_noop_patch", "patch_apply_failed", "syntax_preflight_failed"}:
        return "pretest_patch_failure", 1
    if note_norm == "patch_verifier_rejected":
        return "verifier_rejected", 2
    return "tests_still_failing", 3


def _adaptive_pipeline_stage_before(
    *,
    previous_failure_context: dict[str, Any] | None,
    current_run: RunResult | None,
) -> tuple[str, int]:
    if isinstance(previous_failure_context, dict):
        previous_note = str(previous_failure_context.get("note") or "")
        if previous_note:
            return _adaptive_pipeline_stage_for_outcome(note=previous_note, success=False)
    if current_run is None:
        return "tests_still_failing", 3
    if current_run.timed_out:
        return "tests_still_failing", 3
    if int(current_run.returncode) == 0:
        return "success", 4
    return "tests_still_failing", 3


def _sync_adaptive_progress_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    progress_state: AdaptiveProgressState,
    no_progress_streak: int,
    recomputed_after_patch: bool = False,
) -> None:
    fields = {
        "adaptive_progress_made": bool(progress_state.made),
        "adaptive_progress_reason": str(progress_state.reason),
        "adaptive_progress_score": float(progress_state.score),
        "adaptive_progress_component_signature": float(progress_state.component_signature),
        "adaptive_progress_component_new_evidence": float(progress_state.component_new_evidence),
        "adaptive_progress_component_pytest_fail_reduction": float(
            progress_state.component_pytest_fail_reduction
        ),
        "adaptive_progress_component_stage_advance": float(progress_state.component_stage_advance),
        "adaptive_progress_component_verifier_score": float(
            progress_state.component_verifier_score
        ),
        "adaptive_progress_component_scope_alignment": float(
            progress_state.component_scope_alignment
        ),
        "adaptive_pytest_before_failed_total": progress_state.pytest_before_failed_total,
        "adaptive_pytest_after_failed_total": progress_state.pytest_after_failed_total,
        "adaptive_pipeline_stage_before": str(progress_state.stage_before),
        "adaptive_pipeline_stage_after": str(progress_state.stage_after),
        "adaptive_pipeline_stage_before_ord": int(progress_state.stage_before_ord),
        "adaptive_pipeline_stage_after_ord": int(progress_state.stage_after_ord),
        "adaptive_no_progress_streak": int(max(0, int(no_progress_streak))),
    }
    attempt_metadata.update(fields)
    patch_meta.update(fields)
    patch_meta["adaptive_progress_recomputed_after_patch"] = bool(recomputed_after_patch)


def _recompute_adaptive_progress_for_attempt_outcome(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    note: str,
    success: bool,
    signature_repeated: bool,
    new_evidence_count: int,
    before_run: RunResult | None,
    after_run: RunResult | None,
    previous_failure_context: dict[str, Any] | None,
    current_no_progress_streak: int,
    recomputed_after_patch: bool,
) -> tuple[AdaptiveProgressState, int]:
    progress_state = _adaptive_progress_signal(
        note=note,
        success=success,
        signature_repeated=signature_repeated,
        new_evidence_count=new_evidence_count,
        before_run=before_run,
        after_run=after_run,
        previous_failure_context=previous_failure_context,
        patch_meta=patch_meta,
    )
    next_no_progress_streak = 0 if progress_state.made else max(0, int(current_no_progress_streak))
    _sync_adaptive_progress_metadata(
        attempt_metadata=attempt_metadata,
        patch_meta=patch_meta,
        progress_state=progress_state,
        no_progress_streak=next_no_progress_streak,
        recomputed_after_patch=recomputed_after_patch,
    )
    return progress_state, next_no_progress_streak


def _adaptive_should_stop_for_no_progress(
    *,
    no_progress_streak: int,
    stop_window: int,
    helper_budget_exhausted: bool,
    helper_saturation_guarded: bool,
    helper_decision_reason: str,
) -> tuple[bool, str]:
    if int(no_progress_streak) < max(1, int(stop_window)):
        return False, "streak_below_window"
    if helper_budget_exhausted:
        return True, "helper_budget_exhausted"
    if helper_saturation_guarded:
        return True, "helper_saturation_guard"
    if helper_decision_reason in {
        "llm_declined_helper",
        "llm_invalid_response_no_helper",
        "llm_snapshot_request_denied_attempt1",
        "request_context_unavailable",
    }:
        return True, helper_decision_reason
    return False, "waiting_for_more_signal"


def _mark_adaptive_progress_stop(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    is_adaptive_condition: bool,
    adaptive_progress_stop_candidate: bool,
    adaptive_progress_stop_reason: str,
    note: str,
) -> bool:
    triggered = bool(
        is_adaptive_condition
        and adaptive_progress_stop_candidate
        and note
        in {
            "hard_stop_no_progress",
            "no_patch_proposed",
            "patcher_request_failed",
            "forced_noop_patch",
            "patch_verifier_rejected",
            "syntax_preflight_failed",
            "patch_apply_failed",
            "tests_still_failing",
        }
    )
    attempt_metadata["adaptive_progress_stop_triggered"] = triggered
    patch_meta["adaptive_progress_stop_triggered"] = triggered
    patch_meta["adaptive_progress_stop_reason"] = adaptive_progress_stop_reason
    if triggered:
        patch_meta["adaptive_progress_stop_note"] = note
    return triggered


def _apply_no_progress_token_cap_candidate(
    *,
    is_adaptive_condition: bool,
    adaptive_progress_made: bool,
    adaptive_no_progress_token_sum: int,
    attempt_tokens_total: int,
    adaptive_no_progress_token_cap: int,
    adaptive_progress_stop_candidate: bool,
    adaptive_progress_stop_reason: str,
) -> tuple[bool, str, int, bool]:
    next_sum = max(0, int(adaptive_no_progress_token_sum))
    if not is_adaptive_condition:
        return adaptive_progress_stop_candidate, adaptive_progress_stop_reason, next_sum, False
    if adaptive_progress_made:
        next_sum = 0
    else:
        next_sum += max(0, int(attempt_tokens_total))

    cap = max(0, int(adaptive_no_progress_token_cap))
    cap_hit = cap > 0 and next_sum >= cap
    if cap_hit:
        return True, "no_progress_token_cap", next_sum, True
    return adaptive_progress_stop_candidate, adaptive_progress_stop_reason, next_sum, False


def _extract_exception_info(
    *,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
) -> tuple[str, str, int, bool]:
    """Extract (exception_type, message, frame_count, crash_in_user_code)."""
    if snapshot and isinstance(snapshot.get("exception"), dict):
        exc_info = snapshot["exception"]
        exc_type = str(exc_info.get("type", "Unknown"))
        exc_msg = str(exc_info.get("message", ""))
        frames = snapshot.get("frames")
        frame_count = len(frames) if isinstance(frames, list) else 0
        crash_in_user_code = False
        if isinstance(frames, list) and len(frames) > 0:
            crash_frame = frames[0]
            if isinstance(crash_frame, dict):
                crash_in_user_code = crash_frame.get("file_rel") is not None
        # Strip module prefix (e.g., "builtins.ValueError" -> "ValueError")
        if "." in exc_type:
            exc_type = exc_type.rsplit(".", 1)[-1]
        return exc_type, exc_msg, frame_count, crash_in_user_code

    # Fallback: parse from stdout/stderr
    for text in (stdout, stderr):
        if not text:
            continue
        m = _STDOUT_PYTEST_ERROR_RE.search(text)
        if m:
            exc_type = m.group(1).rsplit(".", 1)[-1]
            return exc_type, m.group(2).strip(), 0, False
        m = _STDOUT_TRACEBACK_ERROR_RE.search(text)
        if m:
            exc_type = m.group(1).rsplit(".", 1)[-1]
            return exc_type, m.group(2).strip(), 0, False

    return "Unknown", "", 0, False


def _traceback_informativeness_score(
    *,
    exception_type: str,
    exception_message: str,
    frame_count: int,
    crash_in_user_code: bool,
) -> PreGateResult:
    """Score traceback informativeness and return a PreGateResult with all features."""
    # Signal 1: Exception type specificity (weight 0.30)
    f_type_specificity = _EXCEPTION_TYPE_SPECIFICITY.get(
        exception_type, _EXCEPTION_TYPE_SPECIFICITY_DEFAULT
    )

    # Signal 2: Message contains diagnostic values (weight 0.25)
    f_message_diagnostic = 0.0
    if exception_message:
        for pattern in _DIAGNOSTIC_MESSAGE_PATTERNS:
            if pattern.search(exception_message):
                f_message_diagnostic = 1.0
                break
        if f_message_diagnostic == 0.0:
            f_message_diagnostic = 0.3  # Non-empty but generic
    # else: empty message → 0.0

    # Signal 3: Crash frame in user code (weight 0.15)
    f_crash_in_user_code = 1.0 if crash_in_user_code else 0.0

    # Signal 4: Assertion message quality (weight 0.15)
    if exception_type == "AssertionError":
        if exception_message and _ASSERTION_VALUE_PATTERN.search(exception_message):
            f_assertion_quality = 1.0
        elif exception_message:
            f_assertion_quality = 0.5
        else:
            f_assertion_quality = 0.0
    else:
        f_assertion_quality = 0.5  # Neutral for non-assertions

    # Signal 5: Frame count / depth (weight 0.15)
    # frame_count=0 means no structured data (stdout fallback) — treat as neutral.
    if frame_count == 0:
        f_frame_depth = 0.5
    elif frame_count <= 3:
        f_frame_depth = 1.0
    elif frame_count <= 6:
        f_frame_depth = 0.7
    elif frame_count <= 10:
        f_frame_depth = 0.4
    else:
        f_frame_depth = 0.2

    score = (
        0.30 * f_type_specificity
        + 0.25 * f_message_diagnostic
        + 0.15 * f_crash_in_user_code
        + 0.15 * f_assertion_quality
        + 0.15 * f_frame_depth
    )
    score = min(1.0, max(0.0, score))

    tier = (
        "A"
        if f_type_specificity >= 0.8
        else ("B" if f_type_specificity >= 0.5 else ("C" if f_type_specificity >= 0.3 else "D"))
    )
    reason = f"{exception_type}_tier{tier}"

    return PreGateResult(
        should_skip=False,  # Caller decides based on threshold
        score=score,
        exception_type=exception_type,
        reason=reason,
        f_type_specificity=f_type_specificity,
        f_message_diagnostic=f_message_diagnostic,
        f_crash_in_user_code=f_crash_in_user_code,
        f_assertion_quality=f_assertion_quality,
        f_frame_depth=f_frame_depth,
    )


def _traceback_informativeness_pre_gate(
    *,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    threshold: float,
) -> PreGateResult:
    """Return a PreGateResult with should_skip set based on threshold."""
    exc_type, exc_msg, frame_count, crash_in_user = _extract_exception_info(
        snapshot=snapshot,
        stdout=stdout,
        stderr=stderr,
    )
    result = _traceback_informativeness_score(
        exception_type=exc_type,
        exception_message=exc_msg,
        frame_count=frame_count,
        crash_in_user_code=crash_in_user,
    )
    return result._replace(should_skip=result.score >= threshold)


def _adaptive_prompt_budget_for_attempt(
    *,
    base_budget: dict[str, int],
    helper_materialized_context: bool,
    profile: str,
) -> dict[str, int]:
    out = dict(base_budget)
    if not helper_materialized_context or profile != "compact":
        return out

    def _cap(value: int, limit: int) -> int:
        if value <= 0:
            return value
        return min(value, limit)

    out["total_chars"] = _cap(int(out.get("total_chars", 0)), 120_000)
    out["retry_delta_chars"] = _cap(int(out.get("retry_delta_chars", 0)), 12_000)
    out["stdout_chars"] = _cap(int(out.get("stdout_chars", 0)), 16_000)
    out["snapshot_chars"] = _cap(int(out.get("snapshot_chars", 0)), 20_000)
    out["evidence_chars"] = _cap(int(out.get("evidence_chars", 0)), 32_000)
    out["retry_history_chars"] = _cap(int(out.get("retry_history_chars", 0)), 8_000)
    return out


def _openai_preflight_total_char_cap(
    *,
    context_window_tokens: int,
    max_tokens: int,
    reserve_tokens: int = OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
) -> int:
    return openai_preflight_total_char_cap(
        context_window_tokens=context_window_tokens,
        max_tokens=max_tokens,
        reserve_tokens=reserve_tokens,
    )


def _is_snapshot_condition(condition: Condition) -> bool:
    return condition in {
        Condition.LAYERED_ADAPTIVE_GATED,
        Condition.LAYERED_ADAPTIVE_LLM,
    }


def _effective_patch_verifier_scope(*, configured_scope: str, is_adaptive_condition: bool) -> str:
    """Resolve configured patch-verifier scope into an attempt-effective scope."""
    if configured_scope == PATCH_VERIFIER_SCOPE_DISABLED:
        return PATCH_VERIFIER_SCOPE_DISABLED
    if configured_scope == PATCH_VERIFIER_SCOPE_ALL:
        return PATCH_VERIFIER_SCOPE_ALL
    if configured_scope == PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY and is_adaptive_condition:
        return PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY
    return PATCH_VERIFIER_SCOPE_DISABLED


def run_one_condition(
    *,
    case: CaseSpec,
    condition: Condition,
    workdir: Path,
    context_dir_base: Path,
    inputs: RunConditionInputs | None = None,
    **legacy_kwargs: Any,
) -> list[dict[str, Any]]:
    from evals.eval_attempt_flow import run_one_condition_impl

    return run_one_condition_impl(
        case=case,
        condition=condition,
        workdir=workdir,
        context_dir_base=context_dir_base,
        inputs=inputs,
        **legacy_kwargs,
    )


def _run_pytest_pooled(
    *,
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    output_format: str,
    execution: ExecutionConfig,
    pool: _EvalContainerPool,
) -> RunResult:
    """Run pytest in a pooled container via ``exec_run``."""
    env = _base_pytest_env(
        output_format=output_format,
        snapshot_condition=_is_snapshot_condition(condition),
    )
    if execution.container_mount_repo_src:
        env["PYTHONPATH"] = "/repo_src"
    env.update(execution.container_env)

    command = _build_pytest_args(
        python_cmd="python",
        python_args=python_args,
        condition=condition,
    )

    container = pool.acquire()
    try:
        return container.exec_pytest(
            host_cwd=cwd,
            command=command,
            env=env,
            timeout_sec=timeout_sec,
        )
    finally:
        pool.release(container)


def run_pytest(
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    output_format: str,
    *,
    execution: ExecutionConfig | None = None,
    progress_update: Callable[[str, dict[str, Any] | None], None] | None = None,
    progress_phase_label: str = "",
) -> RunResult:
    """Run pytest for *condition* using the active execution backend.

    Delegates to ``_run_pytest_local`` or ``_run_pytest_testcontainers``
    depending on the ``ExecutionConfig.executor`` setting.
    """
    report_name = junit_report_name(python_args)
    if report_name is not None:
        # ATT-01: nothing planted under a report name may survive into the verdict.
        clear_junit_reports(cwd)
    active_execution = execution if execution is not None else get_active_execution_config()
    if active_execution.executor == "local":
        result = _run_pytest_local(
            cwd=cwd,
            python_args=python_args,
            condition=condition,
            timeout_sec=timeout_sec,
            output_format=output_format,
            progress_update=progress_update,
            progress_phase_label=progress_phase_label,
        )
    elif active_execution.executor == "testcontainers":
        pool = _ACTIVE_CONTAINER_POOL
        if pool is not None:
            result = _run_pytest_pooled(
                cwd=cwd,
                python_args=python_args,
                condition=condition,
                timeout_sec=timeout_sec,
                output_format=output_format,
                execution=active_execution,
                pool=pool,
            )
        else:
            result = _run_pytest_testcontainers(
                cwd=cwd,
                python_args=python_args,
                condition=condition,
                timeout_sec=timeout_sec,
                output_format=output_format,
                execution=active_execution,
            )
    else:
        raise RuntimeError(f"unsupported executor: {active_execution.executor!r}")
    if report_name is None:
        return result
    # The verdict comes from the report, read on the host after any executor.
    return replace(
        result, junit_requested=True, junit_counts=consume_junit_counts(cwd, report_name)
    )


def _build_pytest_args(
    *, python_cmd: str, python_args: list[str], condition: Condition
) -> list[str]:
    args = [python_cmd, *python_args]
    if _is_snapshot_condition(condition):
        if "-m" in python_args and "pytest" in python_args:
            idx = python_args.index("pytest")
            args = [
                python_cmd,
                *python_args[: idx + 1],
                "-p",
                "llmdebug.pytest_plugin",
                *python_args[idx + 1 :],
            ]
        else:
            args = [python_cmd, *python_args, "-p", "llmdebug.pytest_plugin"]
    return args


def _base_pytest_env(*, output_format: str, snapshot_condition: bool) -> dict[str, str]:
    env = {
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LLMDEBUG_OUTPUT_FORMAT": output_format,
    }
    if _ACTIVE_RUN_SETTINGS.deterministic and _ACTIVE_RUN_SETTINGS.seed is not None:
        env["PYTHONHASHSEED"] = str(_ACTIVE_RUN_SETTINGS.seed)
        env.setdefault("TZ", "UTC")
    if snapshot_condition:
        env.setdefault("LLMDEBUG_DEBUG", "0")
        env.setdefault("LLMDEBUG_INCLUDE_GIT", "false")
        # Disable built-in hypothesis/category hints: the model should infer diagnosis itself.
        env["LLMDEBUG_CATEGORIZE_ERRORS"] = "0"
    return env


def _run_pytest_local(
    *,
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    output_format: str,
    progress_update: Callable[[str, dict[str, Any] | None], None] | None = None,
    progress_phase_label: str = "",
) -> RunResult:
    """Run pytest as a local subprocess with timeout handling."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    env.update(
        _base_pytest_env(
            output_format=output_format,
            snapshot_condition=_is_snapshot_condition(condition),
        )
    )
    args = _build_pytest_args(
        python_cmd=sys.executable,
        python_args=python_args,
        condition=condition,
    )

    monitored = run_monitored_subprocess(
        cmd=args,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
        progress_callback=progress_update,
        progress_event="pytest_activity",
        progress_base_payload={
            "owner": "pytest",
            "phase_label": str(progress_phase_label or "pytest"),
            "source": "phase",
        },
    )
    if monitored.timed_out:
        timeout_message = f"pytest timed out after {timeout_sec}s"
        stderr = (
            timeout_message if not monitored.stderr else f"{monitored.stderr}\n{timeout_message}"
        )
        return RunResult(
            returncode=PYTEST_TIMEOUT_RETURNCODE,
            duration_sec=float(monitored.duration_sec),
            stdout=monitored.stdout,
            stderr=stderr,
            timed_out=True,
            timeout_sec=timeout_sec,
        )
    return RunResult(
        returncode=int(monitored.returncode),
        duration_sec=float(monitored.duration_sec),
        stdout=monitored.stdout,
        stderr=monitored.stderr,
        timed_out=False,
        timeout_sec=None,
    )


def _status_code_from_wait_result(result: Any) -> int:
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        raw = result.get("StatusCode")
        if isinstance(raw, int):
            return raw
        try:
            return int(raw)
        except Exception:
            return 1
    return 1


def _is_container_wait_timeout(exc: Exception) -> bool:
    name = type(exc).__name__
    if "ReadTimeout" in name:
        return True
    text = str(exc).lower()
    return "read timed out" in text or "timed out" in text


def _run_pytest_testcontainers(
    *,
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    output_format: str,
    execution: ExecutionConfig,
) -> RunResult:
    """Run pytest inside a one-shot Docker container via testcontainers."""
    try:
        from testcontainers.core.container import DockerContainer
    except Exception as exc:
        raise RuntimeError(
            "testcontainers execution requested but dependency is missing "
            f"({type(exc).__name__}: {exc}). Install llmdebug[evals]."
        ) from exc

    container_env = _base_pytest_env(
        output_format=output_format,
        snapshot_condition=_is_snapshot_condition(condition),
    )
    if execution.container_mount_repo_src:
        container_env["PYTHONPATH"] = "/repo_src"
    container_env.update(execution.container_env)

    command = _build_pytest_args(
        python_cmd="python",
        python_args=python_args,
        condition=condition,
    )
    tmpfs_size = max(1, int(execution.container_tmpfs_size_mb))
    kwargs: dict[str, Any] = {
        "working_dir": "/workspace",
        "network_mode": execution.container_network,
        "read_only": True,
        "tmpfs": {"/tmp": f"rw,noexec,nosuid,nodev,size={tmpfs_size}m"},
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "pids_limit": int(execution.container_pids_limit),
        "mem_limit": execution.container_memory,
        "nano_cpus": int(float(execution.container_cpus) * 1_000_000_000),
        "user": execution.container_user,
        "labels": {
            EVAL_CONTAINER_LABEL_KEY: EVAL_CONTAINER_LABEL_VALUE,
        },
    }

    container = (
        DockerContainer(execution.container_image)
        .with_volume_mapping(str(cwd), "/workspace", mode="rw")
        .with_command(command)
        .with_kwargs(**kwargs)
    )
    if execution.container_mount_repo_src:
        container = container.with_volume_mapping(str(_SRC_ROOT), "/repo_src", mode="ro")
    for key, value in sorted(container_env.items()):
        container = container.with_env(key, value)

    start = time.perf_counter()
    stdout = ""
    stderr = ""
    timed_out = False
    returncode = 1
    try:
        container.start()
        wrapped = container.get_wrapped_container()
        if wrapped is None:
            raise RuntimeError("testcontainer started without wrapped container handle")
        try:
            wait_result = wrapped.wait(timeout=timeout_sec)
            returncode = _status_code_from_wait_result(wait_result)
        except Exception as exc:
            if _is_container_wait_timeout(exc):
                timed_out = True
                returncode = PYTEST_TIMEOUT_RETURNCODE
                try:
                    wrapped.kill()
                except Exception:
                    pass
            else:
                raise RuntimeError(f"container wait failed: {type(exc).__name__}: {exc}") from exc

        out_raw, err_raw = container.get_logs()
        stdout = _coerce_subprocess_output(out_raw)
        stderr = _coerce_subprocess_output(err_raw)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "testcontainers execution failed: "
            f"{type(exc).__name__}: {exc} "
            f"(image={execution.container_image!r}, network={execution.container_network!r})"
        ) from exc
    finally:
        try:
            container.stop()
        except Exception:
            pass

    end = time.perf_counter()
    if timed_out:
        timeout_message = f"pytest timed out after {timeout_sec}s"
        stderr = timeout_message if not stderr else f"{stderr}\n{timeout_message}"
        return RunResult(
            returncode=PYTEST_TIMEOUT_RETURNCODE,
            duration_sec=float(end - start),
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            timeout_sec=timeout_sec,
        )
    return RunResult(
        returncode=int(returncode),
        duration_sec=float(end - start),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        timeout_sec=None,
    )


def _truncate_persisted_pytest_output(
    text: str,
    *,
    tail_lines: int,
    hard_char_cap: int | None = None,
) -> str:
    normalized = text or ""
    normalized_tail_lines = max(0, int(tail_lines))
    if normalized_tail_lines > 0 and normalized:
        had_trailing_newline = normalized.endswith("\n")
        lines = normalized.splitlines()
        total_lines = len(lines)
        if total_lines > normalized_tail_lines:
            dropped_lines = total_lines - normalized_tail_lines
            normalized = "\n".join(lines[-normalized_tail_lines:])
            if had_trailing_newline:
                normalized += "\n"
            normalized = (
                f"...[llmdebug: truncated {dropped_lines} earlier line(s)]...\n{normalized}"
            )

    resolved_hard_char_cap = (
        PERSISTED_PYTEST_OUTPUT_HARD_CHAR_CAP if hard_char_cap is None else hard_char_cap
    )
    normalized_hard_char_cap = max(0, int(resolved_hard_char_cap))
    if normalized_hard_char_cap > 0 and len(normalized) > normalized_hard_char_cap:
        dropped_chars = len(normalized) - normalized_hard_char_cap
        normalized = (
            f"...[llmdebug: truncated {dropped_chars} earlier character(s)]...\n"
            f"{normalized[-normalized_hard_char_cap:]}"
        )
    return normalized


def _truncate_snapshot_string_for_artifact(value: str, *, max_chars: int) -> str:
    normalized_max_chars = max(0, int(max_chars))
    if normalized_max_chars <= 0 or len(value) <= normalized_max_chars:
        return value
    dropped_chars = len(value) - normalized_max_chars
    marker = f"...[llmdebug: snapshot string truncated {dropped_chars} chars]..."
    if normalized_max_chars <= len(marker):
        return marker[:normalized_max_chars]
    keep_chars = normalized_max_chars - len(marker)
    head_chars = keep_chars // 2
    tail_chars = keep_chars - head_chars
    return value[:head_chars] + marker + value[-tail_chars:]


def _truncate_snapshot_for_artifact(
    snapshot: dict[str, Any], *, max_string_chars: int
) -> dict[str, Any]:
    normalized_max_string_chars = max(0, int(max_string_chars))

    def _rewrite(value: Any) -> Any:
        if isinstance(value, str):
            return _truncate_snapshot_string_for_artifact(
                value, max_chars=normalized_max_string_chars
            )
        if isinstance(value, dict):
            return {key: _rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        return value

    rewritten = _rewrite(snapshot)
    return rewritten if isinstance(rewritten, dict) else dict(snapshot)


def _write_run_outputs(
    out_dir: Path,
    run: RunResult,
    *,
    prefix: str,
    tail_lines: int = DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
) -> None:
    stdout = _truncate_persisted_pytest_output(run.stdout or "", tail_lines=tail_lines)
    stderr = _truncate_persisted_pytest_output(run.stderr or "", tail_lines=tail_lines)
    (out_dir / f"{prefix}_stdout.txt").write_text(stdout, encoding="utf-8")
    (out_dir / f"{prefix}_stderr.txt").write_text(stderr, encoding="utf-8")
    combined = stdout + ("\n" if stdout and stderr else "") + stderr
    combined = _truncate_persisted_pytest_output(combined, tail_lines=0)
    (out_dir / f"{prefix}_combined.txt").write_text(combined, encoding="utf-8")


def _remaining_attempt_propose_budget_sec(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, float(deadline_monotonic) - time.monotonic())


def _build_attempt_budget_exhausted_patch_result(
    *,
    patcher: Any,
    remaining_budget_sec: float,
) -> Any:
    from evals.patchers.base import PatchResult

    meta: dict[str, Any] = {
        "error": "request_budget_exceeded",
        "effective_timeout_sec": 0.0,
        "remaining_budget_sec": max(0.0, float(remaining_budget_sec)),
        "llm_call_count": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "transport_retry_count": 0,
        "attempt_shared_propose_deadline_enabled": True,
        "attempt_shared_propose_budget_exhausted": True,
        "attempt_shared_propose_budget_remaining_sec": max(0.0, float(remaining_budget_sec)),
    }
    patcher_name = str(getattr(patcher, "name", "") or "")
    if patcher_name:
        meta["patcher"] = patcher_name
    model_name = str(getattr(patcher, "model", "") or "")
    if model_name:
        meta["model"] = model_name
    base_url = str(getattr(patcher, "base_url", "") or "")
    if base_url:
        meta["base_url"] = base_url
    response_mode = str(getattr(patcher, "response_mode", "") or "")
    if response_mode:
        meta["response_mode"] = response_mode
    return PatchResult(diff=None, meta=meta)


def _build_attempt_budget_exhausted_plan_result(
    *,
    patcher: Any,
    remaining_budget_sec: float,
) -> Any:
    from evals.patchers.base import PatchPlanResult

    meta: dict[str, Any] = {
        "error": "request_budget_exceeded",
        "effective_timeout_sec": 0.0,
        "remaining_budget_sec": max(0.0, float(remaining_budget_sec)),
        "llm_call_count": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "transport_retry_count": 0,
        "attempt_shared_propose_deadline_enabled": True,
        "attempt_shared_propose_budget_exhausted": True,
        "attempt_shared_propose_budget_remaining_sec": max(0.0, float(remaining_budget_sec)),
    }
    patcher_name = str(getattr(patcher, "name", "") or "")
    if patcher_name:
        meta["patcher"] = patcher_name
    model_name = str(getattr(patcher, "model", "") or "")
    if model_name:
        meta["model"] = model_name
    base_url = str(getattr(patcher, "base_url", "") or "")
    if base_url:
        meta["base_url"] = base_url
    response_mode = str(getattr(patcher, "response_mode", "") or "")
    if response_mode:
        meta["response_mode"] = response_mode
    return PatchPlanResult(plan_text=None, raw_response=None, meta=meta)


def _patcher_with_shared_attempt_budget(
    *,
    patcher: Any,
    attempt_deadline_monotonic: float | None,
) -> tuple[Any, float | None, float | None, bool]:
    remaining_budget_sec = _remaining_attempt_propose_budget_sec(attempt_deadline_monotonic)
    if remaining_budget_sec is None:
        return patcher, None, None, False

    if str(getattr(patcher, "name", "")).strip() != "openai":
        return patcher, remaining_budget_sec, None, False

    raw_max_propose_sec = getattr(patcher, "max_propose_sec", 0.0)
    try:
        configured_max_propose_sec = float(raw_max_propose_sec)
    except (TypeError, ValueError):
        configured_max_propose_sec = 0.0
    if configured_max_propose_sec <= 0:
        return patcher, remaining_budget_sec, None, False

    clamped_max_propose_sec = max(0.0, min(configured_max_propose_sec, remaining_budget_sec))
    if abs(clamped_max_propose_sec - configured_max_propose_sec) < 1e-6:
        return patcher, remaining_budget_sec, clamped_max_propose_sec, False

    try:
        clamped_patcher = replace(patcher, max_propose_sec=clamped_max_propose_sec)
    except Exception:
        return patcher, remaining_budget_sec, configured_max_propose_sec, False
    return clamped_patcher, remaining_budget_sec, clamped_max_propose_sec, True


def _propose_patch_with_snapshot_context(
    *,
    patcher: Any,
    prompt: str,
    context_dir: Path,
    cwd: Path,
    snapshot_full_text: str | None,
    snapshot_artifact_text: str | None,
    system_prompt_override: str | None = None,
    attempt_deadline_monotonic: float | None = None,
) -> Any:
    """Provide full snapshot context during propose, then restore artifact-capped file."""
    from evals.patchers.base import PatchResult

    (
        effective_patcher,
        remaining_budget_sec,
        call_budget_sec,
        budget_clamped,
    ) = _patcher_with_shared_attempt_budget(
        patcher=patcher,
        attempt_deadline_monotonic=attempt_deadline_monotonic,
    )
    if remaining_budget_sec is not None and remaining_budget_sec <= 0:
        return _build_attempt_budget_exhausted_patch_result(
            patcher=patcher,
            remaining_budget_sec=remaining_budget_sec,
        )
    if (
        effective_patcher is not patcher
        and hasattr(patcher, "_current_progress_callback")
        and hasattr(effective_patcher, "set_progress_callback")
    ):
        effective_patcher.set_progress_callback(patcher._current_progress_callback())

    def _call_propose_patch() -> Any:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "context_dir": context_dir,
            "cwd": cwd,
        }
        if system_prompt_override is not None:
            kwargs["system_prompt_override"] = system_prompt_override
        return effective_patcher.propose_patch(**kwargs)

    if (
        snapshot_full_text is None
        or snapshot_artifact_text is None
        or snapshot_full_text == snapshot_artifact_text
    ):
        result = _call_propose_patch()
    else:
        snapshot_path = context_dir / "snapshot.json"
        snapshot_path.write_text(snapshot_full_text, encoding="utf-8")
        try:
            result = _call_propose_patch()
        finally:
            snapshot_path.write_text(snapshot_artifact_text, encoding="utf-8")

    if not hasattr(result, "diff") or not hasattr(result, "meta"):
        return result

    result_meta = result.meta
    if not isinstance(result_meta, dict):
        return result
    enriched_meta = dict(result_meta)
    if remaining_budget_sec is not None:
        enriched_meta["attempt_shared_propose_deadline_enabled"] = True
        enriched_meta["attempt_shared_propose_budget_remaining_sec"] = remaining_budget_sec
        if call_budget_sec is not None:
            enriched_meta["attempt_shared_propose_call_budget_sec"] = call_budget_sec
        if budget_clamped:
            enriched_meta["attempt_shared_propose_budget_clamped"] = True
    return PatchResult(diff=result.diff, meta=enriched_meta)


def _plan_patch_with_snapshot_context(
    *,
    patcher: Any,
    prompt: str,
    context_dir: Path,
    cwd: Path,
    snapshot_full_text: str | None,
    snapshot_artifact_text: str | None,
    attempt_deadline_monotonic: float | None = None,
) -> Any:
    """Provide full snapshot context during repair planning, then restore artifact-capped file."""
    from evals.patchers.base import PatchPlanResult

    (
        effective_patcher,
        remaining_budget_sec,
        call_budget_sec,
        budget_clamped,
    ) = _patcher_with_shared_attempt_budget(
        patcher=patcher,
        attempt_deadline_monotonic=attempt_deadline_monotonic,
    )
    if remaining_budget_sec is not None and remaining_budget_sec <= 0:
        return _build_attempt_budget_exhausted_plan_result(
            patcher=patcher,
            remaining_budget_sec=remaining_budget_sec,
        )
    if (
        effective_patcher is not patcher
        and hasattr(patcher, "_current_progress_callback")
        and hasattr(effective_patcher, "set_progress_callback")
    ):
        effective_patcher.set_progress_callback(patcher._current_progress_callback())

    def _call_plan_patch() -> Any:
        return effective_patcher.plan_patch(prompt=prompt, context_dir=context_dir, cwd=cwd)

    if (
        snapshot_full_text is None
        or snapshot_artifact_text is None
        or snapshot_full_text == snapshot_artifact_text
    ):
        result = _call_plan_patch()
    else:
        snapshot_path = context_dir / "snapshot.json"
        snapshot_path.write_text(snapshot_full_text, encoding="utf-8")
        try:
            result = _call_plan_patch()
        finally:
            snapshot_path.write_text(snapshot_artifact_text, encoding="utf-8")

    if not hasattr(result, "plan_text") or not hasattr(result, "meta"):
        return result

    result_meta = result.meta
    if not isinstance(result_meta, dict):
        return result
    enriched_meta = dict(result_meta)
    if remaining_budget_sec is not None:
        enriched_meta["attempt_shared_propose_deadline_enabled"] = True
        enriched_meta["attempt_shared_propose_budget_remaining_sec"] = remaining_budget_sec
        if call_budget_sec is not None:
            enriched_meta["attempt_shared_propose_call_budget_sec"] = call_budget_sec
        if budget_clamped:
            enriched_meta["attempt_shared_propose_budget_clamped"] = True
    return PatchPlanResult(
        plan_text=result.plan_text,
        raw_response=getattr(result, "raw_response", None),
        meta=enriched_meta,
    )


def _coerce_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _timeout_fields(run: RunResult) -> dict[str, Any]:
    if run.timed_out:
        return {
            "timed_out": True,
            "timeout_sec": run.timeout_sec,
        }
    return {}


def _enrich_failure_context(
    *,
    base: dict[str, Any],
    note: str,
    patch_meta: dict[str, Any] | None,
    run: RunResult,
) -> dict[str, Any]:
    enriched = dict(base)
    enriched["note"] = str(note or "")
    enriched["timed_out"] = bool(run.timed_out)
    enriched["timeout_sec"] = int(run.timeout_sec) if isinstance(run.timeout_sec, int) else None
    meta = patch_meta if isinstance(patch_meta, dict) else {}
    verifier_reasons_raw = meta.get("patch_verifier_reasons")
    verifier_reasons = (
        [str(item) for item in verifier_reasons_raw]
        if isinstance(verifier_reasons_raw, list)
        else []
    )
    enriched["patch_verifier_rejected"] = bool(meta.get("patch_verifier_enabled")) and (
        meta.get("patch_verifier_passed") is False
    )
    enriched["patch_verifier_reasons"] = verifier_reasons
    enriched["patch_verifier_scope"] = str(meta.get("patch_verifier_scope") or "")
    enriched["patch_verifier_score"] = _adaptive_optional_int(meta.get("patch_verifier_score"))
    enriched["patch_verifier_source_scope_matched"] = _adaptive_optional_bool(
        meta.get("patch_verifier_source_scope_matched")
    )
    enriched["adaptive_progress_reason"] = str(meta.get("adaptive_progress_reason") or "")
    enriched["adaptive_helper_failure_kind"] = str(meta.get("adaptive_helper_failure_kind") or "")
    enriched["adaptive_helper_outcome"] = str(meta.get("adaptive_helper_outcome") or "")
    enriched["adaptive_helper_need_more_context"] = _adaptive_optional_bool(
        meta.get("adaptive_helper_need_more_context")
    )
    return enriched


def read_latest_snapshot(out_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read the most recent llmdebug snapshot from *out_dir*.

    Returns:
        Tuple of (snapshot_dict_or_None, error_string_or_None).
    """
    try:
        from llmdebug.output import get_latest_snapshot

        snap = get_latest_snapshot(str(out_dir))
        return snap, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _build_failure_signature_for_attempt(*, run: RunResult, snapshot: dict[str, Any] | None) -> str:
    """Compute a stable failure signature for an attempt outcome."""
    try:
        from llmdebug.rca_state import build_failure_signature

        return build_failure_signature(snapshot, stdout=run.stdout, stderr=run.stderr)
    except Exception as e:
        sys.stderr.write(
            f"llmdebug eval: build_failure_signature failed, using text fallback: "
            f"{type(e).__name__}: {e}\n"
        )
        text = (run.stderr or "").strip() or (run.stdout or "").strip() or "unknown_failure"
        head = "\n".join([line for line in text.splitlines() if line.strip()][-3:])
        digest = xxh64_hexdigest_text(head)[:16]
        return f"TextFailure:{digest}"


def _build_structured_previous_attempt_diff(
    previous_failure_context: dict[str, Any] | None,
    *,
    current_failure_context: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a structured diff between consecutive attempt failures.

    Compares snapshots (when available) or falls back to text-based
    comparison of pytest output.
    """
    if not previous_failure_context:
        return None

    prev_attempt = int(previous_failure_context.get("attempt", 0))
    curr_attempt = int(current_failure_context.get("attempt", 0))
    prev_signature = str(previous_failure_context.get("failure_signature") or "")
    curr_signature = str(current_failure_context.get("failure_signature") or "")

    prev_snapshot = previous_failure_context.get("snapshot")
    curr_snapshot = current_failure_context.get("snapshot")
    if isinstance(prev_snapshot, dict) and isinstance(curr_snapshot, dict):
        try:
            from llmdebug.snapshot_diff import diff_snapshots, structured_diff_summary

            diff = diff_snapshots(prev_snapshot, curr_snapshot)
            structured = structured_diff_summary(diff)
            structured["attempt_prev"] = f"attempt_{prev_attempt:02d}"
            structured["attempt_curr"] = f"attempt_{curr_attempt:02d}"
            return structured
        except Exception as e:
            sys.stderr.write(
                f"llmdebug eval: structured diff failed, using text fallback: "
                f"{type(e).__name__}: {e}\n"
            )
            pass

    prev_output = (
        str(previous_failure_context.get("stderr") or "")
        + "\n"
        + str(previous_failure_context.get("stdout") or "")
    ).strip()
    curr_output = (
        str(current_failure_context.get("stderr") or "")
        + "\n"
        + str(current_failure_context.get("stdout") or "")
    ).strip()

    return {
        "attempt_prev": f"attempt_{prev_attempt:02d}",
        "attempt_curr": f"attempt_{curr_attempt:02d}",
        "failure_signature_same": prev_signature == curr_signature,
        "evidence_delta": {
            "new_frames": [],
            "dropped_frames": [],
            "message_changed": prev_output != curr_output,
        },
        "change_delta": {
            "touched_files": [],
            "touched_functions": [],
        },
        "hypothesis_delta": "unknown",
        "verification_delta": {
            "old_status": "unknown",
            "new_status": "unknown",
            "status_changed": False,
        },
    }


from evals.attempt_verification import (  # noqa: E402
    apply_diff_with_optional_recovery,
    evaluate_patch_sanitize,
    evaluate_syntax_preflight,
    make_syntax_preflight_state,
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


def _infer_test_source_scope(
    *,
    cwd: Path,
    crash_file: str | None,
    failing_test_file: str | None,
    enabled: bool,
    file_index_lines: Sequence[str] | None = None,
) -> list[str]:
    if not enabled:
        return []

    candidates: list[str] = []
    if isinstance(crash_file, str) and crash_file and _is_test_file_path(crash_file):
        candidates.append(crash_file)
    if (
        isinstance(failing_test_file, str)
        and failing_test_file
        and _is_test_file_path(failing_test_file)
        and failing_test_file not in candidates
    ):
        candidates.append(failing_test_file)

    inferred: list[str] = []
    for rel in candidates:
        for neighbor in _import_neighbors(root=cwd, rel_path=rel, limit=12):
            normalized = str(PurePosixPath(neighbor.replace("\\", "/")))
            if normalized not in inferred:
                inferred.append(normalized)
        for neighbor in _extract_subprocess_script_neighbors(
            root=cwd,
            rel_path=rel,
            limit=12,
            file_index_lines=file_index_lines,
        ):
            normalized = str(PurePosixPath(neighbor.replace("\\", "/")))
            if normalized not in inferred:
                inferred.append(normalized)
    return inferred


def _score_patch_relevance(
    *,
    diff: str,
    cwd: Path,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    min_score: int = PATCH_VERIFIER_MIN_SCORE,
    infer_test_source_scope: bool = True,
    file_index_lines: Sequence[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Heuristic relevance score for a proposed diff.

    Checks whether the patch touches the crash file, its import
    neighbors, the inferred source scope, or test files.  Returns
    (passed, details_dict) with ``passed=True`` when the score
    meets ``min_score``.
    """
    patch_files = _extract_candidate_patch_files(diff)
    crash_file = _crash_file_from_snapshot(snapshot)
    crash_neighbors = (
        _import_neighbors(root=cwd, rel_path=crash_file, limit=12) if crash_file else []
    )
    crash_neighbor_set = {str(PurePosixPath(path.replace("\\", "/"))) for path in crash_neighbors}
    failing_test_file = _detect_failing_test_file(
        stdout=stdout,
        stderr=stderr,
        root=cwd,
        file_index_lines=file_index_lines,
    )
    inferred_source_scope = _infer_test_source_scope(
        cwd=cwd,
        crash_file=crash_file,
        failing_test_file=failing_test_file,
        enabled=infer_test_source_scope,
        file_index_lines=file_index_lines,
    )
    inferred_source_scope_set = {
        str(PurePosixPath(path.replace("\\", "/"))) for path in inferred_source_scope
    }

    substantive_change = _patch_contains_substantive_change(diff)
    matches_crash_file = _patch_mentions_crash_file(patch_files=patch_files, crash_file=crash_file)
    matches_crash_neighbor = any(path in crash_neighbor_set for path in patch_files)
    matches_inferred_source_scope = _patch_mentions_scope_files(
        patch_files=patch_files,
        scope_files=inferred_source_scope_set,
    )
    touches_test_file = any(_is_test_file_path(path) for path in patch_files)
    touches_failing_test_file = _patch_mentions_failing_test_context(
        patch_files=patch_files,
        failing_test_file=failing_test_file,
    )

    score = 0
    reasons: list[str] = []
    if substantive_change:
        score += 1
        reasons.append("substantive_change")
    else:
        score -= 2
        reasons.append("no_substantive_change")

    if crash_file:
        if matches_crash_file:
            score += 2
            reasons.append("matches_crash_file")
        elif matches_crash_neighbor:
            score += 1
            reasons.append("matches_crash_neighbor")
        elif matches_inferred_source_scope:
            score += 1
            reasons.append("matches_inferred_source_scope")
        else:
            score -= 1
            reasons.append("misses_crash_scope")
    else:
        if matches_inferred_source_scope:
            score += 1
            reasons.append("matches_inferred_source_scope")
        else:
            reasons.append("no_crash_file_available")

    if touches_failing_test_file:
        score -= 2
        reasons.append("touches_failing_test_file")
    elif touches_test_file:
        score -= 1
        reasons.append("touches_test_file")
    else:
        score += 1
        reasons.append("non_test_edit")

    normalized_min_score = max(0, int(min_score))
    passed = bool(score >= normalized_min_score)

    return passed, {
        "patch_verifier_passed": passed,
        "patch_verifier_score": int(score),
        "patch_verifier_min_score": int(normalized_min_score),
        "patch_verifier_reasons": reasons,
        "patch_verifier_patch_files": patch_files,
        "patch_verifier_crash_file": crash_file,
        "patch_verifier_crash_file_matched": matches_crash_file,
        "patch_verifier_crash_neighbor_matched": matches_crash_neighbor,
        "patch_verifier_crash_neighbors": sorted(crash_neighbor_set),
        "patch_verifier_inferred_source_scope": sorted(inferred_source_scope_set),
        "patch_verifier_source_scope_matched": matches_inferred_source_scope,
        "patch_verifier_failing_test_file": failing_test_file,
        "patch_verifier_touches_test_file": touches_test_file,
        "patch_verifier_touches_failing_test_file": touches_failing_test_file,
        "patch_verifier_substantive_change": substantive_change,
    }


if __name__ == "__main__":
    raise SystemExit(main())
