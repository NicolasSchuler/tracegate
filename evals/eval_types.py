"""Data types, enums, and constants for the eval harness.

Extracted from ``evals.run_eval`` to keep the main orchestrator lean.
Re-exported by ``evals.run_eval`` for backward compatibility.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from evals.adaptive_budget_online_policy import (
    BUDGET_POLICY_ALGOS,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_MODES,
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
    ONLINE_POLICY_ALGOS,
    ONLINE_POLICY_MODES,
    PolicyDecision,
)
from evals.lambda_rlm_backend import (
    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
    DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
)
from evals.oracle_integrity import PytestCounts
from evals.policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    POLICY_FEATURE_SCALING_PROFILES,
)
from evals.runtime_resolution import (
    DEFAULT_CONTAINER_IMAGE_FALLBACK,
)
from evals.runtime_resolution import (
    OPENAI_API_KEY_ENV_FALLBACKS as RUNTIME_OPENAI_API_KEY_ENV_FALLBACKS,
)

if TYPE_CHECKING:
    from evals.patchers.payload_types import ContextRequestEntry, ContextRequestPayload
else:
    ContextRequestEntry = dict[str, Any]  # type: ignore[misc,assignment]
    ContextRequestPayload = dict[str, Any]  # type: ignore[misc,assignment]


OPENAI_API_KEY_ENV_FALLBACKS = RUNTIME_OPENAI_API_KEY_ENV_FALLBACKS


class Condition(str, Enum):
    """Supported eval conditions for the repair benchmark harness."""

    BASE_REPAIR = "base_repair"
    COT_REPAIR = "cot_repair"
    LAYERED_ADAPTIVE_GATED = "layered_adaptive_gated"
    LAYERED_ADAPTIVE_LLM = "layered_adaptive_llm"
    LDB = "ldb"
    INTERVENOR = "intervenor"
    TRACECODER = "tracecoder"
    LAMBDA_RLM = "lambda_rlm"
    RLM = "rlm"


BASELINE_CONDITION_BY_NAME = {
    "ldb": Condition.LDB,
    "intervenor": Condition.INTERVENOR,
    "tracecoder": Condition.TRACECODER,
    "lambda_rlm": Condition.LAMBDA_RLM,
    "rlm": Condition.RLM,
}


def baseline_condition_for_name(name: str) -> Condition:
    normalized = str(name or "").strip().lower()
    try:
        return BASELINE_CONDITION_BY_NAME[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(BASELINE_CONDITION_BY_NAME))
        raise ValueError(
            f"unknown baseline name {normalized!r}; expected one of: {available}"
        ) from exc


def is_baseline_condition(condition: Condition) -> bool:
    return condition in set(BASELINE_CONDITION_BY_NAME.values())


CONDITION_FAMILY_NATIVE = "native"
CONDITION_FAMILY_ADAPTIVE = "adaptive"
CONDITION_FAMILY_EXTERNAL_BASELINE = "external_baseline"
CONDITION_FAMILY_UNKNOWN = "unknown"
CONDITION_FAMILIES: dict[str, str] = {
    Condition.BASE_REPAIR.value: CONDITION_FAMILY_NATIVE,
    Condition.COT_REPAIR.value: CONDITION_FAMILY_NATIVE,
    Condition.LAYERED_ADAPTIVE_GATED.value: CONDITION_FAMILY_ADAPTIVE,
    Condition.LAYERED_ADAPTIVE_LLM.value: CONDITION_FAMILY_ADAPTIVE,
    **dict.fromkeys(BASELINE_CONDITION_BY_NAME, CONDITION_FAMILY_EXTERNAL_BASELINE),
}
"""The one condition-family registry for analysis and reporting (2026-10-04 review, 2.1)."""


def normalize_condition_name(value: object) -> str:
    """Canonical condition name in records and reports: the stripped string.

    ``analyze_results`` didn't strip while ``budget_risk_report`` and
    ``extract_failed_cases`` did, so one condition could appear twice.
    """
    return str(value or "").strip()


def condition_family(value: object) -> str:
    """``native``, ``adaptive``, ``external_baseline`` or ``unknown``.

    ``eval_summary`` used to detect adaptive conditions by the substring "adaptive".
    """
    return CONDITION_FAMILIES.get(normalize_condition_name(value), CONDITION_FAMILY_UNKNOWN)


VALID_CATEGORIES = {
    "hidden_state",
    "mutability_aliasing",
    "async_temporal",
    "cross_file_contract",
    "data_shape_runtime",
    "parse_value",
    "path_import_env",
    "traceback_controls",
    # Extended categories for imported datasets.
    "syntax_error",
    "reference_error",
    "logic_error",
    "type_coercion",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_SNAPSHOT_DEPENDENCIES = {"required", "helpful", "none"}
PYTEST_TIMEOUT_RETURNCODE = 124
RETRY_GATE_MAX_CONSECUTIVE_BLOCKS = 1
"""ATT-05 / D9: the no-new-evidence retry gate may block at most once in a row."""
EVAL_RECORD_SCHEMA_VERSION = 7
RECORD_MODES = {"full", "hash", "none"}
CONTEXT_PROTOCOLS = {"monolithic", "staged"}
RETRY_PROMPT_MODES = {"full", "delta", "auto"}
RETRY_CONTEXT_MODES = {"fresh", "history"}
CONTEXT_COMPACTION_MODES = {"off", "auto_overflow", "always"}
CONTEXT_COMPACTION_SCOPE_TOKENS = {"snapshot", "pytest", "files"}
PROMPT_ORDER_MODES = {"current", "crash_first", "evidence_last"}
ADAPTIVE_REVEAL_MODES = {"full", "traceback_first"}
SNAPSHOT_FRAME_PRUNING_MODES = {"off", "heuristic"}
ADAPTIVE_GATE_POLICIES = {"aggressive", "balanced", "conservative"}
ADAPTIVE_HELPER_BUDGET_PROFILES = {"default", "compact"}
OPENAI_RESPONSE_MODES = {"edits", "diff", "full_file"}
ADAPTIVE_OPENAI_HELPER_RESPONSE_MODES = {"inherit", *OPENAI_RESPONSE_MODES}
ADAPTIVE_FEATURE_SCALING_PROFILES = set(POLICY_FEATURE_SCALING_PROFILES)
OPENAI_RESPONSE_MODE_AUTOSWITCH_MODES = {"off", "retry_to_diff"}
ADAPTIVE_ONLINE_POLICY_MODES = set(ONLINE_POLICY_MODES)
ADAPTIVE_ONLINE_POLICY_ALGOS = set(ONLINE_POLICY_ALGOS)
ADAPTIVE_BUDGET_POLICY_MODES = set(BUDGET_POLICY_MODES)
ADAPTIVE_BUDGET_POLICY_ALGOS = set(BUDGET_POLICY_ALGOS)
ADAPTIVE_TEP_MODES = {"off", "legacy", "full"}
OPENAI_CONTEXT_WINDOW_CHARS_PER_TOKEN = 4
OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS = 1024
OPENAI_MIN_PROMPT_TOKENS = 1024
"""Smallest usable prompt budget a run may start with (LLM-06)."""
ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT = 0.08
ADAPTIVE_REWARD_MU_LATENCY_DEFAULT = 0.05
ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT = 1000.0
ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT = 10.0
ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT = 0.5
ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT = 0.15
ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT = 0.10
ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT = 1.0
ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT = 1.0
ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT = 1.0
ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT = 0.5
DEFAULT_ADAPTIVE_ONLINE_ALPHA = 0.6
DEFAULT_ADAPTIVE_ONLINE_EPSILON = 0.10
DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS = 50
DEFAULT_ADAPTIVE_ONLINE_DECAY = 0.95
DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES = "search_only=-1.00,frame_only=-0.50"
DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA = 0.4
DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON = 0.05
DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS = 20
DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY = 0.95
DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS = "1.0,1.1,1.2,1.35,1.5"
DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE = 1
# Auxiliary reward signal weights (enabled by default; disable with --no-adaptive-aux-reward).
ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT = 0.15
ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT = 0.10
ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT = 0.05
ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT = 0.10
ADAPTIVE_TEP_BETA_INIT_DEFAULT = 1.0
ADAPTIVE_TEP_BETA_DECAY_DEFAULT = 0.9
ADAPTIVE_TEP_EQUILIBRIUM_MAX_ITERS_DEFAULT = 20
ADAPTIVE_TEP_EQUILIBRIUM_WINDOW_DEFAULT = 3
ADAPTIVE_TEP_EQUILIBRIUM_VARIANCE_THRESHOLD_DEFAULT = 0.5
ADAPTIVE_TEP_VALIDATION_BUFFER_SIZE_DEFAULT = 4
PATCH_VERIFIER_SCOPE_DISABLED = "disabled"
PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY = "adaptive_only"
PATCH_VERIFIER_SCOPE_ALL = "all"
PATCH_VERIFIER_MIN_SCORE = 2
DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES = 2
DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS = 1
DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP = 150_000
DEFAULT_TRANSPORT_FAIL_FAST_STREAK = 5
DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED = True
DEFAULT_OPENAI_TRANSPORT_RECOVERY_MAX_WAIT_SEC = 60.0
DEFAULT_OPENAI_TRANSPORT_RECOVERY_POLL_SEC = 2.0
DEFAULT_PROGRESS_HEARTBEAT_SEC = 30.0
DEFAULT_PYTEST_OUTPUT_TAIL_LINES = 4_000
PERSISTED_PYTEST_OUTPUT_HARD_CHAR_CAP = 250_000
DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS = 16_000
ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED = "not_invoked"
ADAPTIVE_HELPER_OUTCOME_MATERIALIZED = "materialized"
ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT = "semantic_no_context"
ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY = "compat_reasoning_only"
ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD = "invalid_payload"
ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED = "transport_failed"
ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND = "unavailable_backend"
ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED = "request_budget_exceeded"
ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY = "skipped_policy"
ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL = "skipped_guardrail"
ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED = "skipped_unsupported"
HELPER_STAGE_A_ACTIONS = frozenset(
    {
        ACTION_INVOKE_HELPER,
        ACTION_FRAME_ONLY,
        ACTION_FILE_ONLY,
        ACTION_SEARCH_ONLY,
        ACTION_RETRY_HELPER_STRICT_JSON,
    }
)
HELPER_RESTRICTED_STAGE_A_ACTIONS = frozenset(
    {
        ACTION_FRAME_ONLY,
        ACTION_FILE_ONLY,
        ACTION_SEARCH_ONLY,
    }
)
HELPER_SEED_BUNDLE_ACTIONS = HELPER_STAGE_A_ACTIONS - HELPER_RESTRICTED_STAGE_A_ACTIONS
"""Unrestricted Stage-A actions, the only ones that get the whole-file seed bundle (ATT-04)."""
HELPER_FULL_SNAPSHOT_ACTIONS = frozenset({ACTION_COMPACT_THEN_PATCH})
"""Actions whose Stage-B prompt carries the full snapshot under full reveal without a Stage-A
request: compaction works on the full context (ATT-06, D5)."""
_HELPER_ACTION_REQUEST_MINI_BUDGETS: dict[str, dict[str, int]] = {
    ACTION_FRAME_ONLY: {"frame": 1, "local_var": 2},
    ACTION_FILE_ONLY: {"file": 1},
    ACTION_SEARCH_ONLY: {"search": 2},
}
HELPER_CONTEXT_MATERIALIZING_ACTIONS = frozenset(
    {
        ACTION_INVOKE_HELPER,
        ACTION_FRAME_ONLY,
        ACTION_FILE_ONLY,
        ACTION_SEARCH_ONLY,
        ACTION_RETRY_HELPER_STRICT_JSON,
        ACTION_SNAPSHOT_LITE,
        ACTION_FRAME_LOCALS_LITE,
        ACTION_SNAPSHOT_FULL_LITE,
        ACTION_COMPACT_THEN_PATCH,
    }
)
PROGRESS_VERBOSITIES = {"basic", "detailed", "trace"}
PATCHER_CONTEXT_LIMIT_ERROR_MARKERS = (
    "exceeds the available context size",
    "context size",
    "greater than the context length",
    "tokens to keep from the initial prompt is greater than the context length",
    "maximum context length",
    "prompt is too long",
)
CONTEXT_RECOVERY_FAILURE_KINDS = {"context_limit", "transport"}
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429}
CONTEXT_RECOVERY_PROMPT_BUDGET = {
    "total_chars": 24_000,
    "retry_delta_chars": 4_000,
    "stdout_chars": 4_000,
    "snapshot_chars": 6_000,
    "evidence_chars": 10_000,
    "retry_history_chars": 2_000,
}
VALID_EXECUTORS = {"local", "testcontainers"}
CONTAINER_NETWORK_MODES = {"none", "bridge"}
DEFAULT_EXECUTOR = "testcontainers"
DEFAULT_CONTAINER_IMAGE = DEFAULT_CONTAINER_IMAGE_FALLBACK
PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE = "not_eligible"
PATCH_VERIFIER_RECOVERY_RESULT_SKIPPED_DEADLINE_EXHAUSTED = "skipped_deadline_exhausted"
PATCH_VERIFIER_RECOVERY_RESULT_NO_DIFF = "no_diff"
PATCH_VERIFIER_RECOVERY_RESULT_REQUEST_FAILED = "request_failed"
PATCH_VERIFIER_RECOVERY_RESULT_STILL_REJECTED = "still_rejected"
PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER = "passed_verifier"
DEFAULT_CONTAINER_NETWORK = "none"
DEFAULT_CONTAINER_CPUS = 1.0
DEFAULT_CONTAINER_MEMORY = "1g"
DEFAULT_CONTAINER_PIDS_LIMIT = 256
DEFAULT_CONTAINER_TMPFS_SIZE_MB = 128
DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC = 300
EVAL_CONTAINER_LABEL_KEY = "io.llmdebug.eval.managed"
EVAL_CONTAINER_LABEL_VALUE = "1"
_CONTAINER_ACTIVE_STATES = {"running", "restarting", "paused"}
RUN_EVAL_CONFIG_SECTION = "run_eval"
RUN_EVAL_CONFIG_LOCK_SCHEMA_VERSION = 1
RUN_EVAL_LOCKFILE_REDACTED = "***REDACTED***"
RUN_EVAL_SECRET_CONFIG_LITERAL_KEYS = {"openai_api_key", "adaptive_decision_model_api_key"}
RUN_EVAL_SECRET_CONFIG_ENV_KEYS = {"openai_api_key_env"}
RUN_EVAL_LOCKFILE_REDACTED_KEYS = {"openai_api_key", "adaptive_decision_model_api_key"}
EVAL_RECORD_TYPE_ATTEMPT = "attempt"
EVAL_RECORD_TYPE_INFRA_EVENT = "infra_event"


def _default_helper_action() -> str:
    return ACTION_INVOKE_HELPER


def _helper_action_allowed_request_types(action: str) -> tuple[str, ...]:
    if action == ACTION_FRAME_ONLY:
        return ("frame", "local_var")
    if action == ACTION_FILE_ONLY:
        return ("file",)
    if action == ACTION_SEARCH_ONLY:
        return ("search",)
    return ("file", "search", "frame", "local_var")


def _helper_action_request_mini_budgets(action: str) -> Mapping[str, int]:
    """Return per-request-type budgets for restricted helper actions."""
    return _HELPER_ACTION_REQUEST_MINI_BUDGETS.get(action, {})


def _helper_action_fallback_reason(
    *,
    action: str,
    supports_context_request: bool,
    supports_strict_json_first: bool,
    snapshot_available: bool,
    context_compaction_mode: str,
    supports_context_compaction: bool,
    context_compaction_calls_used: int,
    context_compaction_max_calls: int,
    last_helper_invalid_response: bool,
) -> str:
    """Determine why a helper action cannot execute.

    Returns an empty string when the action is eligible, or a short
    reason token (e.g. ``"snapshot_unavailable"``) when it must fall back.
    """
    if action in HELPER_STAGE_A_ACTIONS and not supports_context_request:
        return "request_context_unsupported"
    if (
        action in {ACTION_SNAPSHOT_LITE, ACTION_FRAME_LOCALS_LITE, ACTION_SNAPSHOT_FULL_LITE}
        and not snapshot_available
    ):
        return "snapshot_unavailable"
    if action == ACTION_COMPACT_THEN_PATCH:
        if str(context_compaction_mode or "off") == "off":
            return "context_compaction_disabled"
        if not supports_context_compaction:
            return "context_compaction_unsupported"
        if max(0, int(context_compaction_max_calls)) <= 0:
            return "context_compaction_budget_exhausted"
        if max(0, int(context_compaction_calls_used)) >= max(0, int(context_compaction_max_calls)):
            return "context_compaction_budget_exhausted"
    if action == ACTION_RETRY_HELPER_STRICT_JSON:
        if not supports_strict_json_first:
            return "strict_json_requires_openai_compatible_patcher"
        if not last_helper_invalid_response:
            return "strict_json_requires_prior_invalid_payload"
    return ""


def _helper_action_availability(
    *,
    supports_context_request: bool,
    supports_strict_json_first: bool,
    snapshot_available: bool,
    context_compaction_mode: str,
    supports_context_compaction: bool,
    context_compaction_calls_used: int,
    context_compaction_max_calls: int,
    last_helper_invalid_response: bool,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return the feasible helper actions and reasons for unavailable ones."""
    available_actions: list[str] = []
    unavailable_action_reasons: dict[str, str] = {}
    for action in ONLINE_POLICY_ACTIONS:
        reason = _helper_action_fallback_reason(
            action=action,
            supports_context_request=supports_context_request,
            supports_strict_json_first=supports_strict_json_first,
            snapshot_available=snapshot_available,
            context_compaction_mode=context_compaction_mode,
            supports_context_compaction=supports_context_compaction,
            context_compaction_calls_used=context_compaction_calls_used,
            context_compaction_max_calls=context_compaction_max_calls,
            last_helper_invalid_response=last_helper_invalid_response,
        )
        if reason:
            unavailable_action_reasons[action] = reason
            continue
        available_actions.append(action)
    if not available_actions:
        available_actions = [ACTION_SKIP_HELPER]
        unavailable_action_reasons.setdefault(
            ACTION_SKIP_HELPER,
            "no_feasible_helper_actions",
        )
    return tuple(available_actions), unavailable_action_reasons


def _filter_context_request_payload_for_action(
    *,
    action: str,
    request_payload: ContextRequestPayload | None,
) -> ContextRequestPayload | None:
    """Restrict a Stage-A request payload to match a restricted action.

    For restricted actions (frame-only, file-only, search-only) the
    payload is filtered to include only the allowed request type.
    """
    if request_payload is None or action not in HELPER_RESTRICTED_STAGE_A_ACTIONS:
        return request_payload
    allowed_types = set(_helper_action_allowed_request_types(action))
    remaining_budget = {
        str(key): max(0, int(value))
        for key, value in _helper_action_request_mini_budgets(action).items()
    }
    filtered_requests: list[ContextRequestEntry] = []
    requests_raw = request_payload.get("requests")
    if isinstance(requests_raw, list):
        for entry in requests_raw:
            if not isinstance(entry, dict):
                continue
            request_type = str(entry.get("type") or "").strip()
            if request_type not in allowed_types:
                continue
            if remaining_budget and remaining_budget.get(request_type, 0) <= 0:
                continue
            filtered_requests.append(cast(ContextRequestEntry, dict(entry)))
            if remaining_budget:
                remaining_budget[request_type] = max(0, remaining_budget.get(request_type, 0) - 1)
            if remaining_budget and all(value <= 0 for value in remaining_budget.values()):
                break
    filtered_payload = cast(ContextRequestPayload, dict(request_payload))
    filtered_payload["requests"] = cast(Any, filtered_requests)
    filtered_payload["need_snapshot"] = False
    filtered_payload["need_compact_context"] = False
    filtered_payload["snapshot_justification"] = ""
    filtered_payload["compact_context_reason"] = ""
    return filtered_payload


def _default_container_user() -> str:
    try:
        uid = getattr(os, "getuid", None)
        gid = getattr(os, "getgid", None)
        if callable(uid) and callable(gid):
            uid_fn = cast(Callable[[], int], uid)
            gid_fn = cast(Callable[[], int], gid)
            return f"{uid_fn()}:{gid_fn()}"
    except Exception:
        pass
    return "0:0"


DEFAULT_CONTAINER_USER = _default_container_user()


@dataclass(frozen=True)
class StageAContextData:
    """Immutable result of a Stage-A context-request round."""

    blocks: list[dict[str, Any]]
    text: str
    used_fallback: bool
    requests_count: int
    chars_returned: int
    request_success: bool | None
    request_payload: ContextRequestPayload | None
    request_raw: str | None
    reason: str
    evidence_items: list[dict[str, Any]] = field(default_factory=list)


def _empty_stage_a_context(*, reason: str = "not_used") -> StageAContextData:
    return StageAContextData(
        blocks=[],
        text="",
        used_fallback=False,
        requests_count=0,
        chars_returned=0,
        request_success=None,
        request_payload=None,
        request_raw=None,
        reason=reason,
        evidence_items=[],
    )


@dataclass(frozen=True)
class ContextCompactionBundle:
    """Immutable bundle of text assembled for context compaction."""

    text: str
    source_sections: list[str]
    source_chars: dict[str, int]
    replace_snapshot: bool
    replace_pytest_output: bool
    replace_stage_a_context: bool
    replace_project_files: bool


@dataclass
class _ActiveRunSettings:
    """Global mutable state for deterministic-seed propagation."""

    deterministic: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class ConfigResolutionState:
    """Tracks which TOML config files were loaded and value provenance."""

    config_paths: list[str] = field(default_factory=list)
    value_sources: dict[str, str] = field(default_factory=dict)
    runtime_resolution: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    """Frozen specification of a single eval case loaded from disk."""

    case_id: str
    path: Path
    category: str
    difficulty: str
    snapshot_dependency: str
    expected_exception: str
    tags: list[str]
    python_args: list[str]
    timeout_sec: int
    description: str | None = None
    # Extended fields for imported cases (all optional, backward-compatible).
    bug_source: str | None = None
    contamination_risk: str | None = None
    expected_fix_lines: int | None = None
    source_id: str | None = None
    difficulty_rationale: str | None = None


@dataclass(frozen=True)
class RunResult:
    """Frozen result of a single pytest execution.

    ``junit_requested`` is set by ``run_pytest`` when the arguments asked for the
    harness JUnit report; ``junit_counts`` then holds its counts, or ``None`` when no
    report was written (ATT-01).
    """

    returncode: int
    duration_sec: float
    stdout: str
    stderr: str
    timed_out: bool = False
    timeout_sec: int | None = None
    junit_requested: bool = False
    junit_counts: PytestCounts | None = None


@dataclass(frozen=True)
class UnitExecutionOutcome:
    """Outcome of running one (case, condition) eval unit."""

    case: CaseSpec
    condition: Condition
    case_records: list[dict[str, Any]] | None
    unit_duration_sec: float
    runtime_error: str | None = None
    records_streamed: bool = False


@dataclass(frozen=True)
class _TrailingResultsTailIssue:
    """Describes a corrupt trailing record found during resume scan."""

    line_no: int
    truncate_offset_bytes: int
    raw_line_bytes: bytes
    parse_error: str


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for the test-execution backend (local or container)."""

    executor: str = DEFAULT_EXECUTOR
    container_image: str = DEFAULT_CONTAINER_IMAGE
    container_image_source: str = "default"
    container_image_implicit_latest: bool = False
    container_network: str = DEFAULT_CONTAINER_NETWORK
    container_cpus: float = DEFAULT_CONTAINER_CPUS
    container_memory: str = DEFAULT_CONTAINER_MEMORY
    container_pids_limit: int = DEFAULT_CONTAINER_PIDS_LIMIT
    container_user: str = DEFAULT_CONTAINER_USER
    container_tmpfs_size_mb: int = DEFAULT_CONTAINER_TMPFS_SIZE_MB
    container_mount_repo_src: bool = True
    container_env: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    container_pool: bool = False
    """Reuse a pool of long-lived containers (one per job) instead of a fresh container per
    pytest call. Off by default: pooled containers share the read-write workdirs mount
    and outlive units, which is weaker isolation than the one-shot containers that
    produced all pre-2026-10-04 results (ORC-02)."""


@dataclass(frozen=True)
class PromptAssemblyConfig:
    """Prompt assembly and evidence-budget controls for a run condition."""

    native_baseline_mode: str = ""
    context_protocol: str = "monolithic"
    context_stage_a_max_requests: int = 6
    context_stage_a_max_chars: int = 120_000
    retry_prompt_mode: str = "auto"
    retry_context_mode: str = "history"
    retry_history_max_chars: int = 48_000
    prompt_retry_reflection: bool = False
    prompt_adaptive_stage_b_cot: bool = False
    prompt_repair_method_scaffold: bool = False
    prompt_retry_plan_then_patch: bool = False
    pytest_output_tail_lines: int = DEFAULT_PYTEST_OUTPUT_TAIL_LINES
    snapshot_artifact_max_string_chars: int = DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS
    prompt_budget_total_chars: int = 180_000
    prompt_budget_retry_delta_chars: int = 16_000
    prompt_budget_stdout_chars: int = 32_000
    prompt_budget_snapshot_chars: int = 32_000
    prompt_budget_evidence_chars: int = 64_000
    prompt_budget_retry_history_chars: int = 12_000
    context_compaction_mode: str = "auto_overflow"
    context_compaction_scope: str = "snapshot,pytest,files"
    context_compaction_max_input_chars: int = 120_000
    context_compaction_target_chars: int = 18_000
    context_compaction_max_calls: int = 1
    prompt_order_mode: str = "current"
    adaptive_reveal_mode: str = "full"
    snapshot_frame_pruning_mode: str = "off"


@dataclass(frozen=True)
class AdaptiveRuntimeConfig:
    """Adaptive helper and policy controls for a run condition."""

    adaptive_gate_policy: str = "balanced"
    adaptive_max_helper_calls: int = 1
    adaptive_budget_topup_max_calls: int = DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS
    adaptive_stagnation_window: int = 1
    adaptive_saturation_threshold: float = 0.75
    adaptive_no_progress_token_cap: int = DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP
    adaptive_helper_budget_profile: str = "default"
    adaptive_openai_helper_response_mode: str = "inherit"
    adaptive_controller_lambda_rlm: bool = False
    adaptive_controller_lambda_rlm_repo_dir: str = ""
    adaptive_controller_lambda_rlm_python: str = ""
    adaptive_controller_lambda_rlm_context_window_chars: int = (
        DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS
    )
    adaptive_controller_lambda_rlm_timeout_sec: int = DEFAULT_LAMBDA_RLM_TIMEOUT_SEC
    adaptive_informativeness_pregate: bool = False
    adaptive_informativeness_pregate_threshold: float = 0.7
    adaptive_online_policy_controller: Any | None = None
    adaptive_online_policy_mode: str = "off"
    adaptive_online_policy_algo: str = "linucb"
    adaptive_online_feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE
    adaptive_online_action_score_biases: str = DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES
    adaptive_online_include_case_labels: bool = False
    adaptive_lite_snapshot_mode: str = "replace"
    adaptive_budget_policy_controller: Any | None = None
    adaptive_budget_policy_mode: str = "on"
    adaptive_budget_policy_algo: str = "linucb"
    adaptive_budget_policy_architecture: str = BUDGET_POLICY_ARCH_TIER_BINARY
    adaptive_budget_feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE
    adaptive_budget_policy_tiers: str = DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS
    adaptive_budget_policy_max_ups_per_case: int = DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE
    adaptive_budget_bwk_run_token_budget: int = 0
    adaptive_budget_reward_rho_budget_exceeded: float = (
        ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT
    )
    adaptive_reward_lambda_tokens: float = ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT
    adaptive_reward_mu_latency: float = ADAPTIVE_REWARD_MU_LATENCY_DEFAULT
    adaptive_reward_token_scale: float = ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT
    adaptive_reward_latency_scale: float = ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT
    adaptive_reward_eta_progress: float = ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT
    adaptive_reward_kappa_stagnation: float = ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT
    adaptive_reward_success_hint: float = ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT
    adaptive_reward_progress_scale: float = ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT
    adaptive_reward_stagnation_scale: float = ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT
    adaptive_reward_materialization_scale: float = ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT
    adaptive_toon_encoding: bool = False
    # Auxiliary reward signals (on by default for adaptive conditions).
    enable_auxiliary_rewards: bool = True
    adaptive_aux_pytest_reduction_weight: float = (
        ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT
    )
    adaptive_aux_evidence_novelty_weight: float = (
        ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT
    )
    adaptive_aux_compaction_efficiency_weight: float = (
        ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT
    )
    adaptive_aux_informativeness_alignment_weight: float = (
        ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT
    )
    adaptive_tep_mode: str = "off"
    adaptive_tep_state_path: str = ""
    adaptive_tep_init_state_path: str = ""
    adaptive_tep_beta_init: float = ADAPTIVE_TEP_BETA_INIT_DEFAULT
    adaptive_tep_beta_decay: float = ADAPTIVE_TEP_BETA_DECAY_DEFAULT
    adaptive_tep_equilibrium_max_iters: int = ADAPTIVE_TEP_EQUILIBRIUM_MAX_ITERS_DEFAULT
    adaptive_tep_equilibrium_window: int = ADAPTIVE_TEP_EQUILIBRIUM_WINDOW_DEFAULT
    adaptive_tep_equilibrium_variance_threshold: float = (
        ADAPTIVE_TEP_EQUILIBRIUM_VARIANCE_THRESHOLD_DEFAULT
    )
    adaptive_tep_validation_buffer_size: int = ADAPTIVE_TEP_VALIDATION_BUFFER_SIZE_DEFAULT
    # TEP configuration (off by default).
    enable_tep: bool = False
    tep_nudge_escalation: bool = True
    tep_equilibrium_check: bool = False
    tep_equilibrium_max_iterations: int = 2
    tep_equilibrium_similarity_threshold: float = 0.85


@dataclass(frozen=True)
class VerificationConfig:
    """Verification and sanitization controls for a run condition."""

    syntax_preflight_enabled: bool = True
    syntax_preflight_max_cycles: int = DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES
    syntax_preflight_fallthrough: bool = True
    patch_sanitize_enabled: bool = True
    patch_verifier_scope: str = PATCH_VERIFIER_SCOPE_ALL
    patch_verifier_min_score: int = PATCH_VERIFIER_MIN_SCORE
    patch_verifier_infer_test_source_scope: bool = True


@dataclass(frozen=True)
class TransportConfig:
    """Transport-window and response-mode controls for a run condition."""

    openai_context_window_tokens: int = 0
    openai_context_window_source: str = ""
    openai_context_window_probe_provider: str = ""
    openai_context_window_probe_error: str = ""
    openai_response_mode_autoswitch: str = "off"
    transport_recovery_gate: Any | None = None
    openai_transport_recovery: bool = DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED


@dataclass(frozen=True)
class RecordingConfig:
    """Recording and deterministic-execution controls for a run condition."""

    run_metadata: dict[str, Any] | None = None
    record_prompt: str = "hash"
    record_evidence: str = "hash"
    deterministic: bool = False
    seed: int | None = None
    record_sink: Callable[[dict[str, Any]], None] | None = None
    progress_update: Callable[[str, dict[str, Any] | None], None] | None = None


@dataclass(frozen=True)
class RunConditionInputs:
    """Inputs for a single ``run_one_condition`` execution."""

    k: int
    output_format: str
    run_id: str
    include_files_context: bool
    force_patch_attempt: bool
    patcher: Any | None = None
    max_context_chars: int = 200_000
    case_fail_fast_on_request_budget_exceeded: bool = False
    native_baseline_mode: str = ""
    context_protocol: str = "monolithic"
    context_stage_a_max_requests: int = 6
    context_stage_a_max_chars: int = 120_000
    retry_prompt_mode: str = "auto"
    retry_context_mode: str = "history"
    retry_history_max_chars: int = 48_000
    prompt_retry_reflection: bool = False
    prompt_adaptive_stage_b_cot: bool = False
    prompt_repair_method_scaffold: bool = False
    prompt_retry_plan_then_patch: bool = False
    pytest_output_tail_lines: int = DEFAULT_PYTEST_OUTPUT_TAIL_LINES
    snapshot_artifact_max_string_chars: int = DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS
    prompt_budget_total_chars: int = 180_000
    prompt_budget_retry_delta_chars: int = 16_000
    prompt_budget_stdout_chars: int = 32_000
    prompt_budget_snapshot_chars: int = 32_000
    prompt_budget_evidence_chars: int = 64_000
    prompt_budget_retry_history_chars: int = 12_000
    context_compaction_mode: str = "auto_overflow"
    context_compaction_scope: str = "snapshot,pytest,files"
    context_compaction_max_input_chars: int = 120_000
    context_compaction_target_chars: int = 18_000
    context_compaction_max_calls: int = 1
    prompt_order_mode: str = "current"
    adaptive_reveal_mode: str = "full"
    snapshot_frame_pruning_mode: str = "off"
    openai_context_window_tokens: int = 0
    openai_context_window_source: str = ""
    openai_context_window_probe_provider: str = ""
    openai_context_window_probe_error: str = ""
    openai_response_mode_autoswitch: str = "off"
    adaptive_gate_policy: str = "balanced"
    adaptive_max_helper_calls: int = 1
    adaptive_budget_topup_max_calls: int = DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS
    adaptive_stagnation_window: int = 1
    adaptive_saturation_threshold: float = 0.75
    adaptive_no_progress_token_cap: int = DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP
    adaptive_helper_budget_profile: str = "default"
    adaptive_openai_helper_response_mode: str = "inherit"
    adaptive_controller_lambda_rlm: bool = False
    adaptive_controller_lambda_rlm_repo_dir: str = ""
    adaptive_controller_lambda_rlm_python: str = ""
    adaptive_controller_lambda_rlm_context_window_chars: int = (
        DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS
    )
    adaptive_controller_lambda_rlm_timeout_sec: int = DEFAULT_LAMBDA_RLM_TIMEOUT_SEC
    adaptive_informativeness_pregate: bool = False
    adaptive_informativeness_pregate_threshold: float = 0.7
    syntax_preflight_enabled: bool = True
    syntax_preflight_max_cycles: int = DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES
    syntax_preflight_fallthrough: bool = True
    patch_sanitize_enabled: bool = True
    patch_verifier_scope: str = PATCH_VERIFIER_SCOPE_ALL
    patch_verifier_min_score: int = PATCH_VERIFIER_MIN_SCORE
    patch_verifier_infer_test_source_scope: bool = True
    adaptive_online_policy_controller: Any | None = None
    adaptive_online_policy_mode: str = "off"
    adaptive_online_policy_algo: str = "linucb"
    adaptive_online_feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE
    adaptive_online_action_score_biases: str = DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES
    adaptive_online_include_case_labels: bool = False
    adaptive_lite_snapshot_mode: str = "replace"
    adaptive_budget_policy_controller: Any | None = None
    adaptive_budget_policy_mode: str = "on"
    adaptive_budget_policy_algo: str = "linucb"
    adaptive_budget_policy_architecture: str = BUDGET_POLICY_ARCH_TIER_BINARY
    adaptive_budget_feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE
    adaptive_budget_policy_tiers: str = DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS
    adaptive_budget_policy_max_ups_per_case: int = DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE
    adaptive_budget_bwk_run_token_budget: int = 0
    adaptive_budget_reward_rho_budget_exceeded: float = (
        ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT
    )
    adaptive_reward_lambda_tokens: float = ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT
    adaptive_reward_mu_latency: float = ADAPTIVE_REWARD_MU_LATENCY_DEFAULT
    adaptive_reward_token_scale: float = ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT
    adaptive_reward_latency_scale: float = ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT
    adaptive_reward_eta_progress: float = ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT
    adaptive_reward_kappa_stagnation: float = ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT
    adaptive_reward_success_hint: float = ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT
    adaptive_reward_progress_scale: float = ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT
    adaptive_reward_stagnation_scale: float = ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT
    adaptive_reward_materialization_scale: float = ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT
    adaptive_toon_encoding: bool = False
    enable_auxiliary_rewards: bool = True
    adaptive_aux_pytest_reduction_weight: float = (
        ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT
    )
    adaptive_aux_evidence_novelty_weight: float = (
        ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT
    )
    adaptive_aux_compaction_efficiency_weight: float = (
        ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT
    )
    adaptive_aux_informativeness_alignment_weight: float = (
        ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT
    )
    adaptive_tep_mode: str = "off"
    adaptive_tep_state_path: str = ""
    adaptive_tep_init_state_path: str = ""
    adaptive_tep_beta_init: float = ADAPTIVE_TEP_BETA_INIT_DEFAULT
    adaptive_tep_beta_decay: float = ADAPTIVE_TEP_BETA_DECAY_DEFAULT
    adaptive_tep_equilibrium_max_iters: int = ADAPTIVE_TEP_EQUILIBRIUM_MAX_ITERS_DEFAULT
    adaptive_tep_equilibrium_window: int = ADAPTIVE_TEP_EQUILIBRIUM_WINDOW_DEFAULT
    adaptive_tep_equilibrium_variance_threshold: float = (
        ADAPTIVE_TEP_EQUILIBRIUM_VARIANCE_THRESHOLD_DEFAULT
    )
    adaptive_tep_validation_buffer_size: int = ADAPTIVE_TEP_VALIDATION_BUFFER_SIZE_DEFAULT
    enable_tep: bool = False
    tep_nudge_escalation: bool = True
    tep_equilibrium_check: bool = False
    tep_equilibrium_max_iterations: int = 2
    tep_equilibrium_similarity_threshold: float = 0.85
    run_metadata: dict[str, Any] | None = None
    record_prompt: str = "hash"
    record_evidence: str = "hash"
    deterministic: bool = False
    seed: int | None = None
    transport_recovery_gate: Any | None = None
    openai_transport_recovery: bool = DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED
    record_sink: Callable[[dict[str, Any]], None] | None = None
    progress_update: Callable[[str, dict[str, Any] | None], None] | None = None

    @classmethod
    def from_components(
        cls,
        *,
        k: int,
        output_format: str,
        run_id: str,
        include_files_context: bool,
        force_patch_attempt: bool,
        patcher: Any | None = None,
        max_context_chars: int = 200_000,
        case_fail_fast_on_request_budget_exceeded: bool = False,
        prompt_assembly: PromptAssemblyConfig | None = None,
        adaptive_runtime: AdaptiveRuntimeConfig | None = None,
        verification: VerificationConfig | None = None,
        transport: TransportConfig | None = None,
        recording: RecordingConfig | None = None,
    ) -> RunConditionInputs:
        prompt = prompt_assembly or PromptAssemblyConfig()
        adaptive = adaptive_runtime or AdaptiveRuntimeConfig()
        verify = verification or VerificationConfig()
        transport_cfg = transport or TransportConfig()
        record = recording or RecordingConfig()
        return cls(
            k=k,
            output_format=output_format,
            run_id=run_id,
            include_files_context=include_files_context,
            force_patch_attempt=force_patch_attempt,
            patcher=patcher,
            max_context_chars=max_context_chars,
            case_fail_fast_on_request_budget_exceeded=case_fail_fast_on_request_budget_exceeded,
            native_baseline_mode=prompt.native_baseline_mode,
            context_protocol=prompt.context_protocol,
            context_stage_a_max_requests=prompt.context_stage_a_max_requests,
            context_stage_a_max_chars=prompt.context_stage_a_max_chars,
            retry_prompt_mode=prompt.retry_prompt_mode,
            retry_context_mode=prompt.retry_context_mode,
            retry_history_max_chars=prompt.retry_history_max_chars,
            prompt_retry_reflection=prompt.prompt_retry_reflection,
            prompt_adaptive_stage_b_cot=prompt.prompt_adaptive_stage_b_cot,
            prompt_repair_method_scaffold=prompt.prompt_repair_method_scaffold,
            prompt_retry_plan_then_patch=prompt.prompt_retry_plan_then_patch,
            pytest_output_tail_lines=prompt.pytest_output_tail_lines,
            snapshot_artifact_max_string_chars=prompt.snapshot_artifact_max_string_chars,
            prompt_budget_total_chars=prompt.prompt_budget_total_chars,
            prompt_budget_retry_delta_chars=prompt.prompt_budget_retry_delta_chars,
            prompt_budget_stdout_chars=prompt.prompt_budget_stdout_chars,
            prompt_budget_snapshot_chars=prompt.prompt_budget_snapshot_chars,
            prompt_budget_evidence_chars=prompt.prompt_budget_evidence_chars,
            prompt_budget_retry_history_chars=prompt.prompt_budget_retry_history_chars,
            context_compaction_mode=prompt.context_compaction_mode,
            context_compaction_scope=prompt.context_compaction_scope,
            context_compaction_max_input_chars=prompt.context_compaction_max_input_chars,
            context_compaction_target_chars=prompt.context_compaction_target_chars,
            context_compaction_max_calls=prompt.context_compaction_max_calls,
            prompt_order_mode=prompt.prompt_order_mode,
            adaptive_reveal_mode=prompt.adaptive_reveal_mode,
            snapshot_frame_pruning_mode=prompt.snapshot_frame_pruning_mode,
            openai_context_window_tokens=transport_cfg.openai_context_window_tokens,
            openai_context_window_source=transport_cfg.openai_context_window_source,
            openai_context_window_probe_provider=transport_cfg.openai_context_window_probe_provider,
            openai_context_window_probe_error=transport_cfg.openai_context_window_probe_error,
            openai_response_mode_autoswitch=transport_cfg.openai_response_mode_autoswitch,
            adaptive_gate_policy=adaptive.adaptive_gate_policy,
            adaptive_max_helper_calls=adaptive.adaptive_max_helper_calls,
            adaptive_budget_topup_max_calls=adaptive.adaptive_budget_topup_max_calls,
            adaptive_stagnation_window=adaptive.adaptive_stagnation_window,
            adaptive_saturation_threshold=adaptive.adaptive_saturation_threshold,
            adaptive_no_progress_token_cap=adaptive.adaptive_no_progress_token_cap,
            adaptive_helper_budget_profile=adaptive.adaptive_helper_budget_profile,
            adaptive_openai_helper_response_mode=adaptive.adaptive_openai_helper_response_mode,
            adaptive_controller_lambda_rlm=adaptive.adaptive_controller_lambda_rlm,
            adaptive_controller_lambda_rlm_repo_dir=adaptive.adaptive_controller_lambda_rlm_repo_dir,
            adaptive_controller_lambda_rlm_python=adaptive.adaptive_controller_lambda_rlm_python,
            adaptive_controller_lambda_rlm_context_window_chars=(
                adaptive.adaptive_controller_lambda_rlm_context_window_chars
            ),
            adaptive_controller_lambda_rlm_timeout_sec=(
                adaptive.adaptive_controller_lambda_rlm_timeout_sec
            ),
            adaptive_informativeness_pregate=adaptive.adaptive_informativeness_pregate,
            adaptive_informativeness_pregate_threshold=(
                adaptive.adaptive_informativeness_pregate_threshold
            ),
            syntax_preflight_enabled=verify.syntax_preflight_enabled,
            syntax_preflight_max_cycles=verify.syntax_preflight_max_cycles,
            syntax_preflight_fallthrough=verify.syntax_preflight_fallthrough,
            patch_sanitize_enabled=verify.patch_sanitize_enabled,
            patch_verifier_scope=verify.patch_verifier_scope,
            patch_verifier_min_score=verify.patch_verifier_min_score,
            patch_verifier_infer_test_source_scope=verify.patch_verifier_infer_test_source_scope,
            adaptive_online_policy_controller=adaptive.adaptive_online_policy_controller,
            adaptive_online_policy_mode=adaptive.adaptive_online_policy_mode,
            adaptive_online_policy_algo=adaptive.adaptive_online_policy_algo,
            adaptive_online_feature_scaling_profile=adaptive.adaptive_online_feature_scaling_profile,
            adaptive_online_action_score_biases=adaptive.adaptive_online_action_score_biases,
            adaptive_online_include_case_labels=adaptive.adaptive_online_include_case_labels,
            adaptive_lite_snapshot_mode=adaptive.adaptive_lite_snapshot_mode,
            adaptive_budget_policy_controller=adaptive.adaptive_budget_policy_controller,
            adaptive_budget_policy_mode=adaptive.adaptive_budget_policy_mode,
            adaptive_budget_policy_algo=adaptive.adaptive_budget_policy_algo,
            adaptive_budget_policy_architecture=adaptive.adaptive_budget_policy_architecture,
            adaptive_budget_feature_scaling_profile=adaptive.adaptive_budget_feature_scaling_profile,
            adaptive_budget_policy_tiers=adaptive.adaptive_budget_policy_tiers,
            adaptive_budget_policy_max_ups_per_case=adaptive.adaptive_budget_policy_max_ups_per_case,
            adaptive_budget_bwk_run_token_budget=adaptive.adaptive_budget_bwk_run_token_budget,
            adaptive_budget_reward_rho_budget_exceeded=(
                adaptive.adaptive_budget_reward_rho_budget_exceeded
            ),
            adaptive_reward_lambda_tokens=adaptive.adaptive_reward_lambda_tokens,
            adaptive_reward_mu_latency=adaptive.adaptive_reward_mu_latency,
            adaptive_reward_token_scale=adaptive.adaptive_reward_token_scale,
            adaptive_reward_latency_scale=adaptive.adaptive_reward_latency_scale,
            adaptive_reward_eta_progress=adaptive.adaptive_reward_eta_progress,
            adaptive_reward_kappa_stagnation=adaptive.adaptive_reward_kappa_stagnation,
            adaptive_reward_success_hint=adaptive.adaptive_reward_success_hint,
            adaptive_reward_progress_scale=adaptive.adaptive_reward_progress_scale,
            adaptive_reward_stagnation_scale=adaptive.adaptive_reward_stagnation_scale,
            adaptive_reward_materialization_scale=adaptive.adaptive_reward_materialization_scale,
            adaptive_toon_encoding=adaptive.adaptive_toon_encoding,
            enable_auxiliary_rewards=adaptive.enable_auxiliary_rewards,
            adaptive_aux_pytest_reduction_weight=adaptive.adaptive_aux_pytest_reduction_weight,
            adaptive_aux_evidence_novelty_weight=adaptive.adaptive_aux_evidence_novelty_weight,
            adaptive_aux_compaction_efficiency_weight=adaptive.adaptive_aux_compaction_efficiency_weight,
            adaptive_aux_informativeness_alignment_weight=adaptive.adaptive_aux_informativeness_alignment_weight,
            adaptive_tep_mode=adaptive.adaptive_tep_mode,
            adaptive_tep_state_path=adaptive.adaptive_tep_state_path,
            adaptive_tep_init_state_path=adaptive.adaptive_tep_init_state_path,
            adaptive_tep_beta_init=adaptive.adaptive_tep_beta_init,
            adaptive_tep_beta_decay=adaptive.adaptive_tep_beta_decay,
            adaptive_tep_equilibrium_max_iters=adaptive.adaptive_tep_equilibrium_max_iters,
            adaptive_tep_equilibrium_window=adaptive.adaptive_tep_equilibrium_window,
            adaptive_tep_equilibrium_variance_threshold=(
                adaptive.adaptive_tep_equilibrium_variance_threshold
            ),
            adaptive_tep_validation_buffer_size=adaptive.adaptive_tep_validation_buffer_size,
            enable_tep=adaptive.enable_tep,
            tep_nudge_escalation=adaptive.tep_nudge_escalation,
            tep_equilibrium_check=adaptive.tep_equilibrium_check,
            tep_equilibrium_max_iterations=adaptive.tep_equilibrium_max_iterations,
            tep_equilibrium_similarity_threshold=adaptive.tep_equilibrium_similarity_threshold,
            run_metadata=record.run_metadata,
            record_prompt=record.record_prompt,
            record_evidence=record.record_evidence,
            deterministic=record.deterministic,
            seed=record.seed,
            transport_recovery_gate=transport_cfg.transport_recovery_gate,
            openai_transport_recovery=transport_cfg.openai_transport_recovery,
            record_sink=record.record_sink,
            progress_update=record.progress_update,
        )

    @property
    def prompt_assembly(self) -> PromptAssemblyConfig:
        return PromptAssemblyConfig(
            native_baseline_mode=self.native_baseline_mode,
            context_protocol=self.context_protocol,
            context_stage_a_max_requests=self.context_stage_a_max_requests,
            context_stage_a_max_chars=self.context_stage_a_max_chars,
            retry_prompt_mode=self.retry_prompt_mode,
            retry_context_mode=self.retry_context_mode,
            retry_history_max_chars=self.retry_history_max_chars,
            prompt_retry_reflection=self.prompt_retry_reflection,
            prompt_adaptive_stage_b_cot=self.prompt_adaptive_stage_b_cot,
            prompt_repair_method_scaffold=self.prompt_repair_method_scaffold,
            prompt_retry_plan_then_patch=self.prompt_retry_plan_then_patch,
            pytest_output_tail_lines=self.pytest_output_tail_lines,
            snapshot_artifact_max_string_chars=self.snapshot_artifact_max_string_chars,
            prompt_budget_total_chars=self.prompt_budget_total_chars,
            prompt_budget_retry_delta_chars=self.prompt_budget_retry_delta_chars,
            prompt_budget_stdout_chars=self.prompt_budget_stdout_chars,
            prompt_budget_snapshot_chars=self.prompt_budget_snapshot_chars,
            prompt_budget_evidence_chars=self.prompt_budget_evidence_chars,
            prompt_budget_retry_history_chars=self.prompt_budget_retry_history_chars,
            context_compaction_mode=self.context_compaction_mode,
            context_compaction_scope=self.context_compaction_scope,
            context_compaction_max_input_chars=self.context_compaction_max_input_chars,
            context_compaction_target_chars=self.context_compaction_target_chars,
            context_compaction_max_calls=self.context_compaction_max_calls,
            prompt_order_mode=self.prompt_order_mode,
            adaptive_reveal_mode=self.adaptive_reveal_mode,
            snapshot_frame_pruning_mode=self.snapshot_frame_pruning_mode,
        )

    @property
    def adaptive_runtime(self) -> AdaptiveRuntimeConfig:
        return AdaptiveRuntimeConfig(
            adaptive_gate_policy=self.adaptive_gate_policy,
            adaptive_max_helper_calls=self.adaptive_max_helper_calls,
            adaptive_budget_topup_max_calls=self.adaptive_budget_topup_max_calls,
            adaptive_stagnation_window=self.adaptive_stagnation_window,
            adaptive_saturation_threshold=self.adaptive_saturation_threshold,
            adaptive_no_progress_token_cap=self.adaptive_no_progress_token_cap,
            adaptive_helper_budget_profile=self.adaptive_helper_budget_profile,
            adaptive_openai_helper_response_mode=self.adaptive_openai_helper_response_mode,
            adaptive_controller_lambda_rlm=self.adaptive_controller_lambda_rlm,
            adaptive_controller_lambda_rlm_repo_dir=self.adaptive_controller_lambda_rlm_repo_dir,
            adaptive_controller_lambda_rlm_python=self.adaptive_controller_lambda_rlm_python,
            adaptive_controller_lambda_rlm_context_window_chars=(
                self.adaptive_controller_lambda_rlm_context_window_chars
            ),
            adaptive_controller_lambda_rlm_timeout_sec=(
                self.adaptive_controller_lambda_rlm_timeout_sec
            ),
            adaptive_informativeness_pregate=self.adaptive_informativeness_pregate,
            adaptive_informativeness_pregate_threshold=(
                self.adaptive_informativeness_pregate_threshold
            ),
            adaptive_online_policy_controller=self.adaptive_online_policy_controller,
            adaptive_online_policy_mode=self.adaptive_online_policy_mode,
            adaptive_online_policy_algo=self.adaptive_online_policy_algo,
            adaptive_online_feature_scaling_profile=self.adaptive_online_feature_scaling_profile,
            adaptive_online_action_score_biases=self.adaptive_online_action_score_biases,
            adaptive_online_include_case_labels=self.adaptive_online_include_case_labels,
            adaptive_lite_snapshot_mode=self.adaptive_lite_snapshot_mode,
            adaptive_budget_policy_controller=self.adaptive_budget_policy_controller,
            adaptive_budget_policy_mode=self.adaptive_budget_policy_mode,
            adaptive_budget_policy_algo=self.adaptive_budget_policy_algo,
            adaptive_budget_policy_architecture=self.adaptive_budget_policy_architecture,
            adaptive_budget_feature_scaling_profile=self.adaptive_budget_feature_scaling_profile,
            adaptive_budget_policy_tiers=self.adaptive_budget_policy_tiers,
            adaptive_budget_policy_max_ups_per_case=self.adaptive_budget_policy_max_ups_per_case,
            adaptive_budget_bwk_run_token_budget=self.adaptive_budget_bwk_run_token_budget,
            adaptive_budget_reward_rho_budget_exceeded=(
                self.adaptive_budget_reward_rho_budget_exceeded
            ),
            adaptive_reward_lambda_tokens=self.adaptive_reward_lambda_tokens,
            adaptive_reward_mu_latency=self.adaptive_reward_mu_latency,
            adaptive_reward_token_scale=self.adaptive_reward_token_scale,
            adaptive_reward_latency_scale=self.adaptive_reward_latency_scale,
            adaptive_reward_eta_progress=self.adaptive_reward_eta_progress,
            adaptive_reward_kappa_stagnation=self.adaptive_reward_kappa_stagnation,
            adaptive_reward_success_hint=self.adaptive_reward_success_hint,
            adaptive_reward_progress_scale=self.adaptive_reward_progress_scale,
            adaptive_reward_stagnation_scale=self.adaptive_reward_stagnation_scale,
            adaptive_reward_materialization_scale=self.adaptive_reward_materialization_scale,
            adaptive_toon_encoding=self.adaptive_toon_encoding,
            enable_auxiliary_rewards=self.enable_auxiliary_rewards,
            adaptive_aux_pytest_reduction_weight=self.adaptive_aux_pytest_reduction_weight,
            adaptive_aux_evidence_novelty_weight=self.adaptive_aux_evidence_novelty_weight,
            adaptive_aux_compaction_efficiency_weight=(
                self.adaptive_aux_compaction_efficiency_weight
            ),
            adaptive_aux_informativeness_alignment_weight=(
                self.adaptive_aux_informativeness_alignment_weight
            ),
            adaptive_tep_mode=self.adaptive_tep_mode,
            adaptive_tep_state_path=self.adaptive_tep_state_path,
            adaptive_tep_init_state_path=self.adaptive_tep_init_state_path,
            adaptive_tep_beta_init=self.adaptive_tep_beta_init,
            adaptive_tep_beta_decay=self.adaptive_tep_beta_decay,
            adaptive_tep_equilibrium_max_iters=self.adaptive_tep_equilibrium_max_iters,
            adaptive_tep_equilibrium_window=self.adaptive_tep_equilibrium_window,
            adaptive_tep_equilibrium_variance_threshold=(
                self.adaptive_tep_equilibrium_variance_threshold
            ),
            adaptive_tep_validation_buffer_size=self.adaptive_tep_validation_buffer_size,
            enable_tep=self.enable_tep,
            tep_nudge_escalation=self.tep_nudge_escalation,
            tep_equilibrium_check=self.tep_equilibrium_check,
            tep_equilibrium_max_iterations=self.tep_equilibrium_max_iterations,
            tep_equilibrium_similarity_threshold=self.tep_equilibrium_similarity_threshold,
        )

    @property
    def verification(self) -> VerificationConfig:
        return VerificationConfig(
            syntax_preflight_enabled=self.syntax_preflight_enabled,
            syntax_preflight_max_cycles=self.syntax_preflight_max_cycles,
            syntax_preflight_fallthrough=self.syntax_preflight_fallthrough,
            patch_sanitize_enabled=self.patch_sanitize_enabled,
            patch_verifier_scope=self.patch_verifier_scope,
            patch_verifier_min_score=self.patch_verifier_min_score,
            patch_verifier_infer_test_source_scope=self.patch_verifier_infer_test_source_scope,
        )

    @property
    def transport(self) -> TransportConfig:
        return TransportConfig(
            openai_context_window_tokens=self.openai_context_window_tokens,
            openai_context_window_source=self.openai_context_window_source,
            openai_context_window_probe_provider=self.openai_context_window_probe_provider,
            openai_context_window_probe_error=self.openai_context_window_probe_error,
            openai_response_mode_autoswitch=self.openai_response_mode_autoswitch,
            transport_recovery_gate=self.transport_recovery_gate,
            openai_transport_recovery=self.openai_transport_recovery,
        )

    @property
    def recording(self) -> RecordingConfig:
        return RecordingConfig(
            run_metadata=self.run_metadata,
            record_prompt=self.record_prompt,
            record_evidence=self.record_evidence,
            deterministic=self.deterministic,
            seed=self.seed,
            record_sink=self.record_sink,
            progress_update=self.progress_update,
        )


@dataclass(frozen=True)
class EvalUnitInputs:
    """Inputs for a single ``_execute_eval_unit`` call."""

    patcher_args: Any
    run_condition: RunConditionInputs
    patcher_override: Any | None = None
    keep_workdirs: bool = False
    progress_verbosity: str = "basic"
    progress_heartbeat_sec: float = DEFAULT_PROGRESS_HEARTBEAT_SEC
    unit_progress_callback: Callable[[str, str, str, dict[str, Any] | None], None] | None = None


_PROGRESS_LEVEL_ORDER = {"basic": 1, "detailed": 2, "trace": 3}
_PROGRESS_PRINT_LOCK = threading.Lock()

ADAPTIVE_ONLINE_REWARD_SCOPE_LEGACY = "helper_local_v1"
ADAPTIVE_BUDGET_REWARD_SCOPE_LEGACY = "budget_local_v1"
ADAPTIVE_ONLINE_REWARD_SCOPE = "helper_credit_v2"
ADAPTIVE_BUDGET_REWARD_SCOPE = "budget_credit_v2"
ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT = 1.0
ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE = 1.0
ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE = 1.0
ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE = 0.5
ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS = 2
ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE = 0.5
ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_HORIZON_TWO_FACTOR = 0.5
ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_CONTEXT_MULTIPLIER = 1.25
ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX = 0.75


@dataclass(frozen=True)
class AdaptivePolicyLocalAccounting:
    """Token and latency accounting for a single adaptive-policy attempt."""

    llm_call_count: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    wall_sec: float = 0.0


EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING = AdaptivePolicyLocalAccounting()

ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS = (
    "propose_budget",
    "max_tokens",
    "helper_allowance",
    "stage_a_requests",
    "stage_a_chars",
    "retry_history_budget",
    "compaction_allowance",
    "response_mode",
)


@dataclass(frozen=True)
class AdaptiveBudgetControlProfile:
    """Budget-control knobs produced by the budget-policy tier multiplier."""

    propose_budget_multiplier: float = 1.0
    propose_budget_sec: float = 0.0
    openai_max_tokens: int = 0
    helper_max_calls: int = 0
    stage_a_max_requests: int = 1
    stage_a_max_chars: int = 1
    retry_history_max_chars: int = 0
    context_compaction_max_calls: int = 0
    response_mode: str = ""


# ---------------------------------------------------------------------------
# P1.1 — Rule-based traceback informativeness pre-gate
# ---------------------------------------------------------------------------

_EXCEPTION_TYPE_SPECIFICITY: dict[str, float] = {
    # Tier A: Always diagnostic (score >= 0.8)
    "SyntaxError": 1.0,
    "IndentationError": 1.0,
    "TabError": 1.0,
    "NameError": 0.9,
    "UnboundLocalError": 0.9,
    "ModuleNotFoundError": 0.9,
    "FileNotFoundError": 0.9,
    "ImportError": 0.85,
    "ZeroDivisionError": 0.8,
    # Tier B: Usually diagnostic (score 0.6-0.8)
    "KeyError": 0.7,
    "IndexError": 0.7,
    "AttributeError": 0.7,
    "StopIteration": 0.6,
    "OverflowError": 0.5,
    # Tier C: Context-dependent (score 0.3-0.5)
    "TypeError": 0.4,
    "RecursionError": 0.35,  # Need to see what's recursing
    "ValueError": 0.3,
    # Tier D: Low informativeness (score <= 0.1)
    "AssertionError": 0.1,
    "TimeoutError": 0.05,
    "RuntimeError": 0.05,
    "CancelledError": 0.05,
    "InvalidStateError": 0.05,
}
_EXCEPTION_TYPE_SPECIFICITY_DEFAULT = 0.2

_DIAGNOSTIC_MESSAGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"name\s+'[^']+'\s+is\s+not\s+defined", re.IGNORECASE),
    re.compile(r"No\s+module\s+named", re.IGNORECASE),
    re.compile(r"invalid\s+syntax", re.IGNORECASE),
    re.compile(r"unexpected\s+indent", re.IGNORECASE),
    re.compile(r"unexpected\s+EOF", re.IGNORECASE),
    re.compile(r"list\s+index\s+out\s+of\s+range", re.IGNORECASE),
    re.compile(r"has\s+no\s+attribute\s+'[^']+'", re.IGNORECASE),
    re.compile(r"division\s+by\s+zero", re.IGNORECASE),
    re.compile(r"not\s+subscriptable", re.IGNORECASE),
    re.compile(r"object\s+is\s+not\s+callable", re.IGNORECASE),
    re.compile(r"cannot\s+import\s+name", re.IGNORECASE),
    re.compile(r"No\s+such\s+file\s+or\s+directory", re.IGNORECASE),
]

_ASSERTION_VALUE_PATTERN = re.compile(
    r"(?:==|!=|>=|<=|>|<|assert|expected|got|actual)\s*\S", re.IGNORECASE
)

_STDOUT_PYTEST_ERROR_RE = re.compile(r"^E\s+(\w+(?:\.\w+)*):\s*(.*)", re.MULTILINE)
_STDOUT_TRACEBACK_ERROR_RE = re.compile(r"^(\w+(?:Error|Exception|Warning)):\s*(.*)", re.MULTILINE)
_PYTEST_SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(failed|error|errors)\b", re.IGNORECASE)
_EVAL_ONLY_CASE_ARTIFACTS = ("case.json", "meta.json", "oracle.patch")


class PreGateResult(NamedTuple):
    """Result of the traceback-informativeness pre-gate check."""

    should_skip: bool
    score: float
    exception_type: str
    reason: str
    f_type_specificity: float
    f_message_diagnostic: float
    f_crash_in_user_code: float
    f_assertion_quality: float
    f_frame_depth: float


@dataclass(frozen=True)
class AdaptiveProgressState:
    """Multi-signal progress assessment for adaptive conditions."""

    made: bool
    reason: str
    score: float
    component_signature: float
    component_new_evidence: float
    component_pytest_fail_reduction: float
    component_stage_advance: float
    component_verifier_score: float
    component_scope_alignment: float
    pytest_before_failed_total: int | None
    pytest_after_failed_total: int | None
    stage_before: str
    stage_after: str
    stage_before_ord: int
    stage_after_ord: int


@dataclass(frozen=True)
class AdaptiveOnlineRewardBreakdown:
    """Decomposed reward components for an online-policy update."""

    reward_scope: str
    reward: float
    reward_success_hint: float
    reward_progress_hint: float
    reward_progress_score: float
    reward_token_penalty: float
    reward_latency_penalty: float
    reward_stagnation_penalty: float
    reward_helper_materialization_bonus: float
    reward_delayed_rescue_bonus: float
    reward_credit_resolved_attempt: int
    reward_credit_horizon_attempts: int
    reward_credit_resolution_reason: str
    # Auxiliary reward components (additive, gated by enable flag).
    reward_aux_total: float = 0.0
    reward_aux_pytest_fail_reduction: float = 0.0
    reward_aux_evidence_novelty_ratio: float = 0.0
    reward_aux_compaction_ratio: float = 0.0
    reward_aux_informativeness_alignment: float = 0.0


@dataclass
class PendingAdaptiveOnlineCredit:
    """Deferred credit awaiting resolution for an online-policy decision."""

    decision: PolicyDecision
    run_id: str
    case_id: str
    condition_value: str
    attempt: int
    note: str
    success: bool
    attempt_metadata: dict[str, Any]
    patch_meta: dict[str, Any]
    record_ref: dict[str, Any] | None
    online_policy_mode_for_attempt: str
    online_policy_algo_for_attempt: str
    online_policy_eligible: bool
    online_policy_override_reason: str
    online_policy_feature_scaling_profile: str
    online_policy_features_raw: dict[str, float]
    online_policy_features_scaled: dict[str, float]
    helper_decision_reason: str
    normalized_adaptive_online_policy_mode: str
    adaptive_no_progress_streak_current: int
    reward_lambda_tokens: float
    reward_mu_latency: float
    reward_token_scale: float
    reward_latency_scale: float
    reward_eta_progress: float
    reward_kappa_stagnation: float
    local_accounting: AdaptivePolicyLocalAccounting
    override_reason: str
    behavior_action: str
    context_materialized: bool
    helper_outcome: str
    progress_state: AdaptiveProgressState
    failure_signature_repeat: bool
    case_difficulty: str
    case_snapshot_dependency: str
    reward_success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT
    reward_progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE
    reward_stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE
    reward_helper_materialization_bonus_scale: float = (
        ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE
    )
    # Auxiliary reward input signals (optional, default 0.0).
    aux_pytest_fail_reduction: float = 0.0
    aux_evidence_novelty_ratio: float = 0.0
    aux_compaction_ratio: float = 0.0
    aux_informativeness_alignment: float = 0.0
    enable_auxiliary_rewards: bool = True
    aux_pytest_reduction_weight: float = ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT
    aux_evidence_novelty_weight: float = ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT
    aux_compaction_efficiency_weight: float = (
        ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT
    )
    aux_informativeness_alignment_weight: float = (
        ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT
    )


@dataclass(frozen=True)
class PromptBudgetResult:
    """Result of applying per-section and total-char prompt budgets."""

    prompt: str
    used_chars: int
    overflow_chars: int
    dropped_sections: list[str]
    section_status: dict[str, str] = field(default_factory=dict)
