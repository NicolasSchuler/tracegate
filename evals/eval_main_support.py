"""Owned support surface for extracted ``evals.run_eval`` main-flow modules."""
# ruff: noqa: F401

from __future__ import annotations

from typing import Any

from evals import run_eval as _legacy
from evals.adaptive_budget_online_policy import AdaptiveBudgetPolicyController, BudgetPolicyConfig
from evals.adaptive_online_policy import AdaptiveOnlinePolicyController, OnlinePolicyConfig
from evals.eval_config import set_active_execution_config
from evals.eval_containers import _EvalContainerPool
from evals.eval_recording import _build_run_metadata, _SynchronizedJsonlWriter
from evals.eval_types import (
    DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    CaseSpec,
    Condition,
    EvalUnitInputs,
    RunConditionInputs,
    UnitExecutionOutcome,
    _ActiveRunSettings,
    is_baseline_condition,
)
from evals.progress_display import _format_duration
from evals.progress_output import RunProgressSummary, build_progress_palette
from evals.run_eval import (
    _baseline_extra_from_args,
    _baseline_patch_inventory,
    _completion_chars_from_response,
    _execute_eval_unit,
    _filter_attempt_records,
    _format_progress_token_usage,
    _is_transport_runtime_error,
    _nonnegative_int,
    _normalize_adaptive_controller_lambda_rlm_settings,
    _normalize_baseline_selection,
    _OpenAITransportRecoveryGate,
    _record_is_transport_request_failure,
    _record_token_usage_state,
    _resolve_conditions_for_run,
    _resolve_external_baseline_runtime_requirements,
    _resolve_openai_context_window_for_run,
    _single_line_error,
    _usage_token_counts_from_patch_meta,
    _validate_external_baseline_runtime_requirements,
    set_active_container_pool,
    set_active_policy_controllers,
    set_active_run_settings,
)


class _DynamicAttrProxy:
    def __init__(self, name: str) -> None:
        self._name = name

    def _value(self) -> Any:
        return getattr(_legacy, self._name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value(), name)

    def __fspath__(self) -> str:
        return self._value().__fspath__()

    def __repr__(self) -> str:
        return repr(self._value())

    def __str__(self) -> str:
        return str(self._value())

    def __truediv__(self, other: object) -> Any:
        return self._value() / other


_REPO_ROOT = _DynamicAttrProxy("_REPO_ROOT")
_RunResultsLock = _DynamicAttrProxy("_RunResultsLock")


def make_patcher(*args: Any, **kwargs: Any) -> Any:
    return _legacy.make_patcher(*args, **kwargs)


def load_cases(*args: Any, **kwargs: Any) -> Any:
    return _legacy.load_cases(*args, **kwargs)


def _build_run_settings_fingerprint(*args: Any, **kwargs: Any) -> Any:
    return _legacy._build_run_settings_fingerprint(*args, **kwargs)


def _write_resolved_run_config_lockfile(*args: Any, **kwargs: Any) -> Any:
    return _legacy._write_resolved_run_config_lockfile(*args, **kwargs)


def _load_existing_run_state(*args: Any, **kwargs: Any) -> Any:
    return _legacy._load_existing_run_state(*args, **kwargs)


def _repair_trailing_results_tail(*args: Any, **kwargs: Any) -> Any:
    return _legacy._repair_trailing_results_tail(*args, **kwargs)


def check_testcontainers_runtime_available(*args: Any, **kwargs: Any) -> Any:
    return _legacy.check_testcontainers_runtime_available(*args, **kwargs)


def cleanup_stale_eval_containers(*args: Any, **kwargs: Any) -> Any:
    return _legacy.cleanup_stale_eval_containers(*args, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(_legacy, name)
