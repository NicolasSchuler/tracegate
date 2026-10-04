"""JSONL recording, results locking, and metadata building for the eval harness.

Extracted from ``evals.run_eval`` to keep the main orchestrator lean.
Re-exported by ``evals.run_eval`` for backward compatibility.
"""

from __future__ import annotations

import argparse
import os
import platform
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baselines.case_information import CASE_INFO_POLICY
from baselines.openai_compat import normalize_compat_profile

from . import _json as json
from .adaptive_online_policy import ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER
from .attempt_metadata import (
    AttemptMetadataRequest,
    _build_attempt_metadata_request_from_legacy_values,
)
from .decision_model.config import decision_model_run_config
from .decision_model.policy import decision_model_metadata_fields
from .eval_config import _execution_config_payload, build_execution_config_from_args
from .eval_evidence import _xxh3_128_jsonlike, _xxh3_128_text
from .eval_types import (
    ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
    ADAPTIVE_BUDGET_REWARD_SCOPE,
    ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    ADAPTIVE_ONLINE_REWARD_SCOPE,
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
    BASELINE_CONDITION_BY_NAME,
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
    DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS,
    DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
    DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
    DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
    DEFAULT_TRANSPORT_FAIL_FAST_STREAK,
    EVAL_RECORD_SCHEMA_VERSION,
    PATCH_VERIFIER_MIN_SCORE,
    PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE,
    PATCH_VERIFIER_SCOPE_ALL,
    PATCH_VERIFIER_SCOPE_DISABLED,
    CaseSpec,
    Condition,
)
from .policy_feature_scaling import DEFAULT_POLICY_FEATURE_SCALING_PROFILE

DEFAULT_REPO_OPENAI_COMPAT_PROFILE = "lmstudio_model_yaml"


def _resolve_openai_compat_settings(
    *,
    prefix: str,
    profile: Any,
    allow_reasoning_fallback: Any,
    enable_thinking: Any,
) -> dict[str, Any]:
    normalized_profile = normalize_compat_profile(profile or DEFAULT_REPO_OPENAI_COMPAT_PROFILE)
    normalized_enable_thinking = bool(enable_thinking)
    normalized_allow_reasoning_fallback = bool(allow_reasoning_fallback)
    if normalized_enable_thinking and not normalized_allow_reasoning_fallback:
        normalized_allow_reasoning_fallback = True
    return {
        f"{prefix}compat_profile": normalized_profile,
        f"{prefix}allow_reasoning_fallback": normalized_allow_reasoning_fallback,
        f"{prefix}enable_thinking": normalized_enable_thinking,
    }


def _native_openai_compat_settings_from_args(*, args: argparse.Namespace) -> dict[str, Any]:
    return _resolve_openai_compat_settings(
        prefix="openai_",
        profile=getattr(args, "openai_compat_profile", DEFAULT_REPO_OPENAI_COMPAT_PROFILE),
        allow_reasoning_fallback=getattr(args, "openai_allow_reasoning_fallback", True),
        enable_thinking=getattr(args, "openai_enable_thinking", True),
    )


def _baseline_openai_compat_settings_from_args(*, args: argparse.Namespace) -> dict[str, Any]:
    return _resolve_openai_compat_settings(
        prefix="baseline_openai_",
        profile=getattr(args, "baseline_openai_compat_profile", DEFAULT_REPO_OPENAI_COMPAT_PROFILE),
        allow_reasoning_fallback=getattr(args, "baseline_openai_allow_reasoning_fallback", True),
        enable_thinking=getattr(args, "baseline_openai_enable_thinking", True),
    )


class _RunResultsLock:
    """File-based cross-process advisory lock for a results JSONL file."""

    def __init__(self, *, path: Path, handle: Any) -> None:
        self.path = path
        self._handle = handle
        self._released = False

    @classmethod
    def acquire(
        cls,
        *,
        results_path: Path,
        run_id: str,
    ) -> tuple[_RunResultsLock | None, str | None, str | None]:
        """Try to acquire an exclusive file lock for *results_path*.

        Returns:
            Tuple of (lock_or_None, error_or_None, warning_or_None).
        """
        lock_path = results_path.with_suffix(results_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                import fcntl
            except ModuleNotFoundError:
                handle.close()
                return (
                    None,
                    None,
                    (
                        "run lock not available on this platform "
                        "(missing fcntl); continuing without cross-process lock."
                    ),
                )
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                errno_value = getattr(exc, "errno", None)
                if errno_value in {11, 35}:
                    return (
                        None,
                        f"Run lock already held for run_id={run_id!r} ({lock_path}).",
                        None,
                    )
                return None, f"failed to acquire run lock: {type(exc).__name__}: {exc}", None
            handle.seek(0)
            handle.truncate()
            payload = {
                "run_id": run_id,
                "pid": os.getpid(),
                "acquired_at_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "results_path": str(results_path),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            return cls(path=lock_path, handle=handle), None, None
        except Exception:
            try:
                handle.close()
            except Exception:
                pass
            raise

    def release(self) -> None:
        """Release the file lock and remove the lock file."""
        if self._released:
            return
        self._released = True
        try:
            try:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            self._handle.close()
        finally:
            try:
                self.path.unlink()
            except OSError:
                pass


class _SynchronizedJsonlWriter:
    """Thread-safe, append-only JSONL writer with lazy file open."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._handle: Any = None

    def append_one(self, record: dict[str, Any]) -> None:
        """Serialize *record* as JSON and append one line to the file."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            if self._handle is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self._path.open("a", encoding="utf-8")
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            if self._handle is None:
                return
            try:
                self._handle.close()
            finally:
                self._handle = None


def _build_run_metadata(
    *,
    args: argparse.Namespace,
    patcher_name: str,
    run_id: str,
    started_at_utc: str,
    run_settings_fingerprint: str,
) -> dict[str, Any]:
    """Assemble the run-level metadata dict embedded in every record."""
    patcher_config = _build_patcher_config(args=args, patcher_name=patcher_name)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "deterministic": bool(args.deterministic),
        "seed": args.seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "llmdebug_version": _resolve_llmdebug_version(),
        "patcher_name": patcher_name,
        "patcher_config": patcher_config,
        "patcher_config_fingerprint": _xxh3_128_jsonlike(patcher_config),
        "run_settings_fingerprint": run_settings_fingerprint,
        "record_policy": {
            "prompt": str(args.record_prompt),
            "evidence": str(args.record_evidence),
        },
    }
    env_capture = _collect_allowed_env(args.record_env)
    if env_capture:
        metadata["env"] = env_capture
    dependency_versions = _collect_dependency_versions(mode=str(args.record_deps))
    if dependency_versions:
        metadata["dependency_versions"] = dependency_versions
    return metadata


def _build_patcher_config(*, args: argparse.Namespace, patcher_name: str) -> dict[str, Any]:
    """Assemble the patcher-config dict capturing all tuning knobs."""
    config: dict[str, Any] = {
        "patcher": str(args.patcher),
        "patcher_name": patcher_name,
        "k": int(args.k),
        "transport_fail_fast_streak": int(
            getattr(args, "transport_fail_fast_streak", DEFAULT_TRANSPORT_FAIL_FAST_STREAK)
        ),
        "case_fail_fast_on_request_budget_exceeded": bool(
            getattr(args, "case_fail_fast_on_request_budget_exceeded", True)
        ),
        "output_format": str(args.output_format),
        "force_patch_attempt": bool(args.force_patch_attempt),
        "deterministic": bool(args.deterministic),
        "seed": args.seed,
        "no_files_context": bool(args.no_files_context),
        "max_context_chars": int(args.max_context_chars),
        "context_protocol": str(args.context_protocol),
        "context_stage_a_max_requests": int(args.context_stage_a_max_requests),
        "context_stage_a_max_chars": int(args.context_stage_a_max_chars),
        "retry_prompt_mode": str(args.retry_prompt_mode),
        "retry_context_mode": str(args.retry_context_mode),
        "retry_history_max_chars": int(args.retry_history_max_chars),
        "native_baseline_mode": str(getattr(args, "native_baseline_mode", "") or ""),
        "case_info_policy": CASE_INFO_POLICY,
        "prompt_retry_reflection": bool(getattr(args, "prompt_retry_reflection", False)),
        "prompt_adaptive_stage_b_cot": bool(getattr(args, "prompt_adaptive_stage_b_cot", False)),
        "prompt_repair_method_scaffold": bool(
            getattr(args, "prompt_repair_method_scaffold", False)
        ),
        "prompt_retry_plan_then_patch": bool(getattr(args, "prompt_retry_plan_then_patch", False)),
        "pytest_output_tail_lines": int(
            getattr(args, "pytest_output_tail_lines", DEFAULT_PYTEST_OUTPUT_TAIL_LINES)
        ),
        "snapshot_artifact_max_string_chars": int(
            getattr(
                args,
                "snapshot_artifact_max_string_chars",
                DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
            )
        ),
        "prompt_budget_total_chars": int(args.prompt_budget_total_chars),
        "prompt_budget_retry_delta_chars": int(args.prompt_budget_retry_delta_chars),
        "prompt_budget_stdout_chars": int(args.prompt_budget_stdout_chars),
        "prompt_budget_snapshot_chars": int(args.prompt_budget_snapshot_chars),
        "prompt_budget_evidence_chars": int(args.prompt_budget_evidence_chars),
        "prompt_budget_retry_history_chars": int(args.prompt_budget_retry_history_chars),
        "context_compaction_mode": str(args.context_compaction_mode),
        "context_compaction_scope": str(args.context_compaction_scope),
        "context_compaction_max_input_chars": int(args.context_compaction_max_input_chars),
        "context_compaction_target_chars": int(args.context_compaction_target_chars),
        "context_compaction_max_calls": int(args.context_compaction_max_calls),
        "prompt_order_mode": str(getattr(args, "prompt_order_mode", "current")),
        "adaptive_reveal_mode": str(getattr(args, "adaptive_reveal_mode", "full")),
        **decision_model_run_config(args),
        "snapshot_frame_pruning_mode": str(getattr(args, "snapshot_frame_pruning_mode", "off")),
        "openai_auto_context_window": bool(getattr(args, "openai_auto_context_window", False)),
        **_native_openai_compat_settings_from_args(args=args),
        "openai_response_mode_autoswitch": str(
            getattr(args, "openai_response_mode_autoswitch", "off")
        ),
        "adaptive_gate_policy": str(getattr(args, "adaptive_gate_policy", "balanced")),
        "adaptive_max_helper_calls": int(getattr(args, "adaptive_max_helper_calls", 1)),
        "adaptive_stagnation_window": int(getattr(args, "adaptive_stagnation_window", 1)),
        "adaptive_saturation_threshold": float(
            getattr(args, "adaptive_saturation_threshold", 0.75)
        ),
        "adaptive_no_progress_token_cap": int(
            getattr(args, "adaptive_no_progress_token_cap", DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP)
        ),
        "adaptive_helper_budget_profile": str(
            getattr(args, "adaptive_helper_budget_profile", "default")
        ),
        "adaptive_budget_topup_max_calls": int(
            getattr(
                args, "adaptive_budget_topup_max_calls", DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS
            )
        ),
        "adaptive_openai_helper_response_mode": str(
            getattr(args, "adaptive_openai_helper_response_mode", "inherit")
        ),
        "adaptive_controller_lambda_rlm": bool(
            getattr(args, "adaptive_controller_lambda_rlm", False)
        ),
        "adaptive_controller_lambda_rlm_repo_dir": str(
            getattr(args, "adaptive_controller_lambda_rlm_repo_dir", "") or ""
        ),
        "adaptive_controller_lambda_rlm_python": str(
            getattr(args, "adaptive_controller_lambda_rlm_python", "") or ""
        ),
        "adaptive_controller_lambda_rlm_context_window_chars": int(
            getattr(args, "adaptive_controller_lambda_rlm_context_window_chars", 100_000)
        ),
        "adaptive_controller_lambda_rlm_timeout_sec": int(
            getattr(args, "adaptive_controller_lambda_rlm_timeout_sec", 120)
        ),
        "adaptive_informativeness_pregate": bool(
            getattr(args, "adaptive_informativeness_pregate", False)
        ),
        "adaptive_informativeness_pregate_threshold": float(
            getattr(args, "adaptive_informativeness_pregate_threshold", 0.7)
        ),
        "adaptive_online_policy_mode": str(getattr(args, "adaptive_online_policy_mode", "off")),
        "adaptive_online_policy_algo": str(getattr(args, "adaptive_online_policy_algo", "linucb")),
        "adaptive_online_alpha": float(
            getattr(args, "adaptive_online_alpha", DEFAULT_ADAPTIVE_ONLINE_ALPHA)
        ),
        "adaptive_online_epsilon": float(
            getattr(args, "adaptive_online_epsilon", DEFAULT_ADAPTIVE_ONLINE_EPSILON)
        ),
        "adaptive_online_l2": float(getattr(args, "adaptive_online_l2", 1.0)),
        "adaptive_online_warmup_decisions": int(
            getattr(
                args, "adaptive_online_warmup_decisions", DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS
            )
        ),
        "adaptive_online_save_every": int(getattr(args, "adaptive_online_save_every", 50)),
        "adaptive_online_decay": float(
            getattr(args, "adaptive_online_decay", DEFAULT_ADAPTIVE_ONLINE_DECAY)
        ),
        "adaptive_online_feature_scaling_profile": str(
            getattr(args, "adaptive_online_feature_scaling_profile", "v1")
        ),
        "adaptive_online_action_score_biases": str(
            getattr(
                args,
                "adaptive_online_action_score_biases",
                DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
            )
        ),
        "adaptive_online_init_state_path": str(
            getattr(args, "adaptive_online_init_state_path", "")
        ),
        "adaptive_online_state_path": str(getattr(args, "adaptive_online_state_path", "")),
        "adaptive_online_events_path": str(getattr(args, "adaptive_online_events_path", "")),
        "adaptive_budget_policy_mode": str(getattr(args, "adaptive_budget_policy_mode", "on")),
        "adaptive_budget_policy_algo": str(getattr(args, "adaptive_budget_policy_algo", "linucb")),
        "adaptive_budget_policy_architecture": str(
            getattr(args, "adaptive_budget_policy_architecture", "tier_binary")
        ),
        "adaptive_budget_policy_tiers": str(
            getattr(args, "adaptive_budget_policy_tiers", DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS)
        ),
        "adaptive_budget_policy_max_ups_per_case": int(
            getattr(
                args,
                "adaptive_budget_policy_max_ups_per_case",
                DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
            )
        ),
        "adaptive_budget_online_alpha": float(
            getattr(args, "adaptive_budget_online_alpha", DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA)
        ),
        "adaptive_budget_online_epsilon": float(
            getattr(args, "adaptive_budget_online_epsilon", DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON)
        ),
        "adaptive_budget_online_l2": float(getattr(args, "adaptive_budget_online_l2", 1.0)),
        "adaptive_budget_online_warmup_decisions": int(
            getattr(args, "adaptive_budget_online_warmup_decisions", 20)
        ),
        "adaptive_budget_online_save_every": int(
            getattr(args, "adaptive_budget_online_save_every", 50)
        ),
        "adaptive_budget_online_decay": float(
            getattr(args, "adaptive_budget_online_decay", DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY)
        ),
        "adaptive_budget_feature_scaling_profile": str(
            getattr(args, "adaptive_budget_feature_scaling_profile", "v1")
        ),
        "adaptive_budget_init_state_path": str(
            getattr(args, "adaptive_budget_init_state_path", "")
        ),
        "adaptive_budget_online_state_path": str(
            getattr(args, "adaptive_budget_online_state_path", "")
        ),
        "adaptive_budget_online_events_path": str(
            getattr(args, "adaptive_budget_online_events_path", "")
        ),
        "adaptive_budget_bwk_run_token_budget": int(
            getattr(args, "adaptive_budget_bwk_run_token_budget", 0)
        ),
        "adaptive_budget_reward_rho_budget_exceeded": float(
            getattr(
                args,
                "adaptive_budget_reward_rho_budget_exceeded",
                ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
            )
        ),
        "adaptive_reward_lambda_tokens": float(
            getattr(args, "adaptive_reward_lambda_tokens", ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT)
        ),
        "adaptive_reward_mu_latency": float(
            getattr(args, "adaptive_reward_mu_latency", ADAPTIVE_REWARD_MU_LATENCY_DEFAULT)
        ),
        "adaptive_reward_token_scale": float(
            getattr(args, "adaptive_reward_token_scale", ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT)
        ),
        "adaptive_reward_latency_scale": float(
            getattr(args, "adaptive_reward_latency_scale", ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT)
        ),
        "adaptive_reward_eta_progress": float(
            getattr(args, "adaptive_reward_eta_progress", ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT)
        ),
        "adaptive_reward_kappa_stagnation": float(
            getattr(
                args,
                "adaptive_reward_kappa_stagnation",
                ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
            )
        ),
        "adaptive_reward_success_hint": float(
            getattr(args, "adaptive_reward_success_hint", ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT)
        ),
        "adaptive_reward_progress_scale": float(
            getattr(args, "adaptive_reward_progress_scale", ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT)
        ),
        "adaptive_reward_stagnation_scale": float(
            getattr(
                args,
                "adaptive_reward_stagnation_scale",
                ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
            )
        ),
        "adaptive_reward_materialization_scale": float(
            getattr(
                args,
                "adaptive_reward_materialization_scale",
                ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
            )
        ),
        "syntax_preflight_enabled": bool(getattr(args, "syntax_preflight_enabled", True)),
        "syntax_preflight_max_cycles": int(
            getattr(args, "syntax_preflight_max_cycles", DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES)
        ),
        "syntax_preflight_fallthrough": bool(getattr(args, "syntax_preflight_fallthrough", True)),
        "patch_sanitize_enabled": bool(getattr(args, "patch_sanitize_enabled", True)),
        "patch_verifier_scope": str(
            getattr(args, "patch_verifier_scope", PATCH_VERIFIER_SCOPE_ALL)
        ),
        "patch_verifier_min_score": int(
            getattr(args, "patch_verifier_min_score", PATCH_VERIFIER_MIN_SCORE)
        ),
        "patch_verifier_infer_test_source_scope": bool(
            getattr(args, "patch_verifier_infer_test_source_scope", True)
        ),
    }
    try:
        config["execution"] = _execution_config_payload(build_execution_config_from_args(args))
    except ValueError:
        pass
    if str(args.patcher) == "openai":
        config.update(
            {
                "openai_base_url": str(args.openai_base_url),
                "openai_model": str(args.openai_model),
                "openai_http2": bool(getattr(args, "openai_http2", True)),
                "openai_timeout_sec": float(args.openai_timeout_sec),
                "openai_transport_max_retries": int(args.openai_transport_max_retries),
                "openai_transport_retry_backoff_base_sec": float(
                    args.openai_transport_retry_backoff_base_sec
                ),
                "openai_transport_retry_backoff_max_sec": float(
                    args.openai_transport_retry_backoff_max_sec
                ),
                "openai_transport_recovery": bool(getattr(args, "openai_transport_recovery", True)),
                "openai_transport_recovery_max_wait_sec": float(
                    getattr(args, "openai_transport_recovery_max_wait_sec", 60.0)
                ),
                "openai_transport_recovery_poll_sec": float(
                    getattr(args, "openai_transport_recovery_poll_sec", 2.0)
                ),
                "openai_max_propose_sec": float(args.openai_max_propose_sec),
                "openai_max_tokens": int(args.openai_max_tokens),
                "openai_plan_patch_max_tokens": int(
                    getattr(args, "openai_plan_patch_max_tokens", 768)
                ),
                "openai_context_window_tokens": int(args.openai_context_window_tokens),
                "openai_auto_context_window": bool(args.openai_auto_context_window),
                "openai_temperature": float(args.openai_temperature),
                "openai_response_mode": str(args.openai_response_mode),
                "openai_response_mode_autoswitch": str(args.openai_response_mode_autoswitch),
                "openai_force_patch_attempt": bool(args.openai_force_patch_attempt),
            }
        )
    elif str(args.patcher) == "baseline":
        config.update(
            {
                "baseline_name": str(getattr(args, "baseline_name", "") or "").strip().lower(),
                "baseline_repo_dir": str(getattr(args, "baseline_repo_dir", "") or "").strip(),
                "baseline_python": str(getattr(args, "baseline_python", "") or "").strip(),
            }
        )
    elif str(args.patcher) == "command":
        config["patch_cmd"] = str(args.patch_cmd or os.getenv("LLMDEBUG_EVAL_PATCH_CMD", ""))
        config["patch_cmd_shell"] = bool(
            args.patch_cmd_shell
            or os.getenv("LLMDEBUG_EVAL_PATCH_CMD_SHELL", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        try:
            env_timeout = float(os.getenv("LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC", "30.0"))
        except ValueError:
            env_timeout = 30.0
        config["patch_cmd_timeout_sec"] = float(
            args.patch_cmd_timeout_sec if args.patch_cmd else env_timeout
        )
    return config


def _resolve_llmdebug_version() -> str:
    try:
        import llmdebug  # type: ignore

        value = getattr(llmdebug, "__version__", None)
        if isinstance(value, str) and value:
            return value
    except Exception:
        pass
    return "unknown"


def _collect_allowed_env(keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in keys:
        key = raw.strip()
        if not key:
            continue
        value = os.environ.get(key)
        if value is not None:
            out[key] = value
    return out


def _collect_dependency_versions(*, mode: str) -> dict[str, str]:
    """Collect installed package versions based on *mode*.

    Modes: ``"none"`` returns empty, ``"runtime"`` returns all installed
    packages, default returns a curated list of key dependencies.
    """
    if mode == "none":
        return {}

    try:
        from importlib import metadata as importlib_metadata
    except ImportError:
        return {}

    if mode == "runtime":
        out: dict[str, str] = {}
        for dist in importlib_metadata.distributions():
            try:
                name = dist.metadata.get("Name")
                version = dist.version
            except Exception:
                continue
            if not name or not version:
                continue
            out[str(name)] = str(version)
        return dict(sorted(out.items()))

    names = (
        "llmdebug",
        "pytest",
        "mcp",
        "openai",
        "numpy",
        "torch",
        "jax",
        "tensorflow",
    )
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def _normalize_openai_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(("/v1", "/v2", "/v3", "/v4")):
        return base
    return base + "/v1"


def _default_run_metadata(*, run_id: str, patcher_name: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "deterministic": False,
        "seed": None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "llmdebug_version": _resolve_llmdebug_version(),
        "patcher_name": patcher_name,
        "patcher_config": {"patcher_name": patcher_name},
        "patcher_config_fingerprint": _xxh3_128_text("default"),
        "run_settings_fingerprint": _xxh3_128_text("default-run-settings"),
        "record_policy": {"prompt": "hash", "evidence": "hash"},
    }


def _build_base_record(
    *,
    case: CaseSpec,
    condition: Condition,
    run_id: str,
    attempt: int,
    patcher_name: str,
    run_metadata: dict[str, Any],
    attempt_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    context_protocol: str = "monolithic",
) -> dict[str, Any]:
    """Construct the skeleton JSONL record shared by every attempt."""
    return {
        "schema_version": EVAL_RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case.case_id,
        "category": case.category,
        "difficulty": case.difficulty,
        "snapshot_dependency": case.snapshot_dependency,
        "expected_exception": case.expected_exception,
        "tags": case.tags,
        "bug_source": case.bug_source,
        "contamination_risk": case.contamination_risk,
        "condition": condition.value,
        "attempt": attempt,
        "patcher": patcher_name,
        "context_protocol": context_protocol,
        "run_metadata": dict(run_metadata),
        "attempt_metadata": dict(attempt_metadata),
        "model_metadata": dict(model_metadata),
    }


def _empty_attempt_metadata(*, attempt: int) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "max_attempts": 0,
        "attempt_ordinal_label": "0/0",
        "prompt_xxh3_128": None,
        "prompt_path": None,
        "evidence_xxh3_128": None,
        "evidence_path": None,
        "stage_a_prompt_xxh3_128": None,
        "stage_a_response_xxh3_128": None,
        "stage_a_prompt_chars": 0,
        "stage_a_requests_count": 0,
        "stage_a_chars_returned": 0,
        "stage_a_request_success": None,
        "stage_b_prompt_chars": None,
        "prompt_chars_total": 0,
        "completion_chars_total": 0,
        "chars_input": 0,
        "chars_output": 0,
        "chars_total": 0,
        "llm_call_count": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "tokens_input": 0,
        "tokens_output": 0,
        "retry_summary": None,
        "context_protocol": "monolithic",
        "retry_prompt_mode": "full",
        "retry_context_mode": "history",
        "retry_history_chars": 0,
        "retry_history_truncated": False,
        "retry_history_turns_included": 0,
        "retry_history_budget_chars": 0,
        "retry_history_budget_applied": False,
        "retry_delta_prompt": False,
        "retry_delta_chars": 0,
        "adaptive_mode": "none",
        "adaptive_policy": "disabled",
        "adaptive_signature_repeat": False,
        "adaptive_stagnation_streak": 0,
        "adaptive_helper_invoked": False,
        "adaptive_helper_action_default": ACTION_INVOKE_HELPER,
        "adaptive_helper_action_selected": ACTION_INVOKE_HELPER,
        "adaptive_helper_action_executed": ACTION_SKIP_HELPER,
        "adaptive_helper_action_fallback_reason": "",
        "adaptive_helper_available_actions": [],
        "adaptive_helper_unavailable_action_reasons": {},
        "adaptive_helper_materialized_context": False,
        "adaptive_helper_outcome": ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
        "adaptive_helper_failure_kind": "",
        "adaptive_helper_retry_count": 0,
        "adaptive_helper_transport_retry_count": 0,
        "adaptive_helper_decision_reason": "",
        "adaptive_stagnation_override_applied": False,
        "adaptive_stagnation_override_reason": "",
        "adaptive_helper_need_more_context": None,
        "adaptive_helper_calls_used_before_attempt": 0,
        "adaptive_helper_calls_used_after_attempt": 0,
        "adaptive_helper_budget_max_calls": 0,
        "adaptive_helper_budget_unlimited": True,
        "adaptive_helper_budget_exhausted": False,
        "adaptive_helper_saturation_guarded": False,
        "adaptive_helper_budget_profile": "default",
        "adaptive_budget_topup_granted": False,
        "adaptive_budget_topup_reason": "",
        "adaptive_budget_default_control_values": {},
        "adaptive_budget_selected_control_values": {},
        "adaptive_budget_applied_control_values": {},
        "adaptive_budget_selected_controls_changed": {},
        "adaptive_budget_applied_controls_changed": {},
        "adaptive_budget_partial_override": False,
        "adaptive_budget_partial_override_errors": {},
        "adaptive_budget_effective_helper_max_calls": 0,
        "adaptive_budget_effective_stage_a_max_requests": 1,
        "adaptive_budget_effective_stage_a_max_chars": 1,
        "adaptive_budget_effective_retry_history_max_chars": 0,
        "adaptive_budget_effective_context_compaction_max_calls": 0,
        "adaptive_budget_effective_openai_max_tokens": 0,
        "adaptive_budget_effective_response_mode": "",
        "adaptive_context_requested": False,
        "adaptive_snapshot_included": False,
        "adaptive_snapshot_lite_included": False,
        "adaptive_snapshot_attempt1_denied": False,
        "adaptive_snapshot_attempt1_denied_reason": "",
        "adaptive_progress_made": False,
        "adaptive_progress_reason": "",
        "adaptive_progress_score": 0.0,
        "adaptive_progress_component_signature": 0.0,
        "adaptive_progress_component_new_evidence": 0.0,
        "adaptive_progress_component_pytest_fail_reduction": 0.0,
        "adaptive_progress_component_stage_advance": 0.0,
        "adaptive_progress_component_verifier_score": 0.0,
        "adaptive_progress_component_scope_alignment": 0.0,
        "adaptive_pytest_before_failed_total": None,
        "adaptive_pytest_after_failed_total": None,
        "adaptive_pipeline_stage_before": "",
        "adaptive_pipeline_stage_after": "",
        "adaptive_pipeline_stage_before_ord": 0,
        "adaptive_pipeline_stage_after_ord": 0,
        "adaptive_no_progress_streak": 0,
        "adaptive_progress_stop_candidate": False,
        "adaptive_progress_stop_triggered": False,
        "adaptive_progress_stop_reason": "",
        "evidence_ids_new_count": 0,
        "evidence_ids_reused_count": 0,
        "evidence_ids_sent_count": 0,
        "stage_a_index_included": False,
        "stage_a_index_reason": "",
        "openai_preflight_prompt_cap_chars": None,
        "openai_preflight_trimmed": False,
        "openai_context_window_source": "",
        "openai_context_window_probe_provider": "",
        "openai_context_window_probe_error": "",
        "openai_response_mode_autoswitch_policy": "off",
        "openai_response_mode_autoswitched": False,
        "openai_response_mode_autoswitch_from": "",
        "openai_response_mode_autoswitch_to": "",
        "openai_response_mode_autoswitch_cycle": 0,
        "openai_response_mode_autoswitch_reason": "",
        "prompt_budget_used_chars": None,
        "prompt_budget_overflow_chars": 0,
        "prompt_budget_dropped_sections": [],
        "context_compaction_mode": "off",
        "context_compaction_scope": [],
        "context_compaction_supported": False,
        "context_compaction_requested": False,
        "context_compaction_request_reason": "",
        "context_compaction_trigger": "",
        "context_compaction_auto_overflow_triggered": False,
        "context_compaction_invoked": False,
        "context_compaction_success": False,
        "context_compaction_error": "",
        "context_compaction_calls_used": 0,
        "context_compaction_max_calls": 0,
        "context_compaction_source_sections": [],
        "context_compaction_source_chars": {},
        "context_compaction_replaced_sections": [],
        "context_compaction_input_chars": 0,
        "context_compaction_target_chars": 0,
        "context_compaction_output_chars": 0,
        "context_compaction_prompt_chars": 0,
        "patch_verifier_enabled": False,
        "patch_verifier_scope": PATCH_VERIFIER_SCOPE_DISABLED,
        "patch_verifier_min_score": PATCH_VERIFIER_MIN_SCORE,
        "patch_verifier_passed": None,
        "patch_verifier_score": None,
        "patch_verifier_reasons": [],
        "patch_verifier_recovery_eligible": False,
        "patch_verifier_recovery_invoked": False,
        "patch_verifier_recovery_result": PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE,
        "patch_verifier_recovery_prompt_chars_before": 0,
        "patch_verifier_recovery_prompt_chars_after": 0,
        "patch_verifier_recovery_duration_sec": 0.0,
        "patch_verifier_recovery_terminal_note": "",
        "patch_verifier_recovery_score_before": None,
        "patch_verifier_recovery_reasons_before": [],
        "patch_verifier_recovery_patch_files_before": [],
        "patch_verifier_recovery_source_scope_matched_before": False,
        "patch_verifier_recovery_raw_meta": None,
        "patcher_request_failed": False,
        "patcher_request_failure_kind": "",
        "patcher_request_failure_error": "",
        "patcher_request_failure_status": None,
        "patcher_request_failure_exc": "",
        "runner_forced_patch_skipped": False,
        "runner_forced_patch_skip_reason": "",
        "patch_sanitize_enabled": False,
        "patch_sanitize_passed": False,
        "patch_sanitize_rejected": False,
        "patch_sanitize_reasons": [],
        "patch_sanitize_error_msg": "",
        "adaptive_informativeness_pregate_fired": False,
        "adaptive_informativeness_pregate_score": 0.0,
        "adaptive_informativeness_pregate_exception_type": "",
        "adaptive_informativeness_pregate_reason": "",
        "adaptive_informativeness_pregate_features": {},
        "adaptive_online_policy_mode": "off",
        "adaptive_online_policy_algo": "",
        "adaptive_online_decision_id": "",
        "adaptive_online_decision_strategy": "",
        "adaptive_online_action_selected": "none",
        "adaptive_online_action_applied": "none",
        "adaptive_online_propensity": 1.0,
        "adaptive_online_behavior_action": "none",
        "adaptive_online_behavior_propensity": 1.0,
        "adaptive_online_override_reason": "",
        "adaptive_online_feature_scaling_profile": DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        "adaptive_online_feature_scaling_caps": {},
        "adaptive_online_feature_scaling_history_sizes": {},
        "adaptive_online_feature_scaling_fallback_to_v1": False,
        "adaptive_online_feature_vector_raw": {},
        "adaptive_online_feature_vector": {},
        "adaptive_online_available_actions": [],
        "adaptive_online_unavailable_action_reasons": {},
        "adaptive_online_action_score_biases": {},
        "adaptive_online_configured_alpha": 0.0,
        "adaptive_online_configured_epsilon": 0.0,
        "adaptive_online_configured_decay": 1.0,
        "adaptive_online_effective_epsilon": 0.0,
        "adaptive_online_warmup_active": False,
        "adaptive_online_decay": 1.0,
        "adaptive_online_eligible": False,
        "adaptive_online_reward_lambda_tokens": ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
        "adaptive_online_reward_mu_latency": ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
        "adaptive_online_reward_token_scale": ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
        "adaptive_online_reward_latency_scale": ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
        "adaptive_online_reward_eta_progress": ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
        "adaptive_online_reward_kappa_stagnation": ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
        "adaptive_online_reward_success_hint_scale": ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
        "adaptive_online_reward_progress_hint_scale": ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
        "adaptive_online_reward_stagnation_penalty_scale": (
            ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT
        ),
        "adaptive_online_reward_helper_materialization_bonus_scale": (
            ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT
        ),
        "adaptive_online_reward_scope": ADAPTIVE_ONLINE_REWARD_SCOPE,
        "adaptive_online_reward": None,
        "adaptive_online_reward_success_hint": 0.0,
        "adaptive_online_reward_progress_hint": 0.0,
        "adaptive_online_reward_progress_score": 0.0,
        "adaptive_online_reward_token_penalty": 0.0,
        "adaptive_online_reward_latency_penalty": 0.0,
        "adaptive_online_reward_stagnation_penalty": 0.0,
        "adaptive_online_reward_helper_materialization_bonus": 0.0,
        "adaptive_online_reward_delayed_rescue_bonus": 0.0,
        "adaptive_online_reward_credit_resolved_attempt": 0,
        "adaptive_online_reward_credit_horizon_attempts": 0,
        "adaptive_online_reward_credit_resolution_reason": "",
        "adaptive_online_local_llm_call_count": 0,
        "adaptive_online_local_tokens_prompt": 0,
        "adaptive_online_local_tokens_completion": 0,
        "adaptive_online_local_tokens_total": 0,
        "adaptive_online_local_wall_sec": 0.0,
        "adaptive_online_updated": False,
        "adaptive_online_update_reason": "",
        "adaptive_online_event_logged": False,
        **decision_model_metadata_fields(None),
        "adaptive_lite_replaced_snapshot": False,
        "adaptive_budget_policy_mode": "off",
        "adaptive_budget_policy_algo": "",
        "adaptive_budget_policy_architecture": "tier_binary",
        "adaptive_budget_decision_id": "",
        "adaptive_budget_decision_strategy": "",
        "adaptive_budget_action_selected": "none",
        "adaptive_budget_action_applied": "none",
        "adaptive_budget_family_selected": "",
        "adaptive_budget_family_applied": "",
        "adaptive_budget_available_families": [],
        "adaptive_budget_unavailable_family_reasons": {},
        "adaptive_budget_family_propensity": 1.0,
        "adaptive_budget_family_decision_strategy": "",
        "adaptive_budget_propensity": 1.0,
        "adaptive_budget_behavior_action": "none",
        "adaptive_budget_behavior_propensity": 1.0,
        "adaptive_budget_override_reason": "",
        "adaptive_budget_feature_scaling_profile": DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        "adaptive_budget_feature_scaling_caps": {},
        "adaptive_budget_feature_scaling_history_sizes": {},
        "adaptive_budget_feature_scaling_fallback_to_v1": False,
        "adaptive_budget_family_feature_scaling_caps": {},
        "adaptive_budget_family_feature_scaling_history_sizes": {},
        "adaptive_budget_family_feature_scaling_fallback_to_v1": False,
        "adaptive_budget_feature_vector_raw": {},
        "adaptive_budget_feature_vector": {},
        "adaptive_budget_configured_alpha": 0.0,
        "adaptive_budget_configured_epsilon": 0.0,
        "adaptive_budget_configured_decay": 1.0,
        "adaptive_budget_effective_epsilon": 0.0,
        "adaptive_budget_warmup_active": False,
        "adaptive_budget_decay": 1.0,
        "adaptive_budget_eligible": False,
        "adaptive_budget_tiers": [1.0],
        "adaptive_budget_tier_index_before": 0,
        "adaptive_budget_tier_index_after": 0,
        "adaptive_budget_selected_multiplier": 1.0,
        "adaptive_budget_applied_multiplier": 1.0,
        "adaptive_budget_default_multiplier": 1.0,
        "adaptive_budget_max_ups_per_case": 0,
        "adaptive_budget_ups_used": 0,
        "adaptive_budget_reward_lambda_tokens": ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
        "adaptive_budget_reward_mu_latency": ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
        "adaptive_budget_reward_token_scale": ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
        "adaptive_budget_reward_latency_scale": ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
        "adaptive_budget_reward_rho_budget_exceeded": ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
        "adaptive_budget_reward_eta_progress": ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
        "adaptive_budget_reward_kappa_stagnation": ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
        "adaptive_budget_reward_success_hint_scale": ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
        "adaptive_budget_reward_progress_hint_scale": ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
        "adaptive_budget_reward_stagnation_penalty_scale": (
            ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT
        ),
        "adaptive_budget_reward_scope": ADAPTIVE_BUDGET_REWARD_SCOPE,
        "adaptive_budget_reward": None,
        "adaptive_budget_reward_success_hint": 0.0,
        "adaptive_budget_reward_progress_hint": 0.0,
        "adaptive_budget_reward_progress_score": 0.0,
        "adaptive_budget_reward_token_penalty": 0.0,
        "adaptive_budget_reward_latency_penalty": 0.0,
        "adaptive_budget_reward_stagnation_penalty": 0.0,
        "adaptive_budget_reward_budget_exceeded_penalty": 0.0,
        "adaptive_budget_request_budget_exceeded": False,
        "adaptive_budget_local_llm_call_count": 0,
        "adaptive_budget_local_tokens_prompt": 0,
        "adaptive_budget_local_tokens_completion": 0,
        "adaptive_budget_local_tokens_total": 0,
        "adaptive_budget_local_wall_sec": 0.0,
        "adaptive_budget_updated": False,
        "adaptive_budget_update_reason": "",
        "adaptive_budget_event_logged": False,
        "attempt_shared_propose_budget_base_sec": 0.0,
        "attempt_shared_propose_budget_multiplier": 1.0,
        "attempt_shared_propose_budget_sec": 0.0,
        "syntax_preflight_enabled": False,
        "syntax_preflight_cycles_used": 0,
        "syntax_preflight_failures": 0,
        "syntax_preflight_last_error_type": "",
        "syntax_preflight_last_error_msg": "",
        "timing_breakdown": {
            "stage_a_llm_sec": 0.0,
            "stage_a_context_fulfillment_sec": 0.0,
            "stage_a_total_sec": 0.0,
            "snapshot_read_sec": 0.0,
            "adaptive_controller_prepare_sec": 0.0,
            "adaptive_controller_summarizer_sec": 0.0,
            "adaptive_controller_sidecar_build_sec": 0.0,
            "adaptive_controller_sidecar_write_sec": 0.0,
            "baseline_case_conversion_sec": 0.0,
            "baseline_sidecar_load_sec": 0.0,
            "context_compaction_llm_sec": 0.0,
            "context_compaction_total_sec": 0.0,
            "prompt_assembly_sec": 0.0,
            "stage_b_llm_sec": 0.0,
            "baseline_internal_total_sec": 0.0,
            "baseline_eval_sec": 0.0,
            "baseline_orchestration_sec": 0.0,
            "patch_apply_sec": 0.0,
            "syntax_preflight_sec": 0.0,
            "test_before_sec": 0.0,
            "test_after_sec": 0.0,
            "recovery_llm_sec": 0.0,
            "recovery_total_sec": 0.0,
            "llmdebug_total_sec": 0.0,
            "llm_total_sec": 0.0,
            "overhead_sec": 0.0,
        },
        "token_breakdown": {
            "stage_a_tokens_prompt": 0,
            "stage_a_tokens_completion": 0,
            "stage_a_tokens_total": 0,
            "stage_b_tokens_prompt": 0,
            "stage_b_tokens_completion": 0,
            "stage_b_tokens_total": 0,
            "compaction_tokens_prompt": 0,
            "compaction_tokens_completion": 0,
            "compaction_tokens_total": 0,
            "recovery_tokens_prompt": 0,
            "recovery_tokens_completion": 0,
            "recovery_tokens_total": 0,
            "tool_response_chars": 0,
            "snapshot_chars": 0,
        },
    }


def _materialize_attempt_metadata_request(
    *,
    request: AttemptMetadataRequest,
) -> dict[str, Any]:
    """Build the per-attempt metadata dict from a typed request."""
    stage_a = request.stage_a
    retry = request.retry
    adaptive = request.adaptive
    prompt_budget = request.prompt_budget
    context_compaction = request.context_compaction
    openai = request.openai
    syntax_preflight = request.syntax_preflight

    metadata = _empty_attempt_metadata(attempt=request.attempt)
    metadata["max_attempts"] = max(0, int(request.max_attempts))
    metadata["attempt_ordinal_label"] = f"{request.attempt}/{max(0, int(request.max_attempts))}"
    metadata["context_protocol"] = request.context_protocol
    metadata["stage_a_prompt_chars"] = int(stage_a.stage_a_prompt_chars)
    metadata["stage_a_requests_count"] = int(stage_a.stage_a_requests_count)
    metadata["stage_a_chars_returned"] = int(stage_a.stage_a_chars_returned)
    metadata["stage_a_request_success"] = stage_a.stage_a_request_success
    metadata["stage_b_prompt_chars"] = len(request.prompt)
    normalized_prompt_chars = max(0, int(request.prompt_chars_total))
    normalized_completion_chars = max(0, int(request.completion_chars_total))
    metadata["prompt_chars_total"] = normalized_prompt_chars
    metadata["completion_chars_total"] = normalized_completion_chars
    metadata["chars_input"] = normalized_prompt_chars
    metadata["chars_output"] = normalized_completion_chars
    metadata["chars_total"] = normalized_prompt_chars + normalized_completion_chars
    metadata["llm_call_count"] = max(0, int(request.llm_call_count))
    metadata["tokens_prompt"] = max(0, int(request.tokens_prompt))
    metadata["tokens_completion"] = max(0, int(request.tokens_completion))
    metadata["tokens_total"] = max(0, int(request.tokens_total))
    metadata["tokens_input"] = metadata["tokens_prompt"]
    metadata["tokens_output"] = metadata["tokens_completion"]
    metadata["retry_prompt_mode"] = retry.retry_prompt_mode
    metadata["retry_context_mode"] = retry.retry_context_mode
    metadata["retry_history_chars"] = int(retry.retry_history_chars)
    metadata["retry_history_truncated"] = bool(retry.retry_history_truncated)
    metadata["retry_history_turns_included"] = int(retry.retry_history_turns_included)
    metadata["retry_history_budget_chars"] = int(retry.retry_history_budget_chars)
    metadata["retry_history_budget_applied"] = bool(retry.retry_history_budget_applied)
    metadata["retry_delta_prompt"] = bool(retry.retry_delta_prompt)
    metadata["retry_delta_chars"] = int(retry.retry_delta_chars)
    metadata["adaptive_mode"] = str(adaptive.adaptive_mode or "none")
    metadata["adaptive_policy"] = str(adaptive.adaptive_policy or "disabled")
    metadata["adaptive_signature_repeat"] = bool(adaptive.adaptive_signature_repeat)
    metadata["adaptive_stagnation_streak"] = int(adaptive.adaptive_stagnation_streak)
    metadata["adaptive_helper_invoked"] = bool(adaptive.adaptive_helper_invoked)
    metadata["adaptive_helper_action_default"] = str(
        adaptive.adaptive_helper_action_default or ACTION_INVOKE_HELPER
    )
    metadata["adaptive_helper_action_selected"] = str(
        adaptive.adaptive_helper_action_selected or ACTION_INVOKE_HELPER
    )
    metadata["adaptive_helper_action_executed"] = str(
        adaptive.adaptive_helper_action_executed or ACTION_SKIP_HELPER
    )
    metadata["adaptive_helper_action_fallback_reason"] = str(
        adaptive.adaptive_helper_action_fallback_reason or ""
    )
    metadata["adaptive_helper_available_actions"] = [
        str(action) for action in adaptive.adaptive_helper_available_actions
    ]
    metadata["adaptive_helper_unavailable_action_reasons"] = {
        str(key): str(value)
        for key, value in adaptive.adaptive_helper_unavailable_action_reasons.items()
    }
    metadata["adaptive_helper_materialized_context"] = bool(
        adaptive.adaptive_helper_materialized_context
    )
    metadata["adaptive_helper_outcome"] = str(
        adaptive.adaptive_helper_outcome or ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED
    )
    metadata["adaptive_helper_failure_kind"] = str(adaptive.adaptive_helper_failure_kind or "")
    metadata["adaptive_helper_retry_count"] = int(adaptive.adaptive_helper_retry_count)
    metadata["adaptive_helper_transport_retry_count"] = int(
        adaptive.adaptive_helper_transport_retry_count
    )
    metadata["adaptive_helper_decision_reason"] = str(
        adaptive.adaptive_helper_decision_reason or ""
    )
    metadata["adaptive_stagnation_override_applied"] = bool(
        adaptive.adaptive_stagnation_override_applied
    )
    metadata["adaptive_stagnation_override_reason"] = str(
        adaptive.adaptive_stagnation_override_reason or ""
    )
    metadata["adaptive_helper_need_more_context"] = adaptive.adaptive_helper_need_more_context
    metadata["adaptive_helper_calls_used_before_attempt"] = int(
        adaptive.adaptive_helper_calls_used_before_attempt
    )
    metadata["adaptive_helper_calls_used_after_attempt"] = int(
        adaptive.adaptive_helper_calls_used_after_attempt
    )
    metadata["adaptive_helper_budget_max_calls"] = int(adaptive.adaptive_helper_budget_max_calls)
    metadata["adaptive_helper_budget_unlimited"] = bool(adaptive.adaptive_helper_budget_unlimited)
    metadata["adaptive_helper_budget_exhausted"] = bool(adaptive.adaptive_helper_budget_exhausted)
    metadata["adaptive_helper_saturation_guarded"] = bool(
        adaptive.adaptive_helper_saturation_guarded
    )
    metadata["adaptive_helper_budget_profile"] = str(
        adaptive.adaptive_helper_budget_profile or "default"
    )
    metadata["adaptive_budget_topup_granted"] = bool(adaptive.adaptive_budget_topup_granted)
    metadata["adaptive_budget_topup_reason"] = str(adaptive.adaptive_budget_topup_reason or "")
    metadata["adaptive_context_requested"] = bool(adaptive.adaptive_context_requested)
    metadata["adaptive_snapshot_included"] = bool(adaptive.adaptive_snapshot_included)
    metadata["adaptive_snapshot_lite_included"] = bool(adaptive.adaptive_snapshot_lite_included)
    metadata["adaptive_snapshot_attempt1_denied"] = bool(adaptive.adaptive_snapshot_attempt1_denied)
    metadata["adaptive_snapshot_attempt1_denied_reason"] = str(
        adaptive.adaptive_snapshot_attempt1_denied_reason or ""
    )
    metadata["adaptive_progress_made"] = bool(adaptive.adaptive_progress_made)
    metadata["adaptive_progress_reason"] = str(adaptive.adaptive_progress_reason or "")
    metadata["adaptive_no_progress_streak"] = int(adaptive.adaptive_no_progress_streak)
    metadata["adaptive_progress_stop_candidate"] = bool(adaptive.adaptive_progress_stop_candidate)
    metadata["adaptive_progress_stop_triggered"] = bool(adaptive.adaptive_progress_stop_triggered)
    metadata["adaptive_progress_stop_reason"] = str(adaptive.adaptive_progress_stop_reason or "")
    metadata["evidence_ids_new_count"] = int(request.evidence_ids_new_count)
    metadata["evidence_ids_reused_count"] = int(request.evidence_ids_reused_count)
    metadata["evidence_ids_sent_count"] = int(request.evidence_ids_sent_count)
    metadata["stage_a_index_included"] = bool(stage_a.stage_a_index_included)
    metadata["stage_a_index_reason"] = str(stage_a.stage_a_index_reason or "")
    metadata["openai_preflight_prompt_cap_chars"] = openai.openai_preflight_prompt_cap_chars
    metadata["openai_preflight_trimmed"] = bool(openai.openai_preflight_trimmed)
    metadata["openai_context_window_source"] = str(openai.openai_context_window_source or "")
    metadata["openai_context_window_probe_provider"] = str(
        openai.openai_context_window_probe_provider or ""
    )
    metadata["openai_context_window_probe_error"] = str(
        openai.openai_context_window_probe_error or ""
    )
    metadata["openai_response_mode_autoswitch_policy"] = str(
        openai.openai_response_mode_autoswitch_policy or "off"
    )
    metadata["openai_response_mode_autoswitched"] = bool(openai.openai_response_mode_autoswitched)
    metadata["openai_response_mode_autoswitch_from"] = str(
        openai.openai_response_mode_autoswitch_from or ""
    )
    metadata["openai_response_mode_autoswitch_to"] = str(
        openai.openai_response_mode_autoswitch_to or ""
    )
    metadata["openai_response_mode_autoswitch_cycle"] = int(
        openai.openai_response_mode_autoswitch_cycle
    )
    metadata["openai_response_mode_autoswitch_reason"] = str(
        openai.openai_response_mode_autoswitch_reason or ""
    )
    metadata["prompt_budget_used_chars"] = prompt_budget.prompt_budget_used_chars
    metadata["prompt_budget_overflow_chars"] = int(prompt_budget.prompt_budget_overflow_chars)
    metadata["prompt_budget_dropped_sections"] = list(prompt_budget.prompt_budget_dropped_sections)
    metadata["context_compaction_mode"] = str(context_compaction.context_compaction_mode or "off")
    metadata["context_compaction_scope"] = list(context_compaction.context_compaction_scope)
    metadata["context_compaction_supported"] = bool(context_compaction.context_compaction_supported)
    metadata["context_compaction_requested"] = bool(context_compaction.context_compaction_requested)
    metadata["context_compaction_request_reason"] = str(
        context_compaction.context_compaction_request_reason or ""
    )
    metadata["context_compaction_trigger"] = str(
        context_compaction.context_compaction_trigger or ""
    )
    metadata["context_compaction_auto_overflow_triggered"] = bool(
        context_compaction.context_compaction_auto_overflow_triggered
    )
    metadata["context_compaction_invoked"] = bool(context_compaction.context_compaction_invoked)
    metadata["context_compaction_success"] = bool(context_compaction.context_compaction_success)
    metadata["context_compaction_error"] = str(context_compaction.context_compaction_error or "")
    metadata["context_compaction_calls_used"] = int(
        context_compaction.context_compaction_calls_used
    )
    metadata["context_compaction_max_calls"] = int(context_compaction.context_compaction_max_calls)
    metadata["context_compaction_source_sections"] = list(
        context_compaction.context_compaction_source_sections
    )
    metadata["context_compaction_source_chars"] = dict(
        context_compaction.context_compaction_source_chars
    )
    metadata["context_compaction_replaced_sections"] = list(
        context_compaction.context_compaction_replaced_sections
    )
    metadata["context_compaction_input_chars"] = int(
        context_compaction.context_compaction_input_chars
    )
    metadata["context_compaction_target_chars"] = int(
        context_compaction.context_compaction_target_chars
    )
    metadata["context_compaction_output_chars"] = int(
        context_compaction.context_compaction_output_chars
    )
    metadata["context_compaction_prompt_chars"] = int(
        context_compaction.context_compaction_prompt_chars
    )
    metadata["adaptive_informativeness_pregate_fired"] = bool(
        adaptive.adaptive_informativeness_pregate_fired
    )
    metadata["adaptive_informativeness_pregate_score"] = float(
        adaptive.adaptive_informativeness_pregate_score
    )
    metadata["adaptive_informativeness_pregate_exception_type"] = str(
        adaptive.adaptive_informativeness_pregate_exception_type or ""
    )
    metadata["adaptive_informativeness_pregate_reason"] = str(
        adaptive.adaptive_informativeness_pregate_reason or ""
    )
    metadata["adaptive_informativeness_pregate_features"] = dict(
        adaptive.adaptive_informativeness_pregate_features
    )
    metadata["syntax_preflight_enabled"] = bool(syntax_preflight.syntax_preflight_enabled)
    metadata["syntax_preflight_cycles_used"] = int(syntax_preflight.syntax_preflight_cycles_used)
    metadata["syntax_preflight_failures"] = int(syntax_preflight.syntax_preflight_failures)
    metadata["syntax_preflight_last_error_type"] = str(
        syntax_preflight.syntax_preflight_last_error_type or ""
    )
    metadata["syntax_preflight_last_error_msg"] = str(
        syntax_preflight.syntax_preflight_last_error_msg or ""
    )

    if request.timing_breakdown is not None:
        base = dict(metadata["timing_breakdown"])
        for key, value in request.timing_breakdown.items():
            if key in base:
                base[key] = max(0.0, float(value))
        metadata["timing_breakdown"] = base
    if request.token_breakdown is not None:
        base = dict(metadata["token_breakdown"])
        for key, value in request.token_breakdown.items():
            if key in base:
                base[key] = max(0, int(value))
        metadata["token_breakdown"] = base

    if request.record_evidence != "none":
        metadata["evidence_xxh3_128"] = _xxh3_128_jsonlike(request.evidence)
    if request.record_evidence == "full":
        evidence_path = request.attempt_dir / "evidence.json"
        evidence_path.write_text(
            json.dumps(request.evidence, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metadata["evidence_path"] = str(evidence_path)

    if request.record_prompt != "none":
        metadata["prompt_xxh3_128"] = _xxh3_128_text(request.prompt)
    if request.record_prompt == "full":
        prompt_path = request.attempt_dir / "prompt.txt"
        prompt_path.write_text(request.prompt, encoding="utf-8")
        metadata["prompt_path"] = str(prompt_path)

    if stage_a.stage_a_prompt:
        metadata["stage_a_prompt_xxh3_128"] = _xxh3_128_text(stage_a.stage_a_prompt)
    stage_a_response_text = ""
    if stage_a.stage_a_raw_response:
        stage_a_response_text = stage_a.stage_a_raw_response
    elif stage_a.stage_a_request_payload is not None:
        stage_a_response_text = json.dumps(
            stage_a.stage_a_request_payload,
            ensure_ascii=False,
            default=str,
        )
    if stage_a_response_text:
        metadata["stage_a_response_xxh3_128"] = _xxh3_128_text(stage_a_response_text)

    return metadata


def _materialize_attempt_metadata(**legacy_kwargs: Any) -> dict[str, Any]:
    """Preserve the historical, warning-free flat keyword interface."""
    request = _build_attempt_metadata_request_from_legacy_values(legacy_kwargs)
    return _materialize_attempt_metadata_request(request=request)


def _build_model_metadata(*, patcher: Any, patch_meta: dict[str, Any] | None) -> dict[str, Any]:
    patcher_name = str(getattr(patcher, "name", "unknown"))
    metadata: dict[str, Any] = {"backend": patcher_name, "name": patcher_name}
    patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
    if patcher_name in BASELINE_CONDITION_BY_NAME:
        baseline_name = str(patch_meta_dict.get("baseline") or patcher_name)
        model = str(patch_meta_dict.get("model") or "")
        base_url = str(patch_meta_dict.get("base_url") or "")
        if baseline_name:
            metadata["baseline_name"] = baseline_name
        if model:
            metadata["model"] = model
        if base_url:
            metadata["base_url"] = base_url
        return metadata
    if patcher_name != "openai":
        return metadata

    requested_seed = patch_meta_dict.get("api_compat_seed_requested")
    if requested_seed is None:
        requested_seed = getattr(patcher, "seed", None)
    acknowledged = patch_meta_dict.get("api_compat_seed_acknowledged")
    metadata.update(
        {
            "backend": "openai",
            "base_url": str(
                patch_meta_dict.get("base_url")
                or _normalize_openai_base_url(str(getattr(patcher, "base_url", "")))
            ),
            "model": str(patch_meta_dict.get("model") or getattr(patcher, "model", "")),
            "temperature": patch_meta_dict.get(
                "temperature", getattr(patcher, "temperature", None)
            ),
            "max_tokens": patch_meta_dict.get("max_tokens", getattr(patcher, "max_tokens", None)),
            "response_mode": str(
                patch_meta_dict.get("response_mode") or getattr(patcher, "response_mode", "")
            ),
            "api_compat_seed_requested": requested_seed,
            "api_compat_seed_acknowledged": bool(acknowledged)
            if acknowledged is not None
            else False,
            "api_compat_seed_response": patch_meta_dict.get("api_compat_seed_response"),
        }
    )
    return metadata
