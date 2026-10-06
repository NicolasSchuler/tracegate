"""Build the patcher selected by ``--patcher``.

Also resolves the OpenAI context window for a run."""

from __future__ import annotations

import argparse
from typing import Any

from evals.baseline_runtime import (
    _baseline_extra_from_args,
    _normalize_adaptive_controller_lambda_rlm_settings,
    _normalize_baseline_selection,
    _resolve_external_baseline_runtime_requirements,
)
from evals.eval_recording import _native_openai_compat_settings_from_args


class PatcherConfigError(ValueError):
    """The patcher can't be built from the given arguments (ORC-12)."""


def make_patcher(args: argparse.Namespace):
    """Instantiate the patcher backend selected by ``--patcher``.

    Returns one of ``OpenAICompatPatcher``, ``HeuristicSnapshotPatcher``,
    ``NullPatcher``, ``CommandPatcher``, or ``BaselinePatcher`` depending
    on *args.patcher*. Units build their patchers from worker threads, so this works
    on a private copy of *args* and raises :class:`PatcherConfigError` rather than
    ``SystemExit``, which escaped the units' ``except Exception`` (ORC-12).
    """
    from evals.patchers import CommandPatcher, HeuristicSnapshotPatcher, NullPatcher
    from evals.patchers.command_patcher import patcher_from_env

    args = argparse.Namespace(**vars(args))
    try:
        _normalize_baseline_selection(args=args, config_state=None)
        _normalize_adaptive_controller_lambda_rlm_settings(args=args, config_state=None)
    except ValueError as exc:
        raise PatcherConfigError(str(exc)) from exc

    if args.patcher == "baseline":
        from baselines.adapter_base import BaselineGenerationSettings
        from evals.patchers.baseline_patcher import build_baseline_patcher

        baseline_name = (getattr(args, "baseline_name", "") or "").strip()
        if not baseline_name:
            raise PatcherConfigError(
                "--baseline-name is required for --patcher baseline. "
                "Available: ldb, intervenor, tracecoder, lambda_rlm, rlm."
            )
        repo_dir = (getattr(args, "baseline_repo_dir", "") or "").strip()
        if not repo_dir:
            raise PatcherConfigError(
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
            raise PatcherConfigError(str(exc)) from exc
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
            # D4 (BAS-03/04/11): the native conditions' sampling, cap and retries.
            generation=BaselineGenerationSettings(
                temperature=float(getattr(args, "openai_temperature", 0.0)),
                seed=getattr(args, "seed", None),
                max_completion_tokens=int(getattr(args, "openai_max_tokens", 2048)),
                transport_max_retries=int(getattr(args, "openai_transport_max_retries", 0)),
                transport_retry_backoff_sec=float(
                    getattr(args, "openai_transport_retry_backoff_base_sec", 1.0)
                ),
            ),
            extra=baseline_extra,
            strict_extra_config=True,
        )

    if args.patcher == "openai":
        from evals.patchers import OpenAICompatPatcher

        model = (args.openai_model or "").strip()
        if not model:
            raise PatcherConfigError(
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
        raise PatcherConfigError(
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
        probe_result: Any = probe_context_window()
        probe_tokens_raw, probe_meta_raw = probe_result
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
