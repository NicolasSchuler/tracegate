"""Bootstrap and orchestration helpers for ``evals.run_eval``.

This module owns run-level preparation, metadata/bootstrap wiring, and unit
scheduling while ``evals.run_eval`` remains the compatibility facade.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import eval_main_support as legacy
from evals.decision_model.config import decision_model_settings_from_args
from evals.decision_model.runtime import build_decision_model_binding
from evals.eval_config import _parse_adaptive_online_action_score_biases
from evals.eval_main_resume import prepare_main_resume
from evals.eval_main_setup import MainFlowCliError, prepare_main_setup
from evals.eval_main_support import (
    AdaptiveBudgetPolicyController,
    AdaptiveOnlinePolicyController,
    BudgetPolicyConfig,
    CaseSpec,
    Condition,
    EvalUnitInputs,
    OnlinePolicyConfig,
    RunProgressSummary,
    UnitExecutionOutcome,
    _ActiveRunSettings,
    _baseline_extra_from_args,
    _baseline_patch_inventory,
    _build_run_metadata,
    _completion_chars_from_response,
    _EvalContainerPool,
    _execute_eval_unit,
    _filter_attempt_records,
    _format_duration,
    _format_progress_token_usage,
    _is_transport_runtime_error,
    _nonnegative_int,
    _OpenAITransportRecoveryGate,
    _record_is_transport_request_failure,
    _record_token_usage_state,
    _resolve_openai_context_window_for_run,
    _single_line_error,
    _SynchronizedJsonlWriter,
    _usage_token_counts_from_patch_meta,
    build_progress_palette,
    is_baseline_condition,
    set_active_execution_config,
)
from evals.eval_main_support import (
    RunConditionInputs as _RunConditionInputs,
)
from evals.native_stage_b_flow import openai_usable_prompt_tokens
from evals.patchers.baseline_patcher import BaselinePatcher, baseline_case_support
from evals.progress_display import DisplayRunSummary, build_startup_sections, create_run_display
from evals.progress_output import _is_infra_interrupted_records, classify_failure_reason
from evals.run_condition_builder import build_run_condition_inputs
from evals.seed_derivation import derive_seed

RunConditionInputs = _RunConditionInputs


_BUDGET_POLICY_CONDITIONS = (Condition.LAYERED_ADAPTIVE_GATED, Condition.LAYERED_ADAPTIVE_LLM)


def _per_condition_budget_configs(
    config: BudgetPolicyConfig, *, conditions: list[Condition], seed: int | None
) -> dict[str, BudgetPolicyConfig]:
    """Budget-policy configs keyed by condition value (POL-05, POL-23).

    With more than one adaptive condition each gets its own state and event files
    (``<stem>.<condition><suffix>``); every controller gets a seed derived from the run
    seed and its condition. Runs without an adaptive condition keep one shared config.
    """
    keys = [condition.value for condition in conditions if condition in _BUDGET_POLICY_CONDITIONS]
    separate_files = len(keys) > 1
    out: dict[str, BudgetPolicyConfig] = {}
    for key in keys or [""]:
        updates: dict[str, Any] = {
            "random_seed": derive_seed(int(seed), "budget", key) if seed is not None else None
        }
        if separate_files:
            for field_name in ("state_path", "events_path"):
                path = getattr(config, field_name)
                if path is not None:
                    updates[field_name] = path.with_name(f"{path.stem}.{key}{path.suffix}")
        out[key] = replace(config, **updates)
    return out


def run_main_flow(
    *, parser: argparse.ArgumentParser, args: argparse.Namespace, argv_tokens: list[str]
) -> int:
    try:
        setup = prepare_main_setup(
            parser=parser,
            args=args,
            argv_tokens=argv_tokens,
        )
        resume = prepare_main_resume(
            setup=setup,
            args=args,
            argv_tokens=argv_tokens,
        )
    except MainFlowCliError as exc:
        print(str(exc))
        return exc.exit_code

    config_state = setup.config_state
    cases = setup.cases
    execution_config = setup.execution_config
    conditions = setup.conditions
    _has_adaptive_condition = setup.has_adaptive_condition
    startup_warnings = setup.startup_warnings
    run_id = setup.run_id
    artifacts_root = setup.artifacts_root
    results_path = setup.results_path
    execution_units = resume.execution_units
    total_units = resume.total_units
    run_settings_fingerprint = resume.run_settings_fingerprint
    resolved_config_lock_path = resume.resolved_config_lock_path
    run_results_lock = resume.run_results_lock
    existing_run_metadata = resume.existing_run_metadata
    resumed_completed_units = resume.resumed_completed_units
    skipped_units = resume.skipped_units
    remaining_units = resume.remaining_units

    results_writer: _SynchronizedJsonlWriter | None = None

    def _close_run_results_io() -> None:
        nonlocal run_results_lock, results_writer
        if results_writer is not None:
            results_writer.close()
            results_writer = None
        if run_results_lock is not None:
            run_results_lock.release()
            run_results_lock = None

    display = create_run_display(
        total_units=total_units,
        verbosity=str(args.progress_verbosity),
    )
    display.render_startup(
        build_startup_sections(
            args=args,
            cases=cases,
            conditions=conditions,
            has_adaptive_condition=_has_adaptive_condition,
            run_id=run_id,
            results_path=results_path,
            config_paths=config_state.config_paths,
            resolved_config_lock_path=resolved_config_lock_path,
            total_units=total_units,
            skipped_units=skipped_units,
            remaining_units=remaining_units,
            execution_config=execution_config,
        )
    )
    for warning in startup_warnings:
        display.add_event(message=warning, style="warning")
    if remaining_units <= 0:
        _close_run_results_io()
        display.render_final_lines(
            ["Nothing to do: all planned case/condition units are already complete."]
        )
        display.close()
        return 0

    results_writer = _SynchronizedJsonlWriter(results_path)
    transport_recovery_gate = _OpenAITransportRecoveryGate(
        enabled=bool(getattr(args, "openai_transport_recovery", True)),
        max_wait_sec=float(getattr(args, "openai_transport_recovery_max_wait_sec", 60.0)),
        poll_sec=float(getattr(args, "openai_transport_recovery_poll_sec", 2.0)),
    )
    run_prompt_chars_total = 0
    run_completion_chars_total = 0
    run_tokens_prompt_total = 0
    run_tokens_completion_total = 0
    run_tokens_total = 0
    run_token_usage_available_records = 0
    run_token_usage_unavailable_records = 0
    jobs = int(args.jobs)
    try:
        patcher = legacy.make_patcher(args)
    except BaseException:
        _close_run_results_io()
        display.close()
        raise
    patcher_name = str(getattr(patcher, "name", str(args.patcher)))
    if patcher_name != str(args.patcher):
        display.add_event(message=f"Patcher resolved: {patcher_name}", style="info")
    try:
        (
            openai_context_window_tokens_effective,
            openai_context_window_info,
        ) = _resolve_openai_context_window_for_run(
            args=args,
            patcher=patcher,
        )
        # LLM-06: fail before any case runs when max_tokens leaves no usable prompt.
        openai_usable_prompt_tokens(
            context_window_tokens=openai_context_window_tokens_effective,
            max_tokens=int(getattr(patcher, "max_tokens", 0) or 0),
        )
    except BaseException:
        _close_run_results_io()
        display.close()
        raise
    openai_context_window_source = str(
        openai_context_window_info.get("openai_context_window_source") or "disabled"
    )
    openai_context_window_probe_provider = str(
        openai_context_window_info.get("openai_context_window_probe_provider") or ""
    )
    openai_context_window_probe_error = str(
        openai_context_window_info.get("openai_context_window_probe_error") or ""
    )
    if patcher_name == "openai":
        display.add_event(
            message=(
                "OpenAI context window guard: "
                f"effective={openai_context_window_tokens_effective} "
                f"source={openai_context_window_source}"
            ),
            style="info",
        )
        if openai_context_window_probe_provider:
            display.add_event(
                message=f"OpenAI context probe provider: {openai_context_window_probe_provider}",
                style="info",
            )
        if openai_context_window_probe_error:
            display.add_event(
                message=f"OpenAI context probe warning: {openai_context_window_probe_error}",
                style="warning",
            )
        display.add_event(
            message=f"OpenAI response autoswitch: {args.openai_response_mode_autoswitch}",
            style="info",
        )

    existing_fingerprint = (
        existing_run_metadata.get("run_settings_fingerprint")
        if isinstance(existing_run_metadata, dict)
        else None
    )
    if isinstance(existing_run_metadata, dict) and existing_fingerprint in {
        None,
        run_settings_fingerprint,
    }:
        run_metadata = dict(existing_run_metadata)
        run_metadata.setdefault("run_id", run_id)
        run_metadata.setdefault("run_settings_fingerprint", run_settings_fingerprint)
    else:
        started_at_utc = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
        run_metadata = _build_run_metadata(
            args=args,
            patcher_name=patcher_name,
            run_id=run_id,
            started_at_utc=started_at_utc,
            run_settings_fingerprint=run_settings_fingerprint,
        )
        if existing_fingerprint is not None:
            # ORC-04: resumed with --allow-settings-drift; keep the link to the old settings.
            run_metadata["settings_drift_from_fingerprint"] = existing_fingerprint
    baseline_adapter = getattr(patcher, "adapter", None)
    if isinstance(patcher, BaselinePatcher) and baseline_adapter is not None:
        run_metadata["baseline_case_support"] = baseline_case_support(
            baseline_adapter, [(case.case_id, case.path) for case in setup.cases]
        )
    if config_state.config_paths:
        run_metadata.setdefault("config_files", list(config_state.config_paths))
    if resolved_config_lock_path is not None:
        run_metadata.setdefault("resolved_config_lock_path", str(resolved_config_lock_path))
    patcher_config = run_metadata.get("patcher_config")
    if isinstance(patcher_config, dict):
        native_baseline_mode = str(getattr(args, "native_baseline_mode", "") or "").strip()
        if native_baseline_mode:
            patcher_config["native_baseline_mode"] = native_baseline_mode
        if len(conditions) == 1:
            patcher_config["resolved_condition"] = conditions[0].value
        elif conditions:
            patcher_config["resolved_conditions"] = [condition.value for condition in conditions]
        patcher_config["openai_auto_context_window"] = bool(
            getattr(args, "openai_auto_context_window", False)
        )
        patcher_config["openai_response_mode_autoswitch"] = str(
            getattr(args, "openai_response_mode_autoswitch", "off")
        )
        patcher_config["openai_context_window_tokens_effective"] = int(
            openai_context_window_tokens_effective
        )
        patcher_config["openai_context_window_source"] = openai_context_window_source
        patcher_config["openai_context_window_probe_provider"] = (
            openai_context_window_probe_provider
        )
        patcher_config["openai_context_window_probe_error"] = openai_context_window_probe_error
        if str(getattr(args, "patcher", "")) == "baseline":
            baseline_name = str(getattr(args, "baseline_name", "") or "").strip().lower()
            patcher_config["baseline_name"] = baseline_name
            patcher_config["baseline_condition"] = (
                conditions[0].value
                if len(conditions) == 1 and is_baseline_condition(conditions[0])
                else ""
            )
            patcher_config["baseline_repo_dir"] = str(
                Path(str(getattr(args, "baseline_repo_dir", "") or "")).expanduser()
            )
            baseline_python = str(getattr(args, "baseline_python", "") or "").strip()
            if not baseline_python:
                candidate = (
                    legacy._REPO_ROOT / "baselines" / "envs" / baseline_name / "bin" / "python"
                )
                if candidate.is_file():
                    baseline_python = str(candidate)
            if baseline_python:
                patcher_config["baseline_python"] = baseline_python
            repo_commit = str(
                getattr(patcher, "repo_commit", "")
                or getattr(getattr(patcher, "adapter", None), "repo_commit", "")
                or ""
            ).strip()
            if repo_commit:
                patcher_config["baseline_repo_commit"] = repo_commit
            compatibility_patches = _baseline_patch_inventory(baseline_name)
            if compatibility_patches:
                patcher_config["baseline_compatibility_patches"] = compatibility_patches
            baseline_extra = _baseline_extra_from_args(args=args, baseline_name=baseline_name)
            if baseline_extra:
                patcher_config["baseline_extra"] = baseline_extra
            patcher_config["baseline_api_timeout_sec"] = float(
                getattr(args, "openai_timeout_sec", 0.0) or 0.0
            )
            baseline_process_timeout = getattr(args, "baseline_process_timeout_sec", None)
            if (
                isinstance(baseline_process_timeout, (int, float))
                and float(baseline_process_timeout) > 0.0
            ):
                patcher_config["baseline_process_timeout_sec"] = float(baseline_process_timeout)

    try:
        online_init_state_path = str(getattr(args, "adaptive_online_init_state_path", "") or "")
        online_action_score_biases = _parse_adaptive_online_action_score_biases(
            getattr(args, "adaptive_online_action_score_biases", "")
        )
        online_policy_config = OnlinePolicyConfig.normalize(
            mode=str(args.adaptive_online_policy_mode),
            algorithm=str(args.adaptive_online_policy_algo),
            alpha=float(args.adaptive_online_alpha),
            epsilon=float(args.adaptive_online_epsilon),
            l2=float(args.adaptive_online_l2),
            warmup_decisions=int(args.adaptive_online_warmup_decisions),
            save_every=int(args.adaptive_online_save_every),
            decay=float(args.adaptive_online_decay),
            feature_scaling_profile=str(args.adaptive_online_feature_scaling_profile),
            init_state_path=(
                Path(online_init_state_path).expanduser().resolve()
                if online_init_state_path.strip()
                else None
            ),
            state_path=Path(str(args.adaptive_online_state_path)).expanduser().resolve(),
            events_path=Path(str(args.adaptive_online_events_path)).expanduser().resolve(),
            # POL-23: each controller gets its own stream, derived from the run seed.
            random_seed=(derive_seed(int(args.seed), "helper") if args.seed is not None else None),
            action_score_biases=online_action_score_biases,
        )
    except BaseException:
        _close_run_results_io()
        display.close()
        raise
    if online_policy_config.mode == "off":
        online_policy_controller: AdaptiveOnlinePolicyController | None = None
    else:
        try:
            decision_model_binding = build_decision_model_binding(
                decision_model_settings_from_args(args),
                artifacts_root=artifacts_root,
            )
            # Pass the binding only when a role is active (keeps the one-argument form intact).
            online_policy_controller = (
                AdaptiveOnlinePolicyController(
                    online_policy_config,
                    decision_model=decision_model_binding,
                )
                if decision_model_binding is not None
                else AdaptiveOnlinePolicyController(online_policy_config)
            )
        except BaseException:
            _close_run_results_io()
            display.close()
            raise
        if online_policy_controller.load_error:
            display.add_event(
                message=(
                    "Online adaptive policy state load warning: "
                    f"{online_policy_controller.load_error}; starting with a fresh state."
                ),
                style="warning",
            )
    if isinstance(patcher_config, dict):
        patcher_config["adaptive_online_warm_start_source"] = online_init_state_path
        patcher_config["adaptive_online_warm_start_result"] = (
            "load_error"
            if online_policy_controller is not None and online_policy_controller.load_error
            else ("loaded" if online_init_state_path else "fresh")
        )

    legacy.set_active_policy_controllers(online=online_policy_controller, budget=None)

    try:
        budget_init_state_path = str(getattr(args, "adaptive_budget_init_state_path", "") or "")
        budget_policy_config = BudgetPolicyConfig.normalize(
            mode=str(args.adaptive_budget_policy_mode),
            algorithm=str(args.adaptive_budget_policy_algo),
            architecture=str(args.adaptive_budget_policy_architecture),
            alpha=float(args.adaptive_budget_online_alpha),
            epsilon=float(args.adaptive_budget_online_epsilon),
            l2=float(args.adaptive_budget_online_l2),
            warmup_decisions=int(args.adaptive_budget_online_warmup_decisions),
            save_every=int(args.adaptive_budget_online_save_every),
            decay=float(args.adaptive_budget_online_decay),
            feature_scaling_profile=str(args.adaptive_budget_feature_scaling_profile),
            init_state_path=(
                Path(budget_init_state_path).expanduser().resolve()
                if budget_init_state_path.strip()
                else None
            ),
            state_path=Path(str(args.adaptive_budget_online_state_path)).expanduser().resolve(),
            events_path=Path(str(args.adaptive_budget_online_events_path)).expanduser().resolve(),
            bwk_run_token_budget=int(getattr(args, "adaptive_budget_bwk_run_token_budget", 0)),
            random_seed=None,
        )
    except BaseException:
        _close_run_results_io()
        display.close()
        raise
    # POL-05: one budget controller per adaptive condition, so the conditions being
    # compared don't train (and draw exploration from) one shared model.
    budget_policy_controllers: dict[str, AdaptiveBudgetPolicyController] = {}
    if budget_policy_config.mode != "off":
        for condition_key, condition_config in _per_condition_budget_configs(
            budget_policy_config, conditions=conditions, seed=args.seed
        ).items():
            try:
                controller = AdaptiveBudgetPolicyController(condition_config)
            except BaseException:
                for created in budget_policy_controllers.values():
                    created.close()
                _close_run_results_io()
                display.close()
                raise
            if controller.load_error:
                display.add_event(
                    message=(
                        "Online adaptive budget policy state load warning: "
                        f"{controller.load_error}; starting with a fresh state."
                    ),
                    style="warning",
                )
            budget_policy_controllers[condition_key] = controller
    budget_policy_controller: AdaptiveBudgetPolicyController | None = next(
        iter(budget_policy_controllers.values()), None
    )
    if isinstance(patcher_config, dict) and budget_policy_controllers:
        patcher_config["adaptive_budget_controllers"] = {
            (key or "shared"): {
                "state_path": str(controller.config.state_path),
                "events_path": str(controller.config.events_path),
            }
            for key, controller in budget_policy_controllers.items()
        }
    if isinstance(patcher_config, dict):
        patcher_config["adaptive_budget_warm_start_source"] = budget_init_state_path
        patcher_config["adaptive_budget_warm_start_result"] = (
            "load_error"
            if budget_policy_controller is not None and budget_policy_controller.load_error
            else ("loaded" if budget_init_state_path else "fresh")
        )

    legacy.set_active_policy_controllers(
        online=online_policy_controller, budget=budget_policy_controller
    )

    previous_execution_config = set_active_execution_config(execution_config)
    previous_run_settings = legacy.set_active_run_settings(
        _ActiveRunSettings(
            deterministic=bool(args.deterministic),
            seed=args.seed if args.deterministic else None,
        )
    )
    run_started = time.monotonic()
    completed_units = 0
    executed_units = 0
    runtime_errors: list[tuple[str, str, str]] = []
    run_progress_summary = RunProgressSummary()
    progress_palette = build_progress_palette()
    transport_fail_fast_streak = max(0, int(args.transport_fail_fast_streak))
    transport_failure_streak = 0
    transport_fail_fast_triggered = False
    transport_fail_fast_reason = ""
    transport_fail_fast_skipped_units = 0
    transport_fail_fast_skip_reported = False

    def _sync_display_summary(*, elapsed_sec: float, eta_sec: float | None) -> None:
        display.update_summary(
            DisplayRunSummary(
                total_units=total_units,
                completed_units=completed_units,
                skipped_units=skipped_units,
                success_units=run_progress_summary.success_units,
                failed_units=run_progress_summary.failed_units,
                interrupted_units=run_progress_summary.interrupted_units,
                runtime_error_units=run_progress_summary.runtime_error_units,
                failure_reasons=run_progress_summary.failure_reasons.copy(),
                elapsed_sec=elapsed_sec,
                eta_sec=eta_sec,
                prompt_chars_total=run_prompt_chars_total,
                completion_chars_total=run_completion_chars_total,
                token_usage_text=_format_progress_token_usage(
                    prompt_tokens=run_tokens_prompt_total,
                    completion_tokens=run_tokens_completion_total,
                    total_tokens=run_tokens_total,
                    available_records=run_token_usage_available_records,
                    unavailable_records=run_token_usage_unavailable_records,
                ),
            )
        )

    def _estimate_remaining_eta_seconds(*, elapsed_sec: float) -> float | None:
        """Estimate wall-clock ETA using observed throughput.

        This is parallel-aware by design: ``executed_units / elapsed`` already
        reflects the effective throughput of the configured worker count, retry
        mix, and runtime overheads.
        """
        remaining_units = total_units - completed_units
        if remaining_units <= 0:
            return 0.0
        if executed_units <= 0 or elapsed_sec <= 0:
            return None
        units_per_sec = executed_units / elapsed_sec
        if units_per_sec <= 0:
            return None
        return remaining_units / units_per_sec

    def _extract_progress_metrics(
        case_records: list[dict[str, Any]],
    ) -> tuple[int, int, float, float, int, int, int, int, int, int, int]:
        """Return attempt/time/usage totals for a case.

        For test time: count the initial ``before_duration_sec`` once, then
        only ``after_duration_sec`` for each attempt (the next attempt's
        "before" is the previous attempt's "after", so this avoids
        double-counting).
        """
        scored_records = _filter_attempt_records(case_records)
        attempts_used = len(scored_records)
        max_attempts = 0
        if scored_records:
            meta = scored_records[0].get("attempt_metadata")
            if isinstance(meta, dict):
                max_attempts = int(meta.get("max_attempts", 0) or 0)

        total_llm_sec = 0.0
        total_test_sec = (
            float(scored_records[0].get("before_duration_sec", 0) or 0) if scored_records else 0.0
        )
        total_prompt_chars = 0
        total_completion_chars = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        token_usage_available_records = 0
        token_usage_unavailable_records = 0
        for r in scored_records:
            pm = r.get("patch_meta")
            attempt_meta = r.get("attempt_metadata")
            if isinstance(pm, dict):
                total_llm_sec += float(pm.get("propose_duration_sec", 0) or 0)
            total_test_sec += float(r.get("after_duration_sec", 0) or 0)
            prompt_chars = (
                _nonnegative_int(
                    attempt_meta.get("prompt_chars_total"),
                    default=_nonnegative_int(pm.get("prompt_chars_total"), default=0)
                    if isinstance(pm, dict)
                    else 0,
                )
                if isinstance(attempt_meta, dict)
                else _nonnegative_int(pm.get("prompt_chars_total"), default=0)
                if isinstance(pm, dict)
                else 0
            )
            completion_chars = (
                _nonnegative_int(
                    attempt_meta.get("completion_chars_total"),
                    default=_nonnegative_int(pm.get("completion_chars_total"), default=0)
                    if isinstance(pm, dict)
                    else _completion_chars_from_response(
                        patch_meta=pm if isinstance(pm, dict) else None
                    ),
                )
                if isinstance(attempt_meta, dict)
                else _nonnegative_int(pm.get("completion_chars_total"), default=0)
                if isinstance(pm, dict)
                else 0
            )
            if isinstance(attempt_meta, dict):
                prompt_tokens = _nonnegative_int(
                    attempt_meta.get("tokens_prompt"),
                    default=_usage_token_counts_from_patch_meta(pm)[0]
                    if isinstance(pm, dict)
                    else 0,
                )
                completion_tokens = _nonnegative_int(
                    attempt_meta.get("tokens_completion"),
                    default=_usage_token_counts_from_patch_meta(pm)[1]
                    if isinstance(pm, dict)
                    else 0,
                )
                tokens_total = _nonnegative_int(
                    attempt_meta.get("tokens_total"),
                    default=_usage_token_counts_from_patch_meta(pm)[2]
                    if isinstance(pm, dict)
                    else 0,
                )
            else:
                prompt_tokens, completion_tokens, tokens_total = (
                    _usage_token_counts_from_patch_meta(pm)
                )
            total_prompt_chars += prompt_chars
            total_completion_chars += completion_chars
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += tokens_total
            token_usage_state = _record_token_usage_state(
                attempt_meta=attempt_meta if isinstance(attempt_meta, dict) else None,
                patch_meta=pm if isinstance(pm, dict) else None,
                prompt_tokens=prompt_tokens,
                total_tokens=tokens_total,
            )
            if token_usage_state == "available":
                token_usage_available_records += 1
            elif token_usage_state == "unavailable":
                token_usage_unavailable_records += 1

        return (
            attempts_used,
            max_attempts,
            total_llm_sec,
            total_test_sec,
            total_prompt_chars,
            total_completion_chars,
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens,
            token_usage_available_records,
            token_usage_unavailable_records,
        )

    def _append_records_safely(case_records: list[dict[str, Any]]) -> None:
        if not case_records:
            return
        if results_writer is None:
            raise RuntimeError("results writer is not initialized")
        for record in case_records:
            results_writer.append_one(record)

    def _on_unit_progress(
        case_id: str,
        condition: str,
        event: str,
        payload: dict[str, Any] | None,
    ) -> None:
        display.update_unit_progress(
            case_id=case_id,
            condition=condition,
            event=event,
            payload=payload,
        )

    def _handle_outcome(outcome: UnitExecutionOutcome) -> bool:
        nonlocal completed_units
        nonlocal executed_units
        nonlocal transport_failure_streak
        nonlocal transport_fail_fast_triggered
        nonlocal transport_fail_fast_reason
        nonlocal run_prompt_chars_total
        nonlocal run_completion_chars_total
        nonlocal run_tokens_prompt_total
        nonlocal run_tokens_completion_total
        nonlocal run_tokens_total
        nonlocal run_token_usage_available_records
        nonlocal run_token_usage_unavailable_records
        unit_duration = max(0.0, float(outcome.unit_duration_sec))
        completed_units += 1
        executed_units += 1
        elapsed = time.monotonic() - run_started
        eta = _estimate_remaining_eta_seconds(elapsed_sec=elapsed)
        progress_pct = 100.0 * (completed_units / total_units) if total_units else 100.0
        case_id = outcome.case.case_id
        condition_value = outcome.condition.value
        case_records = outcome.case_records or []
        if not outcome.records_streamed:
            _append_records_safely(case_records)
        # ORC-17: usage a unit streamed before it failed still counts toward the run.
        (
            attempts_used,
            max_attempts,
            total_llm_sec,
            total_test_sec,
            total_prompt_chars,
            total_completion_chars,
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens,
            token_usage_available_records,
            token_usage_unavailable_records,
        ) = _extract_progress_metrics(case_records)
        run_prompt_chars_total += total_prompt_chars
        run_completion_chars_total += total_completion_chars
        run_tokens_prompt_total += total_prompt_tokens
        run_tokens_completion_total += total_completion_tokens
        run_tokens_total += total_tokens
        run_token_usage_available_records += token_usage_available_records
        run_token_usage_unavailable_records += token_usage_unavailable_records
        if outcome.runtime_error is not None:
            reason = _single_line_error(outcome.runtime_error)
            runtime_errors.append((case_id, condition_value, reason))
            run_progress_summary.record_runtime_error(reason)
            display.finish_unit(
                case_id=case_id,
                condition=condition_value,
                status="error",
                failure_reason=reason,
            )
            _sync_display_summary(elapsed_sec=elapsed, eta_sec=eta)
            display.add_event(
                message=(
                    f"[{completed_units}/{total_units} {progress_pct:.1f}%] "
                    f"ERROR {case_id}/{condition_value} reason={reason} "
                    f"unit={_format_duration(unit_duration)} "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(eta)}"
                ),
                style="error",
            )
            if transport_fail_fast_streak > 0:
                if _is_transport_runtime_error(reason):
                    transport_failure_streak += 1
                else:
                    transport_failure_streak = 0
                if (
                    transport_failure_streak >= transport_fail_fast_streak
                    and not transport_fail_fast_triggered
                ):
                    transport_fail_fast_triggered = True
                    transport_fail_fast_reason = (
                        f"{transport_failure_streak} consecutive transport/crash runtime failures"
                    )
                    display.add_event(
                        message=(
                            "Fail-fast triggered: "
                            f"{transport_fail_fast_reason} "
                            f"(threshold={transport_fail_fast_streak})."
                        ),
                        style="warning",
                    )
                    return True
            return False
        scored_case_records = _filter_attempt_records(case_records)
        terminal_case_records = scored_case_records or case_records
        run_progress_summary.record_case_records(terminal_case_records)
        unit_success = any(bool(rec.get("success")) for rec in scored_case_records)
        unit_interrupted = _is_infra_interrupted_records(terminal_case_records)
        status = "success" if unit_success else ("interrupted" if unit_interrupted else "failed")
        first_note = str(scored_case_records[0].get("note") or "") if scored_case_records else ""
        k_str = "k=0" if first_note == "no_patch_needed" else f"k={attempts_used}/{max_attempts}"
        token_usage_text = _format_progress_token_usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            available_records=token_usage_available_records,
            unavailable_records=token_usage_unavailable_records,
        )
        _, failure_reason_label = classify_failure_reason(
            case_records=scored_case_records or case_records
        )
        if unit_interrupted:
            failure_reason_label = "infra interruption"
        reason_fragment = (
            f" reason={failure_reason_label}" if status in {"failed", "interrupted"} else ""
        )
        display.finish_unit(
            case_id=case_id,
            condition=condition_value,
            status=status,
            failure_reason=failure_reason_label if status in {"failed", "interrupted"} else "",
        )
        _sync_display_summary(elapsed_sec=elapsed, eta_sec=eta)
        display.add_event(
            message=(
                f"[{completed_units}/{total_units} {progress_pct:.1f}%] "
                f"DONE {case_id}/{condition_value} status={status}{reason_fragment} "
                f"{k_str} llm={_format_duration(total_llm_sec)} "
                f"test={_format_duration(total_test_sec)} "
                f"chars=in:{total_prompt_chars}ch,out:{total_completion_chars}ch "
                f"{token_usage_text} "
                f"unit={_format_duration(unit_duration)} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(eta)}"
            ),
            style="success"
            if status == "success"
            else ("warning" if unit_interrupted else "error"),
        )
        # Warn when a failed case used fewer attempts than allowed (early stop).
        if status == "failed" and attempts_used < max_attempts and max_attempts > 0:
            terminal_rec = scored_case_records[-1] if scored_case_records else {}
            terminal_pm = terminal_rec.get("patch_meta") if isinstance(terminal_rec, dict) else None
            early_reason: str | None = None
            if isinstance(terminal_pm, dict):
                if terminal_pm.get("case_fail_fast_triggered"):
                    early_reason = str(terminal_pm.get("case_fail_fast_reason", "fail_fast"))
                elif terminal_pm.get("adaptive_progress_stop_triggered"):
                    early_reason = str(
                        terminal_pm.get(
                            "adaptive_progress_stop_note",
                            terminal_pm.get("adaptive_progress_stop_reason", "adaptive_stop"),
                        )
                    )
            skipped = max_attempts - attempts_used
            reason_str = f": {early_reason}" if early_reason else ""
            display.add_event(
                message=(
                    f"    ^ early stop{reason_str} "
                    f"(skipped {skipped} remaining attempt{'s' if skipped != 1 else ''})"
                ),
                style="warning",
            )
        if transport_fail_fast_streak <= 0:
            return False
        terminal_record = scored_case_records[-1] if scored_case_records else None
        terminal_transport_failure = bool(
            isinstance(terminal_record, dict)
            and _record_is_transport_request_failure(terminal_record)
        )
        if terminal_transport_failure:
            transport_failure_streak += 1
        else:
            transport_failure_streak = 0
        if transport_failure_streak < transport_fail_fast_streak:
            return False
        if not transport_fail_fast_triggered:
            transport_fail_fast_triggered = True
            transport_fail_fast_reason = (
                f"{transport_failure_streak} consecutive transport patcher request failures"
            )
            display.add_event(
                message=(
                    "Fail-fast triggered: "
                    f"{transport_fail_fast_reason} "
                    f"(threshold={transport_fail_fast_streak})."
                ),
                style="warning",
            )
        return True

    pending_units: list[tuple[CaseSpec, Condition]] = []
    for case, condition in execution_units:
        unit_key = (case.case_id, condition.value)
        if unit_key in resumed_completed_units:
            completed_units += 1
            progress_pct = 100.0 * (completed_units / total_units) if total_units else 100.0
            display.skip_unit(case_id=case.case_id, condition=condition.value)
            _sync_display_summary(elapsed_sec=0.0, eta_sec=None)
            display.add_event(
                message=(
                    f"[{completed_units}/{total_units} {progress_pct:.1f}%] "
                    f"SKIP (resumed) {case.case_id}/{condition.value}"
                ),
                style="skip",
            )
            continue
        pending_units.append((case, condition))

    def _stream_result_record(record: dict[str, Any]) -> None:
        if results_writer is None:
            raise RuntimeError("results writer is not initialized")
        results_writer.append_one(record)

    shared_run_condition_inputs = build_run_condition_inputs(
        args=args,
        run_id=run_id,
        run_metadata=run_metadata,
        patcher=None,
        online_policy_controller=online_policy_controller,
        budget_policy_controller=budget_policy_controller,
        openai_context_window_tokens_effective=openai_context_window_tokens_effective,
        openai_context_window_source=openai_context_window_source,
        openai_context_window_probe_provider=openai_context_window_probe_provider,
        openai_context_window_probe_error=openai_context_window_probe_error,
        transport_recovery_gate=transport_recovery_gate,
        record_sink=_stream_result_record,
    )
    unit_progress_verbosity = str(args.progress_verbosity)
    if display.suppress_inline_progress and unit_progress_verbosity != "trace":
        unit_progress_verbosity = "off"
    shared_unit_inputs = EvalUnitInputs(
        patcher_args=args,
        run_condition=shared_run_condition_inputs,
        keep_workdirs=bool(args.keep_workdirs),
        progress_verbosity=unit_progress_verbosity,
        progress_heartbeat_sec=float(args.progress_heartbeat_sec),
        unit_progress_callback=_on_unit_progress,
    )
    unit_inputs_by_condition: dict[str, EvalUnitInputs] = {
        key: replace(
            shared_unit_inputs,
            run_condition=replace(
                shared_run_condition_inputs, adaptive_budget_policy_controller=controller
            ),
        )
        for key, controller in budget_policy_controllers.items()
        if key
    }

    def _unit_inputs_for(condition: Condition) -> EvalUnitInputs:
        return unit_inputs_by_condition.get(condition.value, shared_unit_inputs)

    if (
        bool(args.deterministic)
        and jobs > 1
        and (online_policy_controller is not None or budget_policy_controllers)
    ):
        display.add_event(
            message=(
                "Deterministic mode with --jobs > 1: policy decisions use per-unit random "
                "streams, but learning order still depends on scheduling; use --jobs 1 for "
                "reproducible policy learning."
            ),
            style="warning",
        )
    _sync_display_summary(elapsed_sec=0.0, eta_sec=None)
    display.start_runtime()

    if (
        execution_config.executor == "testcontainers"
        and execution_config.container_pool
        and pending_units
    ):
        pool_size = max(1, jobs)
        display.add_event(
            message=f"Container pool: starting {pool_size} container(s)...",
            style="info",
        )
        try:
            legacy.set_active_container_pool(
                _EvalContainerPool(
                    size=pool_size,
                    execution=execution_config,
                    workdirs_host=artifacts_root / "workdirs",
                )
            )
            display.add_event(
                message=f"Container pool: {pool_size} container(s) ready",
                style="success",
            )
        except Exception as pool_exc:
            display.add_event(
                message=(
                    f"Container pool: failed to start ({type(pool_exc).__name__}: {pool_exc}), "
                    "falling back to per-call containers"
                ),
                style="warning",
            )

    try:
        if jobs == 1:
            for unit_index, (case, condition) in enumerate(pending_units):
                progress_pct = 100.0 * (completed_units / total_units) if total_units else 100.0
                display.start_unit(case_id=case.case_id, condition=condition.value)
                display.add_event(
                    message=(
                        f"[{completed_units + 1}/{total_units} {progress_pct:.1f}%] "
                        f"START {case.case_id}/{condition.value}"
                    ),
                    style="info",
                )
                outcome = _execute_eval_unit(
                    case=case,
                    condition=condition,
                    artifacts_root=artifacts_root,
                    inputs=replace(_unit_inputs_for(condition), patcher_override=patcher),
                )
                should_abort = _handle_outcome(outcome)
                if should_abort:
                    remaining_units = max(0, len(pending_units) - (unit_index + 1))
                    if remaining_units > 0:
                        transport_fail_fast_skipped_units += remaining_units
                        display.add_event(
                            message=f"Fail-fast: skipping {remaining_units} not-yet-started unit(s).",
                            style="warning",
                        )
                    break
        else:
            future_map: dict[Any, tuple[CaseSpec, Condition]] = {}
            pending_iter = iter(pending_units)
            submitted_units = 0

            def _submit_next(executor: ThreadPoolExecutor) -> bool:
                nonlocal submitted_units
                try:
                    case, condition = next(pending_iter)
                except StopIteration:
                    return False
                progress_pct = 100.0 * (completed_units / total_units) if total_units else 100.0
                ordinal = completed_units + len(future_map) + 1
                display.start_unit(case_id=case.case_id, condition=condition.value)
                display.add_event(
                    message=(
                        f"[{ordinal}/{total_units} {progress_pct:.1f}%] "
                        f"START {case.case_id}/{condition.value}"
                    ),
                    style="info",
                )
                future = executor.submit(
                    _execute_eval_unit,
                    case=case,
                    condition=condition,
                    artifacts_root=artifacts_root,
                    inputs=_unit_inputs_for(condition),
                )
                future_map[future] = (case, condition)
                submitted_units += 1
                return True

            with ThreadPoolExecutor(max_workers=jobs) as executor:
                while (
                    len(future_map) < jobs
                    and (not transport_fail_fast_triggered)
                    and _submit_next(executor)
                ):
                    pass
                while future_map:
                    future = next(as_completed(tuple(future_map.keys())))
                    case, condition = future_map[future]
                    del future_map[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = UnitExecutionOutcome(
                            case=case,
                            condition=condition,
                            case_records=None,
                            unit_duration_sec=0.0,
                            runtime_error=f"{type(exc).__name__}: {exc}",
                        )
                    should_abort = _handle_outcome(outcome)
                    if should_abort and (not transport_fail_fast_skip_reported):
                        cancelled = 0
                        for pending_future in list(future_map):
                            if pending_future.cancel():
                                del future_map[pending_future]
                                cancelled += 1
                        unscheduled = max(0, len(pending_units) - submitted_units)
                        skipped_now = cancelled + unscheduled
                        if skipped_now > 0:
                            transport_fail_fast_skipped_units += skipped_now
                            display.add_event(
                                message=(
                                    f"Fail-fast: skipping {skipped_now} pending/unscheduled unit(s)."
                                ),
                                style="warning",
                            )
                        transport_fail_fast_skip_reported = True
                    while (
                        len(future_map) < jobs
                        and (not transport_fail_fast_triggered)
                        and _submit_next(executor)
                    ):
                        pass
    finally:
        active_pool = legacy.set_active_container_pool(None)
        if active_pool is not None:
            active_pool.shutdown()
        if legacy._ACTIVE_ONLINE_POLICY_CONTROLLER is not None:
            try:
                legacy._ACTIVE_ONLINE_POLICY_CONTROLLER.close()
            except Exception as exc:
                display.add_event(
                    message=f"Online adaptive policy close warning: {type(exc).__name__}: {exc}",
                    style="warning",
                )
        for active_budget_controller in budget_policy_controllers.values():
            try:
                active_budget_controller.close()
            except Exception as exc:
                display.add_event(
                    message=(
                        f"Online adaptive budget policy close warning: {type(exc).__name__}: {exc}"
                    ),
                    style="warning",
                )
        legacy.set_active_policy_controllers(online=None, budget=None)
        _close_run_results_io()
        legacy.set_active_run_settings(previous_run_settings)
        set_active_execution_config(previous_execution_config)

    total_elapsed = time.monotonic() - run_started
    errors_suffix = f", errors={len(runtime_errors)}" if runtime_errors else ""
    run_token_usage_text = _format_progress_token_usage(
        prompt_tokens=run_tokens_prompt_total,
        completion_tokens=run_tokens_completion_total,
        total_tokens=run_tokens_total,
        available_records=run_token_usage_available_records,
        unavailable_records=run_token_usage_unavailable_records,
    )
    _sync_display_summary(elapsed_sec=total_elapsed, eta_sec=0.0)
    display.close()
    final_lines = [
        "Progress: finished "
        f"executed={executed_units}, skipped={skipped_units}, total={total_units}, "
        f"elapsed={_format_duration(total_elapsed)} "
        f"chars=in:{run_prompt_chars_total}ch,out:{run_completion_chars_total}ch "
        f"{run_token_usage_text}{errors_suffix}"
    ]
    final_lines.extend(
        run_progress_summary.render_lines(
            skipped_units=skipped_units,
            palette=progress_palette,
        )
    )
    if runtime_errors:
        final_lines.append(
            "Eval finished with "
            f"{len(runtime_errors)} runtime error(s). "
            "Rerun with the same --run-id to retry unfinished units."
        )
    if run_progress_summary.interrupted_units:
        final_lines.append(
            "Eval finished with "
            f"{run_progress_summary.interrupted_units} infra-interrupted unfinished unit(s). "
            "Rerun with the same --run-id to retry them."
        )
    if transport_fail_fast_triggered:
        final_lines.append(
            "Eval aborted early by transport fail-fast: "
            f"{transport_fail_fast_reason} "
            f"(threshold={transport_fail_fast_streak}, "
            f"skipped_units={transport_fail_fast_skipped_units}). "
            "Rerun with --transport-fail-fast-streak=0 to disable."
        )
    display.render_final_lines(final_lines)
    if transport_fail_fast_triggered:
        return 2
    if runtime_errors:
        return 2
    return 0
