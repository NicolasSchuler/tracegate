"""Bootstrap and orchestration helpers for ``evals.run_eval``.

This module owns run-level preparation, metadata/bootstrap wiring, and unit
scheduling while ``evals.run_eval`` remains the compatibility facade.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import wait as futures_wait
from dataclasses import replace
from typing import Any

from evals.attempt_usage import (
    _format_progress_token_usage,
)
from evals.eval_attempt_flow import execute_eval_unit_impl as _execute_eval_unit
from evals.eval_config import (
    set_active_execution_config,
)
from evals.eval_containers import (
    _EvalContainerPool,
)
from evals.eval_main_metadata import (
    RunMetadataInputs,
    build_run_metadata_for_run,
)
from evals.eval_main_outcome import (
    RunUsageTotals,
    extract_unit_progress_metrics,
)
from evals.eval_main_policies import (
    PolicyControllerInputs,
    build_policy_controllers,
)
from evals.eval_main_resume import prepare_main_resume
from evals.eval_main_setup import MainFlowCliError, prepare_main_setup
from evals.eval_recording import (
    _SynchronizedJsonlWriter,
)
from evals.eval_seams import EvalSeams, default_eval_seams
from evals.eval_types import (
    CaseSpec,
    Condition,
    EvalUnitInputs,
    UnitExecutionOutcome,
    _ActiveRunSettings,
)
from evals.eval_types import RunConditionInputs as _RunConditionInputs
from evals.native_stage_b_flow import openai_usable_prompt_tokens
from evals.patcher_factory import (
    PatcherConfigError,
    _resolve_openai_context_window_for_run,
)
from evals.progress_display import (
    DisplayRunSummary,
    _format_duration,
    build_startup_sections,
    create_run_display,
)
from evals.progress_output import (
    RunProgressSummary,
    _is_infra_interrupted_records,
    build_progress_palette,
    classify_failure_reason,
)
from evals.record_keys import KEY_ADAPTIVE_PROGRESS_STOP_NOTE
from evals.run_condition_builder import build_run_condition_inputs
from evals.run_context import RunContext
from evals.transport_failures import (
    _filter_attempt_records,
    _is_transport_runtime_error,
    _OpenAITransportRecoveryGate,
    _record_is_transport_request_failure,
    _single_line_error,
)

RunConditionInputs = _RunConditionInputs


def _with_run_context(unit_inputs: EvalUnitInputs, run_context: RunContext) -> EvalUnitInputs:
    return replace(
        unit_inputs, run_condition=replace(unit_inputs.run_condition, run_context=run_context)
    )


def _interrupt_parallel_units(
    *,
    executor: ThreadPoolExecutor,
    future_map: dict[Any, Any],
    display: Any,
) -> None:
    """Stop scheduling on Ctrl-C, then let in-flight units finish or exit on a second one.

    Running units can't be stopped from outside their threads, and the interpreter joins
    worker threads at exit anyway, so a single Ctrl-C used to wait silently for all of
    them (ORC-09). Their records are streamed and fsynced as they are written, so a
    second Ctrl-C exits at once and ``--resume`` picks the run up again.
    """
    cancelled = sum(1 for future in list(future_map) if future.cancel())
    executor.shutdown(wait=False, cancel_futures=True)
    in_flight = [future for future in future_map if not future.done()]
    display.add_event(
        message=(
            f"Interrupted: cancelled {cancelled} queued unit(s); waiting for "
            f"{len(in_flight)} in-flight unit(s). Press Ctrl-C again to exit now."
        ),
        style="warning",
    )
    try:
        futures_wait(in_flight)
    except KeyboardInterrupt:
        display.add_event(
            message=f"Exiting with {sum(not f.done() for f in in_flight)} unit(s) unfinished.",
            style="warning",
        )
        display.close()
        os._exit(130)


def run_main_flow(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv_tokens: list[str],
    seams: EvalSeams | None = None,
) -> int:
    seams = seams if seams is not None else default_eval_seams()
    try:
        setup = prepare_main_setup(
            parser=parser,
            args=args,
            argv_tokens=argv_tokens,
            seams=seams,
        )
        resume = prepare_main_resume(
            setup=setup,
            args=args,
            argv_tokens=argv_tokens,
            seams=seams,
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
    run_usage = RunUsageTotals()
    jobs = int(args.jobs)
    try:
        patcher = seams.make_patcher(args)
    except PatcherConfigError as exc:
        _close_run_results_io()
        display.close()
        print(f"Invalid arguments: {exc}")
        return 2
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

    build_run_metadata_for_run_result = build_run_metadata_for_run(
        RunMetadataInputs(
            args=args,
            conditions=conditions,
            config_state=config_state,
            existing_run_metadata=existing_run_metadata,
            openai_context_window_probe_error=openai_context_window_probe_error,
            openai_context_window_probe_provider=openai_context_window_probe_provider,
            openai_context_window_source=openai_context_window_source,
            openai_context_window_tokens_effective=openai_context_window_tokens_effective,
            patcher=patcher,
            patcher_name=patcher_name,
            resolved_config_lock_path=resolved_config_lock_path,
            run_id=run_id,
            run_settings_fingerprint=run_settings_fingerprint,
            seams=seams,
            setup=setup,
        )
    )
    patcher_config = build_run_metadata_for_run_result.patcher_config
    run_metadata = build_run_metadata_for_run_result.run_metadata

    build_policy_controllers_result = build_policy_controllers(
        PolicyControllerInputs(
            _close_run_results_io=_close_run_results_io,
            args=args,
            artifacts_root=artifacts_root,
            cases=cases,
            conditions=conditions,
            display=display,
            patcher_config=patcher_config,
        )
    )
    budget_policy_controller = build_policy_controllers_result.budget_policy_controller
    budget_policy_controllers = build_policy_controllers_result.budget_policy_controllers
    online_policy_controller = build_policy_controllers_result.online_policy_controller

    previous_execution_config = set_active_execution_config(execution_config)
    # X-02: passed down explicitly instead of through run_eval module globals.
    run_context = RunContext(
        execution=execution_config,
        settings=_ActiveRunSettings(
            deterministic=bool(args.deterministic),
            seed=args.seed if args.deterministic else None,
        ),
    )
    container_pool: _EvalContainerPool | None = None
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
                prompt_chars_total=run_usage.prompt_chars,
                completion_chars_total=run_usage.completion_chars,
                token_usage_text=_format_progress_token_usage(
                    prompt_tokens=run_usage.prompt_tokens,
                    completion_tokens=run_usage.completion_tokens,
                    total_tokens=run_usage.total_tokens,
                    available_records=run_usage.usage_available_records,
                    unavailable_records=run_usage.usage_unavailable_records,
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
        metrics = extract_unit_progress_metrics(case_records)
        attempts_used = metrics.attempts_used
        max_attempts = metrics.max_attempts
        total_llm_sec = metrics.total_llm_sec
        total_test_sec = metrics.total_test_sec
        total_prompt_chars = metrics.total_prompt_chars
        total_completion_chars = metrics.total_completion_chars
        total_prompt_tokens = metrics.total_prompt_tokens
        total_completion_tokens = metrics.total_completion_tokens
        total_tokens = metrics.total_tokens
        token_usage_available_records = metrics.token_usage_available_records
        token_usage_unavailable_records = metrics.token_usage_unavailable_records
        run_usage.add(metrics)
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
                            KEY_ADAPTIVE_PROGRESS_STOP_NOTE,
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
    shared_run_condition_inputs = replace(shared_run_condition_inputs, run_context=run_context)
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
        seams=seams,
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
            container_pool = _EvalContainerPool(
                size=pool_size,
                execution=execution_config,
                workdirs_host=artifacts_root / "workdirs",
                run_id=run_id,
            )
            run_context = replace(run_context, container_pool=container_pool)
            shared_unit_inputs = _with_run_context(shared_unit_inputs, run_context)
            unit_inputs_by_condition = {
                key: _with_run_context(unit_inputs, run_context)
                for key, unit_inputs in unit_inputs_by_condition.items()
            }
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
                # ORC-11: each unit builds its own patcher, as with --jobs N, so the job
                # count can't become a confounder once a patcher keeps per-instance state.
                outcome = _execute_eval_unit(
                    case=case,
                    condition=condition,
                    artifacts_root=artifacts_root,
                    inputs=_unit_inputs_for(condition),
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

            executor = ThreadPoolExecutor(max_workers=jobs)
            try:
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
            except KeyboardInterrupt:
                _interrupt_parallel_units(executor=executor, future_map=future_map, display=display)
                raise
            finally:
                executor.shutdown(wait=True)
    finally:
        if container_pool is not None:
            container_pool.shutdown()
        if online_policy_controller is not None:
            try:
                online_policy_controller.close()
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
        _close_run_results_io()
        set_active_execution_config(previous_execution_config)

    total_elapsed = time.monotonic() - run_started
    errors_suffix = f", errors={len(runtime_errors)}" if runtime_errors else ""
    run_token_usage_text = _format_progress_token_usage(
        prompt_tokens=run_usage.prompt_tokens,
        completion_tokens=run_usage.completion_tokens,
        total_tokens=run_usage.total_tokens,
        available_records=run_usage.usage_available_records,
        unavailable_records=run_usage.usage_unavailable_records,
    )
    _sync_display_summary(elapsed_sec=total_elapsed, eta_sec=0.0)
    display.close()
    final_lines = [
        "Progress: finished "
        f"executed={executed_units}, skipped={skipped_units}, total={total_units}, "
        f"elapsed={_format_duration(total_elapsed)} "
        f"chars=in:{run_usage.prompt_chars}ch,out:{run_usage.completion_chars}ch "
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
