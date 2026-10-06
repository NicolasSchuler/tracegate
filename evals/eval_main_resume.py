"""Resume/bootstrap helpers for ``evals.run_eval`` main-flow execution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals import eval_main_support as legacy
from evals.decision_model.config import decision_model_settings_from_args
from evals.decision_model.runtime import check_decision_model_health
from evals.eval_main_setup import MainFlowCliError, MainSetupState
from evals.eval_types import (
    DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    CaseSpec,
    Condition,
)


@dataclass(slots=True)
class MainResumeState:
    execution_units: list[tuple[CaseSpec, Condition]]
    total_units: int
    run_settings_fingerprint: str
    resolved_config_lock_path: Path | None
    run_results_lock: Any
    existing_run_metadata: dict[str, Any] | None
    resumed_completed_units: set[tuple[str, str]]
    skipped_units: int
    remaining_units: int


def _release_lock(lock: Any) -> None:
    if lock is not None:
        lock.release()


def prepare_main_resume(
    *,
    setup: MainSetupState,
    args: argparse.Namespace,
    argv_tokens: list[str],
) -> MainResumeState:
    execution_units = [(case, condition) for case in setup.cases for condition in setup.conditions]
    total_units = len(execution_units)
    run_settings_fingerprint = legacy._build_run_settings_fingerprint(
        args=args,
        cases=setup.cases,
        conditions=setup.conditions,
    )
    try:
        resolved_config_lock_path = legacy._write_resolved_run_config_lockfile(
            run_id=setup.run_id,
            args=args,
            argv_tokens=argv_tokens,
            run_settings_fingerprint=run_settings_fingerprint,
            config_state=setup.config_state,
            repo_root=legacy._REPO_ROOT,
        )
    except OSError as exc:
        resolved_config_lock_path = None
        print(f"Warning: failed to write resolved config lock file: {exc}")

    run_results_lock, run_lock_error, run_lock_warning = legacy._RunResultsLock.acquire(
        results_path=setup.results_path,
        run_id=setup.run_id,
    )
    if run_lock_error is not None:
        raise MainFlowCliError(f"Run aborted: {run_lock_error}")
    if run_lock_warning is not None:
        print(f"Run lock warning: {run_lock_warning}")

    try:
        (
            existing_completed_units,
            existing_run_metadata,
            existing_run_settings_fingerprint,
            existing_records_count,
            trailing_tail_issue,
        ) = legacy._load_existing_run_state(
            results_path=setup.results_path,
            run_id=setup.run_id,
            max_attempts_per_unit=int(args.k),
        )
        if trailing_tail_issue is not None:
            try:
                sidecar_path, removed_bytes = legacy._repair_trailing_results_tail(
                    results_path=setup.results_path,
                    run_id=setup.run_id,
                    issue=trailing_tail_issue,
                )
            except OSError as exc:
                _release_lock(run_results_lock)
                raise MainFlowCliError(
                    "Resume aborted: failed to repair malformed trailing JSON record: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            print(
                "Resume repair: "
                f"truncated malformed trailing JSON record at line {trailing_tail_issue.line_no} "
                f"(removed_bytes={removed_bytes}, backup={sidecar_path})"
            )
        if existing_records_count > 0:
            if not existing_run_settings_fingerprint:
                _release_lock(run_results_lock)
                raise MainFlowCliError(
                    "Resume aborted: existing results are missing run_settings_fingerprint; "
                    "cannot safely resume."
                )
            if existing_run_settings_fingerprint != run_settings_fingerprint:
                print(
                    "Resume warning: run settings fingerprint mismatch for existing run_id; "
                    "continuing with existing completed-unit state."
                )
    except ValueError as exc:
        _release_lock(run_results_lock)
        raise MainFlowCliError(f"Resume aborted: {exc}") from exc

    planned_unit_keys = {(case.case_id, condition.value) for case, condition in execution_units}
    resumed_completed_units = {key for key in existing_completed_units if key in planned_unit_keys}
    skipped_units = len(resumed_completed_units)
    remaining_units = total_units - skipped_units

    if remaining_units > 0:
        decision_model_error = check_decision_model_health(decision_model_settings_from_args(args))
        if decision_model_error:
            _release_lock(run_results_lock)
            raise MainFlowCliError(f"Environment check failed: {decision_model_error}")

    if remaining_units > 0 and setup.execution_config.executor == "testcontainers":
        ok, message = legacy.check_testcontainers_runtime_available()
        if not ok:
            _release_lock(run_results_lock)
            raise MainFlowCliError(f"Environment check failed: {message}")
        removed, failed, cleanup_error = legacy.cleanup_stale_eval_containers(
            grace_sec=DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC
        )
        if cleanup_error is not None:
            print(f"Container cleanup skipped: {cleanup_error}")
        elif removed > 0 or failed > 0:
            print(f"Container cleanup: removed={removed}, failed={failed}")

    return MainResumeState(
        execution_units=execution_units,
        total_units=total_units,
        run_settings_fingerprint=run_settings_fingerprint,
        resolved_config_lock_path=resolved_config_lock_path,
        run_results_lock=run_results_lock,
        existing_run_metadata=(
            dict(existing_run_metadata) if isinstance(existing_run_metadata, dict) else None
        ),
        resumed_completed_units=resumed_completed_units,
        skipped_units=skipped_units,
        remaining_units=remaining_units,
    )
