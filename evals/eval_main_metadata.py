"""Run metadata of an eval run: resume-or-build, and the run-time patcher settings.

Moved out of ``run_main_flow`` (X-01).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.baseline_runtime import _baseline_extra_from_args, _baseline_patch_inventory
from evals.decision_model.config import DECISION_MODEL_SERVING_CONFIG_KEY
from evals.eval_recording import _build_run_metadata
from evals.eval_types import is_baseline_condition
from evals.patchers.baseline_patcher import BaselinePatcher, baseline_case_support


@dataclass(frozen=True)
class RunMetadataInputs:
    """Read-only inputs of :func:`build_run_metadata_for_run`."""

    args: Any
    conditions: Any
    config_state: Any
    decision_model_serving: Any
    existing_run_metadata: Any
    openai_context_window_probe_error: Any
    openai_context_window_probe_provider: Any
    openai_context_window_source: Any
    openai_context_window_tokens_effective: Any
    patcher: Any
    patcher_name: Any
    resolved_config_lock_path: Any
    run_id: Any
    run_settings_fingerprint: Any
    seams: Any
    setup: Any


@dataclass
class RunMetadataAssembly:
    """What :func:`build_run_metadata_for_run` hands back to its caller."""

    patcher_config: Any
    run_metadata: Any


def build_run_metadata_for_run(inputs: RunMetadataInputs) -> RunMetadataAssembly:
    """Build or resume the run metadata and annotate its patcher config.

    Keeps the existing metadata when the settings fingerprint matches (or is
    missing), otherwise starts fresh and links the old fingerprint (ORC-04). Adds
    the baseline adapter, config paths, the resolved lockfile and the patcher
    settings that only exist at run time. Moved verbatim from ``run_main_flow`` (X-01).
    """
    args = inputs.args
    conditions = inputs.conditions
    config_state = inputs.config_state
    existing_run_metadata = inputs.existing_run_metadata
    openai_context_window_probe_error = inputs.openai_context_window_probe_error
    openai_context_window_probe_provider = inputs.openai_context_window_probe_provider
    openai_context_window_source = inputs.openai_context_window_source
    openai_context_window_tokens_effective = inputs.openai_context_window_tokens_effective
    patcher = inputs.patcher
    patcher_name = inputs.patcher_name
    resolved_config_lock_path = inputs.resolved_config_lock_path
    run_id = inputs.run_id
    run_settings_fingerprint = inputs.run_settings_fingerprint
    seams = inputs.seams
    setup = inputs.setup
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
        # Resolved by the preflight, not by the flags: the request alias stays the same when
        # the sidecar serves another revision or quantization (see the resume drift check).
        serving = inputs.decision_model_serving
        if isinstance(serving, dict) and serving:
            patcher_config[DECISION_MODEL_SERVING_CONFIG_KEY] = dict(serving)
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
                    seams.repo_root / "baselines" / "envs" / baseline_name / "bin" / "python"
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
    return RunMetadataAssembly(
        patcher_config=patcher_config,
        run_metadata=run_metadata,
    )
