"""Baseline selection and external baseline runtimes.

Upstream repos, interpreters, runtime manifests and which conditions a run uses."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals import _json as json
from evals import lambda_rlm_backend
from evals.eval_recording import _baseline_openai_compat_settings_from_args
from evals.eval_types import Condition, ConfigResolutionState, baseline_condition_for_name
from evals.lambda_rlm_backend import (
    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
)

# Baseline repos, envs and patches live under the checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]


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

    repo_dir = (getattr(args, "adaptive_controller_lambda_rlm_repo_dir", "") or "").strip()
    if not repo_dir:
        manifest_repo_dir = _baseline_runtime_manifest_path(
            "lambda_rlm",
            absolute_key="repo_dir",
            repo_relative_key="repo_dir_repo_relative",
            repo_relative_parts=("baselines", "repos", "lambda-RLM"),
        )
        default_repo_dir = lambda_rlm_backend.default_lambda_rlm_repo_dir()
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
    # ORC-15: duplicates would double the work and one copy was silently discarded later.
    duplicates = sorted({token for token in raw_conditions if raw_conditions.count(token) > 1})
    if duplicates:
        raise ValueError(f"duplicate condition(s): {', '.join(duplicates)}")
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
        if not raw_conditions:
            raise ValueError("no conditions given")
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
