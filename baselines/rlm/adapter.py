"""Repair-adapted standalone upstream RLM baseline adapter."""

from __future__ import annotations

import ast
import base64
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineConfig,
    BaselineResult,
)
from baselines.case_schemas import LambdaRLMCaseInput
from baselines.contracts import BaselineContract
from baselines.pins import get_expected_env_client_compatibility, get_pinned_repo_commit
from evals.adaptive_controller_packet import (
    parse_controller_packet,
    render_hybrid_controller_guidance,
)
from llmdebug import _json as json

from . import env_client as rlm_env_client
from .case_converter import convert_case

_ENV_CLIENT_SCRIPT = Path(__file__).with_name("env_client.py")
_CANONICAL_PROMPT_CONTRACT = "canonical"
_REPAIR_FAIR_PROMPT_CONTRACT = "repair_fair_v1"
_GUARD_STOP_REASONS = frozenset(
    {
        "iteration_budget_exhausted",
        "timeout_exceeded",
        "token_limit_exceeded",
        "error_threshold_exceeded",
        "budget_exceeded",
        "cancelled",
    }
)


@dataclass(frozen=True)
class _CommandOutcome:
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float = 0.0
    timed_out: bool = False
    timeout_sec: int | None = None


@dataclass(frozen=True)
class _UnchangedWorkspaceFeedbackInputs:
    stop_reason: str
    env_error: str
    last_test_result: dict[str, Any] | None
    tool_call_counts: dict[str, int]
    response_present: bool
    salvage_failure: str
    no_write_attempt_streak: int
    write_rejection_count: int = 0
    last_write_rejection: dict[str, Any] | None = None


def _resolve_repo_commit(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _coerce_subprocess_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max(0, int(max_chars)):
        return value
    if max_chars <= 3:
        return value[: max(0, int(max_chars))]
    return value[: max(0, int(max_chars)) - 3].rstrip() + "..."


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _usage_totals(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    total_calls = _nonnegative_int(payload.get("total_calls")) or 0
    input_tokens = _nonnegative_int(payload.get("total_input_tokens")) or 0
    output_tokens = _nonnegative_int(payload.get("total_output_tokens")) or 0
    total_tokens = input_tokens + output_tokens
    return total_calls, input_tokens, output_tokens, total_tokens


def _char_totals(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    total_calls = _nonnegative_int(payload.get("total_calls")) or 0
    input_chars = _nonnegative_int(payload.get("total_input_chars")) or 0
    output_chars = _nonnegative_int(payload.get("total_output_chars")) or 0
    total_chars = input_chars + output_chars
    return total_calls, input_chars, output_chars, total_chars


def _is_guard_stop_reason(stop_reason: str) -> bool:
    return str(stop_reason or "").strip() in _GUARD_STOP_REASONS


_TERMINAL_ENV_ERRORS = frozenset({"openai_compat_reasoning_only", "timeout", "env_client_failed"})


def _env_failure_error(exc: BaseException) -> str:
    return "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "env_client_failed"


class RLMAdapter(BaselineAdapter):
    """Adapter for the standalone upstream RLM baseline."""

    retry_context_prompt_field = "prompt_context"

    honours_generation_contract = True

    default_model: str = "gpt-4o-mini"
    _config: BaselineConfig | None = None
    repo_commit: str = ""

    @property
    def name(self) -> str:
        return "rlm"

    def contract(self) -> BaselineContract:
        return BaselineContract(
            baseline_name=self.name,
            upstream_commit=get_pinned_repo_commit(self.name),
            fidelity_level="generic_api_adaptation",
            repair_scope_policy="single_file_only",
            prompt_contract_kind=self._prompt_contract_kind(),
            output_contract="single_file_repaired_source_or_final_answer_salvage",
            accepted_case_shapes=(
                "single_file_buggy_py",
                "workspace_tests_available",
                "native_pytest_command_available",
            ),
            patch_scope_summary=("env_shim", "telemetry"),
            notes=(
                "Adapts generic upstream RLM to single-file repair with repair-specific final-turn fallback.",
                "Accepts a valid full-file final answer as repaired buggy.py when no workspace write survives.",
            ),
        )

    def setup(self, config: BaselineConfig) -> None:
        self._config = config
        repo = config.repo_dir
        if not repo.is_dir():
            raise RuntimeError(f"RLM repo not found at {repo}. Run baselines/setup_repos.sh first.")
        required = (
            repo / "pyproject.toml",
            repo / "rlm" / "__init__.py",
            repo / "rlm" / "core" / "rlm.py",
            repo / "rlm" / "environments" / "local_repl.py",
            repo / "rlm" / "clients" / "openai.py",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            rendered = ", ".join(str(path.relative_to(repo)) for path in missing)
            raise RuntimeError(f"RLM repo missing required file(s): {rendered}")
        if config.python_executable is not None and not config.python_executable.is_file():
            raise RuntimeError(f"RLM python executable not found: {config.python_executable}")
        self.repo_commit = _resolve_repo_commit(repo)
        expected_compatibility = get_expected_env_client_compatibility(self.name)
        declared_compatibility = str(getattr(rlm_env_client, "_COMPATIBLE_WITH", "") or "").strip()
        if declared_compatibility != expected_compatibility:
            raise RuntimeError(
                "RLM env client compatibility declaration is stale: "
                f"expected {expected_compatibility}, got {declared_compatibility or '<empty>'}. "
                "Update baselines/pins.py or the env_client shim before running evals."
            )
        expected_repo_commit = get_pinned_repo_commit(self.name)
        if self.repo_commit and self.repo_commit != expected_repo_commit:
            raise RuntimeError(
                "RLM repo commit does not match the env-client compatibility pin: "
                f"expected {expected_repo_commit}, got {self.repo_commit}. "
                "Re-run baselines/setup_repos.sh or update the shared pin if the upstream pin changed."
            )

    def validate_extra_config(self, config: BaselineConfig) -> None:
        from baselines.extra_configs import RLMExtraConfig

        RLMExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def _validated_extra_config(self, config: BaselineConfig):
        from baselines.extra_configs import RLMExtraConfig

        return RLMExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def _prompt_contract(self, config: BaselineConfig | None = None) -> str:
        resolved = config or self._config
        if resolved is None:
            return _CANONICAL_PROMPT_CONTRACT
        value = str(resolved.extra.get("baseline_rlm_prompt_contract") or "").strip()
        return value or _CANONICAL_PROMPT_CONTRACT

    def _prompt_contract_kind(self) -> str:
        prompt_contract = self._prompt_contract()
        if prompt_contract == _REPAIR_FAIR_PROMPT_CONTRACT:
            return "adapter_defined_tool_augmented_prompt_with_repair_fallback__repair_fair_v1"
        return "adapter_defined_tool_augmented_prompt_with_repair_fallback"

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> LambdaRLMCaseInput:
        return convert_case(case_dir, case_context=case_context)

    def _emit_runtime_progress(
        self,
        *,
        phase_label: str,
        note: str = "",
        attempts_completed: int | None = None,
        llm_calls_observed: int | None = None,
        eval_duration_sec_observed: float | None = None,
        source: str = "phase",
    ) -> None:
        payload: dict[str, Any] = {
            "owner": "baseline",
            "source": source,
            "phase_label": phase_label,
            "external_operation": "baseline_repair",
            "external_provider": self.name,
        }
        if note:
            payload["note"] = note
        if isinstance(attempts_completed, int):
            payload["attempts_completed"] = max(0, attempts_completed)
        if isinstance(llm_calls_observed, int):
            payload["llm_calls_observed"] = max(0, llm_calls_observed)
        if isinstance(eval_duration_sec_observed, (int, float)):
            payload["eval_duration_sec_observed"] = max(0.0, float(eval_duration_sec_observed))
        self._emit_progress("baseline_progress", payload)

    def run_repair(self, case_input: dict[str, Any], config: BaselineConfig) -> BaselineResult:
        python_args = [str(arg) for arg in case_input.get("python_args", []) if str(arg)]
        if not python_args:
            return BaselineResult(
                repaired_code=None, success=False, meta={"error": "missing_python_args"}
            )

        original_code = str(case_input.get("buggy_code") or "")
        prompt_context = str(case_input.get("prompt_context") or "").strip()
        if not prompt_context:
            return BaselineResult(
                repaired_code=None, success=False, meta={"error": "missing_prompt_context"}
            )

        controller_packet = parse_controller_packet(case_input.get("llmdebug_controller_packet"))
        extra_config = self._validated_extra_config(config)
        prompt_contract = extra_config.baseline_rlm_prompt_contract
        controller_guidance = render_hybrid_controller_guidance(
            controller_packet,
            mode=extra_config.baseline_hybrid_guidance_mode,
            include_history=True,
            include_retry_plan=True,
            max_chars=extra_config.baseline_rlm_controller_guidance_max_chars,
        ).text
        file_rows = _normalize_file_rows(case_input.get("files"))
        if not file_rows:
            return BaselineResult(
                repaired_code=None, success=False, meta={"error": "missing_workspace_files"}
            )

        test_timeout_sec = self._case_timeout(case_input=case_input, config=config)
        generation_timeout_sec = self._generation_timeout(config=config)
        attempt_summaries: list[dict[str, Any]] = []
        accumulated_prompt_tokens = 0
        accumulated_completion_tokens = 0
        accumulated_llm_calls = 0
        accumulated_reported_chars_prompt = 0
        accumulated_reported_chars_completion = 0
        llm_duration_sec = 0.0
        eval_duration_sec = 0.0
        prompt_chars_total = 0
        completion_chars_total = 0
        usage_reported_call_count = 0
        usage_missing_call_count = 0
        usage_available = False
        internal_iteration_count = 0
        internal_subcall_count = 0
        internal_test_run_count = 0
        internal_max_depth_reached = 0
        internal_length_finish_count = 0
        internal_guard_stop_count = 0
        internal_search_exhausted_attempt_count = 0
        internal_workspace_write_success_count = 0
        internal_salvage_success_count = 0
        internal_compat_recovery_count = 0
        last_candidate: str | None = None
        last_trajectory_summary: dict[str, Any] | None = None
        last_error: str | None = None
        last_terminal_outcome: str | None = None
        no_write_attempt_streak = 0
        started_at = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="rlm_precheck_") as tmpdir:
            precheck_dir = Path(tmpdir)
            self._materialize_workspace(work_dir=precheck_dir, file_rows=file_rows)
            self._emit_runtime_progress(
                phase_label="baseline: rlm precheck",
                note="initial pytest precheck",
                attempts_completed=0,
                llm_calls_observed=accumulated_llm_calls,
                eval_duration_sec_observed=eval_duration_sec,
            )
            precheck = self._run_case_tests(
                work_dir=precheck_dir,
                python_args=python_args,
                timeout_sec=test_timeout_sec,
            )
            eval_duration_sec += precheck.duration_sec
            if precheck.returncode == 0:
                metrics = self._build_metrics(
                    prompt_chars_total=0,
                    completion_chars_total=0,
                    accumulated_prompt_tokens=0,
                    accumulated_completion_tokens=0,
                    accumulated_llm_calls=0,
                    accumulated_reported_chars_prompt=0,
                    accumulated_reported_chars_completion=0,
                    llm_duration_sec=0.0,
                    eval_duration_sec=eval_duration_sec,
                    usage_available=False,
                    usage_reported_call_count=0,
                    usage_missing_call_count=0,
                    started_at=started_at,
                    stopped_reason="case_already_passed",
                    internal_iteration_count=0,
                    internal_subcall_count=0,
                    internal_test_run_count=0,
                    internal_max_depth_reached=0,
                    internal_length_finish_count=0,
                    internal_guard_stop_count=0,
                    internal_search_exhausted_attempt_count=0,
                    internal_workspace_write_success_count=0,
                    internal_salvage_success_count=0,
                    internal_compat_recovery_count=0,
                )
                return BaselineResult(
                    repaired_code=original_code,
                    success=True,
                    iterations_used=0,
                    meta={
                        "baseline_meta": "case_already_passed",
                        "initial_test_stdout_tail": precheck.stdout[-2000:],
                        "initial_test_stderr_tail": precheck.stderr[-2000:],
                        "metrics": metrics,
                    },
                )

        retry_feedback = self._format_failure_feedback(precheck)
        for attempt_idx in range(1, config.max_attempts + 1):
            with tempfile.TemporaryDirectory(prefix=f"rlm_attempt_{attempt_idx:02d}_") as tmpdir:
                work_dir = Path(tmpdir)
                self._materialize_workspace(work_dir=work_dir, file_rows=file_rows)
                self._emit_runtime_progress(
                    phase_label="baseline: rlm generation",
                    note=f"attempt {attempt_idx}/{config.max_attempts} generating",
                    attempts_completed=attempt_idx - 1,
                    llm_calls_observed=accumulated_llm_calls,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                root_prompt = self._build_root_prompt(
                    failure_feedback=retry_feedback,
                    controller_guidance=controller_guidance,
                    attempt_idx=attempt_idx,
                    prompt_contract=prompt_contract,
                    failure_feedback_max_chars=extra_config.baseline_rlm_failure_feedback_max_chars,
                )
                prompt_chars_total += len(prompt_context) + len(root_prompt)
                try:
                    env_result = self._invoke_rlm(
                        prompt_context=prompt_context,
                        root_prompt=root_prompt,
                        work_dir=work_dir,
                        python_args=python_args,
                        test_timeout_sec=test_timeout_sec,
                        config=config,
                        timeout_sec=generation_timeout_sec,
                    )
                except (subprocess.TimeoutExpired, RuntimeError) as exc:
                    # BAS-10: keep earlier attempts' candidate, usage and metrics.
                    env_result = {"error": _env_failure_error(exc), "compat_message": str(exc)}
                env_error = str(env_result.get("error") or "").strip()
                if env_error in _TERMINAL_ENV_ERRORS:
                    metrics = self._build_metrics(
                        prompt_chars_total=prompt_chars_total,
                        completion_chars_total=completion_chars_total,
                        accumulated_prompt_tokens=accumulated_prompt_tokens,
                        accumulated_completion_tokens=accumulated_completion_tokens,
                        accumulated_llm_calls=accumulated_llm_calls,
                        accumulated_reported_chars_prompt=accumulated_reported_chars_prompt,
                        accumulated_reported_chars_completion=accumulated_reported_chars_completion,
                        llm_duration_sec=llm_duration_sec,
                        eval_duration_sec=eval_duration_sec,
                        usage_available=usage_available,
                        usage_reported_call_count=usage_reported_call_count,
                        usage_missing_call_count=usage_missing_call_count,
                        started_at=started_at,
                        stopped_reason=env_error,
                        internal_iteration_count=internal_iteration_count,
                        internal_subcall_count=internal_subcall_count,
                        internal_test_run_count=internal_test_run_count,
                        internal_max_depth_reached=internal_max_depth_reached,
                        internal_length_finish_count=internal_length_finish_count,
                        internal_guard_stop_count=internal_guard_stop_count,
                        internal_search_exhausted_attempt_count=(
                            internal_search_exhausted_attempt_count
                        ),
                        internal_workspace_write_success_count=(
                            internal_workspace_write_success_count
                        ),
                        internal_salvage_success_count=internal_salvage_success_count,
                        internal_compat_recovery_count=internal_compat_recovery_count,
                    )
                    # Validate last_candidate — never return prose.
                    _rlm_final = last_candidate
                    if _rlm_final is not None:
                        try:
                            ast.parse(_rlm_final)
                        except SyntaxError:
                            _rlm_final = None
                    return BaselineResult(
                        repaired_code=_rlm_final,
                        iterations_used=attempt_idx,
                        token_usage={
                            "prompt_tokens": accumulated_prompt_tokens,
                            "completion_tokens": accumulated_completion_tokens,
                            "total_tokens": accumulated_prompt_tokens
                            + accumulated_completion_tokens,
                        },
                        meta={
                            "attempts": attempt_summaries,
                            "metrics": metrics,
                            "last_trajectory_summary": last_trajectory_summary,
                            "error": env_error,
                            "terminal_outcome": env_error,
                            "compat_message": str(env_result.get("compat_message") or ""),
                            "compat_meta": dict(
                                cast(dict[str, Any], env_result.get("compat_meta") or {})
                            ),
                        },
                    )
                llm_duration_sec += env_result["execution_time"]
                completion_chars_total += len(env_result["raw_response"])
                usage_summary = env_result["usage_summary"]
                char_summary = env_result["char_summary"]
                call_summary = env_result["call_summary"]
                trajectory_summary = env_result["trajectory_summary"]
                stop_reason = str(env_result.get("stop_reason") or "unknown")
                env_error = str(env_result.get("error") or "").strip()
                llm_calls_attempt, prompt_tokens_attempt, completion_tokens_attempt, _ = (
                    _usage_totals(usage_summary)
                )
                _, input_chars_attempt, output_chars_attempt, _ = _char_totals(char_summary)
                accumulated_llm_calls += llm_calls_attempt
                accumulated_prompt_tokens += prompt_tokens_attempt
                accumulated_completion_tokens += completion_tokens_attempt
                accumulated_reported_chars_prompt += input_chars_attempt
                accumulated_reported_chars_completion += output_chars_attempt
                if (
                    llm_calls_attempt > 0
                    or prompt_tokens_attempt > 0
                    or completion_tokens_attempt > 0
                ):
                    usage_available = True
                    usage_reported_call_count += max(1, llm_calls_attempt)
                else:
                    usage_missing_call_count += 1

                iterations_completed = (
                    _nonnegative_int(trajectory_summary.get("iterations_completed")) or 0
                )
                subcall_count = _nonnegative_int(trajectory_summary.get("subcall_count")) or 0
                test_run_count = _nonnegative_int(trajectory_summary.get("test_run_count")) or 0
                max_depth_reached = (
                    _nonnegative_int(trajectory_summary.get("max_depth_reached")) or 0
                )
                internal_iteration_count += iterations_completed
                internal_subcall_count += subcall_count
                internal_test_run_count += test_run_count
                internal_max_depth_reached = max(internal_max_depth_reached, max_depth_reached)
                call_length_finish_count = (
                    _nonnegative_int(call_summary.get("call_length_finish_count")) or 0
                )
                compat_recovery_count = (
                    _nonnegative_int(call_summary.get("compat_recovery_count")) or 0
                )
                stop_due_to_guard = _is_guard_stop_reason(stop_reason)
                guard_name = stop_reason if stop_due_to_guard else ""
                internal_length_finish_count += call_length_finish_count
                internal_compat_recovery_count += compat_recovery_count
                internal_guard_stop_count += int(stop_due_to_guard)

                attempt_summary: dict[str, Any] = {
                    "attempt": attempt_idx,
                    "llm_call_count": llm_calls_attempt,
                    "execution_time_sec": env_result["execution_time"],
                    "stop_reason": stop_reason,
                    "iterations_completed": iterations_completed,
                    "subcall_count": subcall_count,
                    "test_run_count": test_run_count,
                    "max_depth_reached": max_depth_reached,
                    "raw_response_chars": len(env_result["raw_response"]),
                    "workspace_buggy_changed": bool(env_result["workspace_buggy_changed"]),
                    "stop_due_to_guard": stop_due_to_guard,
                    "call_length_finish_count": call_length_finish_count,
                    "compat_recovery_count": compat_recovery_count,
                }
                if guard_name:
                    attempt_summary["guard_name"] = guard_name
                call_finish_reason_counts = call_summary.get("call_finish_reason_counts")
                if isinstance(call_finish_reason_counts, dict) and call_finish_reason_counts:
                    attempt_summary["call_finish_reason_counts"] = {
                        str(key): int(value)
                        for key, value in cast(
                            dict[object, object], call_finish_reason_counts
                        ).items()
                        if isinstance(key, str) and isinstance(value, int)
                    }
                call_records = call_summary.get("call_records")
                if isinstance(call_records, list) and call_records:
                    attempt_summary["call_records"] = [
                        record for record in call_records if isinstance(record, dict)
                    ]
                compat_recovery_events = call_summary.get("compat_recovery_events")
                if isinstance(compat_recovery_events, list) and compat_recovery_events:
                    attempt_summary["compat_recovery_events"] = [
                        event for event in compat_recovery_events if isinstance(event, dict)
                    ]
                tool_call_counts_normalized: dict[str, int] = {}
                tool_call_counts = trajectory_summary.get("tool_call_counts")
                if isinstance(tool_call_counts, dict) and tool_call_counts:
                    tool_call_counts_dict = cast(dict[object, object], tool_call_counts)
                    tool_call_counts_normalized = {
                        str(key): int(value)
                        for key, value in tool_call_counts_dict.items()
                        if isinstance(key, str) and isinstance(value, int)
                    }
                    attempt_summary["tool_call_counts"] = tool_call_counts_normalized
                write_buggy_py_calls = int(
                    tool_call_counts_normalized.get("write_buggy_py", 0) or 0
                )
                attempt_summary["write_buggy_py_calls"] = write_buggy_py_calls
                attempt_summary["no_write_buggy_py"] = write_buggy_py_calls <= 0
                if write_buggy_py_calls <= 0:
                    no_write_attempt_streak += 1
                else:
                    no_write_attempt_streak = 0
                attempt_summary["no_write_attempt_streak"] = no_write_attempt_streak
                attempt_summary["read_only_tool_only"] = (
                    bool(tool_call_counts_normalized) and write_buggy_py_calls <= 0
                )
                raw_last_test_result = trajectory_summary.get("last_test_result")
                last_test_result: dict[str, Any] | None = (
                    cast(dict[str, Any], raw_last_test_result)
                    if isinstance(raw_last_test_result, dict)
                    else None
                )
                if last_test_result:
                    attempt_summary["internal_test_feedback"] = self._format_internal_test_feedback(
                        last_test_result
                    )
                write_rejection_count = (
                    _nonnegative_int(trajectory_summary.get("write_rejection_count")) or 0
                )
                last_write_rejection_raw = trajectory_summary.get("last_write_rejection")
                last_write_rejection = (
                    cast(dict[str, Any], last_write_rejection_raw)
                    if isinstance(last_write_rejection_raw, dict) and last_write_rejection_raw
                    else None
                )
                if write_rejection_count > 0:
                    attempt_summary["write_rejection_count"] = write_rejection_count
                if last_write_rejection is not None:
                    attempt_summary["last_write_rejection"] = dict(last_write_rejection)
                if env_error:
                    attempt_summary["env_error"] = env_error
                last_trajectory_summary = dict(trajectory_summary)

                raw_response = str(env_result.get("raw_response") or "")
                workspace_candidate = str(env_result.get("workspace_buggy_code") or "")
                candidate_source = "workspace_write_tool"
                salvage_failure = ""
                if not bool(env_result["workspace_buggy_changed"]):
                    salvaged_candidate, salvage_failure = self._salvage_candidate_from_response(
                        raw_response=raw_response,
                        original_code=original_code,
                    )
                    if salvaged_candidate is not None:
                        workspace_candidate = salvaged_candidate
                        candidate_source = "final_answer_salvage"
                        attempt_summary["candidate_salvaged"] = True
                    else:
                        candidate_source = "none"
                attempt_summary["candidate_source"] = candidate_source
                if salvage_failure and candidate_source == "none":
                    attempt_summary["final_answer_salvage_failure"] = salvage_failure
                if candidate_source == "none":
                    retry_feedback = self._retry_feedback_for_unchanged_workspace(
                        _UnchangedWorkspaceFeedbackInputs(
                            stop_reason=stop_reason,
                            env_error=env_error,
                            last_test_result=last_test_result,
                            tool_call_counts=tool_call_counts_normalized,
                            response_present=bool(raw_response.strip()),
                            salvage_failure=salvage_failure,
                            no_write_attempt_streak=no_write_attempt_streak,
                            write_rejection_count=write_rejection_count,
                            last_write_rejection=last_write_rejection,
                        )
                    )
                    attempt_summary["candidate_status"] = (
                        "invalid_workspace_write"
                        if write_rejection_count > 0
                        else "workspace_unchanged"
                    )
                    outcome_classification = (
                        "failure_invalid_workspace_write"
                        if write_rejection_count > 0
                        else (
                            "failure_search_exhausted"
                            if stop_due_to_guard
                            else "failure_workspace_unchanged"
                        )
                    )
                    attempt_summary["outcome_classification"] = outcome_classification
                    if outcome_classification == "failure_search_exhausted":
                        internal_search_exhausted_attempt_count += 1
                    attempt_summaries.append(attempt_summary)
                    self._emit_runtime_progress(
                        phase_label="baseline: rlm generation",
                        note=f"attempt {attempt_idx} left workspace unchanged",
                        attempts_completed=attempt_idx,
                        llm_calls_observed=accumulated_llm_calls,
                        eval_duration_sec_observed=eval_duration_sec,
                    )
                    last_error = str(attempt_summary["candidate_status"])
                    last_terminal_outcome = outcome_classification
                    continue

                attempt_summary["candidate_chars"] = len(workspace_candidate)
                if not self._is_valid_python(workspace_candidate):
                    syntax_feedback = self._syntax_feedback(workspace_candidate)
                    retry_feedback = (
                        "Previous attempt wrote invalid Python to buggy.py.\n\n" + syntax_feedback
                    )
                    attempt_summary["candidate_status"] = "syntax_error"
                    attempt_summary["outcome_classification"] = "failure_syntax_error"
                    attempt_summary["syntax_feedback"] = syntax_feedback
                    attempt_summaries.append(attempt_summary)
                    self._emit_runtime_progress(
                        phase_label="baseline: rlm generation",
                        note=f"attempt {attempt_idx} produced invalid python",
                        attempts_completed=attempt_idx,
                        llm_calls_observed=accumulated_llm_calls,
                        eval_duration_sec_observed=eval_duration_sec,
                    )
                    last_candidate = workspace_candidate
                    last_error = "invalid_python"
                    last_terminal_outcome = "failure_syntax_error"
                    continue

                last_candidate = workspace_candidate
                (work_dir / "buggy.py").write_text(workspace_candidate, encoding="utf-8")
                self._emit_runtime_progress(
                    phase_label="baseline: rlm postcheck",
                    note=f"attempt {attempt_idx}/{config.max_attempts} local tests",
                    attempts_completed=attempt_idx,
                    llm_calls_observed=accumulated_llm_calls,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                test_outcome = self._run_case_tests(
                    work_dir=work_dir,
                    python_args=python_args,
                    timeout_sec=test_timeout_sec,
                )
                eval_duration_sec += test_outcome.duration_sec
                attempt_summary["test_returncode"] = test_outcome.returncode
                attempt_summary["test_duration_sec"] = test_outcome.duration_sec
                if test_outcome.returncode == 0:
                    attempt_summary["candidate_status"] = "passed"
                    outcome_classification = (
                        "success_final_answer_salvage"
                        if candidate_source == "final_answer_salvage"
                        else "success_workspace_write"
                    )
                    attempt_summary["outcome_classification"] = outcome_classification
                    if outcome_classification == "success_workspace_write":
                        internal_workspace_write_success_count += 1
                    else:
                        internal_salvage_success_count += 1
                    attempt_summaries.append(attempt_summary)
                    self._emit_runtime_progress(
                        phase_label="baseline: rlm postcheck",
                        note=f"attempt {attempt_idx} passed local tests",
                        attempts_completed=attempt_idx,
                        llm_calls_observed=accumulated_llm_calls,
                        eval_duration_sec_observed=eval_duration_sec,
                    )
                    metrics = self._build_metrics(
                        prompt_chars_total=prompt_chars_total,
                        completion_chars_total=completion_chars_total,
                        accumulated_prompt_tokens=accumulated_prompt_tokens,
                        accumulated_completion_tokens=accumulated_completion_tokens,
                        accumulated_llm_calls=accumulated_llm_calls,
                        accumulated_reported_chars_prompt=accumulated_reported_chars_prompt,
                        accumulated_reported_chars_completion=accumulated_reported_chars_completion,
                        llm_duration_sec=llm_duration_sec,
                        eval_duration_sec=eval_duration_sec,
                        usage_available=usage_available,
                        usage_reported_call_count=usage_reported_call_count,
                        usage_missing_call_count=usage_missing_call_count,
                        started_at=started_at,
                        stopped_reason="passed_tests",
                        internal_iteration_count=internal_iteration_count,
                        internal_subcall_count=internal_subcall_count,
                        internal_test_run_count=internal_test_run_count,
                        internal_max_depth_reached=internal_max_depth_reached,
                        internal_length_finish_count=internal_length_finish_count,
                        internal_guard_stop_count=internal_guard_stop_count,
                        internal_search_exhausted_attempt_count=(
                            internal_search_exhausted_attempt_count
                        ),
                        internal_workspace_write_success_count=(
                            internal_workspace_write_success_count
                        ),
                        internal_salvage_success_count=internal_salvage_success_count,
                        internal_compat_recovery_count=internal_compat_recovery_count,
                    )
                    metrics["direct_gen_passed"] = attempt_idx == 1
                    metrics["debug_added_value"] = attempt_idx > 1
                    return BaselineResult(
                        repaired_code=workspace_candidate,
                        success=True,
                        iterations_used=attempt_idx,
                        token_usage={
                            "prompt_tokens": accumulated_prompt_tokens,
                            "completion_tokens": accumulated_completion_tokens,
                            "total_tokens": accumulated_prompt_tokens
                            + accumulated_completion_tokens,
                        },
                        meta={
                            "attempts": attempt_summaries,
                            "metrics": metrics,
                            "last_trajectory_summary": last_trajectory_summary,
                            "terminal_outcome": outcome_classification,
                        },
                    )

                retry_feedback = self._format_failure_feedback(test_outcome)
                attempt_summary["candidate_status"] = "tests_still_failing"
                attempt_summary["outcome_classification"] = "failure_tests_still_failing"
                attempt_summary["failure_feedback"] = _truncate(
                    retry_feedback,
                    max_chars=extra_config.baseline_rlm_failure_feedback_meta_max_chars,
                )
                attempt_summaries.append(attempt_summary)
                self._emit_runtime_progress(
                    phase_label="baseline: rlm postcheck",
                    note=f"attempt {attempt_idx} tests still failing",
                    attempts_completed=attempt_idx,
                    llm_calls_observed=accumulated_llm_calls,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                last_error = "tests_still_failing"
                last_terminal_outcome = "failure_tests_still_failing"

        metrics = self._build_metrics(
            prompt_chars_total=prompt_chars_total,
            completion_chars_total=completion_chars_total,
            accumulated_prompt_tokens=accumulated_prompt_tokens,
            accumulated_completion_tokens=accumulated_completion_tokens,
            accumulated_llm_calls=accumulated_llm_calls,
            accumulated_reported_chars_prompt=accumulated_reported_chars_prompt,
            accumulated_reported_chars_completion=accumulated_reported_chars_completion,
            llm_duration_sec=llm_duration_sec,
            eval_duration_sec=eval_duration_sec,
            usage_available=usage_available,
            usage_reported_call_count=usage_reported_call_count,
            usage_missing_call_count=usage_missing_call_count,
            started_at=started_at,
            stopped_reason=last_error or "attempt_budget_exhausted",
            internal_iteration_count=internal_iteration_count,
            internal_subcall_count=internal_subcall_count,
            internal_test_run_count=internal_test_run_count,
            internal_max_depth_reached=internal_max_depth_reached,
            internal_length_finish_count=internal_length_finish_count,
            internal_guard_stop_count=internal_guard_stop_count,
            internal_search_exhausted_attempt_count=internal_search_exhausted_attempt_count,
            internal_workspace_write_success_count=internal_workspace_write_success_count,
            internal_salvage_success_count=internal_salvage_success_count,
            internal_compat_recovery_count=internal_compat_recovery_count,
        )
        # Validate last_candidate — never return prose.
        _rlm_final = last_candidate
        if _rlm_final is not None:
            try:
                ast.parse(_rlm_final)
            except SyntaxError:
                _rlm_final = None
        return BaselineResult(
            repaired_code=_rlm_final,
            iterations_used=len(attempt_summaries),
            token_usage={
                "prompt_tokens": accumulated_prompt_tokens,
                "completion_tokens": accumulated_completion_tokens,
                "total_tokens": accumulated_prompt_tokens + accumulated_completion_tokens,
            },
            meta={
                "attempts": attempt_summaries,
                "metrics": metrics,
                "last_trajectory_summary": last_trajectory_summary,
                "error": last_error or "attempt_budget_exhausted",
                "terminal_outcome": last_terminal_outcome or "attempt_budget_exhausted",
            },
        )

    def _build_root_prompt(
        self,
        *,
        failure_feedback: str,
        controller_guidance: str,
        attempt_idx: int,
        prompt_contract: str = _CANONICAL_PROMPT_CONTRACT,
        failure_feedback_max_chars: int = 3500,
    ) -> str:
        lines = [
            "Repair buggy.py in the provided workspace so the case tests pass.",
            "Use list_files() and read_file(path) to inspect the workspace when needed.",
            "When you change the implementation, call write_buggy_py(content) with the full corrected contents of buggy.py.",
            "Use run_case_tests() to validate your changes inside the workspace before finalizing.",
            "Only modify buggy.py. Do not edit tests or helper files.",
            "You may use llm_query() and rlm_query() for subproblems.",
            "Prefer write_buggy_py(content) to persist your fix in the workspace.",
            "If no workspace write survives, your final answer must be ONLY the full corrected contents of buggy.py as valid Python with no markdown or explanation.",
            "Before finishing, either leave buggy.py updated in the workspace or make FINAL(...) / FINAL_VAR(...) resolve to that same full-file buggy.py contents.",
        ]
        if prompt_contract == _REPAIR_FAIR_PROMPT_CONTRACT:
            lines.extend(
                [
                    "Attempt a minimal direct repair before recursive subcalls.",
                    "Within the first two inner iterations, either call write_buggy_py(content) with a concrete candidate or run_case_tests() on a written candidate.",
                    "Use llm_query() and rlm_query() only after the direct attempt or after reading the most relevant files.",
                ]
            )
        if attempt_idx > 1:
            lines.append(
                f"This is outer repair attempt {attempt_idx}. Do not repeat the previous failure."
            )
        if failure_feedback:
            lines.append("Failure feedback from the previous attempt/test run:")
            lines.append(_truncate(failure_feedback, max_chars=failure_feedback_max_chars))
        if controller_guidance:
            lines.append("Adaptive controller guidance:")
            lines.append(controller_guidance)
        return "\n\n".join(line for line in lines if line.strip())

    def _invoke_rlm(
        self,
        *,
        prompt_context: str,
        root_prompt: str,
        work_dir: Path,
        python_args: list[str],
        test_timeout_sec: int,
        config: BaselineConfig,
        timeout_sec: int,
    ) -> dict[str, Any]:
        repo_dir = config.repo_dir.expanduser().resolve()
        python_executable = config.python_executable or Path(sys.executable)
        payload: dict[str, Any] = {
            "repo_dir": str(repo_dir),
            "model": str(config.model or ""),
            "base_url": str(config.api_base_url or ""),
            "api_key": str(config.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")),
            "api_timeout_sec": float(config.api_timeout_sec or 0.0),
            "baseline_openai_compat_profile": str(
                config.extra.get("baseline_openai_compat_profile") or "auto"
            ),
            "baseline_openai_allow_reasoning_fallback": bool(
                config.extra.get("baseline_openai_allow_reasoning_fallback", False)
            ),
            "baseline_openai_enable_thinking": bool(
                config.extra.get("baseline_openai_enable_thinking", False)
            ),
            "prompt_context": prompt_context,
            "root_prompt": root_prompt,
            "workspace_dir": str(work_dir),
            "test_python": sys.executable,
            "python_args": list(python_args),
            "test_timeout_sec": int(test_timeout_sec),
            "max_depth": int(config.extra.get("baseline_rlm_max_depth") or 2),
            "max_iterations": int(config.extra.get("baseline_rlm_max_iterations") or 12),
            "max_timeout_sec": float(timeout_sec),
            "max_errors": int(config.extra.get("baseline_rlm_max_errors") or 5),
            "compaction": bool(config.extra.get("baseline_rlm_compaction", False)),
            "compaction_threshold_pct": float(
                config.extra.get("baseline_rlm_compaction_threshold_pct") or 0.85
            ),
            "verbose": bool(config.extra.get("baseline_rlm_verbose", False)),
        }
        if config.extra.get("baseline_rlm_max_total_tokens") is not None:
            payload["max_total_tokens"] = int(config.extra["baseline_rlm_max_total_tokens"])
        # D4: the shared generation contract unless the baseline-specific knob is set.
        if config.generation.temperature is not None:
            payload["temperature"] = float(config.generation.temperature)
        if config.generation.max_completion_tokens is not None:
            payload["max_tokens"] = int(config.generation.max_completion_tokens)
        for source_key, target_key in (
            ("baseline_rlm_temperature", "temperature"),
            ("baseline_rlm_top_p", "top_p"),
            ("baseline_rlm_max_tokens", "max_tokens"),
        ):
            value = config.extra.get(source_key)
            if value is not None:
                payload[target_key] = value
        with tempfile.TemporaryDirectory(prefix="rlm_env_") as tmpdir:
            request_path = Path(tmpdir) / "request.json"
            response_path = Path(tmpdir) / "response.json"
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    str(python_executable),
                    str(_ENV_CLIENT_SCRIPT),
                    str(request_path),
                    str(response_path),
                ],
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_sec)),
                cwd=str(repo_dir),
                env=os.environ.copy(),
            )
            if completed.returncode != 0:
                detail = completed.stderr[-1200:] if completed.stderr else completed.stdout[-1200:]
                raise RuntimeError(f"RLM env client failed: {detail}")
            raw_payload = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise RuntimeError(f"RLM env client returned {type(raw_payload).__name__}")
        payload_dict = cast(dict[str, Any], raw_payload)
        usage_summary = payload_dict.get("usage_summary")
        char_summary = payload_dict.get("char_summary")
        call_summary = payload_dict.get("call_summary")
        trajectory_summary = payload_dict.get("trajectory_summary")
        return {
            "raw_response": str(payload_dict.get("response") or ""),
            "execution_time": float(payload_dict.get("execution_time") or 0.0),
            "usage_summary": usage_summary if isinstance(usage_summary, dict) else {},
            "char_summary": char_summary if isinstance(char_summary, dict) else {},
            "call_summary": call_summary if isinstance(call_summary, dict) else {},
            "trajectory_summary": trajectory_summary
            if isinstance(trajectory_summary, dict)
            else {},
            "stop_reason": str(payload_dict.get("stop_reason") or ""),
            "error": str(payload_dict.get("error") or ""),
            "compat_message": str(payload_dict.get("compat_message") or ""),
            "compat_meta": payload_dict.get("compat_meta")
            if isinstance(payload_dict.get("compat_meta"), dict)
            else {},
            "workspace_buggy_code": str(payload_dict.get("workspace_buggy_code") or ""),
            "workspace_buggy_changed": bool(payload_dict.get("workspace_buggy_changed", False)),
        }

    def _build_metrics(
        self,
        *,
        prompt_chars_total: int,
        completion_chars_total: int,
        accumulated_prompt_tokens: int,
        accumulated_completion_tokens: int,
        accumulated_llm_calls: int,
        accumulated_reported_chars_prompt: int,
        accumulated_reported_chars_completion: int,
        llm_duration_sec: float,
        eval_duration_sec: float,
        usage_available: bool,
        usage_reported_call_count: int,
        usage_missing_call_count: int,
        started_at: float,
        stopped_reason: str,
        internal_iteration_count: int,
        internal_subcall_count: int,
        internal_test_run_count: int,
        internal_max_depth_reached: int,
        internal_length_finish_count: int,
        internal_guard_stop_count: int,
        internal_search_exhausted_attempt_count: int,
        internal_workspace_write_success_count: int,
        internal_salvage_success_count: int,
        internal_compat_recovery_count: int,
    ) -> dict[str, Any]:
        orchestration_total = max(0.0, time.perf_counter() - started_at)
        metrics: dict[str, Any] = {
            "llm_call_count": max(0, int(accumulated_llm_calls)),
            "llm_duration_sec": max(0.0, float(llm_duration_sec)),
            "eval_duration_sec": max(0.0, float(eval_duration_sec)),
            "internal_total_duration_sec": orchestration_total,
            "orchestration_duration_sec": max(
                0.0,
                orchestration_total
                - max(0.0, float(llm_duration_sec))
                - max(0.0, float(eval_duration_sec)),
            ),
            "usage_available": bool(usage_available and accumulated_llm_calls > 0),
            "usage_reported_call_count": max(0, int(usage_reported_call_count)),
            "usage_missing_call_count": max(0, int(usage_missing_call_count)),
            "reported_tokens_prompt": max(0, int(accumulated_prompt_tokens)),
            "reported_tokens_completion": max(0, int(accumulated_completion_tokens)),
            "reported_tokens_total": max(
                0, int(accumulated_prompt_tokens + accumulated_completion_tokens)
            ),
            "prompt_chars_total": max(0, int(prompt_chars_total)),
            "completion_chars_total": max(0, int(completion_chars_total)),
            "chars_input": max(0, int(prompt_chars_total)),
            "chars_output": max(0, int(completion_chars_total)),
            "chars_total": max(0, int(prompt_chars_total + completion_chars_total)),
            "reported_chars_prompt": max(0, int(accumulated_reported_chars_prompt)),
            "reported_chars_completion": max(0, int(accumulated_reported_chars_completion)),
            "reported_chars_total": max(
                0, int(accumulated_reported_chars_prompt + accumulated_reported_chars_completion)
            ),
            "usage_source": "rlm_internal_summary",
            "stopped_reason": str(stopped_reason or "unknown"),
            "method_instrumentation": False,
            "method_two_step_repair": False,
            "internal_iteration_count": max(0, int(internal_iteration_count)),
            "internal_subcall_count": max(0, int(internal_subcall_count)),
            "internal_test_run_count": max(0, int(internal_test_run_count)),
            "internal_max_depth_reached": max(0, int(internal_max_depth_reached)),
            "internal_length_finish_count": max(0, int(internal_length_finish_count)),
            "internal_guard_stop_count": max(0, int(internal_guard_stop_count)),
            "internal_search_exhausted_attempt_count": max(
                0, int(internal_search_exhausted_attempt_count)
            ),
            "internal_workspace_write_success_count": max(
                0, int(internal_workspace_write_success_count)
            ),
            "internal_salvage_success_count": max(0, int(internal_salvage_success_count)),
            "internal_compat_recovery_count": max(0, int(internal_compat_recovery_count)),
        }
        return metrics

    def _generation_timeout(self, *, config: BaselineConfig) -> int:
        process_timeout = max(1, int(config.process_timeout_sec or 1.0))
        raw_timeout = config.extra.get("baseline_rlm_timeout_sec")
        if isinstance(raw_timeout, (int, float)) and float(raw_timeout) > 0:
            return min(max(1, int(float(raw_timeout))), process_timeout)
        return process_timeout

    def _case_timeout(self, *, case_input: dict[str, Any], config: BaselineConfig) -> int:
        raw_timeout = case_input.get("timeout_sec")
        if isinstance(raw_timeout, int) and raw_timeout > 0:
            return raw_timeout
        return max(1, int(config.process_timeout_sec or 1.0))

    def _materialize_workspace(
        self,
        *,
        work_dir: Path,
        file_rows: list[dict[str, object]],
    ) -> None:
        for row in file_rows:
            relative_path = Path(str(row.get("path", "")))
            if not relative_path.parts:
                continue
            target = work_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            encoding = str(row.get("encoding", "utf-8"))
            content = row.get("content", "")
            if encoding == "base64":
                target.write_bytes(base64.b64decode(str(content)))
            else:
                target.write_text(str(content), encoding="utf-8")

    def _run_case_tests(
        self,
        *,
        work_dir: Path,
        python_args: list[str],
        timeout_sec: int,
    ) -> _CommandOutcome:
        started_at = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, *python_args],
                capture_output=True,
                text=True,
                cwd=str(work_dir),
                timeout=timeout_sec,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            return _CommandOutcome(
                returncode=124,
                stdout=_coerce_subprocess_output(exc.stdout),
                stderr=_coerce_subprocess_output(exc.stderr),
                duration_sec=time.perf_counter() - started_at,
                timed_out=True,
                timeout_sec=timeout_sec,
            )
        return _CommandOutcome(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_sec=time.perf_counter() - started_at,
        )

    def _format_failure_feedback(self, outcome: _CommandOutcome) -> str:
        parts: list[str] = []
        if outcome.timed_out:
            parts.append(f"Command timed out after {outcome.timeout_sec} seconds.")
        parts.append(f"Exit code: {outcome.returncode}")
        stdout_tail = outcome.stdout[-4000:].strip()
        stderr_tail = outcome.stderr[-4000:].strip()
        if stdout_tail:
            parts.append(f"stdout:\n{stdout_tail}")
        if stderr_tail:
            parts.append(f"stderr:\n{stderr_tail}")
        return "\n\n".join(parts)

    def _format_internal_test_feedback(self, payload: dict[str, Any]) -> str:
        return self._format_failure_feedback(
            _CommandOutcome(
                returncode=int(payload.get("returncode") or 0),
                stdout=str(payload.get("stdout") or ""),
                stderr=str(payload.get("stderr") or ""),
                duration_sec=float(payload.get("duration_sec") or 0.0),
                timed_out=bool(payload.get("timed_out", False)),
                timeout_sec=int(payload.get("timeout_sec") or 0) or None,
            )
        )

    def _retry_feedback_for_unchanged_workspace(
        self,
        inputs: _UnchangedWorkspaceFeedbackInputs,
    ) -> str:
        write_buggy_py_calls = int(inputs.tool_call_counts.get("write_buggy_py", 0) or 0)
        parts = [
            "Previous attempt did not leave a modified buggy.py in the workspace.",
            "You must call write_buggy_py(content) with the full corrected file contents before finishing.",
            f"RLM stop reason: {inputs.stop_reason}",
        ]
        if inputs.no_write_attempt_streak > 0:
            parts.append(f"No-write attempt streak: {inputs.no_write_attempt_streak}")
        if inputs.write_rejection_count > 0:
            parts.append(
                "write_buggy_py(content) calls were rejected by workspace guards before they could persist."
            )
        if inputs.last_write_rejection:
            reason = str(inputs.last_write_rejection.get("reason") or "").strip()
            detail = str(inputs.last_write_rejection.get("detail") or "").strip()
            if reason:
                parts.append(f"Last rejected write reason: {reason}")
            if detail:
                parts.append(f"Last rejected write detail: {_truncate(detail, max_chars=400)}")
        if write_buggy_py_calls <= 0:
            parts.append("This attempt did not call write_buggy_py(content).")
        else:
            parts.append(
                "write_buggy_py(content) was called, but buggy.py still matched the original file at the end of the attempt."
            )
        if inputs.tool_call_counts:
            observed_tools = ", ".join(
                f"{name}={count}" for name, count in sorted(inputs.tool_call_counts.items())
            )
            parts.append(f"Observed tool usage: {observed_tools}")
        if not inputs.response_present:
            parts.append("You also produced no final code output that could be salvaged.")
        elif inputs.salvage_failure == "response_matches_original_buggy_py":
            parts.append(
                "Your final answer matched the original buggy.py, so it did not count as a repair."
            )
        elif inputs.salvage_failure == "response_not_valid_python":
            parts.append("Your final answer was not salvageable as valid full-file Python.")
        elif inputs.salvage_failure == "response_not_salvageable":
            parts.append("Your final answer was not recognizable as full buggy.py contents.")
        parts.append(
            "Prefer write_buggy_py(content). If you stop without a workspace write, return ONLY the full corrected contents of buggy.py as valid Python with no markdown or explanation."
        )
        if inputs.env_error:
            parts.append(f"RLM error: {inputs.env_error}")
        if inputs.last_test_result:
            parts.append("Most recent internal test feedback:")
            parts.append(self._format_internal_test_feedback(inputs.last_test_result))
        return "\n\n".join(part for part in parts if part)

    def _is_valid_python(self, candidate: str) -> bool:
        try:
            ast.parse(candidate)
        except SyntaxError:
            return False
        return True

    def _syntax_feedback(self, candidate: str) -> str:
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            line = exc.text.strip() if exc.text else ""
            return f"{exc.msg} at line {exc.lineno}, column {exc.offset}.\n{line}"
        return ""

    def _salvage_candidate_from_response(
        self,
        *,
        raw_response: str,
        original_code: str,
    ) -> tuple[str | None, str]:
        response_text = str(raw_response or "").strip()
        if not response_text:
            return None, "empty_response"

        saw_same_as_original = False
        saw_non_python = False
        saw_not_full_file = False
        for candidate in self._candidate_variants_from_response(response_text):
            if candidate.rstrip() == original_code.rstrip():
                saw_same_as_original = True
                continue
            if not self._is_valid_python(candidate):
                saw_non_python = True
                continue
            if not self._is_plausible_full_file_candidate(candidate):
                saw_not_full_file = True
                continue
            return candidate, "final_answer_salvage"

        if saw_same_as_original:
            return None, "response_matches_original_buggy_py"
        if saw_non_python:
            return None, "response_not_valid_python"
        if saw_not_full_file:
            return None, "response_not_salvageable"
        return None, "response_not_salvageable"

    def _candidate_variants_from_response(self, raw_response: str) -> list[str]:
        pending = [str(raw_response or "").strip()]
        seen: set[str] = set()
        variants: list[str] = []
        while pending:
            candidate = pending.pop(0).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            variants.append(candidate)
            pending.extend(self._extract_embedded_final_answer_wrappers(candidate))
            pending.extend(self._extract_embedded_code_fences(candidate))
            final_inner = self._unwrap_final_answer(candidate)
            if final_inner:
                pending.append(final_inner)
            fenced_inner = self._unwrap_code_fence(candidate)
            if fenced_inner:
                pending.append(fenced_inner)
        return variants

    def _extract_embedded_final_answer_wrappers(self, candidate: str) -> list[str]:
        wrappers: list[str] = []
        for match in re.finditer(r"(?m)^\s*(FINAL(?:_VAR)?)\(", candidate):
            extracted = self._extract_balanced_wrapper(candidate, start_idx=match.start())
            if extracted:
                wrappers.append(extracted)
        return wrappers

    def _extract_balanced_wrapper(self, candidate: str, *, start_idx: int) -> str | None:
        open_paren_idx = candidate.find("(", start_idx)
        if open_paren_idx < 0:
            return None
        depth = 0
        in_string: str | None = None
        triple_quoted = False
        idx = open_paren_idx
        while idx < len(candidate):
            if in_string is not None:
                if triple_quoted:
                    if candidate.startswith(in_string * 3, idx):
                        idx += 3
                        in_string = None
                        triple_quoted = False
                        continue
                    idx += 1
                    continue
                if candidate[idx] == "\\":
                    idx += 2
                    continue
                if candidate[idx] == in_string:
                    idx += 1
                    in_string = None
                    continue
                idx += 1
                continue
            if candidate.startswith("'''", idx) or candidate.startswith('"""', idx):
                in_string = candidate[idx]
                triple_quoted = True
                idx += 3
                continue
            if candidate[idx] in {"'", '"'}:
                in_string = candidate[idx]
                triple_quoted = False
                idx += 1
                continue
            if candidate[idx] == "#":
                newline_idx = candidate.find("\n", idx)
                if newline_idx < 0:
                    break
                idx = newline_idx + 1
                continue
            if candidate[idx] == "(":
                depth += 1
            elif candidate[idx] == ")":
                depth -= 1
                if depth == 0:
                    return candidate[start_idx : idx + 1].strip()
            idx += 1
        return None

    def _extract_embedded_code_fences(self, candidate: str) -> list[str]:
        fences: list[str] = []
        for match in re.finditer(r"```[^\n`]*\n(.*?)```", candidate, flags=re.DOTALL):
            inner = match.group(1).strip()
            if inner:
                fences.append(inner)
        return fences

    def _unwrap_final_answer(self, candidate: str) -> str | None:
        if candidate.startswith("FINAL(") and candidate.endswith(")"):
            return candidate[len("FINAL(") : -1].strip()
        return None

    def _unwrap_code_fence(self, candidate: str) -> str | None:
        if not candidate.startswith("```") or not candidate.endswith("```"):
            return None
        first_newline = candidate.find("\n")
        closing = candidate.rfind("```")
        if first_newline < 0 or closing <= first_newline:
            return None
        return candidate[first_newline + 1 : closing].strip()

    def _is_plausible_full_file_candidate(self, candidate: str) -> bool:
        try:
            module = ast.parse(candidate)
        except SyntaxError:
            return False
        if not module.body:
            return False
        # Single-expression files are only plausible repair payloads when the
        # expression is an actual call such as `print(...)`. Reject scalar
        # placeholders unwrapped from FINAL(...) like `done` or `"code"`.
        if len(module.body) == 1 and isinstance(module.body[0], ast.Expr):
            return self._is_plausible_expression_only_module(module.body[0].value)
        return True

    def _is_plausible_expression_only_module(self, expr: ast.expr) -> bool:
        return isinstance(expr, ast.Call) and not self._is_unresolved_final_answer_wrapper(expr)

    def _is_unresolved_final_answer_wrapper(self, expr: ast.expr) -> bool:
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id in {"FINAL", "FINAL_VAR"}
        )


def _normalize_file_rows(raw: Any) -> list[dict[str, object]]:
    file_rows: list[dict[str, object]] = []
    if not isinstance(raw, list):
        return file_rows
    for row in cast(list[object], raw):
        if not isinstance(row, dict):
            continue
        row_mapping = cast(dict[object, object], row)
        normalized_row: dict[str, object] = {}
        for key, value in row_mapping.items():
            normalized_row[str(key)] = value
        file_rows.append(normalized_row)
    return file_rows
