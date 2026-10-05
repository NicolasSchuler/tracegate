"""lambda-RLM raw baseline adapter."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from baselines._runtime import (
    _CandidateWorkspaceMixin,
    _char_totals,
    _env_failure_error,
    _nonnegative_int,
    _normalize_file_rows,
    _resolve_repo_commit,
    _RuntimeFeedbackMixin,
    _truncate,
    _usage_totals,
)
from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineConfig,
    BaselineResult,
    observed_repo_state,
)
from baselines.case_schemas import LambdaRLMCaseInput
from baselines.contracts import BaselineContract
from baselines.controller_packet import (
    parse_controller_packet,
    render_hybrid_controller_guidance,
)
from baselines.pins import get_expected_env_client_compatibility, get_pinned_repo_commit
from llmdebug import _json as json

from . import env_client as lambda_env_client
from .case_converter import convert_case

_ENV_CLIENT_SCRIPT = Path(__file__).with_name("env_client.py")
_FENCED_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CANONICAL_PROMPT_CONTRACT = "canonical"
_REPAIR_FAIR_PROMPT_CONTRACT = "repair_fair_v1"


_TERMINAL_ENV_ERRORS = frozenset({"openai_compat_reasoning_only", "timeout", "env_client_failed"})


class LambdaRLMAdapter(_RuntimeFeedbackMixin, _CandidateWorkspaceMixin, BaselineAdapter):
    """Adapter for a raw lambda-RLM repair baseline."""

    retry_context_prompt_field = "prompt_context"

    honours_generation_contract = True

    default_model: str = "gpt-4o-mini"
    _config: BaselineConfig | None = None
    repo_commit: str = ""

    @property
    def name(self) -> str:
        return "lambda_rlm"

    def contract(self) -> BaselineContract:
        return BaselineContract(
            baseline_name=self.name,
            upstream_commit=get_pinned_repo_commit(self.name),
            fidelity_level="generic_api_adaptation",
            repair_scope_policy="single_file_only",
            prompt_contract_kind=self._prompt_contract_kind(),
            output_contract="single_file_repaired_source",
            accepted_case_shapes=(
                "single_file_buggy_py",
                "workspace_tests_available",
                "native_pytest_command_available",
            ),
            patch_scope_summary=("env_shim", "telemetry"),
            notes=("Adapts generic lambda-RLM prompt completion to single-file code repair.",),
        )

    def setup(self, config: BaselineConfig) -> None:
        self._config = config
        repo = config.repo_dir
        if not repo.is_dir():
            raise RuntimeError(
                f"lambda-RLM repo not found at {repo}. Run baselines/setup_repos.sh first."
            )
        required = (
            repo / "pyproject.toml",
            repo / "rlm" / "__init__.py",
            repo / "rlm" / "lambda_rlm.py",
            repo / "rlm" / "clients" / "openai.py",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            rendered = ", ".join(str(path.relative_to(repo)) for path in missing)
            raise RuntimeError(f"lambda-RLM repo missing required file(s): {rendered}")
        if config.python_executable is not None and not config.python_executable.is_file():
            raise RuntimeError(
                f"lambda-RLM python executable not found: {config.python_executable}"
            )
        self.repo_commit = _resolve_repo_commit(repo)
        self.repo_state = observed_repo_state(repo)
        expected_compatibility = get_expected_env_client_compatibility(self.name)
        declared_compatibility = str(
            getattr(lambda_env_client, "_COMPATIBLE_WITH", "") or ""
        ).strip()
        if declared_compatibility != expected_compatibility:
            raise RuntimeError(
                "lambda-RLM env client compatibility declaration is stale: "
                f"expected {expected_compatibility}, got {declared_compatibility or '<empty>'}. "
                "Update baselines/pins.py or the env_client shim before running evals."
            )
        expected_repo_commit = get_pinned_repo_commit(self.name)
        if self.repo_commit and self.repo_commit != expected_repo_commit:
            raise RuntimeError(
                "lambda-RLM repo commit does not match the env-client compatibility pin: "
                f"expected {expected_repo_commit}, got {self.repo_commit}. "
                "Re-run baselines/setup_repos.sh or update the shared pin if the upstream pin changed."
            )

    def validate_extra_config(self, config: BaselineConfig) -> None:
        from baselines.extra_configs import LambdaRLMExtraConfig

        LambdaRLMExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def _validated_extra_config(self, config: BaselineConfig):
        from baselines.extra_configs import LambdaRLMExtraConfig

        return LambdaRLMExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def _prompt_contract(self, config: BaselineConfig | None = None) -> str:
        resolved = config or self._config
        if resolved is None:
            return _CANONICAL_PROMPT_CONTRACT
        value = str(resolved.extra.get("baseline_lambda_prompt_contract") or "").strip()
        return value or _CANONICAL_PROMPT_CONTRACT

    def _prompt_contract_kind(self) -> str:
        prompt_contract = self._prompt_contract()
        if prompt_contract == _REPAIR_FAIR_PROMPT_CONTRACT:
            return "adapter_defined_code_repair_prompt__repair_fair_v1"
        return "adapter_defined_code_repair_prompt"

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> LambdaRLMCaseInput:
        return convert_case(case_dir, case_context=case_context)

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
        prompt_contract = extra_config.baseline_lambda_prompt_contract
        controller_guidance = render_hybrid_controller_guidance(
            controller_packet,
            mode=extra_config.baseline_hybrid_guidance_mode,
            include_history=True,
            include_retry_plan=True,
            max_chars=extra_config.baseline_lambda_controller_guidance_max_chars,
        ).text
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
        internal_task_detection_call_count = 0
        internal_repl_llm_query_call_count = 0
        internal_length_finish_count = 0
        internal_compat_recovery_count = 0
        internal_planned_decomposition_attempt_count = 0
        internal_executed_decomposition_attempt_count = 0
        last_candidate: str | None = None
        last_plan_metadata: dict[str, Any] | None = None
        last_error: str | None = None
        started_at = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="lambda_rlm_") as tmpdir:
            work_dir = Path(tmpdir)
            self._materialize_workspace(
                work_dir=work_dir,
                file_rows=_normalize_file_rows(case_input.get("files")),
            )
            self._emit_runtime_progress(
                phase_label="baseline: lambda_rlm precheck",
                note="initial pytest precheck",
                attempts_completed=0,
                llm_calls_observed=accumulated_llm_calls,
                eval_duration_sec_observed=eval_duration_sec,
            )
            precheck = self._run_case_tests(
                work_dir=work_dir,
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
                    internal_task_detection_call_count=0,
                    internal_repl_llm_query_call_count=0,
                    internal_length_finish_count=0,
                    internal_compat_recovery_count=0,
                    internal_planned_decomposition_attempt_count=0,
                    internal_executed_decomposition_attempt_count=0,
                    started_at=started_at,
                    stopped_reason="case_already_passed",
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
                self._emit_runtime_progress(
                    phase_label="baseline: lambda_rlm generation",
                    note=f"attempt {attempt_idx}/{config.max_attempts} generating",
                    attempts_completed=attempt_idx - 1,
                    llm_calls_observed=accumulated_llm_calls,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                prompt = self._build_prompt(
                    prompt_context=prompt_context,
                    failure_feedback=retry_feedback,
                    controller_guidance=controller_guidance,
                    attempt_idx=attempt_idx,
                    prompt_contract=prompt_contract,
                    failure_feedback_max_chars=extra_config.baseline_lambda_failure_feedback_max_chars,
                )
                prompt_chars_total += len(prompt)
                try:
                    env_result = self._invoke_lambda(
                        prompt=prompt,
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
                        internal_task_detection_call_count=internal_task_detection_call_count,
                        internal_repl_llm_query_call_count=internal_repl_llm_query_call_count,
                        internal_length_finish_count=internal_length_finish_count,
                        internal_compat_recovery_count=internal_compat_recovery_count,
                        internal_planned_decomposition_attempt_count=(
                            internal_planned_decomposition_attempt_count
                        ),
                        internal_executed_decomposition_attempt_count=(
                            internal_executed_decomposition_attempt_count
                        ),
                        started_at=started_at,
                        stopped_reason=env_error,
                    )
                    return BaselineResult(
                        repaired_code=last_candidate,
                        success=False,
                        iterations_used=max(0, attempt_idx - 1),
                        token_usage={
                            "prompt_tokens": accumulated_prompt_tokens,
                            "completion_tokens": accumulated_completion_tokens,
                            "total_tokens": accumulated_prompt_tokens
                            + accumulated_completion_tokens,
                        },
                        meta={
                            "attempts": attempt_summaries,
                            "metrics": metrics,
                            "last_plan_metadata": last_plan_metadata,
                            "error": env_error,
                            "compat_message": str(env_result.get("compat_message") or ""),
                            "compat_meta": dict(
                                cast(dict[str, Any], env_result.get("compat_meta") or {})
                            ),
                        },
                    )
                llm_duration_sec += env_result["execution_time"]
                completion_chars_total += len(env_result["raw_response"])
                response_text = env_result["raw_response"]
                usage_summary = env_result["usage_summary"]
                char_summary = env_result["char_summary"]
                plan_metadata = env_result["plan_metadata"]
                call_summary = env_result["call_summary"]
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
                task_detection_call_count = (
                    _nonnegative_int(call_summary.get("task_detection_call_count")) or 0
                )
                repl_llm_query_call_count = (
                    _nonnegative_int(call_summary.get("repl_llm_query_call_count")) or 0
                )
                call_length_finish_count = (
                    _nonnegative_int(call_summary.get("call_length_finish_count")) or 0
                )
                compat_recovery_count = (
                    _nonnegative_int(call_summary.get("compat_recovery_count")) or 0
                )
                planned_decomposition = bool(call_summary.get("planned_decomposition", False))
                executed_decomposition = bool(call_summary.get("executed_decomposition", False))
                internal_task_detection_call_count += task_detection_call_count
                internal_repl_llm_query_call_count += repl_llm_query_call_count
                internal_length_finish_count += call_length_finish_count
                internal_compat_recovery_count += compat_recovery_count
                internal_planned_decomposition_attempt_count += int(planned_decomposition)
                internal_executed_decomposition_attempt_count += int(executed_decomposition)
                attempt_summary: dict[str, Any] = {
                    "attempt": attempt_idx,
                    "llm_call_count": llm_calls_attempt,
                    "execution_time_sec": env_result["execution_time"],
                    "task_detection_call_count": task_detection_call_count,
                    "repl_llm_query_call_count": repl_llm_query_call_count,
                    "call_length_finish_count": call_length_finish_count,
                    "compat_recovery_count": compat_recovery_count,
                    "planned_decomposition": planned_decomposition,
                    "executed_decomposition": executed_decomposition,
                }
                if plan_metadata:
                    attempt_summary["plan_metadata"] = plan_metadata
                    last_plan_metadata = dict(plan_metadata)
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
                for key in ("planned_depth", "k_star", "tau_star"):
                    value = _nonnegative_int(call_summary.get(key))
                    if value is not None:
                        attempt_summary[key] = value
                task_type = str(call_summary.get("task_type") or "")
                if task_type:
                    attempt_summary["task_type"] = task_type
                candidate = self._extract_candidate_code(response_text)
                attempt_summary["raw_response_chars"] = len(response_text)
                attempt_summary["candidate_chars"] = len(candidate)
                if not candidate.strip():
                    retry_feedback = (
                        "Previous attempt returned no candidate code. "
                        "Return ONLY the full corrected Python source code for buggy.py. "
                        "Do not explain the bug. Do not add commentary. Just output the code."
                    )
                    attempt_summary["candidate_status"] = "empty_response"
                    attempt_summaries.append(attempt_summary)
                    self._emit_runtime_progress(
                        phase_label="baseline: lambda_rlm generation",
                        note=f"attempt {attempt_idx} returned no candidate",
                        attempts_completed=attempt_idx,
                        llm_calls_observed=accumulated_llm_calls,
                        eval_duration_sec_observed=eval_duration_sec,
                    )
                    last_error = "empty_response"
                    continue
                if not self._is_valid_python(candidate):
                    syntax_feedback = self._syntax_feedback(candidate)
                    retry_feedback = (
                        "Previous attempt returned invalid Python for buggy.py. "
                        "You MUST return ONLY valid Python source code, nothing else.\n\n"
                        + syntax_feedback
                    )
                    attempt_summary["candidate_status"] = "syntax_error"
                    attempt_summary["syntax_feedback"] = syntax_feedback
                    attempt_summaries.append(attempt_summary)
                    self._emit_runtime_progress(
                        phase_label="baseline: lambda_rlm generation",
                        note=f"attempt {attempt_idx} produced invalid python",
                        attempts_completed=attempt_idx,
                        llm_calls_observed=accumulated_llm_calls,
                        eval_duration_sec_observed=eval_duration_sec,
                    )
                    last_candidate = candidate
                    last_error = "invalid_python"
                    continue
                last_candidate = candidate
                (work_dir / "buggy.py").write_text(candidate, encoding="utf-8")
                self._emit_runtime_progress(
                    phase_label="baseline: lambda_rlm postcheck",
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
                    attempt_summaries.append(attempt_summary)
                    self._emit_runtime_progress(
                        phase_label="baseline: lambda_rlm postcheck",
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
                        internal_task_detection_call_count=internal_task_detection_call_count,
                        internal_repl_llm_query_call_count=internal_repl_llm_query_call_count,
                        internal_length_finish_count=internal_length_finish_count,
                        internal_compat_recovery_count=internal_compat_recovery_count,
                        internal_planned_decomposition_attempt_count=(
                            internal_planned_decomposition_attempt_count
                        ),
                        internal_executed_decomposition_attempt_count=(
                            internal_executed_decomposition_attempt_count
                        ),
                        started_at=started_at,
                        stopped_reason="passed_tests",
                    )
                    metrics["direct_gen_passed"] = attempt_idx == 1
                    metrics["debug_added_value"] = attempt_idx > 1
                    return BaselineResult(
                        repaired_code=candidate,
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
                            "last_plan_metadata": last_plan_metadata,
                        },
                    )
                retry_feedback = self._format_failure_feedback(test_outcome)
                attempt_summary["candidate_status"] = "tests_still_failing"
                attempt_summary["failure_feedback"] = _truncate(
                    retry_feedback,
                    max_chars=extra_config.baseline_lambda_failure_feedback_meta_max_chars,
                )
                attempt_summaries.append(attempt_summary)
                self._emit_runtime_progress(
                    phase_label="baseline: lambda_rlm postcheck",
                    note=f"attempt {attempt_idx} tests still failing",
                    attempts_completed=attempt_idx,
                    llm_calls_observed=accumulated_llm_calls,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                last_error = "tests_still_failing"

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
                internal_task_detection_call_count=internal_task_detection_call_count,
                internal_repl_llm_query_call_count=internal_repl_llm_query_call_count,
                internal_length_finish_count=internal_length_finish_count,
                internal_compat_recovery_count=internal_compat_recovery_count,
                internal_planned_decomposition_attempt_count=(
                    internal_planned_decomposition_attempt_count
                ),
                internal_executed_decomposition_attempt_count=(
                    internal_executed_decomposition_attempt_count
                ),
                started_at=started_at,
                stopped_reason=last_error or "attempt_budget_exhausted",
            )
            # Validate the last candidate before returning — never return
            # prose or invalid Python as repaired_code.
            final_code: str | None = last_candidate
            if final_code is not None:
                try:
                    ast.parse(final_code)
                except SyntaxError:
                    final_code = None
            return BaselineResult(
                repaired_code=final_code,
                success=False,
                iterations_used=len(attempt_summaries),
                token_usage={
                    "prompt_tokens": accumulated_prompt_tokens,
                    "completion_tokens": accumulated_completion_tokens,
                    "total_tokens": accumulated_prompt_tokens + accumulated_completion_tokens,
                },
                meta={
                    "attempts": attempt_summaries,
                    "metrics": metrics,
                    "last_plan_metadata": last_plan_metadata,
                    "error": last_error or "attempt_budget_exhausted",
                },
            )

    def _build_prompt(
        self,
        *,
        prompt_context: str,
        failure_feedback: str,
        controller_guidance: str,
        attempt_idx: int,
        prompt_contract: str = _CANONICAL_PROMPT_CONTRACT,
        failure_feedback_max_chars: int = 3500,
    ) -> str:
        question_lines = [
            "Return ONLY the full corrected Python source code for buggy.py.",
            "Your response must be valid Python that can replace the buggy.py file.",
            "Do NOT include explanations, comments about what you changed, or any other text.",
            "Keep the fix minimal and preserve unrelated behavior.",
        ]
        if prompt_contract == _REPAIR_FAIR_PROMPT_CONTRACT:
            question_lines.extend(
                [
                    "Prefer the smallest plausible direct repair immediately.",
                    "Do not spend output budget restating the task, explaining the bug, or writing analysis.",
                ]
            )
        if attempt_idx > 1:
            question_lines.append(
                f"This is outer repair attempt {attempt_idx}. Do not repeat the previous failure."
            )
            if prompt_contract == _REPAIR_FAIR_PROMPT_CONTRACT:
                question_lines.append(
                    "Revise the previous fix directly using the failure feedback below."
                )
        if failure_feedback:
            question_lines.append("Failure feedback from the previous attempt/test run:")
            question_lines.append(_truncate(failure_feedback, max_chars=failure_feedback_max_chars))
        if controller_guidance:
            question_lines.append("Adaptive controller guidance:")
            question_lines.append(controller_guidance)
        question = "\n\n".join(line for line in question_lines if line.strip())
        return f"Context:\n{prompt_context}\n\nQuestion: {question}\n\nAnswer:"

    def _invoke_lambda(
        self,
        *,
        prompt: str,
        config: BaselineConfig,
        timeout_sec: int,
    ) -> dict[str, Any]:
        repo_dir = config.repo_dir.expanduser().resolve()
        python_executable = config.python_executable or Path(sys.executable)
        payload: dict[str, Any] = {
            "repo_dir": str(repo_dir),
            "model": str(config.model or ""),
            "prompt": prompt,
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
            "context_window_chars": int(
                config.extra.get("baseline_lambda_context_window_chars") or 100_000
            ),
            "accuracy_target": float(config.extra.get("baseline_lambda_accuracy_target") or 0.80),
            "a_leaf": float(config.extra.get("baseline_lambda_a_leaf") or 0.95),
            "a_compose": float(config.extra.get("baseline_lambda_a_compose") or 0.90),
            "verbose": bool(config.extra.get("baseline_lambda_verbose", False)),
        }
        force_task_type = config.extra.get("baseline_lambda_force_task_type")
        if force_task_type:
            payload["force_task_type"] = str(force_task_type)
        # D4: the shared generation contract unless the baseline-specific knob is set.
        if config.generation.temperature is not None:
            payload["temperature"] = float(config.generation.temperature)
        if config.generation.max_completion_tokens is not None:
            payload["max_tokens"] = int(config.generation.max_completion_tokens)
        for key in (
            "baseline_lambda_temperature",
            "baseline_lambda_top_p",
            "baseline_lambda_max_tokens",
            "baseline_lambda_stream",
        ):
            value = config.extra.get(key)
            if value is not None:
                payload[key.removeprefix("baseline_lambda_")] = value
        with tempfile.TemporaryDirectory(prefix="lambda_rlm_env_") as tmpdir:
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
                raise RuntimeError(f"lambda-RLM env client failed: {detail}")
            raw_payload = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, dict):
            raise RuntimeError(f"lambda-RLM env client returned {type(raw_payload).__name__}")
        payload_dict = cast(dict[str, Any], raw_payload)
        usage_summary = payload_dict.get("usage_summary")
        char_summary = payload_dict.get("char_summary")
        plan_metadata = payload_dict.get("plan_metadata")
        call_summary = payload_dict.get("call_summary")
        return {
            "error": str(payload_dict.get("error") or ""),
            "compat_message": str(payload_dict.get("compat_message") or ""),
            "compat_meta": payload_dict.get("compat_meta")
            if isinstance(payload_dict.get("compat_meta"), dict)
            else {},
            "raw_response": str(payload_dict.get("response") or ""),
            "execution_time": float(payload_dict.get("execution_time") or 0.0),
            "usage_summary": usage_summary if isinstance(usage_summary, dict) else {},
            "char_summary": char_summary if isinstance(char_summary, dict) else {},
            "plan_metadata": plan_metadata if isinstance(plan_metadata, dict) else {},
            "call_summary": call_summary if isinstance(call_summary, dict) else {},
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
        internal_task_detection_call_count: int,
        internal_repl_llm_query_call_count: int,
        internal_length_finish_count: int,
        internal_compat_recovery_count: int,
        internal_planned_decomposition_attempt_count: int,
        internal_executed_decomposition_attempt_count: int,
        started_at: float,
        stopped_reason: str,
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
            "usage_source": "lambda_rlm_internal_summary",
            "stopped_reason": str(stopped_reason or "unknown"),
            "method_instrumentation": False,
            "method_two_step_repair": False,
            "internal_task_detection_call_count": max(0, int(internal_task_detection_call_count)),
            "internal_repl_llm_query_call_count": max(0, int(internal_repl_llm_query_call_count)),
            "internal_length_finish_count": max(0, int(internal_length_finish_count)),
            "internal_compat_recovery_count": max(0, int(internal_compat_recovery_count)),
            "internal_planned_decomposition_attempt_count": max(
                0, int(internal_planned_decomposition_attempt_count)
            ),
            "internal_executed_decomposition_attempt_count": max(
                0, int(internal_executed_decomposition_attempt_count)
            ),
        }
        return metrics

    def _generation_timeout(self, *, config: BaselineConfig) -> int:
        process_timeout = max(1, int(config.process_timeout_sec or 1.0))
        raw_timeout = config.extra.get("baseline_lambda_timeout_sec")
        if isinstance(raw_timeout, (int, float)) and float(raw_timeout) > 0:
            return min(max(1, int(float(raw_timeout))), process_timeout)
        return process_timeout

    def _extract_candidate_code(self, response_text: str) -> str:
        matches = list(_FENCED_CODE_RE.finditer(response_text))
        if matches:
            return matches[-1].group(1).strip()
        stripped = str(response_text or "").strip()
        if stripped.lower().startswith("buggy.py"):
            _, _, remainder = stripped.partition("\n")
            return remainder.strip()
        return stripped
