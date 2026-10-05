"""INTERVENOR baseline adapter.

This bridge keeps INTERVENOR's teacher/learner chain-of-repair idea and its
official prompt assets, but runs the loop through adapter-owned orchestration
so it can execute llmdebug benchmark cases in a temporary workspace with the
case's native test command.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from baselines._runtime import (
    _coerce_subprocess_output,
    _CommandOutcome,
    _resolve_repo_commit,
    _RuntimeFeedbackMixin,
)
from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineConfig,
    BaselineResult,
    observed_repo_state,
    require_pinned_commit,
)
from baselines.case_schemas import IntervenorCaseInput
from baselines.contracts import BaselineContract
from baselines.controller_packet import (
    HYBRID_GUIDANCE_MODE_FAIR_V1,
    HYBRID_GUIDANCE_MODE_FIXED_V1,
    ControllerPacketV1,
    parse_controller_packet,
    render_hybrid_controller_guidance,
)
from baselines.openai_compat import (
    OpenAICompatResponseError,
    apply_compat_request_profile,
    compat_config_from_extra,
    compat_payload_fields,
    normalize_chat_completion_response,
    require_final_content,
    usage_include_extra_body,
)
from baselines.pins import get_pinned_repo_commit

from .case_converter import convert_case

_ENV_CLIENT_SCRIPT = Path(__file__).with_name("env_client.py")


_FENCED_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_python_code(text: str) -> str | None:
    """Extract valid Python code from an LLM response.

    Strategy:
    1. Prefer fenced code blocks and return the first syntactically valid one.
    2. If no valid fenced block exists, still return the most plausible fenced
       candidate so local tests can surface SyntaxError feedback on the next
       repair turn.
    3. If no fenced blocks exist, strip outer fences and fall back to the whole
       text, again preferring valid Python but preserving syntax-broken code.
    """
    # Try fenced code blocks first (handles prose wrapping and thinking tokens).
    matches = list(_FENCED_CODE_RE.finditer(text))
    if matches:
        fallback_candidate: str | None = None
        for m in reversed(matches):
            candidate = m.group(1).strip()
            if not candidate:
                continue
            if fallback_candidate is None:
                fallback_candidate = candidate
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                continue
        if fallback_candidate:
            return fallback_candidate

    # Fallback: strip outer fences and try the whole text.
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    candidate = "\n".join(lines).strip()
    if candidate:
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            return candidate

    return None


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    source: dict[str, Any]
    if isinstance(usage, dict):
        usage_mapping = cast(dict[object, object], usage)
        source = {}
        for key, value in usage_mapping.items():
            source[str(key)] = value
    else:
        source = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "total_input_tokens": getattr(usage, "total_input_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_output_tokens": getattr(usage, "total_output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    result: dict[str, int] = {}
    prompt_tokens = _usage_int(
        source,
        "prompt_tokens",
        "input_tokens",
        "total_input_tokens",
    )
    completion_tokens = _usage_int(
        source,
        "completion_tokens",
        "output_tokens",
        "total_output_tokens",
    )
    total_tokens = _usage_int(source, "total_tokens")
    if isinstance(prompt_tokens, int):
        result["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        result["completion_tokens"] = completion_tokens
    if isinstance(total_tokens, int):
        result["total_tokens"] = total_tokens
    elif isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        result["total_tokens"] = prompt_tokens + completion_tokens
    return result


def _usage_int(source: dict[str, Any], *keys: str) -> int | None:
    fallback_zero: int | None = None
    for key in keys:
        value = source.get(key)
        if isinstance(value, int):
            if value > 0:
                return value
            fallback_zero = 0
    return fallback_zero


def _accumulate_usage(total: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    merged = dict(total)
    for key, value in delta.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _truncate_guidance_text(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max(0, int(max_chars)):
        return value
    if max_chars <= 3:
        return value[: max(0, int(max_chars))]
    return value[: max(0, int(max_chars)) - 3].rstrip() + "..."


def _normalize_generation_result(
    value: tuple[str, dict[str, int]]
    | tuple[str, dict[str, int], float]
    | tuple[str, dict[str, int], float, dict[str, Any]],
) -> tuple[str, dict[str, int], float, dict[str, Any]]:
    meta: dict[str, Any] = {}
    if len(value) == 4:
        text, usage, duration_sec, raw_meta = value
        meta = dict(raw_meta)
        return text, _usage_dict(usage), max(0.0, float(duration_sec)), meta
    if len(value) == 3:
        text, usage, duration_sec = value
        return text, _usage_dict(usage), max(0.0, float(duration_sec)), meta
    text, usage = value
    return text, _usage_dict(usage), 0.0, meta


def _attempt_metrics(attempt_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_duration_sec_total = 0.0
    learner_duration_sec_total = 0.0
    postcheck_duration_sec_total = 0.0
    learner_empty_response_count = 0
    attempts_with_postcheck_count = 0
    teacher_finish_reason_length_count = 0
    learner_finish_reason_length_count = 0
    teacher_finish_reason_known_count = 0
    learner_finish_reason_known_count = 0
    for attempt in attempt_summaries:
        teacher_raw = attempt.get("teacher_duration_sec")
        if isinstance(teacher_raw, (int, float)):
            teacher_duration_sec_total += max(0.0, float(teacher_raw))
        learner_raw = attempt.get("learner_duration_sec")
        if isinstance(learner_raw, (int, float)):
            learner_duration_sec_total += max(0.0, float(learner_raw))
        postcheck_raw = attempt.get("postcheck_duration_sec")
        if isinstance(postcheck_raw, (int, float)):
            postcheck_duration_sec_total += max(0.0, float(postcheck_raw))
        if bool(attempt.get("learner_response_empty")):
            learner_empty_response_count += 1
        if "postcheck_returncode" in attempt:
            attempts_with_postcheck_count += 1
        teacher_finish_reason = str(attempt.get("teacher_finish_reason") or "").strip()
        if teacher_finish_reason:
            teacher_finish_reason_known_count += 1
            if teacher_finish_reason == "length":
                teacher_finish_reason_length_count += 1
        learner_finish_reason = str(attempt.get("learner_finish_reason") or "").strip()
        if learner_finish_reason:
            learner_finish_reason_known_count += 1
            if learner_finish_reason == "length":
                learner_finish_reason_length_count += 1
    return {
        "teacher_duration_sec_total": teacher_duration_sec_total,
        "learner_duration_sec_total": learner_duration_sec_total,
        "postcheck_duration_sec_total": postcheck_duration_sec_total,
        "learner_empty_response_count": learner_empty_response_count,
        "attempts_with_postcheck_count": attempts_with_postcheck_count,
        "teacher_finish_reason_length_count": teacher_finish_reason_length_count,
        "learner_finish_reason_length_count": learner_finish_reason_length_count,
        "teacher_finish_reason_known_count": teacher_finish_reason_known_count,
        "learner_finish_reason_known_count": learner_finish_reason_known_count,
    }


def _response_finish_reason(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        first_choice = cast(object, choices[0])
        value = getattr(first_choice, "finish_reason", None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            dumped_mapping = cast(dict[str, Any], dumped)
            raw_choices = dumped_mapping.get("choices")
            if isinstance(raw_choices, list) and raw_choices:
                choice = cast(object, raw_choices[0])
                if isinstance(choice, dict):
                    choice_mapping = cast(dict[str, Any], choice)
                    value = str(choice_mapping.get("finish_reason") or "").strip()
                    if value:
                        return value
    return ""


def _reported_token_metrics(accumulated_usage: dict[str, int]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for source_key, target_key in (
        ("prompt_tokens", "reported_tokens_prompt"),
        ("completion_tokens", "reported_tokens_completion"),
        ("total_tokens", "reported_tokens_total"),
    ):
        value = accumulated_usage.get(source_key)
        if isinstance(value, int) and value >= 0:
            metrics[target_key] = value
    return metrics


def _reported_char_metrics(
    *, prompt_chars_total: int, completion_chars_total: int
) -> dict[str, int]:
    normalized_prompt = max(0, int(prompt_chars_total))
    normalized_completion = max(0, int(completion_chars_total))
    normalized_total = normalized_prompt + normalized_completion
    return {
        "prompt_chars_total": normalized_prompt,
        "completion_chars_total": normalized_completion,
        "chars_total": normalized_total,
        "reported_chars_prompt": normalized_prompt,
        "reported_chars_completion": normalized_completion,
        "reported_chars_total": normalized_total,
    }


def _keep_progress_on_generation_failure(
    *,
    exc: BaseException,
    current_code: str,
    original_code: str,
    failure: BaselineResult,
) -> BaselineResult:
    """BAS-10: a failed model call keeps earlier attempts' candidate and usage.

    Compatibility errors keep their ``openai_compat_reasoning_only`` label; anything else
    (timeouts, transport errors) is recorded as ``generation_failed`` with the exception.
    """
    candidate = current_code if current_code != original_code else None
    if isinstance(exc, OpenAICompatResponseError):
        return replace(failure, repaired_code=candidate)
    meta = dict(failure.meta)
    meta.pop("compat_message", None)
    meta.pop("compat_meta", None)
    meta["error"] = "generation_failed"
    meta["generation_exc"] = f"{type(exc).__name__}: {exc}"
    return replace(failure, repaired_code=candidate, meta=meta)


def _generation_temperature(config: BaselineConfig | None) -> float:
    """Shared D4 temperature; INTERVENOR's own default is 0.0 (BAS-03)."""
    if config is None or config.generation.temperature is None:
        return 0.0
    return float(config.generation.temperature)


def _generation_max_tokens(config: BaselineConfig | None, adapter_default: int) -> int:
    """Shared D4 completion cap when set, else the teacher/learner default (BAS-04)."""
    if config is None or config.generation.max_completion_tokens is None:
        return int(adapter_default)
    return int(config.generation.max_completion_tokens)


def _generation_seed_kwargs(config: BaselineConfig | None) -> dict[str, Any]:
    if config is None or config.generation.seed is None:
        return {}
    return {"seed": int(config.generation.seed)}


class IntervenorAdapter(_RuntimeFeedbackMixin, BaselineAdapter):
    """Adapter for the INTERVENOR teacher-learner debugging baseline."""

    retry_context_prompt_field = "prompt"

    honours_generation_contract = True

    default_model: str = "gpt-4o-mini"
    _config: BaselineConfig | None = None
    _teacher_prompt_template: str = ""
    _learner_prompt_template: str = ""
    repo_commit: str = ""

    @property
    def name(self) -> str:
        return "intervenor"

    def contract(self) -> BaselineContract:
        return BaselineContract(
            baseline_name=self.name,
            upstream_commit=get_pinned_repo_commit(self.name),
            fidelity_level="prompt_asset_port",
            repair_scope_policy="single_file_only",
            prompt_contract_kind="upstream_prompt_assets_plus_adapter_instructions",
            output_contract="single_file_repaired_source",
            accepted_case_shapes=(
                "single_file_buggy_py",
                "workspace_tests_available",
                "native_pytest_command_available",
            ),
            patch_scope_summary=("adapter_orchestration",),
            notes=(
                "Uses upstream teacher/learner prompt assets with adapter-owned orchestration.",
            ),
        )

    def setup(self, config: BaselineConfig) -> None:
        """Validate the repo clone and load the official prompt assets."""
        self._config = config
        repo = config.repo_dir

        if not repo.is_dir():
            msg = f"INTERVENOR repo not found at {repo}. Run baselines/setup_repos.sh first."
            raise RuntimeError(msg)
        teacher_prompt_path = repo / "prompt" / "code_teacher.txt"
        learner_prompt_path = repo / "prompt" / "code_learner.txt"
        missing = [
            path for path in (teacher_prompt_path, learner_prompt_path) if not path.is_file()
        ]
        if missing:
            rendered = ", ".join(str(path.relative_to(repo)) for path in missing)
            raise RuntimeError(f"INTERVENOR repo missing required prompt asset(s): {rendered}")

        self._teacher_prompt_template = teacher_prompt_path.read_text(encoding="utf-8")
        self._learner_prompt_template = learner_prompt_path.read_text(encoding="utf-8")
        self.repo_commit = _resolve_repo_commit(repo)
        self.repo_state = observed_repo_state(repo)
        require_pinned_commit(
            name=self.name,
            observed_commit=self.repo_commit,
            pinned_commit=get_pinned_repo_commit(self.name),
        )

    def validate_extra_config(self, config: BaselineConfig) -> None:
        from baselines.extra_configs import IntervenorExtraConfig

        IntervenorExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def _validated_extra_config(self, config: BaselineConfig):
        from baselines.extra_configs import IntervenorExtraConfig

        return IntervenorExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> IntervenorCaseInput:
        """Convert llmdebug case → INTERVENOR adapter input."""
        return convert_case(case_dir, case_context=case_context)

    def _final_metrics(
        self,
        *,
        llm_call_count: int,
        llm_duration_sec: float,
        eval_duration_sec: float,
        started_at: float,
        usage_available: bool,
        accumulated_usage: dict[str, int],
        usage_reported_call_count: int,
        usage_missing_call_count: int,
        prompt_chars_total: int,
        completion_chars_total: int,
        attempt_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metrics = self._build_metrics(
            llm_call_count=llm_call_count,
            llm_duration_sec=llm_duration_sec,
            eval_duration_sec=eval_duration_sec,
            internal_total_duration_sec=time.perf_counter() - started_at,
            usage_available=usage_available and llm_call_count > 0 and bool(accumulated_usage),
            usage_reported_call_count=usage_reported_call_count,
            usage_missing_call_count=usage_missing_call_count,
            accumulated_usage=accumulated_usage,
            prompt_chars_total=prompt_chars_total,
            completion_chars_total=completion_chars_total,
        )
        metrics.update(_attempt_metrics(attempt_summaries))
        return metrics

    def run_repair(
        self,
        case_input: dict[str, Any],
        config: BaselineConfig,
    ) -> BaselineResult:
        """Run INTERVENOR's CoR loop with repo-sourced prompts and local tests."""
        if not self._teacher_prompt_template or not self._learner_prompt_template:
            raise RuntimeError(
                "INTERVENOR adapter is not set up. Call setup() before run_repair()."
            )

        python_args = [str(arg) for arg in case_input.get("python_args", []) if str(arg)]
        if not python_args:
            return BaselineResult(
                repaired_code=None,
                success=False,
                meta={"error": "missing_python_args"},
            )

        client = None if config.python_executable is not None else self._build_client(config)
        original_code = str(case_input.get("buggy_code", ""))
        current_code = original_code
        accumulated_usage: dict[str, int] = {}
        usage_available = True
        usage_reported_call_count = 0
        usage_missing_call_count = 0
        llm_call_count = 0
        llm_duration_sec = 0.0
        eval_duration_sec = 0.0
        prompt_chars_total = 0
        completion_chars_total = 0
        attempt_summaries: list[dict[str, Any]] = []
        extra_config = self._validated_extra_config(config)
        test_timeout_sec = self._case_timeout(case_input=case_input, config=config)
        started_at = time.perf_counter()
        controller_packet = parse_controller_packet(case_input.get("llmdebug_controller_packet"))
        if extra_config.baseline_hybrid_guidance_mode == HYBRID_GUIDANCE_MODE_FAIR_V1:
            canonical_guidance = render_hybrid_controller_guidance(
                controller_packet,
                mode=extra_config.baseline_hybrid_guidance_mode,
                include_history=True,
                include_retry_plan=True,
            ).text
            teacher_guidance = canonical_guidance
            learner_guidance = canonical_guidance
        elif extra_config.baseline_hybrid_guidance_mode == HYBRID_GUIDANCE_MODE_FIXED_V1:
            shared_guidance = render_hybrid_controller_guidance(
                controller_packet,
                mode=extra_config.baseline_hybrid_guidance_mode,
                include_history=False,
                include_retry_plan=False,
                max_chars=min(
                    extra_config.baseline_intervenor_teacher_guidance_max_chars,
                    extra_config.baseline_intervenor_learner_guidance_max_chars,
                ),
            ).text
            teacher_guidance = shared_guidance
            learner_guidance = shared_guidance
        else:
            teacher_guidance = self._render_controller_guidance(
                controller_packet=controller_packet,
                focus="teacher",
                max_chars=extra_config.baseline_intervenor_teacher_guidance_max_chars,
            )
            learner_guidance = self._render_controller_guidance(
                controller_packet=controller_packet,
                focus="learner",
                max_chars=extra_config.baseline_intervenor_learner_guidance_max_chars,
            )

        with tempfile.TemporaryDirectory(prefix="intervenor_") as tmpdir:
            work_dir = Path(tmpdir)
            raw_file_rows = case_input.get("files", [])
            file_rows: list[dict[str, object]] = []
            if isinstance(raw_file_rows, list):
                for row in cast(list[object], raw_file_rows):
                    if isinstance(row, dict):
                        row_mapping = cast(dict[object, object], row)
                        normalized_row: dict[str, object] = {}
                        for key, value in row_mapping.items():
                            normalized_row[str(key)] = value
                        file_rows.append(normalized_row)
            self._materialize_workspace(work_dir=work_dir, file_rows=file_rows)

            self._emit_runtime_progress(
                phase_label="baseline: intervenor precheck",
                note="initial pytest precheck",
                attempts_completed=0,
                llm_calls_observed=llm_call_count,
                eval_duration_sec_observed=eval_duration_sec,
            )
            precheck = self._run_case_tests(
                work_dir=work_dir,
                python_args=python_args,
                timeout_sec=test_timeout_sec,
            )
            eval_duration_sec += precheck.duration_sec
            if precheck.returncode == 0:
                metrics = self._final_metrics(
                    llm_call_count=llm_call_count,
                    llm_duration_sec=llm_duration_sec,
                    eval_duration_sec=eval_duration_sec,
                    usage_available=False,
                    accumulated_usage=accumulated_usage,
                    usage_reported_call_count=usage_reported_call_count,
                    usage_missing_call_count=usage_missing_call_count,
                    prompt_chars_total=prompt_chars_total,
                    completion_chars_total=completion_chars_total,
                    started_at=started_at,
                    attempt_summaries=attempt_summaries,
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

            deadline_exceeded = False
            for attempt_idx in range(1, config.max_attempts + 1):
                deadline_sec = float(config.process_timeout_sec or 0.0)
                if deadline_sec > 0 and time.perf_counter() - started_at >= deadline_sec:
                    # BAS-08: one wall-clock deadline for the whole run_repair call.
                    deadline_exceeded = True
                    break
                failure_summary = self._format_failure_feedback(precheck)
                self._emit_runtime_progress(
                    phase_label="baseline: intervenor teacher",
                    note=f"attempt {attempt_idx}/{config.max_attempts} teacher",
                    attempts_completed=attempt_idx - 1,
                    llm_calls_observed=llm_call_count,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                teacher_prompt = self._render_teacher_prompt(
                    buggy_code=current_code,
                    error_message=failure_summary,
                    controller_guidance=teacher_guidance,
                )
                try:
                    teacher_text, teacher_usage, teacher_duration_sec, teacher_response_meta = (
                        _normalize_generation_result(
                            self._generate_text(
                                client=client,
                                model=config.model or "gpt-4o-mini",
                                prompt=teacher_prompt,
                                max_tokens=extra_config.baseline_intervenor_teacher_max_tokens,
                                config=config,
                            )
                        )
                    )
                except Exception as exc:  # BAS-10: keep the last candidate and usage
                    return _keep_progress_on_generation_failure(
                        exc=exc,
                        current_code=current_code,
                        original_code=original_code,
                        failure=self._compatibility_failure_result(
                            error_meta=getattr(exc, "meta", {}),
                            error_message=str(exc),
                            attempts=attempt_summaries,
                            iterations_used=attempt_idx - 1,
                            llm_call_count=llm_call_count,
                            llm_duration_sec=llm_duration_sec,
                            eval_duration_sec=eval_duration_sec,
                            usage_available=usage_available,
                            accumulated_usage=accumulated_usage,
                            usage_reported_call_count=usage_reported_call_count,
                            usage_missing_call_count=usage_missing_call_count,
                            prompt_chars_total=prompt_chars_total,
                            completion_chars_total=completion_chars_total,
                            started_at=started_at,
                        ),
                    )
                llm_call_count += 1
                llm_duration_sec += teacher_duration_sec
                prompt_chars_total += len(teacher_prompt)
                completion_chars_total += len(teacher_text)
                if teacher_usage:
                    usage_reported_call_count += 1
                    accumulated_usage = _accumulate_usage(accumulated_usage, teacher_usage)
                else:
                    usage_available = False
                    usage_missing_call_count += 1

                self._emit_runtime_progress(
                    phase_label="baseline: intervenor learner",
                    note=f"attempt {attempt_idx}/{config.max_attempts} learner",
                    attempts_completed=attempt_idx - 1,
                    llm_calls_observed=llm_call_count,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                learner_prompt = self._render_learner_prompt(
                    buggy_code=current_code,
                    error_message=failure_summary,
                    repair_method=teacher_text,
                    task_context=str(case_input.get("prompt", "")),
                    controller_guidance=learner_guidance,
                    append_output_contract=extra_config.baseline_intervenor_append_output_contract,
                )
                try:
                    learner_text, learner_usage, learner_duration_sec, learner_response_meta = (
                        _normalize_generation_result(
                            self._generate_text(
                                client=client,
                                model=config.model or "gpt-4o-mini",
                                prompt=learner_prompt,
                                max_tokens=extra_config.baseline_intervenor_learner_max_tokens,
                                config=config,
                            )
                        )
                    )
                except Exception as exc:  # BAS-10: keep the last candidate and usage
                    return _keep_progress_on_generation_failure(
                        exc=exc,
                        current_code=current_code,
                        original_code=original_code,
                        failure=self._compatibility_failure_result(
                            error_meta=getattr(exc, "meta", {}),
                            error_message=str(exc),
                            attempts=attempt_summaries,
                            iterations_used=attempt_idx - 1,
                            llm_call_count=llm_call_count,
                            llm_duration_sec=llm_duration_sec,
                            eval_duration_sec=eval_duration_sec,
                            usage_available=usage_available,
                            accumulated_usage=accumulated_usage,
                            usage_reported_call_count=usage_reported_call_count,
                            usage_missing_call_count=usage_missing_call_count,
                            prompt_chars_total=prompt_chars_total,
                            completion_chars_total=completion_chars_total,
                            started_at=started_at,
                        ),
                    )
                llm_call_count += 1
                llm_duration_sec += learner_duration_sec
                prompt_chars_total += len(learner_prompt)
                completion_chars_total += len(learner_text)
                if learner_usage:
                    usage_reported_call_count += 1
                    accumulated_usage = _accumulate_usage(accumulated_usage, learner_usage)
                else:
                    usage_available = False
                    usage_missing_call_count += 1

                repaired_code = _extract_python_code(learner_text)
                attempt_meta: dict[str, Any] = {
                    "attempt": attempt_idx,
                    "feedback": failure_summary,
                    "teacher_response": teacher_text,
                    "learner_response_empty": not bool(repaired_code),
                    "teacher_duration_sec": teacher_duration_sec,
                    "learner_duration_sec": learner_duration_sec,
                }
                teacher_finish_reason = str(
                    teacher_response_meta.get("finish_reason") or ""
                ).strip()
                if teacher_finish_reason:
                    attempt_meta["teacher_finish_reason"] = teacher_finish_reason
                    attempt_meta["teacher_response_saturated"] = teacher_finish_reason == "length"
                learner_finish_reason = str(
                    learner_response_meta.get("finish_reason") or ""
                ).strip()
                if learner_finish_reason:
                    attempt_meta["learner_finish_reason"] = learner_finish_reason
                    attempt_meta["learner_response_saturated"] = learner_finish_reason == "length"
                if not repaired_code:
                    attempt_meta["error"] = "learner_empty_response"
                    attempt_summaries.append(attempt_meta)
                    self._emit_runtime_progress(
                        phase_label="baseline: intervenor learner",
                        note=f"attempt {attempt_idx} empty learner response",
                        attempts_completed=attempt_idx,
                        llm_calls_observed=llm_call_count,
                        eval_duration_sec_observed=eval_duration_sec,
                    )
                    metrics = self._final_metrics(
                        llm_call_count=llm_call_count,
                        llm_duration_sec=llm_duration_sec,
                        eval_duration_sec=eval_duration_sec,
                        usage_available=usage_available,
                        accumulated_usage=accumulated_usage,
                        usage_reported_call_count=usage_reported_call_count,
                        usage_missing_call_count=usage_missing_call_count,
                        prompt_chars_total=prompt_chars_total,
                        completion_chars_total=completion_chars_total,
                        started_at=started_at,
                        attempt_summaries=attempt_summaries,
                    )
                    return BaselineResult(
                        repaired_code=current_code if current_code != original_code else None,
                        success=False,
                        iterations_used=attempt_idx,
                        token_usage=(
                            accumulated_usage
                            if usage_available and llm_call_count > 0 and accumulated_usage
                            else {}
                        ),
                        meta={
                            "error": "learner_empty_response",
                            "attempts": attempt_summaries,
                            "metrics": metrics,
                        },
                    )

                current_code = repaired_code
                (work_dir / "buggy.py").write_text(current_code, encoding="utf-8")
                self._emit_runtime_progress(
                    phase_label="baseline: intervenor postcheck",
                    note=f"attempt {attempt_idx}/{config.max_attempts} local tests",
                    attempts_completed=attempt_idx,
                    llm_calls_observed=llm_call_count,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                postcheck = self._run_case_tests(
                    work_dir=work_dir,
                    python_args=python_args,
                    timeout_sec=test_timeout_sec,
                )
                eval_duration_sec += postcheck.duration_sec
                attempt_meta["postcheck_returncode"] = postcheck.returncode
                attempt_meta["postcheck_duration_sec"] = postcheck.duration_sec
                attempt_summaries.append(attempt_meta)
                if postcheck.returncode == 0:
                    self._emit_runtime_progress(
                        phase_label="baseline: intervenor postcheck",
                        note=f"attempt {attempt_idx} passed local tests",
                        attempts_completed=attempt_idx,
                        llm_calls_observed=llm_call_count,
                        eval_duration_sec_observed=eval_duration_sec,
                    )
                    metrics = self._final_metrics(
                        llm_call_count=llm_call_count,
                        llm_duration_sec=llm_duration_sec,
                        eval_duration_sec=eval_duration_sec,
                        usage_available=usage_available,
                        accumulated_usage=accumulated_usage,
                        usage_reported_call_count=usage_reported_call_count,
                        usage_missing_call_count=usage_missing_call_count,
                        prompt_chars_total=prompt_chars_total,
                        completion_chars_total=completion_chars_total,
                        started_at=started_at,
                        attempt_summaries=attempt_summaries,
                    )
                    return BaselineResult(
                        repaired_code=current_code,
                        success=True,
                        iterations_used=attempt_idx,
                        token_usage=(
                            accumulated_usage
                            if usage_available and llm_call_count > 0 and accumulated_usage
                            else {}
                        ),
                        meta={
                            "attempts": attempt_summaries,
                            "test_stdout_tail": postcheck.stdout[-2000:],
                            "test_stderr_tail": postcheck.stderr[-2000:],
                            "metrics": metrics,
                        },
                    )
                self._emit_runtime_progress(
                    phase_label="baseline: intervenor postcheck",
                    note=f"attempt {attempt_idx} tests still failing",
                    attempts_completed=attempt_idx,
                    llm_calls_observed=llm_call_count,
                    eval_duration_sec_observed=eval_duration_sec,
                )
                precheck = postcheck

        metrics = self._final_metrics(
            llm_call_count=llm_call_count,
            llm_duration_sec=llm_duration_sec,
            eval_duration_sec=eval_duration_sec,
            usage_available=usage_available,
            accumulated_usage=accumulated_usage,
            usage_reported_call_count=usage_reported_call_count,
            usage_missing_call_count=usage_missing_call_count,
            prompt_chars_total=prompt_chars_total,
            completion_chars_total=completion_chars_total,
            started_at=started_at,
            attempt_summaries=attempt_summaries,
        )
        return BaselineResult(
            repaired_code=current_code if current_code != original_code else None,
            success=False,
            iterations_used=len(attempt_summaries) if deadline_exceeded else config.max_attempts,
            token_usage=(
                accumulated_usage
                if usage_available and llm_call_count > 0 and accumulated_usage
                else {}
            ),
            meta={
                "error": "deadline_exceeded" if deadline_exceeded else "max_attempts_exhausted",
                "attempts": attempt_summaries,
                "last_feedback": self._format_failure_feedback(precheck),
                "test_stdout_tail": precheck.stdout[-2000:],
                "test_stderr_tail": precheck.stderr[-2000:],
                "metrics": metrics,
            },
        )

    def _build_client(self, config: BaselineConfig) -> Any:
        try:
            module = import_module("openai")
        except ImportError as exc:
            raise RuntimeError("openai package is required for the INTERVENOR adapter") from exc
        openai_factory = getattr(module, "OpenAI", None)
        if openai_factory is None:
            raise RuntimeError("openai.OpenAI is required for the INTERVENOR adapter")

        base_url = config.api_base_url
        if base_url and not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        return openai_factory(
            base_url=base_url or None,
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            timeout=config.api_timeout_sec,
            max_retries=int(config.generation.transport_max_retries),  # BAS-11
        )

    def _generate_text(
        self,
        *,
        client: Any,
        model: str,
        prompt: str,
        max_tokens: int,
        config: BaselineConfig | None = None,
    ) -> tuple[str, dict[str, int], float, dict[str, Any]]:
        if config and config.python_executable is not None:
            return self._generate_text_via_env_python(
                python_executable=config.python_executable,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                config=config,
            )
        compat_config = compat_config_from_extra((config.extra if config else {}) or {})
        started_at = time.perf_counter()
        request_kwargs, request_meta = apply_compat_request_profile(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": _generation_temperature(config),
                "max_tokens": _generation_max_tokens(config, max_tokens),
                **_generation_seed_kwargs(config),
                "extra_body": usage_include_extra_body(config.api_base_url if config else ""),
            },
            compat_config=compat_config,
            base_url=str(config.api_base_url if config else ""),
        )
        response = client.chat.completions.create(
            **request_kwargs,
        )
        normalized = normalize_chat_completion_response(
            response,
            compat_config=compat_config,
            base_url=str(config.api_base_url if config else ""),
            request_meta=request_meta,
        )
        text = require_final_content(
            normalized,
            base_url=str(config.api_base_url if config else ""),
        )
        finish_reason = _response_finish_reason(response)
        return (
            text,
            _usage_dict(getattr(response, "usage", None)),
            time.perf_counter() - started_at,
            {"finish_reason": finish_reason} if finish_reason else {},
        )

    def _generate_text_via_env_python(
        self,
        *,
        python_executable: Path,
        model: str,
        prompt: str,
        max_tokens: int,
        config: BaselineConfig,
    ) -> tuple[str, dict[str, int], float, dict[str, Any]]:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": _generation_max_tokens(config, max_tokens),
            "temperature": _generation_temperature(config),
            "seed": config.generation.seed,
            "max_retries": int(config.generation.transport_max_retries),
            "base_url": config.api_base_url,
            "api_key": config.api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            "timeout_sec": config.api_timeout_sec,
            # BAS-19: the resolved shared defaults, so the env client behaves like
            # the in-process path for the same config.
            **compat_payload_fields(compat_config_from_extra(config.extra)),
        }
        with tempfile.TemporaryDirectory(prefix="intervenor_client_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "request.json"
            output_path = tmp_path / "response.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            started_at = time.perf_counter()
            result = subprocess.run(
                [
                    str(python_executable),
                    str(_ENV_CLIENT_SCRIPT),
                    str(input_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=max(1, int(config.api_timeout_sec or 1.0)),
                cwd=str(config.repo_dir),
                env=os.environ.copy(),
            )
            if result.returncode != 0:
                detail = result.stderr[-1000:] if result.stderr else result.stdout[-1000:]
                raise RuntimeError(f"INTERVENOR env client failed: {detail}")
            raw_response_payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(raw_response_payload, dict):
            raise RuntimeError("INTERVENOR env client returned a non-object payload")
        response_payload = cast(dict[str, Any], raw_response_payload)
        if str(response_payload.get("error") or ""):
            compat_meta = response_payload.get("compat_meta")
            raise OpenAICompatResponseError(
                str(response_payload.get("compat_message") or "OpenAI-compatible response error"),
                meta=cast(dict[str, Any], compat_meta) if isinstance(compat_meta, dict) else {},
            )
        text = str(response_payload.get("text", ""))
        usage_raw = response_payload.get("usage")
        usage: dict[str, int] = {}
        if isinstance(usage_raw, dict):
            usage_mapping = cast(dict[object, object], usage_raw)
            for key, value in usage_mapping.items():
                if isinstance(value, int):
                    usage[str(key)] = value
        meta_raw = response_payload.get("meta")
        meta = cast(dict[str, Any], meta_raw) if isinstance(meta_raw, dict) else {}
        return text, usage, time.perf_counter() - started_at, meta

    def _compatibility_failure_result(
        self,
        *,
        error_meta: dict[str, Any],
        error_message: str,
        attempts: list[dict[str, Any]],
        iterations_used: int,
        llm_call_count: int,
        llm_duration_sec: float,
        eval_duration_sec: float,
        usage_available: bool,
        accumulated_usage: dict[str, int],
        usage_reported_call_count: int,
        usage_missing_call_count: int,
        prompt_chars_total: int,
        completion_chars_total: int,
        started_at: float,
    ) -> BaselineResult:
        metrics = self._final_metrics(
            llm_call_count=llm_call_count,
            llm_duration_sec=llm_duration_sec,
            eval_duration_sec=eval_duration_sec,
            usage_available=usage_available,
            accumulated_usage=accumulated_usage,
            usage_reported_call_count=usage_reported_call_count,
            usage_missing_call_count=usage_missing_call_count,
            prompt_chars_total=prompt_chars_total,
            completion_chars_total=completion_chars_total,
            started_at=started_at,
            attempt_summaries=attempts,
        )
        return BaselineResult(
            repaired_code=None,
            success=False,
            iterations_used=max(0, int(iterations_used)),
            token_usage=(
                accumulated_usage
                if usage_available and llm_call_count > 0 and accumulated_usage
                else {}
            ),
            meta={
                "error": "openai_compat_reasoning_only",
                "compat_message": error_message,
                "compat_meta": dict(error_meta),
                "attempts": attempts,
                "metrics": metrics,
            },
        )

    def _build_metrics(
        self,
        *,
        llm_call_count: int,
        llm_duration_sec: float,
        eval_duration_sec: float,
        internal_total_duration_sec: float,
        usage_available: bool,
        usage_reported_call_count: int,
        usage_missing_call_count: int,
        accumulated_usage: dict[str, int],
        prompt_chars_total: int,
        completion_chars_total: int,
    ) -> dict[str, Any]:
        normalized_llm_sec = max(0.0, float(llm_duration_sec))
        normalized_eval_sec = max(0.0, float(eval_duration_sec))
        normalized_total_sec = max(0.0, float(internal_total_duration_sec))
        orchestration_duration_sec = max(
            0.0,
            normalized_total_sec - normalized_llm_sec - normalized_eval_sec,
        )
        metrics: dict[str, Any] = {
            "llm_call_count": max(0, int(llm_call_count)),
            "llm_duration_sec": normalized_llm_sec,
            "eval_duration_sec": normalized_eval_sec,
            "internal_total_duration_sec": normalized_total_sec,
            "orchestration_duration_sec": orchestration_duration_sec,
            "usage_available": bool(usage_available),
            "usage_reported_call_count": max(0, int(usage_reported_call_count)),
            "usage_missing_call_count": max(0, int(usage_missing_call_count)),
        }
        metrics.update(_reported_token_metrics(accumulated_usage))
        metrics.update(
            _reported_char_metrics(
                prompt_chars_total=prompt_chars_total,
                completion_chars_total=completion_chars_total,
            )
        )
        if usage_available:
            metrics["usage_source"] = "provider_reported"
        elif usage_reported_call_count > 0:
            metrics["usage_source"] = "partial_provider_reported"
        elif llm_call_count > 0:
            metrics["usage_source"] = "not_reported"
        return metrics

    def _render_teacher_prompt(
        self,
        *,
        buggy_code: str,
        error_message: str,
        controller_guidance: str = "",
    ) -> str:
        prompt = self._teacher_prompt_template.replace("%%%buggy_code%%%", buggy_code).replace(
            "%%%error_message%%%", error_message
        )
        if controller_guidance:
            prompt = f"{prompt}\n\nAdaptive controller guidance:\n{controller_guidance}"
        return prompt

    def _render_learner_prompt(
        self,
        *,
        buggy_code: str,
        error_message: str,
        repair_method: str,
        task_context: str,
        controller_guidance: str = "",
        append_output_contract: bool = True,
    ) -> str:
        prompt = self._learner_prompt_template.replace("%%%buggy_code%%%", buggy_code)
        prompt = prompt.replace("%%%error_message%%%", error_message)
        prompt = prompt.replace("%%%repair_method%%%", repair_method)
        if task_context:
            prompt = f"{prompt}\n\nTask context:\n{task_context}"
        if controller_guidance:
            prompt = f"{prompt}\n\nAdaptive controller guidance:\n{controller_guidance}"
        if not append_output_contract:
            return prompt
        return (
            f"{prompt}\n\n"
            "YOUR TASK: Output the full corrected Python source of buggy.py. "
            "Do not include markdown fences, explanations, or any text that is not Python code."
        )

    def _render_controller_guidance(
        self,
        *,
        controller_packet: ControllerPacketV1 | None,
        focus: str,
        max_chars: int | None = None,
    ) -> str:
        if controller_packet is None:
            return ""

        lines: list[str] = []
        state = controller_packet.controller_state
        action = state.helper_action_executed
        decision_reason = state.helper_decision_reason
        helper_outcome = state.helper_outcome
        if action or decision_reason or helper_outcome:
            lines.append(
                "controller_state: "
                f"action={action or 'none'} reason={decision_reason or 'n/a'} "
                f"outcome={helper_outcome or 'n/a'}"
            )

        verifier = controller_packet.verifier_report
        if verifier.status and verifier.status != "none":
            lines.append(f"verifier_status: {verifier.status}")
        if verifier.reasons:
            rendered_reasons = ", ".join(str(item) for item in verifier.reasons[:4] if str(item))
            if rendered_reasons:
                lines.append(f"verifier_reasons: {rendered_reasons}")
        last_error_type = verifier.last_error_type
        last_error_msg = _truncate_guidance_text(
            verifier.last_error_msg,
            max_chars=300,
        )
        if last_error_type or last_error_msg:
            lines.append(
                f"retry_constraint: {last_error_type or 'VerifierError'} {last_error_msg}".strip()
            )

        evidence_bundle = controller_packet.evidence_bundle
        evidence_summary = evidence_bundle.summary
        if evidence_summary is not None:
            if evidence_summary.crash_anchor:
                lines.append(f"crash_anchor: {evidence_summary.crash_anchor}")
            if evidence_summary.top_facts:
                rendered_facts = [
                    _truncate_guidance_text(fact, max_chars=180)
                    for fact in evidence_summary.top_facts[:4]
                ]
                lines.append(
                    "evidence_summary: " + " | ".join(fact for fact in rendered_facts if fact)
                )
            if focus == "learner" and evidence_summary.selected_local_names:
                lines.append(
                    "local_focus: "
                    + ", ".join(str(item) for item in evidence_summary.selected_local_names)
                )
        ast_context = evidence_bundle.ast_context
        if ast_context is not None and focus == "learner":
            path = ast_context.path
            symbol_name = ast_context.symbol_name
            node_type = ast_context.node_type
            line_span = ast_context.line_span
            snippet = _truncate_guidance_text(
                ast_context.snippet,
                max_chars=900,
            )
            header = f"ast_focus: {path}"
            if line_span:
                header = f"{header} lines={line_span}"
            if symbol_name or node_type:
                header = f"{header} symbol={symbol_name or node_type}"
            lines.append(header)
            if snippet:
                lines.append(f"ast_snippet:\n{snippet}")

        if evidence_bundle.items:
            rendered_items: list[str] = []
            for row in evidence_bundle.items[:4]:
                kind = row.kind or "evidence"
                locator = row.locator or row.path or ""
                summary = _truncate_guidance_text(
                    row.summary,
                    max_chars=180,
                )
                if summary:
                    rendered_items.append(
                        f"{kind} {locator}".strip() + (f": {summary}" if summary else "")
                    )
            if rendered_items:
                lines.append("selected_evidence: " + " | ".join(rendered_items))

        snapshot_reduction = evidence_bundle.snapshot_reduction
        if snapshot_reduction is not None and snapshot_reduction.crash_frame_summary:
            lines.append(f"snapshot_focus: {snapshot_reduction.crash_frame_summary}")

        history_bundle = controller_packet.history_bundle
        previous_note = history_bundle.previous_note
        if previous_note:
            lines.append(f"previous_attempt_note: {previous_note}")
        if focus == "teacher":
            if history_bundle.lessons:
                rendered_lessons: list[str] = []
                for lesson in history_bundle.lessons[:3]:
                    prefix = (
                        f"attempt {lesson.attempt}"
                        if isinstance(lesson.attempt, int) and lesson.attempt > 0
                        else lesson.source
                    )
                    rendered_lessons.append(f"{prefix}: {lesson.lesson}")
                if rendered_lessons:
                    lines.append("history_lessons: " + " | ".join(rendered_lessons))
            retry_history_text = _truncate_guidance_text(
                history_bundle.retry_history_text,
                max_chars=800,
            )
            if retry_history_text:
                lines.append(f"attempt_history:\n{retry_history_text}")
            repo_history = history_bundle.repo_history
            if repo_history is not None:
                blame_summary = _truncate_guidance_text(
                    repo_history.crash_file_blame_summary,
                    max_chars=200,
                )
                if blame_summary:
                    lines.append(f"blame_hint: {blame_summary}")
                commit_summaries = [
                    summary
                    for summary in (
                        _truncate_guidance_text(row.summary, max_chars=140)
                        for row in repo_history.recent_commits[:2]
                    )
                    if summary
                ]
                if commit_summaries:
                    lines.append("recent_commits: " + " | ".join(commit_summaries))

        solver_hints = controller_packet.solver_hints
        if solver_hints.recommended_focus:
            lines.append(
                "recommended_focus: "
                + ", ".join(str(item) for item in solver_hints.recommended_focus)
            )
        strategy_summary = _truncate_guidance_text(
            solver_hints.strategy_summary,
            max_chars=220,
        )
        if strategy_summary:
            lines.append(f"strategy_summary: {strategy_summary}")
        if solver_hints.decomposition:
            rendered_steps: list[str] = []
            for step in solver_hints.decomposition[:3]:
                title = str(step.get("title") or "").strip()
                question = _truncate_guidance_text(
                    str(step.get("question") or ""),
                    max_chars=200,
                )
                if title and question:
                    rendered_steps.append(f"{title}: {question}")
            if rendered_steps:
                lines.append("decomposition: " + " | ".join(rendered_steps))
        if focus == "learner":
            retry_plan = _truncate_guidance_text(
                solver_hints.retry_plan,
                max_chars=800,
            )
            if retry_plan:
                lines.append(f"retry_plan:\n{retry_plan}")

        if not lines:
            return ""
        resolved_max_chars = max_chars if isinstance(max_chars, int) and max_chars > 0 else None
        if resolved_max_chars is None:
            resolved_max_chars = 4_000 if focus == "teacher" else 6_000
        return _truncate_guidance_text("\n".join(lines), max_chars=resolved_max_chars)

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
                import base64

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
        env = os.environ.copy()
        started_at = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, *python_args],
                capture_output=True,
                text=True,
                cwd=str(work_dir),
                timeout=timeout_sec,
                env=env,
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

    def extract_diff(self, case_dir: Path, result: BaselineResult) -> str | None:
        """Produce unified diff from INTERVENOR's repaired code."""
        return super().extract_diff(case_dir, result)
