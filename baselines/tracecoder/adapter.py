"""TraceCoder baseline adapter.

Runs the patched upstream TraceCoder CLI against one generated llmdebug
dataset row at a time and translates its CSV output back into the llmdebug
baseline adapter contract.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineConfig,
    BaselineResult,
)
from baselines.case_schemas import TraceCoderCaseInput
from baselines.contracts import BaselineContract
from baselines.pins import get_pinned_repo_commit
from evals.adaptive_controller_packet import (
    parse_controller_packet,
    render_hybrid_controller_guidance,
)
from evals.subprocess_monitor import run_monitored_subprocess

from .case_converter import compact_problem_prompt, convert_case

_TRACECODER_API_TIMEOUT_ENV = "TRACECODER_OPENAI_TIMEOUT_SEC"
_TRACECODER_ANALYSIS_MAX_TOKENS_ENV = "TRACECODER_ANALYSIS_MAX_TOKENS"
_TRACECODER_IMPLEMENTATION_MAX_TOKENS_ENV = "TRACECODER_IMPLEMENTATION_MAX_TOKENS"
_TRACECODER_PROBLEM_SUMMARY_MAX_CHARS_ENV = "TRACECODER_PROBLEM_SUMMARY_MAX_CHARS"
_TRACECODER_EXACT_TEST_CONTEXT_MAX_CHARS_ENV = "TRACECODER_EXACT_TEST_CONTEXT_MAX_CHARS"
_TRACECODER_FAILURE_FEEDBACK_MAX_CHARS_ENV = "TRACECODER_FAILURE_FEEDBACK_MAX_CHARS"
_TRACECODER_FAILURE_FEEDBACK_TAIL_MAX_CHARS_ENV = "TRACECODER_FAILURE_FEEDBACK_TAIL_MAX_CHARS"
_TRACECODER_INSTRUMENTATION_OUTPUT_MAX_LINES_ENV = "TRACECODER_INSTRUMENTATION_OUTPUT_MAX_LINES"
_TRACECODER_INSTRUMENTATION_OUTPUT_MAX_CHARS_ENV = "TRACECODER_INSTRUMENTATION_OUTPUT_MAX_CHARS"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _has_llmdebug_support(repo: Path) -> bool:
    try:
        config_text = (repo / "config.py").read_text(encoding="utf-8")
        processor_text = (repo / "problem_processor.py").read_text(encoding="utf-8")
        reporting_text = (repo / "reporting.py").read_text(encoding="utf-8")
        generation_text = (repo / "src" / "generation.py").read_text(encoding="utf-8")
        evaluation_path = repo / "datasets" / "llmdebug" / "evaluation.py"
        exact_test_context_path = repo / "src" / "exact_test_context.py"
    except OSError:
        return False
    return (
        '"llmdebug"' in config_text
        and "--dataset-path" in config_text
        and "repair_seeded" in processor_text
        and "execution_mode" in processor_text
        and "method_instrumentation" in processor_text
        and "method_two_step_repair" in processor_text
        and "direct_gen_passed" in processor_text
        and "final_passed" in processor_text
        and "stopped_reason" in processor_text
        and "total_llm_duration_sec" in processor_text
        and "total_eval_duration_sec" in processor_text
        and "build_exact_test_context" in processor_text
        and "exact_test_context_meta" in processor_text
        and "build_llmdebug_failure_feedback" in reporting_text
        and "LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE" in generation_text
        and "openai_compat_reasoning_only" in generation_text
        and _TRACECODER_ANALYSIS_MAX_TOKENS_ENV in generation_text
        and _TRACECODER_IMPLEMENTATION_MAX_TOKENS_ENV in generation_text
        and exact_test_context_path.is_file()
        and evaluation_path.is_file()
    )


def _has_metrics_sidecar_support(repo: Path) -> bool:
    try:
        config_text = (repo / "config.py").read_text(encoding="utf-8")
        reporting_text = (repo / "reporting.py").read_text(encoding="utf-8")
        runner_text = (repo / "trace_learn_coder.py").read_text(encoding="utf-8")
    except OSError:
        return False
    return (
        "--llmdebug-metrics-path" in config_text
        and "save_metrics_sidecar" in reporting_text
        and "llmdebug_metrics_path" in runner_text
    )


def _parse_json_field(raw: object) -> object:
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return raw
    if text[0] not in '[{"':
        return raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


def _parse_optional_bool(raw: object) -> bool | None:
    if isinstance(raw, bool):
        return raw
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _parse_int(raw: object) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _parse_float(raw: object) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _read_metrics_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else None


def _parse_metrics_sidecar(
    payload: dict[str, Any] | None,
    *,
    sidecar_path: Path,
) -> tuple[dict[str, Any], dict[str, int], int | None, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}, {}, None, {}

    llm_call_count = max(0, int(_parse_int(payload.get("llm_call_count")) or 0))
    llm_duration_sec = max(0.0, float(_parse_float(payload.get("llm_duration_sec")) or 0.0))
    eval_duration_sec = max(0.0, float(_parse_float(payload.get("eval_duration_sec")) or 0.0))
    internal_total_duration_sec = max(
        0.0,
        float(
            _parse_float(payload.get("internal_total_duration_sec"))
            or _parse_float(payload.get("processing_time_seconds"))
            or 0.0
        ),
    )
    orchestration_duration_sec = max(
        0.0,
        float(
            _parse_float(payload.get("orchestration_duration_sec"))
            or (internal_total_duration_sec - llm_duration_sec - eval_duration_sec)
        ),
    )
    usage_available_raw = _parse_optional_bool(payload.get("usage_available"))
    usage_reported_call_count = max(
        0,
        int(_parse_int(payload.get("usage_reported_call_count")) or 0),
    )
    usage_missing_call_count = max(
        0,
        int(_parse_int(payload.get("usage_missing_call_count")) or 0),
    )
    prompt_tokens = _parse_int(payload.get("tokens_prompt"))
    completion_tokens = _parse_int(payload.get("tokens_completion"))
    total_tokens = _parse_int(payload.get("tokens_total"))
    reported_prompt = _parse_int(payload.get("reported_tokens_prompt"))
    reported_completion = _parse_int(payload.get("reported_tokens_completion"))
    reported_total = _parse_int(payload.get("reported_tokens_total"))
    prompt_chars_total = _parse_int(payload.get("prompt_chars_total"))
    completion_chars_total = _parse_int(payload.get("completion_chars_total"))
    chars_total = _parse_int(payload.get("chars_total"))
    reported_prompt_chars = _parse_int(payload.get("reported_chars_prompt"))
    reported_completion_chars = _parse_int(payload.get("reported_chars_completion"))
    reported_chars_total = _parse_int(payload.get("reported_chars_total"))
    if reported_prompt is None:
        reported_prompt = prompt_tokens
    if reported_completion is None:
        reported_completion = completion_tokens
    if reported_total is None:
        reported_total = total_tokens
    if reported_prompt_chars is None:
        reported_prompt_chars = prompt_chars_total
    if reported_completion_chars is None:
        reported_completion_chars = completion_chars_total
    if reported_chars_total is None:
        reported_chars_total = chars_total
    stopped_reason = str(payload.get("stopped_reason") or "").strip()
    method_instrumentation = _parse_optional_bool(payload.get("method_instrumentation"))
    method_two_step_repair = _parse_optional_bool(payload.get("method_two_step_repair"))
    direct_gen_passed = _parse_optional_bool(payload.get("direct_gen_passed"))
    debug_added_value = _parse_optional_bool(payload.get("debug_added_value"))
    attempts_completed = _parse_int(payload.get("attempts_completed"))
    usage_source = str(payload.get("usage_source") or "").strip()
    llm_timeout_count = _parse_int(payload.get("llm_timeout_count"))
    last_llm_failure_kind = str(payload.get("last_llm_failure_kind") or "").strip()
    last_llm_failure_stage = str(payload.get("last_llm_failure_stage") or "").strip()
    compat_message = str(payload.get("compat_message") or "").strip()
    compat_meta_keys = (
        "compat_profile_requested",
        "compat_profile_applied",
        "direct_output_requested",
        "allow_reasoning_fallback",
        "content_present",
        "reasoning_present",
        "empty_content_with_reasoning",
        "used_reasoning_fallback",
        "reasoning_chars",
        "reasoning_content_chars",
        "final_text_chars",
        "finish_reason",
    )
    compat_meta = {key: payload[key] for key in compat_meta_keys if key in payload}

    metrics: dict[str, Any] = {
        "llm_call_count": llm_call_count,
        "llm_duration_sec": llm_duration_sec,
        "eval_duration_sec": eval_duration_sec,
        "internal_total_duration_sec": internal_total_duration_sec,
        "orchestration_duration_sec": orchestration_duration_sec,
        "usage_reported_call_count": usage_reported_call_count,
        "usage_missing_call_count": usage_missing_call_count,
    }
    if isinstance(usage_available_raw, bool):
        metrics["usage_available"] = usage_available_raw
    if isinstance(reported_prompt, int):
        metrics["reported_tokens_prompt"] = max(0, reported_prompt)
    if isinstance(reported_completion, int):
        metrics["reported_tokens_completion"] = max(0, reported_completion)
    if isinstance(reported_total, int):
        metrics["reported_tokens_total"] = max(0, reported_total)
    if isinstance(prompt_chars_total, int):
        metrics["prompt_chars_total"] = max(0, prompt_chars_total)
        metrics["chars_input"] = max(0, prompt_chars_total)
    if isinstance(completion_chars_total, int):
        metrics["completion_chars_total"] = max(0, completion_chars_total)
        metrics["chars_output"] = max(0, completion_chars_total)
    if isinstance(chars_total, int):
        metrics["chars_total"] = max(0, chars_total)
    elif isinstance(prompt_chars_total, int) and isinstance(completion_chars_total, int):
        metrics["chars_total"] = max(0, prompt_chars_total) + max(0, completion_chars_total)
    if isinstance(reported_prompt_chars, int):
        metrics["reported_chars_prompt"] = max(0, reported_prompt_chars)
    if isinstance(reported_completion_chars, int):
        metrics["reported_chars_completion"] = max(0, reported_completion_chars)
    if isinstance(reported_chars_total, int):
        metrics["reported_chars_total"] = max(0, reported_chars_total)
    elif isinstance(reported_prompt_chars, int) and isinstance(reported_completion_chars, int):
        metrics["reported_chars_total"] = max(0, reported_prompt_chars) + max(
            0, reported_completion_chars
        )
    if stopped_reason:
        metrics["stopped_reason"] = stopped_reason
    if isinstance(method_instrumentation, bool):
        metrics["method_instrumentation"] = method_instrumentation
    if isinstance(method_two_step_repair, bool):
        metrics["method_two_step_repair"] = method_two_step_repair
    if isinstance(direct_gen_passed, bool):
        metrics["direct_gen_passed"] = direct_gen_passed
    if isinstance(debug_added_value, bool):
        metrics["debug_added_value"] = debug_added_value
    if usage_source:
        metrics["usage_source"] = usage_source
    if isinstance(llm_timeout_count, int):
        metrics["llm_timeout_count"] = max(0, llm_timeout_count)
    if last_llm_failure_kind:
        metrics["last_llm_failure_kind"] = last_llm_failure_kind
    if last_llm_failure_stage:
        metrics["last_llm_failure_stage"] = last_llm_failure_stage
    if compat_message:
        metrics["compat_message"] = compat_message
    if compat_meta:
        metrics["compat_meta"] = compat_meta

    usage: dict[str, int] = {}
    if usage_available_raw is True and prompt_tokens is not None and completion_tokens is not None:
        usage["prompt_tokens"] = max(0, prompt_tokens)
        usage["completion_tokens"] = max(0, completion_tokens)
        usage["total_tokens"] = max(
            0,
            total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
        )

    meta: dict[str, Any] = {
        "tracecoder_metrics_sidecar": str(sidecar_path),
    }
    if compat_message:
        meta["compat_message"] = compat_message
    if compat_meta:
        meta["compat_meta"] = compat_meta
    return metrics, usage, attempts_completed, meta


def _resolve_token_usage(
    *,
    fallback_usage: dict[str, int],
    sidecar_metrics: dict[str, Any],
    sidecar_usage: dict[str, int],
) -> dict[str, int]:
    sidecar_usage_available = sidecar_metrics.get("usage_available")
    if sidecar_usage_available is False:
        return {}
    if sidecar_usage:
        return sidecar_usage
    return dict(fallback_usage)


def _live_progress_payload_for_metrics_sidecar(metrics_sidecar_path: Path) -> dict[str, Any] | None:
    sidecar_payload = _read_metrics_sidecar(metrics_sidecar_path)
    metrics, _, attempts_completed, _ = _parse_metrics_sidecar(
        sidecar_payload,
        sidecar_path=metrics_sidecar_path,
    )
    if not isinstance(sidecar_payload, dict) and not metrics and attempts_completed is None:
        return None

    note_parts: list[str] = []
    status = (
        str(sidecar_payload.get("status") or "").strip()
        if isinstance(sidecar_payload, dict)
        else ""
    )
    if status:
        note_parts.append(f"status={status}")
    stopped_reason = str(metrics.get("stopped_reason") or "").strip()
    if stopped_reason:
        note_parts.append(f"stopped={stopped_reason}")
    direct_gen_passed = metrics.get("direct_gen_passed")
    if isinstance(direct_gen_passed, bool):
        note_parts.append(f"direct_gen={str(direct_gen_passed).lower()}")
    last_llm_failure_kind = str(metrics.get("last_llm_failure_kind") or "").strip()
    last_llm_failure_stage = str(metrics.get("last_llm_failure_stage") or "").strip()
    if last_llm_failure_kind:
        if last_llm_failure_stage:
            note_parts.append(f"last_llm_failure={last_llm_failure_kind}@{last_llm_failure_stage}")
        else:
            note_parts.append(f"last_llm_failure={last_llm_failure_kind}")

    payload: dict[str, Any] = {
        "owner": "baseline",
        "source": "sidecar",
        "phase_label": "baseline: tracecoder subprocess",
        "waiting_for": "tracecoder_subprocess",
        "external_operation": "baseline_repair",
        "external_provider": "tracecoder",
    }
    if isinstance(attempts_completed, int):
        payload["attempts_completed"] = max(0, attempts_completed)
    llm_call_count = metrics.get("llm_call_count")
    if isinstance(llm_call_count, int):
        payload["llm_calls_observed"] = max(0, llm_call_count)
    eval_duration_sec = metrics.get("eval_duration_sec")
    if isinstance(eval_duration_sec, (int, float)):
        payload["eval_duration_sec_observed"] = max(0.0, float(eval_duration_sec))
    if note_parts:
        payload["note"] = " ".join(note_parts)
    return payload


def _case_test_timeout_sec(case_input: Mapping[str, Any], config: BaselineConfig) -> int:
    """Per-case test timeout; falls back to the process timeout like the other adapters
    (BAS-12: it used to fall back to 10 s when the case context was missing)."""
    return int(case_input.get("timeout_sec") or 0) or max(1, int(config.process_timeout_sec or 0))


class TraceCoderAdapter(BaselineAdapter):
    """Adapter for the patched upstream TraceCoder baseline."""

    default_model: str = "gpt-4o-mini"
    _config: BaselineConfig | None = None
    repo_commit: str = ""
    _has_metrics_sidecar_support: bool = False

    @property
    def name(self) -> str:
        return "tracecoder"

    def setup(self, config: BaselineConfig) -> None:
        self._config = config
        repo = config.repo_dir
        if not repo.is_dir():
            raise RuntimeError(
                f"TraceCoder repo not found at {repo}. Run baselines/setup_repos.sh first."
            )
        for required in (
            "trace_learn_coder.py",
            "problem_processor.py",
            "src/generation.py",
            "src/traceRunner.py",
            "src/exact_test_context.py",
        ):
            if not (repo / required).is_file():
                raise RuntimeError(f"TraceCoder missing required file: {required}")
        if not _has_llmdebug_support(repo):
            raise RuntimeError(
                "TraceCoder repo is missing llmdebug compatibility support. "
                "Re-run baselines/setup_repos.sh or apply baselines/tracecoder/patches/0001-add-llmdebug-repair-mode.patch."
            )
        self._has_metrics_sidecar_support = _has_metrics_sidecar_support(repo)
        self.repo_commit = _resolve_repo_commit(repo)

    def validate_extra_config(self, config: BaselineConfig) -> None:
        from baselines.extra_configs import TraceCoderExtraConfig

        TraceCoderExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def contract(self) -> BaselineContract:
        return BaselineContract(
            baseline_name=self.name,
            upstream_commit=get_pinned_repo_commit(self.name),
            fidelity_level="upstream_patched",
            repair_scope_policy="single_file_only",
            prompt_contract_kind="patched_tracecoder_dataset_prompt",
            output_contract="single_file_repaired_source",
            accepted_case_shapes=(
                "single_file_buggy_py",
                "workspace_tests_available",
                "repair_seeded_tracecoder_row",
            ),
            patch_scope_summary=("compat", "telemetry", "behavioral_adaptation"),
            notes=(
                "TraceCoder runs in patched llmdebug dataset mode rather than raw upstream datasets.",
            ),
        )

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> TraceCoderCaseInput:
        return convert_case(case_dir, case_context=case_context)

    def run_repair(self, case_input: dict[str, Any], config: BaselineConfig) -> BaselineResult:
        repo = config.repo_dir
        extra_config = self._validated_extra_config(config)
        original_buggy_code = str(case_input.get("buggy_code") or "")
        with tempfile.TemporaryDirectory(prefix="tracecoder_") as tmpdir:
            work_dir = Path(tmpdir)
            dataset_path = work_dir / "dataset.jsonl"
            output_csv = work_dir / "result.csv"
            metrics_sidecar_path = work_dir / "result.csv.llmdebug_metrics.json"
            dataset_row = dict(case_input)
            controller_packet = parse_controller_packet(
                case_input.get("llmdebug_controller_packet")
            )
            controller_guidance = render_hybrid_controller_guidance(
                controller_packet,
                mode=extra_config.baseline_hybrid_guidance_mode,
                include_history=True,
                include_retry_plan=True,
                max_chars=extra_config.baseline_tracecoder_controller_guidance_max_chars,
            ).text
            complete_prompt = compact_problem_prompt(
                str(dataset_row.get("complete_prompt") or ""),
                max_chars=extra_config.baseline_tracecoder_problem_summary_max_chars,
            )
            if controller_guidance:
                dataset_row["complete_prompt"] = (
                    f"{complete_prompt}\n\nLLMDebug controller guidance:\n{controller_guidance}"
                )
            else:
                dataset_row["complete_prompt"] = complete_prompt
            dataset_path.write_text(json.dumps(dataset_row) + "\n", encoding="utf-8")

            env = os.environ.copy()
            env.update(config.generation.to_env())  # D4; upstream doesn't read these yet
            env["OPENAI_API_KEY"] = config.api_key or env.get("OPENAI_API_KEY", "sk-placeholder")
            if config.api_base_url:
                env["OPENAI_BASE_URL"] = config.api_base_url
            api_timeout_sec = float(config.api_timeout_sec or 0.0)
            if api_timeout_sec > 0.0:
                env[_TRACECODER_API_TIMEOUT_ENV] = str(api_timeout_sec)
            env[_TRACECODER_ANALYSIS_MAX_TOKENS_ENV] = str(
                extra_config.baseline_tracecoder_analysis_max_tokens
            )
            env[_TRACECODER_IMPLEMENTATION_MAX_TOKENS_ENV] = str(
                extra_config.baseline_tracecoder_implementation_max_tokens
            )
            env[_TRACECODER_PROBLEM_SUMMARY_MAX_CHARS_ENV] = str(
                extra_config.baseline_tracecoder_problem_summary_max_chars
            )
            env[_TRACECODER_EXACT_TEST_CONTEXT_MAX_CHARS_ENV] = str(
                extra_config.baseline_tracecoder_exact_test_context_max_chars
            )
            env[_TRACECODER_FAILURE_FEEDBACK_MAX_CHARS_ENV] = str(
                extra_config.baseline_tracecoder_failure_feedback_max_chars
            )
            env[_TRACECODER_FAILURE_FEEDBACK_TAIL_MAX_CHARS_ENV] = str(
                extra_config.baseline_tracecoder_failure_feedback_tail_max_chars
            )
            env[_TRACECODER_INSTRUMENTATION_OUTPUT_MAX_LINES_ENV] = str(
                extra_config.baseline_tracecoder_instrumentation_output_max_lines
            )
            env[_TRACECODER_INSTRUMENTATION_OUTPUT_MAX_CHARS_ENV] = str(
                extra_config.baseline_tracecoder_instrumentation_output_max_chars
            )
            env["LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE"] = str(
                config.extra.get("baseline_openai_compat_profile") or "auto"
            )
            env["LLMDEBUG_BASELINE_OPENAI_ALLOW_REASONING_FALLBACK"] = (
                "1"
                if bool(config.extra.get("baseline_openai_allow_reasoning_fallback", False))
                else "0"
            )
            env["LLMDEBUG_BASELINE_OPENAI_ENABLE_THINKING"] = (
                "1" if bool(config.extra.get("baseline_openai_enable_thinking", False)) else "0"
            )
            existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
            env["PYTHONPATH"] = (
                f"{_PROJECT_ROOT}:{existing_pythonpath}"
                if existing_pythonpath
                else str(_PROJECT_ROOT)
            )
            fake_responses_path = str(
                config.extra.get("tracecoder_fake_responses_path") or ""
            ).strip()
            if fake_responses_path:
                env["TRACECODER_FAKE_RESPONSES_PATH"] = fake_responses_path

            timeout_sec = _case_test_timeout_sec(case_input, config)
            cmd = [
                str(config.python_executable or Path(sys.executable)),
                str(repo / "trace_learn_coder.py"),
                "-d",
                "llmdebug",
                "--dataset-path",
                str(dataset_path),
                "-m",
                config.model or "gpt-4o-mini",
                "--max-problems",
                "1",
                "--max-debug-attempts",
                str(config.max_attempts),
                "--max-no-improvement-streak",
                str(extra_config.baseline_tracecoder_max_no_improvement_streak),
                "--timeout",
                str(timeout_sec),
                "-o",
                str(output_csv),
            ]
            if self._has_metrics_sidecar_support:
                cmd.extend(["--llmdebug-metrics-path", str(metrics_sidecar_path)])

            monitored = run_monitored_subprocess(
                cmd=cmd,
                cwd=repo,
                env=env,
                timeout_sec=max(1, int(config.process_timeout_sec or 1.0)),
                progress_callback=self._current_progress_callback(),
                progress_event="baseline_progress",
                progress_base_payload={
                    "owner": "baseline",
                    "phase_label": "baseline: tracecoder subprocess",
                    "waiting_for": "tracecoder_subprocess",
                    "external_operation": "baseline_repair",
                    "external_provider": "tracecoder",
                    "source": "phase",
                },
                state_supplier=(
                    (lambda: _live_progress_payload_for_metrics_sidecar(metrics_sidecar_path))
                    if self._has_metrics_sidecar_support
                    else None
                ),
            )
            if monitored.timed_out:
                return self._normalize_failed_unchanged_candidate(
                    self._build_timeout_result(
                        output_csv=output_csv,
                        metrics_sidecar_path=metrics_sidecar_path,
                        timeout_sec=float(config.process_timeout_sec or 0.0),
                        stdout=monitored.stdout,
                        stderr=monitored.stderr,
                    ),
                    original_buggy_code=original_buggy_code,
                )

            return self._normalize_failed_unchanged_candidate(
                self._parse_output(
                    output_csv=output_csv,
                    metrics_sidecar_path=metrics_sidecar_path,
                    stdout=monitored.stdout,
                    stderr=monitored.stderr,
                    returncode=monitored.returncode,
                ),
                original_buggy_code=original_buggy_code,
            )

    def _build_timeout_result(
        self,
        *,
        output_csv: Path,
        metrics_sidecar_path: Path,
        timeout_sec: float,
        stdout: str,
        stderr: str,
    ) -> BaselineResult:
        sidecar_payload = _read_metrics_sidecar(metrics_sidecar_path)
        sidecar_metrics, sidecar_usage, sidecar_iterations, sidecar_meta = _parse_metrics_sidecar(
            sidecar_payload,
            sidecar_path=metrics_sidecar_path,
        )
        if output_csv.is_file():
            parsed = self._parse_output(
                output_csv=output_csv,
                metrics_sidecar_path=metrics_sidecar_path,
                stdout=stdout,
                stderr=stderr,
                returncode=-1,
            )
            meta = dict(parsed.meta)
            meta.update(sidecar_meta)
            parsed_error = meta.get("error")
            meta["error"] = "timeout"
            meta["timeout_sec"] = timeout_sec
            if parsed_error and parsed_error != "timeout":
                meta["timeout_parse_error"] = parsed_error
            metrics = meta.get("metrics")
            if not isinstance(metrics, dict):
                if sidecar_metrics:
                    meta["metrics"] = sidecar_metrics
                else:
                    meta["metrics"] = {"llm_call_count": 0, "usage_available": False}
            return BaselineResult(
                repaired_code=None,
                success=False,
                iterations_used=(
                    int(sidecar_iterations)
                    if isinstance(sidecar_iterations, int)
                    else parsed.iterations_used
                ),
                token_usage=_resolve_token_usage(
                    fallback_usage=parsed.token_usage,
                    sidecar_metrics=sidecar_metrics,
                    sidecar_usage=sidecar_usage,
                ),
                meta=meta,
            )

        meta: dict[str, Any] = {
            "error": "timeout",
            "timeout_sec": timeout_sec,
            "stdout_tail": stdout[-2000:] if stdout else "",
            "stderr_tail": stderr[-2000:] if stderr else "",
        }
        meta.update(sidecar_meta)
        if sidecar_metrics:
            meta["metrics"] = sidecar_metrics
        return BaselineResult(
            repaired_code=None,
            success=False,
            iterations_used=int(sidecar_iterations or 0),
            token_usage=sidecar_usage if sidecar_usage else {},
            meta={
                **meta,
                **(
                    {}
                    if sidecar_metrics
                    else {
                        "metrics": {
                            "llm_call_count": 0,
                            "usage_available": False,
                            "usage_source": "not_reported",
                        }
                    }
                ),
            },
        )

    def extract_diff(self, case_dir: Path, result: BaselineResult) -> str | None:
        return super().extract_diff(case_dir, result)

    def _normalize_failed_unchanged_candidate(
        self,
        result: BaselineResult,
        *,
        original_buggy_code: str,
    ) -> BaselineResult:
        repaired_code = str(result.repaired_code or "")
        if result.success or not repaired_code.strip():
            return result
        if repaired_code.strip() != str(original_buggy_code or "").strip():
            return result
        meta = dict(result.meta)
        meta["candidate_status"] = "unchanged_original_code"
        meta["artifact_error"] = "unchanged_original_code"
        if not str(meta.get("error") or "").strip():
            meta["error"] = "unchanged_original_code"
        return replace(result, repaired_code=None, meta=meta)

    def _validated_extra_config(self, config: BaselineConfig):
        from baselines.extra_configs import TraceCoderExtraConfig

        return TraceCoderExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def _parse_output(
        self,
        *,
        output_csv: Path,
        metrics_sidecar_path: Path | None = None,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> BaselineResult:
        resolved_metrics_sidecar_path = (
            metrics_sidecar_path
            if isinstance(metrics_sidecar_path, Path)
            else output_csv.with_name(f"{output_csv.name}.llmdebug_metrics.json")
        )
        meta: dict[str, Any] = {
            "returncode": returncode,
            "stdout_tail": stdout[-2000:] if stdout else "",
            "stderr_tail": stderr[-2000:] if stderr else "",
        }
        sidecar_payload = _read_metrics_sidecar(resolved_metrics_sidecar_path)
        sidecar_metrics, sidecar_usage, sidecar_iterations, sidecar_meta = _parse_metrics_sidecar(
            sidecar_payload,
            sidecar_path=resolved_metrics_sidecar_path,
        )
        meta.update(sidecar_meta)
        if (
            str(sidecar_metrics.get("last_llm_failure_kind") or "").strip()
            == "openai_compat_reasoning_only"
            or str(meta.get("compat_message") or "").strip()
        ):
            meta["error"] = "openai_compat_reasoning_only"
        if not output_csv.is_file():
            meta["error"] = "missing_output_csv"
            if sidecar_metrics:
                meta["metrics"] = sidecar_metrics
            return BaselineResult(
                repaired_code=None,
                success=False,
                iterations_used=int(sidecar_iterations or 0),
                token_usage=sidecar_usage if sidecar_usage else {},
                meta=meta,
            )

        with output_csv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
        if row is None:
            meta["error"] = "empty_output_csv"
            if sidecar_metrics:
                meta["metrics"] = sidecar_metrics
            return BaselineResult(
                repaired_code=None,
                success=False,
                iterations_used=int(sidecar_iterations or 0),
                token_usage=sidecar_usage if sidecar_usage else {},
                meta=meta,
            )

        final_code = str(row.get("final_code") or "")
        final_eval = _parse_json_field(row.get("final_eval"))
        direct_gen_eval = _parse_json_field(row.get("direct_gen_eval"))
        debug_session_log = _parse_json_field(row.get("debug_session_log"))
        prompt_tokens = _parse_int(row.get("total_prompt_tokens"))
        completion_tokens = _parse_int(row.get("total_completion_tokens"))
        prompt_chars_total = _parse_int(row.get("prompt_chars_total"))
        completion_chars_total = _parse_int(row.get("completion_chars_total"))
        chars_total = _parse_int(row.get("chars_total"))
        reported_prompt_chars = _parse_int(row.get("reported_chars_prompt"))
        reported_completion_chars = _parse_int(row.get("reported_chars_completion"))
        reported_chars_total = _parse_int(row.get("reported_chars_total"))
        total_llm_calls = _parse_int(row.get("total_llm_calls"))
        llm_duration_sec = _parse_float(row.get("total_llm_duration_sec"))
        eval_duration_sec = _parse_float(row.get("total_eval_duration_sec"))
        processing_time_seconds = _parse_float(row.get("processing_time_seconds"))
        missing_fields: list[str] = []
        invalid_fields: list[str] = []
        required_bool_fields = (
            "method_instrumentation",
            "method_two_step_repair",
            "direct_gen_passed",
            "final_passed",
        )
        parsed_required_bools: dict[str, bool] = {}
        for key in required_bool_fields:
            raw_value = row.get(key)
            parsed_value = _parse_optional_bool(raw_value)
            if raw_value in (None, ""):
                missing_fields.append(key)
                continue
            if parsed_value is None:
                invalid_fields.append(key)
                continue
            parsed_required_bools[key] = parsed_value
        stopped_reason_raw = row.get("stopped_reason")
        stopped_reason = str(stopped_reason_raw or "").strip()
        if stopped_reason_raw in (None, ""):
            missing_fields.append("stopped_reason")
        elif not stopped_reason:
            invalid_fields.append("stopped_reason")
        if missing_fields or invalid_fields:
            meta["error"] = "baseline_incompatible_output_schema"
            meta["tracecoder_output_csv"] = str(output_csv)
            if missing_fields:
                meta["schema_missing_fields"] = missing_fields
            if invalid_fields:
                meta["schema_invalid_fields"] = invalid_fields
            if sidecar_metrics:
                meta["metrics"] = sidecar_metrics
            return BaselineResult(
                repaired_code=None,
                success=False,
                iterations_used=int(sidecar_iterations or 0),
                token_usage=sidecar_usage if sidecar_usage else {},
                meta=meta,
            )

        usage: dict[str, int] = {}
        if prompt_tokens is not None:
            usage["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            usage["completion_tokens"] = completion_tokens
        if prompt_tokens is not None and completion_tokens is not None:
            usage["total_tokens"] = prompt_tokens + completion_tokens
        usage_available = "total_tokens" in usage
        if reported_prompt_chars is None:
            reported_prompt_chars = prompt_chars_total
        if reported_completion_chars is None:
            reported_completion_chars = completion_chars_total
        if reported_chars_total is None:
            reported_chars_total = chars_total

        debug_session_entries = (
            cast(list[object], debug_session_log) if isinstance(debug_session_log, list) else []
        )
        iterations_used = len(debug_session_entries)
        normalized_llm_duration_sec = max(0.0, float(llm_duration_sec or 0.0))
        normalized_eval_duration_sec = max(0.0, float(eval_duration_sec or 0.0))
        normalized_processing_time_seconds = max(0.0, float(processing_time_seconds or 0.0))
        method_instrumentation = parsed_required_bools["method_instrumentation"]
        method_two_step_repair = parsed_required_bools["method_two_step_repair"]
        direct_gen_passed = parsed_required_bools["direct_gen_passed"]
        final_passed = parsed_required_bools["final_passed"]
        orchestration_duration_sec = max(
            0.0,
            normalized_processing_time_seconds
            - normalized_llm_duration_sec
            - normalized_eval_duration_sec,
        )

        meta.update(
            {
                "tracecoder_output_csv": str(output_csv),
                "final_eval": final_eval,
                "direct_gen_eval": direct_gen_eval,
                "debug_session_log": debug_session_log,
                "stopped_reason": stopped_reason,
                "method_instrumentation": method_instrumentation,
                "method_two_step_repair": method_two_step_repair,
                "processing_time_seconds": normalized_processing_time_seconds,
                "metrics": {
                    "llm_call_count": max(0, int(total_llm_calls or 0)),
                    "llm_duration_sec": normalized_llm_duration_sec,
                    "eval_duration_sec": normalized_eval_duration_sec,
                    "internal_total_duration_sec": normalized_processing_time_seconds,
                    "orchestration_duration_sec": orchestration_duration_sec,
                    "usage_available": usage_available,
                    "stopped_reason": stopped_reason,
                    "method_instrumentation": method_instrumentation,
                    "method_two_step_repair": method_two_step_repair,
                    "direct_gen_passed": direct_gen_passed,
                    "debug_added_value": final_passed and not direct_gen_passed,
                    **({"usage_source": "provider_reported"} if usage_available else {}),
                },
            }
        )
        metrics = cast(dict[str, Any], meta["metrics"])
        if isinstance(prompt_chars_total, int):
            metrics["prompt_chars_total"] = max(0, prompt_chars_total)
            metrics["chars_input"] = max(0, prompt_chars_total)
        if isinstance(completion_chars_total, int):
            metrics["completion_chars_total"] = max(0, completion_chars_total)
            metrics["chars_output"] = max(0, completion_chars_total)
        if isinstance(chars_total, int):
            metrics["chars_total"] = max(0, chars_total)
        elif isinstance(prompt_chars_total, int) and isinstance(completion_chars_total, int):
            metrics["chars_total"] = max(0, prompt_chars_total) + max(0, completion_chars_total)
        if isinstance(reported_prompt_chars, int):
            metrics["reported_chars_prompt"] = max(0, reported_prompt_chars)
        if isinstance(reported_completion_chars, int):
            metrics["reported_chars_completion"] = max(0, reported_completion_chars)
        if isinstance(reported_chars_total, int):
            metrics["reported_chars_total"] = max(0, reported_chars_total)
        elif isinstance(reported_prompt_chars, int) and isinstance(reported_completion_chars, int):
            metrics["reported_chars_total"] = max(0, reported_prompt_chars) + max(
                0, reported_completion_chars
            )
        if sidecar_metrics:
            merged_metrics = dict(metrics)
            merged_metrics.update(sidecar_metrics)
            metrics = merged_metrics
            meta["metrics"] = metrics
        if (
            str(metrics.get("last_llm_failure_kind") or "").strip()
            == "openai_compat_reasoning_only"
            or str(meta.get("compat_message") or "").strip()
        ):
            meta["error"] = "openai_compat_reasoning_only"

        # Validate final_code — never return prose.
        if final_code:
            try:
                ast.parse(final_code)
            except SyntaxError:
                final_code = None

        return BaselineResult(
            repaired_code=final_code or None,
            success=final_passed,
            iterations_used=(
                int(sidecar_iterations) if isinstance(sidecar_iterations, int) else iterations_used
            ),
            token_usage=_resolve_token_usage(
                fallback_usage=usage if usage_available else {},
                sidecar_metrics=sidecar_metrics,
                sidecar_usage=sidecar_usage,
            ),
            meta=meta,
        )
