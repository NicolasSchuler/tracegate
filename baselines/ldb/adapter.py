"""LDB (LLMDebugger) baseline adapter.

Runs LDB's iterative debugging loop on llmdebug eval cases.  LDB instruments
the buggy code with runtime tracing, feeds execution traces to an LLM for
diagnosis, and generates a repair.

The adapter invokes LDB's core logic as a subprocess to avoid dependency
conflicts (LDB pins different versions of openai, etc.).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineConfig,
    BaselineResult,
    observed_repo_state,
    require_pinned_commit,
)
from baselines.case_schemas import LDBCaseInput
from baselines.contracts import BaselineContract
from baselines.pins import get_pinned_repo_commit
from evals.adaptive_controller_packet import (
    parse_controller_packet,
    render_hybrid_controller_guidance,
)
from evals.subprocess_monitor import run_monitored_subprocess

from .case_converter import convert_case

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_API_TIMEOUT_PATTERN = re.compile(
    r"\b(APITimeoutError|request timed out|timed out while requesting)\b",
    flags=re.IGNORECASE,
)
_API_ERROR_PATTERN = re.compile(
    r"\b("
    r"APIConnectionError|APIStatusError|RateLimitError|InternalServerError|"
    r"ServiceUnavailableError|Connection error"
    r")\b",
    flags=re.IGNORECASE,
)
_UPSTREAM_LOCAL_MODELS = frozenset({"starcoder", "codellama"})


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


def _resolve_upstream_model_selector(requested_model: str | None) -> tuple[str, str | None]:
    raw_model = str(requested_model or "").strip()
    if not raw_model:
        return "gpt-4", None

    lowered = raw_model.lower()
    if "gpt-4" in lowered or "gpt-3.5" in lowered or lowered in _UPSTREAM_LOCAL_MODELS:
        return raw_model, raw_model

    # Upstream LDB only recognizes a narrow model-name set, so keep its GPT path
    # while preserving the real OpenAI-compatible model id for outbound requests.
    return "gpt-4", raw_model


def _has_llmdebug_dataset_support(repo: Path) -> bool:
    try:
        utils_text = (repo / "programming" / "utils.py").read_text(encoding="utf-8")
        ldb_text = (repo / "programming" / "ldb.py").read_text(encoding="utf-8")
        model_text = (repo / "programming" / "generators" / "model.py").read_text(encoding="utf-8")
        py_generate_text = (repo / "programming" / "generators" / "py_generate.py").read_text(
            encoding="utf-8"
        )
    except OSError:
        return False
    return (
        '"llmdebug"' in utils_text
        and '"llmdebug"' in ldb_text
        and "llm_duration_sec" in ldb_text
        and "usage_available" in ldb_text
        and "is_passing" in ldb_text
        and "test_feedback" in ldb_text
        and "_record_metrics" in model_text
        and "LLMDEBUG_BASELINE_OPENAI_COMPAT_PROFILE" in model_text
        and "LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL" in model_text
        and "compat_error" in model_text
        and "super().__init__(model_name)" in model_text
        and 'dataset_type in ["HumanEval", "MBPP", "llmdebug"]' in py_generate_text
    )


def _parse_float(raw: object) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _parse_int(raw: object) -> int | None:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


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


def _coerce_subprocess_text(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def _failed_feedback_count(raw: object) -> int:
    if isinstance(raw, list):
        return len(cast(list[object], raw))
    if raw is None:
        return 0
    if isinstance(raw, str) and not raw.strip():
        return 0
    return 1


def _summarize_test_feedback(raw: object) -> dict[str, int]:
    counts = [_failed_feedback_count(item) for item in cast(list[object], raw)]
    improvement_steps = 0
    regression_steps = 0
    stagnation_steps = 0
    for previous, current in pairwise(counts):
        if current < previous:
            improvement_steps += 1
        elif current > previous:
            regression_steps += 1
        else:
            stagnation_steps += 1
    return {
        "test_feedback_steps": len(counts),
        "test_feedback_nonempty_steps": sum(1 for count in counts if count > 0),
        "test_feedback_improvement_steps": improvement_steps,
        "test_feedback_regression_steps": regression_steps,
        "test_feedback_stagnation_steps": stagnation_steps,
        "test_feedback_final_fail_count": counts[-1] if counts else 0,
    }


def _read_metrics_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _infer_root_cause_error(
    *,
    meta: dict[str, Any],
    stdout: str,
    stderr: str,
) -> str:
    if str(meta.get("compat_message") or "").strip():
        return "openai_compat_reasoning_only"
    combined = "\n".join(
        part
        for part in (
            str(meta.get("stderr_tail") or ""),
            str(meta.get("stdout_tail") or ""),
            stderr,
            stdout,
        )
        if str(part).strip()
    )
    if not combined:
        return ""
    if _API_TIMEOUT_PATTERN.search(combined):
        return "api_timeout"
    if _API_ERROR_PATTERN.search(combined):
        return "api_error"
    return ""


def _apply_artifact_error(
    *,
    meta: dict[str, Any],
    artifact_error: str,
    stdout: str,
    stderr: str,
) -> None:
    root_cause_error = _infer_root_cause_error(meta=meta, stdout=stdout, stderr=stderr)
    if root_cause_error and root_cause_error != artifact_error:
        meta["error"] = root_cause_error
        meta["root_cause_error"] = root_cause_error
        meta["artifact_error"] = artifact_error
        return
    meta["error"] = artifact_error


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
        float(_parse_float(payload.get("orchestration_duration_sec")) or 0.0),
    )
    usage_reported_call_count = max(
        0,
        int(_parse_int(payload.get("usage_reported_call_count")) or 0),
    )
    usage_missing_call_count = max(
        0,
        int(_parse_int(payload.get("usage_missing_call_count")) or 0),
    )
    usage_available = (
        llm_call_count > 0
        and bool(_parse_optional_bool(payload.get("usage_available")))
        and usage_missing_call_count == 0
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
    iterations_used = _parse_int(payload.get("attempts_completed"))
    usage_source = str(payload.get("usage_source") or "").strip()
    is_passing = _parse_optional_bool(payload.get("is_passing"))
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
        "usage_available": usage_available,
        "usage_reported_call_count": usage_reported_call_count,
        "usage_missing_call_count": usage_missing_call_count,
    }
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
    if usage_source:
        metrics["usage_source"] = usage_source
    if isinstance(is_passing, bool):
        metrics["is_passing"] = is_passing
    if compat_message:
        metrics["compat_message"] = compat_message
    if compat_meta:
        metrics["compat_meta"] = compat_meta

    usage: dict[str, int] = {}
    if usage_available and prompt_tokens is not None and completion_tokens is not None:
        usage["prompt_tokens"] = max(0, prompt_tokens)
        usage["completion_tokens"] = max(0, completion_tokens)
        usage["total_tokens"] = max(
            0,
            total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
        )

    meta: dict[str, Any] = {
        "ldb_metrics_sidecar": str(sidecar_path),
    }
    if compat_message:
        meta["compat_message"] = compat_message
    if compat_meta:
        meta["compat_meta"] = compat_meta
    return metrics, usage, iterations_used, meta


def _live_progress_payload_for_log_dir(log_dir: Path) -> dict[str, Any] | None:
    sidecar_path = log_dir / "llmdebug_metrics.json"
    sidecar_payload = _read_metrics_sidecar(sidecar_path)
    metrics, _, iterations_used, _ = _parse_metrics_sidecar(
        sidecar_payload,
        sidecar_path=sidecar_path,
    )
    if not isinstance(sidecar_payload, dict) and not metrics and iterations_used is None:
        return None

    note_parts: list[str] = []
    status = (
        str(sidecar_payload.get("status") or "").strip()
        if isinstance(sidecar_payload, dict)
        else ""
    )
    if status:
        note_parts.append(f"status={status}")
    is_passing = metrics.get("is_passing")
    if isinstance(is_passing, bool):
        note_parts.append(f"is_passing={str(is_passing).lower()}")

    payload: dict[str, Any] = {
        "owner": "baseline",
        "source": "sidecar",
        "phase_label": "baseline: ldb subprocess",
        "waiting_for": "ldb_subprocess",
        "external_operation": "baseline_repair",
        "external_provider": "ldb",
    }
    if isinstance(iterations_used, int):
        payload["attempts_completed"] = max(0, iterations_used)
    llm_call_count = metrics.get("llm_call_count")
    if isinstance(llm_call_count, int):
        payload["llm_calls_observed"] = max(0, llm_call_count)
    eval_duration_sec = metrics.get("eval_duration_sec")
    if isinstance(eval_duration_sec, (int, float)):
        payload["eval_duration_sec_observed"] = max(0.0, float(eval_duration_sec))
    if note_parts:
        payload["note"] = " ".join(note_parts)
    return payload


class LDBAdapter(BaselineAdapter):
    """Adapter for the LDB (LLMDebugger) debugging baseline."""

    default_model: str = "gpt-4"
    _config: BaselineConfig | None = None
    repo_commit: str = ""

    @property
    def name(self) -> str:
        return "ldb"

    def contract(self) -> BaselineContract:
        return BaselineContract(
            baseline_name=self.name,
            upstream_commit=get_pinned_repo_commit(self.name),
            fidelity_level="upstream_patched",
            repair_scope_policy="single_file_only",
            prompt_contract_kind="upstream_native_prompt_contract",
            output_contract="single_file_repaired_source",
            accepted_case_shapes=(
                "single_file_buggy_py",
                "synchronous_top_level_function",
                "from_buggy_import_symbol_pytest_shape",
            ),
            patch_scope_summary=("compat", "telemetry", "behavioral_adaptation"),
            notes=(
                "Strict callable subset only; unsupported cases fail explicitly.",
                "Adapter lowers llmdebug cases into upstream HumanEval-like LDB rows.",
            ),
        )

    def setup(self, config: BaselineConfig) -> None:
        """Validate the LDB repo clone and configuration."""
        self._config = config
        repo = config.repo_dir

        if not repo.is_dir():
            msg = f"LDB repo not found at {repo}. Run baselines/setup_repos.sh first."
            raise RuntimeError(msg)

        programming_dir = repo / "programming"
        if not programming_dir.is_dir():
            msg = f"LDB programming/ directory not found at {programming_dir}."
            raise RuntimeError(msg)

        # Check key files exist.
        for required in ["ldb.py", "main.py"]:
            if not (programming_dir / required).exists():
                msg = f"LDB missing required file: programming/{required}"
                raise RuntimeError(msg)
        if not _has_llmdebug_dataset_support(repo):
            msg = (
                "LDB repo is missing llmdebug compatibility support. "
                "Re-run baselines/setup_repos.sh or apply baselines/ldb/patches/0001-add-llmdebug-dataset-support.patch."
            )
            raise RuntimeError(msg)
        self.repo_commit = _resolve_repo_commit(repo)
        self.repo_state = observed_repo_state(repo)
        require_pinned_commit(
            name=self.name,
            observed_commit=self.repo_commit,
            pinned_commit=get_pinned_repo_commit(self.name),
        )

    def validate_extra_config(self, config: BaselineConfig) -> None:
        from baselines.extra_configs import LDBExtraConfig

        LDBExtraConfig.validate(
            config.extra,
            adapter_name=self.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> LDBCaseInput:
        """Convert llmdebug case → LDB input format."""
        case_id = case_context.case_id if case_context else case_dir.name
        return convert_case(case_dir, case_id=case_id)

    def run_repair(
        self,
        case_input: dict[str, Any],
        config: BaselineConfig,
    ) -> BaselineResult:
        """Run LDB's debugging loop on a single case.

        Creates a temporary workspace, writes LDB input files, and
        executes the LDB debugging script as a subprocess.
        """
        assert self._config is not None
        repo = config.repo_dir

        with tempfile.TemporaryDirectory(prefix="ldb_") as tmpdir:
            work_dir = Path(tmpdir)
            dataset_file = work_dir / "dataset.jsonl"
            seed_file = work_dir / "seed.jsonl"
            test_file = work_dir / "tests.jsonl"
            root_dir = work_dir / "logs"
            root_dir.mkdir()
            run_name = str(case_input["task_id"]).replace("/", "__")
            log_dir = root_dir / run_name
            controller_packet = parse_controller_packet(
                case_input.get("llmdebug_controller_packet")
            )
            prompt_text = str(case_input["prompt"])
            controller_guidance = render_hybrid_controller_guidance(
                controller_packet,
                mode=str(config.extra.get("baseline_hybrid_guidance_mode") or "tuned_v1"),
                include_history=True,
                include_retry_plan=True,
                max_chars=2_000,
            ).text
            if controller_guidance:
                prompt_text = (
                    f"{prompt_text}\n\nLLMDebug controller guidance:\n{controller_guidance}"
                )

            dataset_row = {
                "task_id": case_input["task_id"],
                "prompt": prompt_text,
                "entry_point": case_input["entry_point"],
                "test": case_input["test"],
            }
            seed_row = {
                "task_id": case_input["task_id"],
                "solution": case_input["seed"],
            }
            test_row = {
                "task_id": case_input["task_id"],
                "given_tests": case_input["given_tests"],
            }
            for path, row in (
                (dataset_file, dataset_row),
                (seed_file, seed_row),
                (test_file, test_row),
            ):
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            # Build the LDB subprocess command.
            env = os.environ.copy()
            env.update(config.generation.to_env())  # D4; upstream doesn't read these yet
            env["OPENAI_API_KEY"] = config.api_key or env.get("OPENAI_API_KEY", "sk-placeholder")
            if config.api_base_url:
                env["OPENAI_BASE_URL"] = config.api_base_url
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
            cli_model_name, requested_model_name = _resolve_upstream_model_selector(config.model)
            if requested_model_name is not None:
                env["LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL"] = requested_model_name
            else:
                env.pop("LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL", None)
            existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
            env["PYTHONPATH"] = (
                f"{_PROJECT_ROOT}:{existing_pythonpath}"
                if existing_pythonpath
                else str(_PROJECT_ROOT)
            )

            cmd = [
                str(config.python_executable or Path(sys.executable)),
                str(repo / "programming" / "main.py"),
                "--run_name",
                run_name,
                "--root_dir",
                str(root_dir),
                "--dataset_path",
                str(dataset_file),
                "--strategy",
                "ldb",
                "--model",
                cli_model_name,
                "--max_iters",
                str(config.max_attempts),
                "--level",
                "block",
                "--seedfile",
                str(seed_file),
                "--testfile",
                str(test_file),
                "--n_proc",
                "1",
                "--pass_at_k",
                "1",
            ]

            # Add port if base URL has one.
            if config.api_base_url:
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(config.api_base_url)
                    if parsed.port:
                        cmd.extend(["--port", str(parsed.port)])
                except Exception:
                    pass

            try:
                monitored = run_monitored_subprocess(
                    cmd=cmd,
                    cwd=repo / "programming",
                    env=env,
                    timeout_sec=max(1, int(config.process_timeout_sec or 1.0)),
                    progress_callback=self._current_progress_callback(),
                    progress_event="baseline_progress",
                    progress_base_payload={
                        "owner": "baseline",
                        "phase_label": "baseline: ldb subprocess",
                        "waiting_for": "ldb_subprocess",
                        "external_operation": "baseline_repair",
                        "external_provider": "ldb",
                        "source": "phase",
                    },
                    state_supplier=lambda: _live_progress_payload_for_log_dir(log_dir),
                )
            except FileNotFoundError as exc:
                return BaselineResult(
                    repaired_code=None,
                    success=False,
                    meta={"error": "ldb_not_found", "exc": str(exc)},
                )
            if monitored.timed_out:
                return self._build_timeout_result(
                    log_dir=log_dir,
                    timeout_sec=float(config.process_timeout_sec or 0.0),
                    stdout=monitored.stdout,
                    stderr=monitored.stderr,
                )
            if monitored.returncode == -999999:  # pragma: no cover - defensive unreachable guard
                raise RuntimeError("unexpected sentinel returncode from monitored subprocess")
            if monitored.returncode == 127 and not (repo / "programming").exists():
                return BaselineResult(
                    repaired_code=None,
                    success=False,
                    meta={"error": "ldb_not_found", "exc": "programming directory missing"},
                )
            if (
                monitored.returncode == 127
                and not Path(str(config.python_executable or Path(sys.executable))).exists()
            ):
                return BaselineResult(
                    repaired_code=None,
                    success=False,
                    meta={"error": "ldb_not_found", "exc": "python executable missing"},
                )

            # Parse LDB output.
            return self._parse_output(
                log_dir=log_dir,
                stdout=monitored.stdout,
                stderr=monitored.stderr,
                returncode=monitored.returncode,
            )

    def _build_timeout_result(
        self,
        *,
        log_dir: Path,
        timeout_sec: float,
        stdout: str,
        stderr: str,
    ) -> BaselineResult:
        sidecar_path = log_dir / "llmdebug_metrics.json"
        sidecar_payload = _read_metrics_sidecar(sidecar_path)
        sidecar_metrics, sidecar_usage, sidecar_iterations, sidecar_meta = _parse_metrics_sidecar(
            sidecar_payload,
            sidecar_path=sidecar_path,
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
            meta=meta,
        )

    def _parse_output(
        self,
        *,
        log_dir: Path,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> BaselineResult:
        """Parse LDB's output JSONL to extract the repaired code."""
        meta: dict[str, Any] = {
            "returncode": returncode,
            "stdout_tail": stdout[-2000:] if stdout else "",
            "stderr_tail": stderr[-2000:] if stderr else "",
        }
        sidecar_path = log_dir / "llmdebug_metrics.json"
        sidecar_payload = _read_metrics_sidecar(sidecar_path)
        sidecar_metrics, sidecar_usage, sidecar_iterations, sidecar_meta = _parse_metrics_sidecar(
            sidecar_payload,
            sidecar_path=sidecar_path,
        )
        meta.update(sidecar_meta)
        if str(meta.get("compat_message") or "").strip():
            meta["error"] = "openai_compat_reasoning_only"

        jsonl_files = sorted(log_dir.rglob("*.jsonl"))
        if not jsonl_files:
            _apply_artifact_error(
                meta=meta,
                artifact_error="no_output_jsonl",
                stdout=stdout,
                stderr=stderr,
            )
            if sidecar_metrics:
                meta["metrics"] = sidecar_metrics
            return BaselineResult(
                repaired_code=None,
                success=False,
                iterations_used=int(sidecar_iterations or 0),
                token_usage=sidecar_usage if sidecar_usage else {},
                meta=meta,
            )

        # Read the last line of the first output file.
        last_row_metrics: dict[str, Any] | None = None
        last_row_usage: dict[str, int] = {}
        for jsonl_file in jsonl_files:
            try:
                lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
                if not lines:
                    continue
                last_result = json.loads(lines[-1])
                raw_is_passing = last_result.get("is_passing")
                raw_test_feedback = last_result.get("test_feedback")
                missing_fields: list[str] = []
                invalid_fields: list[str] = []
                is_passing = _parse_optional_bool(raw_is_passing)
                if "is_passing" not in last_result:
                    missing_fields.append("is_passing")
                elif is_passing is None:
                    invalid_fields.append("is_passing")
                if "test_feedback" not in last_result:
                    missing_fields.append("test_feedback")
                elif not isinstance(raw_test_feedback, list):
                    invalid_fields.append("test_feedback")
                if missing_fields or invalid_fields:
                    meta["error"] = "baseline_incompatible_output_schema"
                    meta["ldb_output_file"] = str(jsonl_file)
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
                repaired = last_result.get("solution") or last_result.get("completion", "")
                # Validate repaired code — never return prose.
                if repaired:
                    try:
                        ast.parse(repaired)
                    except SyntaxError:
                        repaired = ""
                is_solved = last_result.get("is_solved", False)
                debug_iter = last_result.get("debug_iter", 0)
                llm_call_count = _parse_int(last_result.get("llm_call_count"))
                llm_duration_sec = _parse_float(last_result.get("llm_duration_sec"))
                eval_duration_sec = _parse_float(last_result.get("eval_duration_sec"))
                processing_time_seconds = _parse_float(last_result.get("processing_time_seconds"))
                prompt_tokens = _parse_int(last_result.get("tokens_prompt"))
                completion_tokens = _parse_int(last_result.get("tokens_completion"))
                total_tokens = _parse_int(last_result.get("tokens_total"))
                prompt_chars_total = _parse_int(last_result.get("prompt_chars_total"))
                completion_chars_total = _parse_int(last_result.get("completion_chars_total"))
                chars_total = _parse_int(last_result.get("chars_total"))
                reported_prompt_chars = _parse_int(last_result.get("reported_chars_prompt"))
                reported_completion_chars = _parse_int(last_result.get("reported_chars_completion"))
                reported_chars_total = _parse_int(last_result.get("reported_chars_total"))
                normalized_llm_call_count = max(0, int(llm_call_count or 0))
                usage_available = normalized_llm_call_count > 0 and bool(
                    _parse_optional_bool(last_result.get("usage_available"))
                )
                if reported_prompt_chars is None:
                    reported_prompt_chars = prompt_chars_total
                if reported_completion_chars is None:
                    reported_completion_chars = completion_chars_total
                if reported_chars_total is None:
                    reported_chars_total = chars_total
                test_feedback_summary = _summarize_test_feedback(raw_test_feedback)
                usage: dict[str, int] = {}
                if usage_available and prompt_tokens is not None and completion_tokens is not None:
                    usage["prompt_tokens"] = prompt_tokens
                    usage["completion_tokens"] = completion_tokens
                    usage["total_tokens"] = (
                        total_tokens
                        if total_tokens is not None
                        else prompt_tokens + completion_tokens
                    )
                normalized_llm_duration_sec = max(0.0, float(llm_duration_sec or 0.0))
                normalized_eval_duration_sec = max(0.0, float(eval_duration_sec or 0.0))
                normalized_processing_time_seconds = max(0.0, float(processing_time_seconds or 0.0))
                orchestration_duration_sec = max(
                    0.0,
                    normalized_processing_time_seconds
                    - normalized_llm_duration_sec
                    - normalized_eval_duration_sec,
                )

                # BAS-15: metrics describe the attempt whether or not it found a repair.
                metrics = {
                    "llm_call_count": normalized_llm_call_count,
                    "llm_duration_sec": normalized_llm_duration_sec,
                    "eval_duration_sec": normalized_eval_duration_sec,
                    "internal_total_duration_sec": normalized_processing_time_seconds,
                    "orchestration_duration_sec": orchestration_duration_sec,
                    "usage_available": bool(usage_available and usage),
                    "is_passing": is_passing,
                    **test_feedback_summary,
                    **({"usage_source": "provider_reported"} if usage_available and usage else {}),
                }
                if isinstance(prompt_chars_total, int):
                    metrics["prompt_chars_total"] = max(0, prompt_chars_total)
                    metrics["chars_input"] = max(0, prompt_chars_total)
                if isinstance(completion_chars_total, int):
                    metrics["completion_chars_total"] = max(0, completion_chars_total)
                    metrics["chars_output"] = max(0, completion_chars_total)
                if isinstance(chars_total, int):
                    metrics["chars_total"] = max(0, chars_total)
                elif isinstance(prompt_chars_total, int) and isinstance(
                    completion_chars_total, int
                ):
                    metrics["chars_total"] = max(0, prompt_chars_total) + max(
                        0, completion_chars_total
                    )
                if isinstance(reported_prompt_chars, int):
                    metrics["reported_chars_prompt"] = max(0, reported_prompt_chars)
                if isinstance(reported_completion_chars, int):
                    metrics["reported_chars_completion"] = max(0, reported_completion_chars)
                if isinstance(reported_chars_total, int):
                    metrics["reported_chars_total"] = max(0, reported_chars_total)
                elif isinstance(reported_prompt_chars, int) and isinstance(
                    reported_completion_chars, int
                ):
                    metrics["reported_chars_total"] = max(0, reported_prompt_chars) + max(
                        0, reported_completion_chars
                    )
                if sidecar_metrics:
                    merged_metrics = dict(metrics)
                    merged_metrics.update(sidecar_metrics)
                    metrics = merged_metrics
                last_row_metrics = metrics
                last_row_usage = sidecar_usage if sidecar_usage else usage
                if repaired:
                    meta["ldb_output_file"] = str(jsonl_file)
                    meta["metrics"] = metrics
                    if (
                        str(meta.get("compat_message") or "").strip()
                        or str(last_result.get("error") or "").strip()
                        == "openai_compat_reasoning_only"
                    ):
                        meta["error"] = "openai_compat_reasoning_only"
                    exact_usage = sidecar_usage if sidecar_usage else usage
                    iterations_used = (
                        int(sidecar_iterations)
                        if isinstance(sidecar_iterations, int)
                        else int(debug_iter)
                    )
                    return BaselineResult(
                        repaired_code=repaired,
                        success=bool(is_solved),
                        iterations_used=iterations_used,
                        token_usage=exact_usage if exact_usage else {},
                        meta=meta,
                    )
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        meta["error"] = "no_solution_in_output"
        if last_row_metrics is not None:
            meta["metrics"] = last_row_metrics
        elif sidecar_metrics:
            meta["metrics"] = sidecar_metrics
        return BaselineResult(
            repaired_code=None,
            success=False,
            iterations_used=int(sidecar_iterations or 0),
            token_usage=last_row_usage or (sidecar_usage if sidecar_usage else {}),
            meta=meta,
        )

    def extract_diff(self, case_dir: Path, result: BaselineResult) -> str | None:
        """Produce unified diff from LDB's repaired code."""
        return super().extract_diff(case_dir, result)
