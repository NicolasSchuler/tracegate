"""Run pytest for an eval attempt (local, one-shot or pooled containers) and keep its output."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from baselines.subprocess_monitor import run_monitored_subprocess
from evals.adaptive_progress import _adaptive_optional_bool, _adaptive_optional_int
from evals.eval_config import get_active_execution_config
from evals.eval_containers import _EvalContainerPool
from evals.eval_types import (
    DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
    EVAL_CONTAINER_LABEL_KEY,
    EVAL_CONTAINER_LABEL_VALUE,
    PERSISTED_PYTEST_OUTPUT_HARD_CHAR_CAP,
    PYTEST_TIMEOUT_RETURNCODE,
    Condition,
    ExecutionConfig,
    RunResult,
    _ActiveRunSettings,
)
from evals.oracle_integrity import clear_junit_reports, consume_junit_counts, junit_report_name
from evals.record_keys import TimeoutFields
from evals.run_context import RunContext
from llmdebug._hashing import xxh64_hexdigest_text

# Prepended to the pytest subprocess's PYTHONPATH so cases import this checkout's llmdebug.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _is_snapshot_condition(condition: Condition) -> bool:
    return condition in {
        Condition.LAYERED_ADAPTIVE_GATED,
        Condition.LAYERED_ADAPTIVE_LLM,
    }


def _run_pytest_pooled(
    *,
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    base_env: dict[str, str],
    execution: ExecutionConfig,
    pool: _EvalContainerPool,
) -> RunResult:
    """Run pytest in a pooled container via ``exec_run``; *base_env* from ``_base_pytest_env``."""
    env = dict(base_env)
    if execution.container_mount_repo_src:
        env["PYTHONPATH"] = "/repo_src"
    env.update(execution.container_env)

    command = _build_pytest_args(
        python_cmd="python",
        python_args=python_args,
        condition=condition,
    )

    container = pool.acquire()
    try:
        return container.exec_pytest(
            host_cwd=cwd,
            command=command,
            env=env,
            timeout_sec=timeout_sec,
        )
    finally:
        pool.release(container)


def run_pytest(
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    output_format: str,
    *,
    run_context: RunContext | None = None,
    progress_update: Callable[[str, dict[str, Any] | None], None] | None = None,
    progress_phase_label: str = "",
) -> RunResult:
    """Run pytest for *condition* using the run's execution backend.

    Delegates to ``_run_pytest_local``, ``_run_pytest_pooled`` or
    ``_run_pytest_testcontainers`` depending on ``ExecutionConfig.executor`` and
    whether *run_context* carries a container pool. Without a context, the
    entrypoint's active execution config applies (validate_cases, importers).
    """
    context = (
        run_context
        if run_context is not None
        else RunContext(execution=get_active_execution_config())
    )
    report_name = junit_report_name(python_args)
    if report_name is not None:
        # ATT-01: nothing planted under a report name may survive into the verdict.
        clear_junit_reports(cwd)
    active_execution = context.execution
    base_env = _base_pytest_env(
        output_format=output_format,
        snapshot_condition=_is_snapshot_condition(condition),
        settings=context.settings,
    )
    if active_execution.executor == "local":
        result = _run_pytest_local(
            cwd=cwd,
            python_args=python_args,
            condition=condition,
            timeout_sec=timeout_sec,
            base_env=base_env,
            progress_update=progress_update,
            progress_phase_label=progress_phase_label,
        )
    elif active_execution.executor == "testcontainers":
        pool = context.container_pool
        if pool is not None:
            result = _run_pytest_pooled(
                cwd=cwd,
                python_args=python_args,
                condition=condition,
                timeout_sec=timeout_sec,
                base_env=base_env,
                execution=active_execution,
                pool=pool,
            )
        else:
            result = _run_pytest_testcontainers(
                cwd=cwd,
                python_args=python_args,
                condition=condition,
                timeout_sec=timeout_sec,
                base_env=base_env,
                execution=active_execution,
            )
    else:
        raise RuntimeError(f"unsupported executor: {active_execution.executor!r}")
    if report_name is None:
        return result
    # The verdict comes from the report, read on the host after any executor.
    return replace(
        result, junit_requested=True, junit_counts=consume_junit_counts(cwd, report_name)
    )


def _build_pytest_args(
    *, python_cmd: str, python_args: list[str], condition: Condition
) -> list[str]:
    args = [python_cmd, *python_args]
    if _is_snapshot_condition(condition):
        if "-m" in python_args and "pytest" in python_args:
            idx = python_args.index("pytest")
            args = [
                python_cmd,
                *python_args[: idx + 1],
                "-p",
                "llmdebug.pytest_plugin",
                *python_args[idx + 1 :],
            ]
        else:
            args = [python_cmd, *python_args, "-p", "llmdebug.pytest_plugin"]
    return args


def _base_pytest_env(
    *,
    output_format: str,
    snapshot_condition: bool,
    settings: _ActiveRunSettings | None = None,
) -> dict[str, str]:
    env = {
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LLMDEBUG_OUTPUT_FORMAT": output_format,
    }
    run_settings = settings if settings is not None else _ActiveRunSettings()
    if run_settings.deterministic and run_settings.seed is not None:
        env["PYTHONHASHSEED"] = str(run_settings.seed)
        env.setdefault("TZ", "UTC")
    if snapshot_condition:
        env.setdefault("LLMDEBUG_DEBUG", "0")
        env.setdefault("LLMDEBUG_INCLUDE_GIT", "false")
        # Disable built-in hypothesis/category hints: the model should infer diagnosis itself.
        env["LLMDEBUG_CATEGORIZE_ERRORS"] = "0"
    return env


def _run_pytest_local(
    *,
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    base_env: dict[str, str],
    progress_update: Callable[[str, dict[str, Any] | None], None] | None = None,
    progress_phase_label: str = "",
) -> RunResult:
    """Run pytest as a local subprocess with timeout handling; *base_env* as for pooled runs."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC_ROOT), env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    env.update(base_env)
    args = _build_pytest_args(
        python_cmd=sys.executable,
        python_args=python_args,
        condition=condition,
    )

    monitored = run_monitored_subprocess(
        cmd=args,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
        progress_callback=progress_update,
        progress_event="pytest_activity",
        progress_base_payload={
            "owner": "pytest",
            "phase_label": str(progress_phase_label or "pytest"),
            "source": "phase",
        },
    )
    if monitored.timed_out:
        timeout_message = f"pytest timed out after {timeout_sec}s"
        stderr = (
            timeout_message if not monitored.stderr else f"{monitored.stderr}\n{timeout_message}"
        )
        return RunResult(
            returncode=PYTEST_TIMEOUT_RETURNCODE,
            duration_sec=float(monitored.duration_sec),
            stdout=monitored.stdout,
            stderr=stderr,
            timed_out=True,
            timeout_sec=timeout_sec,
        )
    return RunResult(
        returncode=int(monitored.returncode),
        duration_sec=float(monitored.duration_sec),
        stdout=monitored.stdout,
        stderr=monitored.stderr,
        timed_out=False,
        timeout_sec=None,
    )


def _status_code_from_wait_result(result: Any) -> int:
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        raw = result.get("StatusCode")
        if isinstance(raw, int):
            return raw
        if raw is None:
            return 1
        try:
            return int(raw)
        except Exception:
            return 1
    return 1


def _is_container_wait_timeout(exc: Exception) -> bool:
    name = type(exc).__name__
    if "ReadTimeout" in name:
        return True
    text = str(exc).lower()
    return "read timed out" in text or "timed out" in text


def _run_pytest_testcontainers(
    *,
    cwd: Path,
    python_args: list[str],
    condition: Condition,
    timeout_sec: int,
    base_env: dict[str, str],
    execution: ExecutionConfig,
) -> RunResult:
    """Run pytest inside a one-shot Docker container; *base_env* as for pooled runs."""
    try:
        from testcontainers.core.container import (  # pyright: ignore[reportMissingImports]
            DockerContainer,
        )
    except Exception as exc:
        raise RuntimeError(
            "testcontainers execution requested but dependency is missing "
            f"({type(exc).__name__}: {exc}). Install llmdebug[evals]."
        ) from exc

    container_env = dict(base_env)
    if execution.container_mount_repo_src:
        container_env["PYTHONPATH"] = "/repo_src"
    container_env.update(execution.container_env)

    command = _build_pytest_args(
        python_cmd="python",
        python_args=python_args,
        condition=condition,
    )
    tmpfs_size = max(1, int(execution.container_tmpfs_size_mb))
    kwargs: dict[str, Any] = {
        "working_dir": "/workspace",
        "network_mode": execution.container_network,
        "read_only": True,
        "tmpfs": {"/tmp": f"rw,noexec,nosuid,nodev,size={tmpfs_size}m"},
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "pids_limit": int(execution.container_pids_limit),
        "mem_limit": execution.container_memory,
        "nano_cpus": int(float(execution.container_cpus) * 1_000_000_000),
        "user": execution.container_user,
        "labels": {
            EVAL_CONTAINER_LABEL_KEY: EVAL_CONTAINER_LABEL_VALUE,
        },
    }

    container = (
        DockerContainer(execution.container_image)
        .with_volume_mapping(str(cwd), "/workspace", mode="rw")
        .with_command(command)
        .with_kwargs(**kwargs)
    )
    if execution.container_mount_repo_src:
        container = container.with_volume_mapping(str(_SRC_ROOT), "/repo_src", mode="ro")
    for key, value in sorted(container_env.items()):
        container = container.with_env(key, value)

    start = time.perf_counter()
    stdout = ""
    stderr = ""
    timed_out = False
    returncode = 1
    try:
        container.start()
        wrapped = container.get_wrapped_container()
        if wrapped is None:
            raise RuntimeError("testcontainer started without wrapped container handle")
        try:
            wait_result = wrapped.wait(timeout=timeout_sec)
            returncode = _status_code_from_wait_result(wait_result)
        except Exception as exc:
            if _is_container_wait_timeout(exc):
                timed_out = True
                returncode = PYTEST_TIMEOUT_RETURNCODE
                try:
                    wrapped.kill()
                except Exception:
                    pass
            else:
                raise RuntimeError(f"container wait failed: {type(exc).__name__}: {exc}") from exc

        out_raw, err_raw = container.get_logs()
        stdout = _coerce_subprocess_output(out_raw)
        stderr = _coerce_subprocess_output(err_raw)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "testcontainers execution failed: "
            f"{type(exc).__name__}: {exc} "
            f"(image={execution.container_image!r}, network={execution.container_network!r})"
        ) from exc
    finally:
        try:
            container.stop()
        except Exception:
            pass

    end = time.perf_counter()
    if timed_out:
        timeout_message = f"pytest timed out after {timeout_sec}s"
        stderr = timeout_message if not stderr else f"{stderr}\n{timeout_message}"
        return RunResult(
            returncode=PYTEST_TIMEOUT_RETURNCODE,
            duration_sec=float(end - start),
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            timeout_sec=timeout_sec,
        )
    return RunResult(
        returncode=int(returncode),
        duration_sec=float(end - start),
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        timeout_sec=None,
    )


def _truncate_persisted_pytest_output(
    text: str,
    *,
    tail_lines: int,
    hard_char_cap: int | None = None,
) -> str:
    normalized = text or ""
    normalized_tail_lines = max(0, int(tail_lines))
    if normalized_tail_lines > 0 and normalized:
        had_trailing_newline = normalized.endswith("\n")
        lines = normalized.splitlines()
        total_lines = len(lines)
        if total_lines > normalized_tail_lines:
            dropped_lines = total_lines - normalized_tail_lines
            normalized = "\n".join(lines[-normalized_tail_lines:])
            if had_trailing_newline:
                normalized += "\n"
            normalized = (
                f"...[llmdebug: truncated {dropped_lines} earlier line(s)]...\n{normalized}"
            )

    resolved_hard_char_cap = (
        PERSISTED_PYTEST_OUTPUT_HARD_CHAR_CAP if hard_char_cap is None else hard_char_cap
    )
    normalized_hard_char_cap = max(0, int(resolved_hard_char_cap))
    if normalized_hard_char_cap > 0 and len(normalized) > normalized_hard_char_cap:
        dropped_chars = len(normalized) - normalized_hard_char_cap
        normalized = (
            f"...[llmdebug: truncated {dropped_chars} earlier character(s)]...\n"
            f"{normalized[-normalized_hard_char_cap:]}"
        )
    return normalized


def _truncate_snapshot_string_for_artifact(value: str, *, max_chars: int) -> str:
    normalized_max_chars = max(0, int(max_chars))
    if normalized_max_chars <= 0 or len(value) <= normalized_max_chars:
        return value
    dropped_chars = len(value) - normalized_max_chars
    marker = f"...[llmdebug: snapshot string truncated {dropped_chars} chars]..."
    if normalized_max_chars <= len(marker):
        return marker[:normalized_max_chars]
    keep_chars = normalized_max_chars - len(marker)
    head_chars = keep_chars // 2
    tail_chars = keep_chars - head_chars
    return value[:head_chars] + marker + value[-tail_chars:]


def _truncate_snapshot_for_artifact(
    snapshot: dict[str, Any], *, max_string_chars: int
) -> dict[str, Any]:
    normalized_max_string_chars = max(0, int(max_string_chars))

    def _rewrite(value: Any) -> Any:
        if isinstance(value, str):
            return _truncate_snapshot_string_for_artifact(
                value, max_chars=normalized_max_string_chars
            )
        if isinstance(value, dict):
            return {key: _rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        return value

    rewritten = _rewrite(snapshot)
    return rewritten if isinstance(rewritten, dict) else dict(snapshot)


def _write_run_outputs(
    out_dir: Path,
    run: RunResult,
    *,
    prefix: str,
    tail_lines: int = DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
) -> None:
    stdout = _truncate_persisted_pytest_output(run.stdout or "", tail_lines=tail_lines)
    stderr = _truncate_persisted_pytest_output(run.stderr or "", tail_lines=tail_lines)
    (out_dir / f"{prefix}_stdout.txt").write_text(stdout, encoding="utf-8")
    (out_dir / f"{prefix}_stderr.txt").write_text(stderr, encoding="utf-8")
    combined = stdout + ("\n" if stdout and stderr else "") + stderr
    combined = _truncate_persisted_pytest_output(combined, tail_lines=0)
    (out_dir / f"{prefix}_combined.txt").write_text(combined, encoding="utf-8")


def _coerce_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _timeout_fields(run: RunResult) -> TimeoutFields:
    if run.timed_out:
        return {
            "timed_out": True,
            "timeout_sec": run.timeout_sec,
        }
    return {}


def _enrich_failure_context(
    *,
    base: dict[str, Any],
    note: str,
    patch_meta: dict[str, Any] | None,
    run: RunResult,
) -> dict[str, Any]:
    enriched = dict(base)
    enriched["note"] = str(note or "")
    enriched["timed_out"] = bool(run.timed_out)
    enriched["timeout_sec"] = int(run.timeout_sec) if isinstance(run.timeout_sec, int) else None
    meta = patch_meta if isinstance(patch_meta, dict) else {}
    verifier_reasons_raw = meta.get("patch_verifier_reasons")
    verifier_reasons = (
        [str(item) for item in verifier_reasons_raw]
        if isinstance(verifier_reasons_raw, list)
        else []
    )
    enriched["patch_verifier_rejected"] = bool(meta.get("patch_verifier_enabled")) and (
        meta.get("patch_verifier_passed") is False
    )
    enriched["patch_verifier_reasons"] = verifier_reasons
    enriched["patch_verifier_scope"] = str(meta.get("patch_verifier_scope") or "")
    enriched["patch_verifier_score"] = _adaptive_optional_int(meta.get("patch_verifier_score"))
    enriched["patch_verifier_source_scope_matched"] = _adaptive_optional_bool(
        meta.get("patch_verifier_source_scope_matched")
    )
    enriched["adaptive_progress_reason"] = str(meta.get("adaptive_progress_reason") or "")
    enriched["adaptive_helper_failure_kind"] = str(meta.get("adaptive_helper_failure_kind") or "")
    enriched["adaptive_helper_outcome"] = str(meta.get("adaptive_helper_outcome") or "")
    enriched["adaptive_helper_need_more_context"] = _adaptive_optional_bool(
        meta.get("adaptive_helper_need_more_context")
    )
    return enriched


def read_latest_snapshot(out_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read the most recent llmdebug snapshot from *out_dir*.

    Returns:
        Tuple of (snapshot_dict_or_None, error_string_or_None).
    """
    try:
        from llmdebug.output import get_latest_snapshot

        snap = get_latest_snapshot(str(out_dir))
        return snap, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _build_failure_signature_for_attempt(*, run: RunResult, snapshot: dict[str, Any] | None) -> str:
    """Compute a stable failure signature for an attempt outcome."""
    try:
        from llmdebug.rca_state import build_failure_signature

        # Raw (pre-RCA-12) signatures: failure_signature_repeat drives snapshot reveal,
        # stagnation features and the no-progress stop, so normalizing them would change
        # adaptive results versus earlier runs. Switching is a research decision.
        return build_failure_signature(
            snapshot, stdout=run.stdout, stderr=run.stderr, normalize=False
        )
    except Exception as e:
        sys.stderr.write(
            f"llmdebug eval: build_failure_signature failed, using text fallback: "
            f"{type(e).__name__}: {e}\n"
        )
        text = (run.stderr or "").strip() or (run.stdout or "").strip() or "unknown_failure"
        head = "\n".join([line for line in text.splitlines() if line.strip()][-3:])
        digest = xxh64_hexdigest_text(head)[:16]
        return f"TextFailure:{digest}"
