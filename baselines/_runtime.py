"""Runtime helpers shared by the baseline adapters (X-04).

Repo commits, subprocess output, usage and character totals, workspace rows, and the
lenient number parsing of upstream CSV/JSON results. Each used to be copied into two
to five adapters.
"""

from __future__ import annotations

import ast
import base64
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from baselines.adapter_base import BaselineConfig


@dataclass(frozen=True)
class _CommandOutcome:
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float = 0.0
    timed_out: bool = False
    timeout_sec: int | None = None


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


def _env_failure_error(exc: BaseException) -> str:
    return "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "env_client_failed"


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


class _RuntimeFeedbackMixin:
    """Runtime progress, failure feedback and per-case timeouts of the iterative adapters.

    Shared by the adapters that inherit it (X-04); the bodies moved unchanged.
    """

    if TYPE_CHECKING:
        # Provided by BaselineAdapter, which every host class also inherits.
        name: str

        def _emit_progress(self, event: str, payload: dict[str, Any] | None = None) -> None: ...

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

    def _case_timeout(self, *, case_input: dict[str, Any], config: BaselineConfig) -> int:
        raw_timeout = case_input.get("timeout_sec")
        if isinstance(raw_timeout, int) and raw_timeout > 0:
            return raw_timeout
        return max(1, int(config.process_timeout_sec or 1.0))


class _CandidateWorkspaceMixin:
    """Materialize a candidate repair into a workspace and run the case's tests on it.

    Shared by the adapters that inherit it (X-04); the bodies moved unchanged.
    """

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
