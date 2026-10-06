"""Shared progress-summary formatting for eval runs.

This module collapses noisy per-attempt metadata into stable outcome counters
and human-readable failure reasons. The normalization order matters: runtime
transport failures outrank case-record metadata, then timeout/context-limit
signals outrank lower-level patcher notes.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from evals.eval_types import PATCHER_CONTEXT_LIMIT_ERROR_MARKERS
from evals.record_keys import KEY_ADAPTIVE_PROGRESS_STOP_NOTE, KEY_INFRA_RESUME_ELIGIBLE

_ANSI_RESET = "\x1b[0m"
_ANSI_GREEN = "\x1b[32m"
_ANSI_RED = "\x1b[31m"
_ANSI_YELLOW = "\x1b[33m"
_ANSI_CYAN = "\x1b[36m"
_REASON_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_SUCCESS_NOTES = {"success", "no_patch_needed"}
_TRANSPORT_RUNTIME_ERROR_MARKERS = (
    "segmentation fault",
    "connection refused",
    "connection reset",
    "brokenpipeerror",
    "remotedisconnected",
    "temporarily unavailable",
    "server disconnected",
    "failed to establish a new connection",
)


def _stdout_is_tty(stream: TextIO | None = None) -> bool:
    target = sys.stdout if stream is None else stream
    isatty = getattr(target, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


@dataclass(frozen=True)
class ProgressPalette:
    """Tiny ANSI palette wrapper used by plain-text progress summaries."""

    enabled: bool

    def _wrap(self, text: str, code: str) -> str:
        if not self.enabled or not text:
            return text
        return f"{code}{text}{_ANSI_RESET}"

    def success_count(self, value: int) -> str:
        text = str(value)
        return self._wrap(text, _ANSI_GREEN) if value > 0 else text

    def failure_count(self, value: int) -> str:
        text = str(value)
        return self._wrap(text, _ANSI_RED) if value > 0 else text

    def warning_count(self, value: int) -> str:
        text = str(value)
        return self._wrap(text, _ANSI_YELLOW) if value > 0 else text

    def info_count(self, value: int) -> str:
        text = str(value)
        return self._wrap(text, _ANSI_CYAN) if value > 0 else text


def build_progress_palette(stream: TextIO | None = None) -> ProgressPalette:
    """Enable colors only for TTY output when ``NO_COLOR`` is unset."""
    return ProgressPalette(
        enabled=_stdout_is_tty(stream) and os.getenv("NO_COLOR") is None,
    )


def _normalize_reason_token(raw: str) -> str:
    normalized = _REASON_TOKEN_RE.sub("_", str(raw).strip().lower()).strip("_")
    return normalized or "failed"


def _humanize_reason(token: str) -> str:
    normalized = _normalize_reason_token(token)
    if normalized == "transport_runtime_error":
        return "transport/runtime error"
    if normalized == "runtime_error":
        return "runtime error"
    if normalized.startswith("patch_verifier_"):
        return "patch verifier: " + normalized.removeprefix("patch_verifier_").replace("_", " ")
    return normalized.replace("_", " ")


def _looks_like_timeout(meta: Mapping[str, Any]) -> bool:
    if bool(meta.get("timed_out")):
        return True
    error = _normalize_reason_token(str(meta.get("error") or ""))
    return error in {"timeout", "timed_out"}


def _looks_like_context_limit(meta: Mapping[str, Any]) -> bool:
    for key in ("body", "message", "error_message", "stdout", "stderr"):
        raw = meta.get(key)
        if not isinstance(raw, str):
            continue
        lowered = raw.lower()
        if any(marker in lowered for marker in PATCHER_CONTEXT_LIMIT_ERROR_MARKERS):
            return True
    return False


_SENTINEL_REASONS: frozenset[str] = frozenset(
    {
        "not_adaptive_condition",
    }
)


def _is_infra_interrupted_records(case_records: Sequence[Mapping[str, Any]]) -> bool:
    if not case_records:
        return False
    saw_infra = False
    for record in case_records:
        patch_meta = record.get("patch_meta")
        resume_eligible = isinstance(patch_meta, Mapping) and bool(
            patch_meta.get(KEY_INFRA_RESUME_ELIGIBLE)
        )
        if str(record.get("record_type") or "attempt").strip() == "infra_event":
            saw_infra = saw_infra or resume_eligible
            continue
        if not resume_eligible:
            return False
        saw_infra = True
    return saw_infra


def _first_reason(values: Sequence[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text not in _SENTINEL_REASONS:
            return text
    return ""


def normalize_failure_reason(
    *,
    runtime_error: str | None = None,
    case_records: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Collapse runtime and case metadata into one normalized failure label.

    Precedence is deliberate:
    runtime transport errors > timeout/context-budget signals >
    fail-fast/adaptive stop reasons > verifier reasons > generic error/note
    fields > ``failed`` fallback.
    """
    runtime_error_text = str(runtime_error or "").strip()
    if runtime_error_text:
        lowered = runtime_error_text.lower()
        if any(marker in lowered for marker in _TRANSPORT_RUNTIME_ERROR_MARKERS):
            return "transport_runtime_error"
        return "runtime_error"

    if not case_records:
        return "failed"

    terminal_record = case_records[-1]
    patch_meta_raw = terminal_record.get("patch_meta")
    patch_meta = patch_meta_raw if isinstance(patch_meta_raw, Mapping) else {}
    attempt_meta_raw = terminal_record.get("attempt_metadata")
    attempt_meta = attempt_meta_raw if isinstance(attempt_meta_raw, Mapping) else {}
    baseline_meta_raw = patch_meta.get("baseline_meta")
    baseline_meta = baseline_meta_raw if isinstance(baseline_meta_raw, Mapping) else {}

    if any(
        _looks_like_timeout(meta)
        for meta in (terminal_record, attempt_meta, patch_meta, baseline_meta)
    ):
        return "timeout"

    if any(_looks_like_context_limit(meta) for meta in (patch_meta, attempt_meta, baseline_meta)):
        return "context_limit"

    case_fail_fast_reason = _first_reason(
        [
            patch_meta.get("case_fail_fast_reason"),
            attempt_meta.get("case_fail_fast_reason"),
        ]
    )
    if case_fail_fast_reason:
        return _normalize_reason_token(case_fail_fast_reason)

    adaptive_stop_reason = _first_reason(
        [
            patch_meta.get(KEY_ADAPTIVE_PROGRESS_STOP_NOTE),
            patch_meta.get("adaptive_progress_stop_reason"),
            attempt_meta.get(KEY_ADAPTIVE_PROGRESS_STOP_NOTE),
            attempt_meta.get("adaptive_progress_stop_reason"),
        ]
    )
    if adaptive_stop_reason:
        return _normalize_reason_token(adaptive_stop_reason)

    verifier_reasons_raw = patch_meta.get("patch_verifier_reasons")
    if isinstance(verifier_reasons_raw, Sequence) and not isinstance(
        verifier_reasons_raw, (str, bytes)
    ):
        for item in verifier_reasons_raw:
            item_text = str(item or "").strip()
            if item_text:
                return f"patch_verifier_{_normalize_reason_token(item_text)}"

    for meta in (patch_meta, attempt_meta, baseline_meta):
        error_text = str(meta.get("error") or "").strip()
        if error_text:
            return _normalize_reason_token(error_text)

    note = str(terminal_record.get("note") or "").strip()
    if note and note not in _SUCCESS_NOTES:
        return _normalize_reason_token(note)

    return "failed"


def humanize_failure_reason(reason: str) -> str:
    """Convert a normalized failure token into UI-facing text."""
    return _humanize_reason(reason)


def classify_failure_reason(
    *,
    runtime_error: str | None = None,
    case_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, str]:
    """Return both the normalized token and its human-readable label."""
    normalized = normalize_failure_reason(
        runtime_error=runtime_error,
        case_records=case_records,
    )
    return normalized, _humanize_reason(normalized)


@dataclass
class RunProgressSummary:
    """Mutable aggregate of per-unit outcomes used by progress UIs."""

    success_units: int = 0
    failed_units: int = 0
    interrupted_units: int = 0
    runtime_error_units: int = 0
    failure_reasons: Counter[str] = field(default_factory=Counter)

    def record_runtime_error(self, reason: str) -> None:
        self.runtime_error_units += 1
        self.failure_reasons[normalize_failure_reason(runtime_error=reason)] += 1

    def record_case_records(self, case_records: Sequence[Mapping[str, Any]]) -> None:
        if any(bool(record.get("success")) for record in case_records):
            self.success_units += 1
            return
        if _is_infra_interrupted_records(case_records):
            self.interrupted_units += 1
            return
        self.failed_units += 1
        self.failure_reasons[normalize_failure_reason(case_records=case_records)] += 1

    def render_lines(
        self,
        *,
        skipped_units: int,
        palette: ProgressPalette,
        top_n: int = 3,
    ) -> list[str]:
        outcome_line = (
            "Outcome summary: "
            f"success={palette.success_count(self.success_units)} "
            f"failed={palette.failure_count(self.failed_units)} "
            f"interrupted={palette.warning_count(self.interrupted_units)} "
            f"runtime_errors={palette.warning_count(self.runtime_error_units)} "
            f"skipped={palette.info_count(skipped_units)}"
        )
        if not self.failure_reasons:
            return [outcome_line, "Top failure reasons: none"]

        ranked_reasons = sorted(
            self.failure_reasons.items(),
            key=lambda item: (-item[1], item[0]),
        )[: max(1, int(top_n))]
        reasons_text = ", ".join(
            f"{_humanize_reason(reason)}={count}" for reason, count in ranked_reasons
        )
        return [outcome_line, f"Top failure reasons: {reasons_text}"]
