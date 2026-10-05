"""Classify patcher request failures, and recover from OpenAI transport outages."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from evals.attempt_usage import _nonnegative_int
from evals.eval_types import (
    EVAL_RECORD_TYPE_ATTEMPT,
    EVAL_RECORD_TYPE_INFRA_EVENT,
    PATCHER_CONTEXT_LIMIT_ERROR_MARKERS,
    RETRYABLE_HTTP_STATUS_CODES,
)
from evals.record_keys import KEY_INFRA_RESUME_ELIGIBLE


def _single_line_error(raw: str, *, max_chars: int = 240) -> str:
    text = " | ".join(segment.strip() for segment in str(raw).splitlines() if segment.strip())
    if not text:
        text = str(raw).strip() or "unknown runtime error"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _is_transport_runtime_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    markers = (
        "segmentation fault",
        "connection refused",
        "connection reset",
        "brokenpipeerror",
        "remotedisconnected",
        "temporarily unavailable",
        "server disconnected",
        "failed to establish a new connection",
    )
    return any(marker in text for marker in markers)


_INFRA_HTTP_STATUS_RE = re.compile(r"\b(?:status|code)\s*[:= -]?\s*(50[234])\b", re.IGNORECASE)


_INFRA_FAILURE_TEXT_MARKERS: tuple[tuple[str, str], ...] = (
    ("loading model", "backend_unavailable"),
    ("unavailable_error", "backend_unavailable"),
    ("temporarily unavailable", "backend_unavailable"),
    ("service unavailable", "backend_unavailable"),
    ("server disconnected", "transport"),
    ("connection refused", "transport"),
    ("connection reset", "transport"),
    ("brokenpipeerror", "transport"),
    ("remotedisconnected", "transport"),
    ("failed to establish a new connection", "transport"),
)


def _patcher_infra_failure_kind_from_meta(meta: Mapping[str, Any] | None) -> str:
    """Classify patcher metadata into a normalized infra failure kind."""
    if not isinstance(meta, Mapping):
        return ""
    meta_candidates = [meta]
    baseline_meta = meta.get("baseline_meta")
    if isinstance(baseline_meta, Mapping):
        meta_candidates.append(baseline_meta)
    for candidate in meta_candidates:
        error = str(candidate.get("error") or "").strip()
        if error in {"request_failed", "api_timeout"}:
            return "transport"
        if error == "http_error":
            status = _nonnegative_int(candidate.get("status"), default=0)
            if status in {502, 503, 504}:
                return "backend_unavailable"
            if _is_retryable_transport_http_status(status):
                return "transport"
        if error not in {"repair_failed", "api_error"}:
            continue

        status = _nonnegative_int(candidate.get("status"), default=0)
        if status in {502, 503, 504}:
            return "backend_unavailable"
        if _is_retryable_transport_http_status(status):
            return "transport"

        text_fields = [
            str(candidate.get("body") or ""),
            str(candidate.get("message") or ""),
            str(candidate.get("stderr") or ""),
            str(candidate.get("stdout") or ""),
            str(candidate.get("stderr_tail") or ""),
            str(candidate.get("stdout_tail") or ""),
            str(candidate.get("exc") or ""),
        ]
        combined = " ".join(part for part in text_fields if part).lower()
        if not combined:
            continue
        status_match = _INFRA_HTTP_STATUS_RE.search(combined)
        if status_match:
            status_code = int(status_match.group(1))
            if status_code in {502, 503, 504}:
                return "backend_unavailable"
            if _is_retryable_transport_http_status(status_code):
                return "transport"
        for marker, kind in _INFRA_FAILURE_TEXT_MARKERS:
            if marker in combined:
                return kind
        if error == "api_error":
            return "transport"
    return ""


def _record_is_transport_request_failure(record: dict[str, Any]) -> bool:
    if _record_type(record) != EVAL_RECORD_TYPE_ATTEMPT:
        return False
    note = str(record.get("note") or "")
    if note != "patcher_request_failed":
        return False
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return False
    if bool(patch_meta.get("patcher_request_failed")):
        return str(patch_meta.get("patcher_request_failure_kind") or "") == "transport"
    failed, kind = _classify_patcher_request_failure(patch_meta)
    return bool(failed and kind == "transport")


def _record_type(record: Mapping[str, Any]) -> str:
    raw = str(record.get("record_type") or "").strip()
    if raw == EVAL_RECORD_TYPE_INFRA_EVENT:
        return EVAL_RECORD_TYPE_INFRA_EVENT
    return EVAL_RECORD_TYPE_ATTEMPT


def _is_attempt_record(record: Mapping[str, Any]) -> bool:
    return _record_type(record) == EVAL_RECORD_TYPE_ATTEMPT


def _filter_attempt_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _is_attempt_record(record)]


def _result_meta_dict(result: Any) -> dict[str, Any]:
    meta = getattr(result, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _result_with_meta(result: Any, meta: dict[str, Any]) -> Any:
    from evals.patchers.base import (
        ContextCompactionResult,
        ContextRequestResult,
        PatchPlanResult,
        PatchResult,
    )

    if isinstance(result, PatchResult):
        return PatchResult(diff=result.diff, meta=meta)
    if isinstance(result, PatchPlanResult):
        return PatchPlanResult(
            plan_text=result.plan_text,
            raw_response=result.raw_response,
            meta=meta,
        )
    if isinstance(result, ContextRequestResult):
        return ContextRequestResult(
            request_json=result.request_json,
            raw_response=result.raw_response,
            meta=meta,
        )
    if isinstance(result, ContextCompactionResult):
        return ContextCompactionResult(
            compacted_text=result.compacted_text,
            raw_response=result.raw_response,
            meta=meta,
        )
    return result


class _OpenAITransportRecoveryGate:
    """Coordinate OpenAI transport outage recovery across worker threads."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_wait_sec: float,
        poll_sec: float,
    ) -> None:
        self._enabled = bool(enabled)
        self._max_wait_sec = max(0.0, float(max_wait_sec))
        self._poll_sec = max(0.1, float(poll_sec))
        self._condition = threading.Condition()
        self._outage_active = False
        self._outage_generation = 0
        self._leader_claimed_generation = 0
        self._last_outcome: dict[str, Any] | None = None

    def wait_if_open(self) -> None:
        if not self._enabled:
            return
        with self._condition:
            while self._outage_active:
                self._condition.wait(timeout=self._poll_sec)

    def recover(
        self,
        *,
        patcher: Any,
        operation_name: str,
    ) -> dict[str, Any]:
        if not self._enabled or self._max_wait_sec <= 0:
            return {
                "attempted": False,
                "recovered": False,
                "timed_out": True,
                "wait_sec": 0.0,
                "probe_attempts": [],
                "probe_meta": {},
                "operation": operation_name,
                "leader": False,
                "generation": 0,
            }

        started_at = time.monotonic()
        deadline = started_at + self._max_wait_sec
        with self._condition:
            if not self._outage_active:
                self._outage_active = True
                self._outage_generation += 1
                self._leader_claimed_generation = self._outage_generation
                generation = self._outage_generation
                leader = True
            else:
                generation = self._outage_generation
                leader = False

        if not leader:
            with self._condition:
                while self._outage_active and self._outage_generation == generation:
                    self._condition.wait(timeout=self._poll_sec)
                outcome = dict(self._last_outcome or {})
            outcome.setdefault("attempted", True)
            outcome.setdefault("operation", operation_name)
            outcome["leader"] = False
            return outcome

        probe_attempts: list[dict[str, Any]] = []
        recovered = False
        probe_meta: dict[str, Any] = {}
        while time.monotonic() < deadline:
            probe_started = time.monotonic()
            try:
                ready, raw_probe_meta = patcher.probe_server_ready()
            except Exception as exc:
                ready = False
                raw_probe_meta = {
                    "error": "server_ready_probe_failed",
                    "exc": f"{type(exc).__name__}: {exc}",
                    "server_ready": False,
                }
            probe_meta = dict(raw_probe_meta or {})
            probe_attempts.append(
                {
                    "provider": str(probe_meta.get("server_ready_probe_provider") or ""),
                    "error": str(probe_meta.get("error") or ""),
                    "status": probe_meta.get("status"),
                    "duration_sec": max(0.0, time.monotonic() - probe_started),
                }
            )
            if ready:
                recovered = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self._poll_sec, remaining))
        finished_at = time.monotonic()

        outcome = {
            "attempted": True,
            "recovered": recovered,
            "timed_out": not recovered,
            "wait_sec": max(0.0, min(self._max_wait_sec, finished_at - started_at)),
            "probe_attempts": probe_attempts,
            "probe_meta": probe_meta,
            "operation": operation_name,
            "leader": True,
            "generation": generation,
        }
        with self._condition:
            self._last_outcome = dict(outcome)
            if self._outage_generation == generation:
                self._outage_active = False
                self._condition.notify_all()
        return outcome


def _request_context_failure_kind_from_meta(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return ""
    explicit = str(meta.get("request_context_failure_kind") or "").strip()
    if explicit:
        return explicit
    error = str(meta.get("error") or "").strip()
    if error == "request_budget_exceeded":
        return "request_budget_exceeded"
    if error == "openai_compat_reasoning_only":
        return "compat_reasoning_only"
    if bool(meta.get("request_context_backend_unavailable")):
        return "backend_unavailable"
    if error == "invalid_request_context_payload":
        return "invalid_payload"
    if error == "request_failed":
        return "transport"
    if error == "http_error":
        status = _nonnegative_int(meta.get("status"), default=0)
        if status in {502, 503, 504}:
            return "backend_unavailable"
        return "transport"
    return ""


def _request_context_recovery_failure_kind_from_meta(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return ""
    error = str(meta.get("error") or "").strip()
    if error == "request_failed":
        return "transport"
    if error == "http_error":
        status = _nonnegative_int(meta.get("status"), default=0)
        if status in {502, 503, 504}:
            return "backend_unavailable"
        if _is_retryable_transport_http_status(status):
            return "transport"
    return ""


def _request_context_retry_counts_from_meta(meta: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(meta, dict):
        return 0, 0
    retry_count = _nonnegative_int(meta.get("request_context_retry_count"), default=0)
    transport_retry_count = _nonnegative_int(
        meta.get("request_context_transport_retry_count"),
        default=0,
    )
    return retry_count, transport_retry_count


def _is_context_limit_http_error_message(body: Any) -> bool:
    text = str(body or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in PATCHER_CONTEXT_LIMIT_ERROR_MARKERS)


def _is_retryable_transport_http_status(status: Any) -> bool:
    if not isinstance(status, int):
        return False
    return status in RETRYABLE_HTTP_STATUS_CODES or (500 <= status < 600)


def _classify_patcher_request_failure(meta: dict[str, Any]) -> tuple[bool, str]:
    """Classify a patcher request error as (failed, kind).

    Returns ``(True, "transport")`` or ``(True, "context_limit")``
    when the error is a recognized failure mode, else ``(False, "")``.
    """
    seen_meta_ids: set[int] = set()
    candidate: dict[str, Any] | Any = meta
    while isinstance(candidate, dict) and id(candidate) not in seen_meta_ids:
        seen_meta_ids.add(id(candidate))
        error = str(candidate.get("error") or "").strip()
        if error == "openai_compat_reasoning_only":
            return True, "compat_reasoning_only"
        if error in {"request_failed", "request_budget_exceeded", "bad_request_url"}:
            return True, "transport"
        if error == "http_error" and _is_context_limit_http_error_message(candidate.get("body")):
            return True, "context_limit"
        if error == "http_error" and _is_retryable_transport_http_status(candidate.get("status")):
            return True, "transport"
        infra_kind = _patcher_infra_failure_kind_from_meta(candidate)
        if infra_kind:
            return True, infra_kind
        candidate = candidate.get("fallback_meta")
    return False, ""


def _transport_recovery_failure_kind_from_meta(meta: dict[str, Any] | None) -> str:
    return _patcher_infra_failure_kind_from_meta(meta)


def _sync_patcher_request_failure_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    failed: bool,
    kind: str = "",
    error: str = "",
    status: int | None = None,
    exc: str = "",
    runner_forced_patch_skipped: bool = False,
    runner_forced_patch_skip_reason: str = "",
) -> None:
    values: dict[str, Any] = {
        "patcher_request_failed": bool(failed),
        "patcher_request_failure_kind": str(kind or ""),
        "patcher_request_failure_error": str(error or ""),
        "patcher_request_failure_status": int(status) if isinstance(status, int) else None,
        "patcher_request_failure_exc": str(exc or ""),
        "runner_forced_patch_skipped": bool(runner_forced_patch_skipped),
        "runner_forced_patch_skip_reason": str(runner_forced_patch_skip_reason or ""),
    }
    patch_meta.update(values)
    attempt_metadata.update(values)


def _record_is_infra_resume_eligible(record: Mapping[str, Any]) -> bool:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, Mapping):
        return False
    return bool(patch_meta.get(KEY_INFRA_RESUME_ELIGIBLE))
