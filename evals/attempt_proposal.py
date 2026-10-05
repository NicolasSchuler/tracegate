"""Patch and plan proposal under an attempt's shared wall-clock budget.

Also checks that a deterministic run's backend acknowledged the seed."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from evals.patchers.payload_types import as_patcher_meta


def _measure_attempt_wall_clock_sec(attempt_start_monotonic: float) -> float:
    return max(0.0, float(time.perf_counter() - attempt_start_monotonic))


def _ensure_deterministic_seed_ack(
    *,
    deterministic: bool,
    seed: int | None,
    patcher_name: str,
    patch_meta: dict[str, Any],
    case_id: str,
    condition: str,
    attempt: int,
) -> None:
    if not deterministic or patcher_name != "openai":
        return

    requested = patch_meta.get("api_compat_seed_requested")
    acknowledged = patch_meta.get("api_compat_seed_acknowledged")
    if requested != seed:
        raise RuntimeError(
            "deterministic seed mismatch for "
            f"{case_id}/{condition}/attempt={attempt}: requested={requested!r}, expected={seed!r}"
        )
    if acknowledged is not True:
        error = patch_meta.get("error")
        body = patch_meta.get("body")
        exc = patch_meta.get("exc")
        details = error or body or exc or "backend did not acknowledge seed support"
        raise RuntimeError(
            f"deterministic mode failed for {case_id}/{condition}/attempt={attempt}: {details}"
        )


def _remaining_attempt_propose_budget_sec(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, float(deadline_monotonic) - time.monotonic())


def _build_attempt_budget_exhausted_patch_result(
    *,
    patcher: Any,
    remaining_budget_sec: float,
) -> Any:
    from evals.patchers.base import PatchResult

    meta: dict[str, Any] = {
        "error": "request_budget_exceeded",
        "effective_timeout_sec": 0.0,
        "remaining_budget_sec": max(0.0, float(remaining_budget_sec)),
        "llm_call_count": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "transport_retry_count": 0,
        "attempt_shared_propose_deadline_enabled": True,
        "attempt_shared_propose_budget_exhausted": True,
        "attempt_shared_propose_budget_remaining_sec": max(0.0, float(remaining_budget_sec)),
    }
    patcher_name = str(getattr(patcher, "name", "") or "")
    if patcher_name:
        meta["patcher"] = patcher_name
    model_name = str(getattr(patcher, "model", "") or "")
    if model_name:
        meta["model"] = model_name
    base_url = str(getattr(patcher, "base_url", "") or "")
    if base_url:
        meta["base_url"] = base_url
    response_mode = str(getattr(patcher, "response_mode", "") or "")
    if response_mode:
        meta["response_mode"] = response_mode
    return PatchResult(diff=None, meta=as_patcher_meta(meta))


def _build_attempt_budget_exhausted_plan_result(
    *,
    patcher: Any,
    remaining_budget_sec: float,
) -> Any:
    from evals.patchers.base import PatchPlanResult

    meta: dict[str, Any] = {
        "error": "request_budget_exceeded",
        "effective_timeout_sec": 0.0,
        "remaining_budget_sec": max(0.0, float(remaining_budget_sec)),
        "llm_call_count": 0,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "transport_retry_count": 0,
        "attempt_shared_propose_deadline_enabled": True,
        "attempt_shared_propose_budget_exhausted": True,
        "attempt_shared_propose_budget_remaining_sec": max(0.0, float(remaining_budget_sec)),
    }
    patcher_name = str(getattr(patcher, "name", "") or "")
    if patcher_name:
        meta["patcher"] = patcher_name
    model_name = str(getattr(patcher, "model", "") or "")
    if model_name:
        meta["model"] = model_name
    base_url = str(getattr(patcher, "base_url", "") or "")
    if base_url:
        meta["base_url"] = base_url
    response_mode = str(getattr(patcher, "response_mode", "") or "")
    if response_mode:
        meta["response_mode"] = response_mode
    return PatchPlanResult(plan_text=None, raw_response=None, meta=as_patcher_meta(meta))


def _patcher_with_shared_attempt_budget(
    *,
    patcher: Any,
    attempt_deadline_monotonic: float | None,
) -> tuple[Any, float | None, float | None, bool]:
    remaining_budget_sec = _remaining_attempt_propose_budget_sec(attempt_deadline_monotonic)
    if remaining_budget_sec is None:
        return patcher, None, None, False

    if str(getattr(patcher, "name", "")).strip() != "openai":
        return patcher, remaining_budget_sec, None, False

    raw_max_propose_sec = getattr(patcher, "max_propose_sec", 0.0)
    try:
        configured_max_propose_sec = float(raw_max_propose_sec)
    except (TypeError, ValueError):
        configured_max_propose_sec = 0.0
    if configured_max_propose_sec <= 0:
        return patcher, remaining_budget_sec, None, False

    clamped_max_propose_sec = max(0.0, min(configured_max_propose_sec, remaining_budget_sec))
    if abs(clamped_max_propose_sec - configured_max_propose_sec) < 1e-6:
        return patcher, remaining_budget_sec, clamped_max_propose_sec, False

    try:
        clamped_patcher = replace(patcher, max_propose_sec=clamped_max_propose_sec)
    except Exception:
        return patcher, remaining_budget_sec, configured_max_propose_sec, False
    return clamped_patcher, remaining_budget_sec, clamped_max_propose_sec, True


def _propose_patch_with_snapshot_context(
    *,
    patcher: Any,
    prompt: str,
    context_dir: Path,
    cwd: Path,
    snapshot_full_text: str | None,
    snapshot_artifact_text: str | None,
    system_prompt_override: str | None = None,
    attempt_deadline_monotonic: float | None = None,
) -> Any:
    """Provide full snapshot context during propose, then restore artifact-capped file."""
    from evals.patchers.base import PatchResult

    (
        effective_patcher,
        remaining_budget_sec,
        call_budget_sec,
        budget_clamped,
    ) = _patcher_with_shared_attempt_budget(
        patcher=patcher,
        attempt_deadline_monotonic=attempt_deadline_monotonic,
    )
    if remaining_budget_sec is not None and remaining_budget_sec <= 0:
        return _build_attempt_budget_exhausted_patch_result(
            patcher=patcher,
            remaining_budget_sec=remaining_budget_sec,
        )
    if (
        effective_patcher is not patcher
        and hasattr(patcher, "_current_progress_callback")
        and hasattr(effective_patcher, "set_progress_callback")
    ):
        effective_patcher.set_progress_callback(patcher._current_progress_callback())

    def _call_propose_patch() -> Any:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "context_dir": context_dir,
            "cwd": cwd,
        }
        if system_prompt_override is not None:
            kwargs["system_prompt_override"] = system_prompt_override
        return effective_patcher.propose_patch(**kwargs)

    if (
        snapshot_full_text is None
        or snapshot_artifact_text is None
        or snapshot_full_text == snapshot_artifact_text
    ):
        result = _call_propose_patch()
    else:
        snapshot_path = context_dir / "snapshot.json"
        snapshot_path.write_text(snapshot_full_text, encoding="utf-8")
        try:
            result = _call_propose_patch()
        finally:
            snapshot_path.write_text(snapshot_artifact_text, encoding="utf-8")

    if not hasattr(result, "diff") or not hasattr(result, "meta"):
        return result

    result_meta = result.meta
    if not isinstance(result_meta, dict):
        return result
    enriched_meta = dict(result_meta)
    if remaining_budget_sec is not None:
        enriched_meta["attempt_shared_propose_deadline_enabled"] = True
        enriched_meta["attempt_shared_propose_budget_remaining_sec"] = remaining_budget_sec
        if call_budget_sec is not None:
            enriched_meta["attempt_shared_propose_call_budget_sec"] = call_budget_sec
        if budget_clamped:
            enriched_meta["attempt_shared_propose_budget_clamped"] = True
    return PatchResult(diff=result.diff, meta=as_patcher_meta(enriched_meta))


def _plan_patch_with_snapshot_context(
    *,
    patcher: Any,
    prompt: str,
    context_dir: Path,
    cwd: Path,
    snapshot_full_text: str | None,
    snapshot_artifact_text: str | None,
    attempt_deadline_monotonic: float | None = None,
) -> Any:
    """Provide full snapshot context during repair planning, then restore artifact-capped file."""
    from evals.patchers.base import PatchPlanResult

    (
        effective_patcher,
        remaining_budget_sec,
        call_budget_sec,
        budget_clamped,
    ) = _patcher_with_shared_attempt_budget(
        patcher=patcher,
        attempt_deadline_monotonic=attempt_deadline_monotonic,
    )
    if remaining_budget_sec is not None and remaining_budget_sec <= 0:
        return _build_attempt_budget_exhausted_plan_result(
            patcher=patcher,
            remaining_budget_sec=remaining_budget_sec,
        )
    if (
        effective_patcher is not patcher
        and hasattr(patcher, "_current_progress_callback")
        and hasattr(effective_patcher, "set_progress_callback")
    ):
        effective_patcher.set_progress_callback(patcher._current_progress_callback())

    def _call_plan_patch() -> Any:
        return effective_patcher.plan_patch(prompt=prompt, context_dir=context_dir, cwd=cwd)

    if (
        snapshot_full_text is None
        or snapshot_artifact_text is None
        or snapshot_full_text == snapshot_artifact_text
    ):
        result = _call_plan_patch()
    else:
        snapshot_path = context_dir / "snapshot.json"
        snapshot_path.write_text(snapshot_full_text, encoding="utf-8")
        try:
            result = _call_plan_patch()
        finally:
            snapshot_path.write_text(snapshot_artifact_text, encoding="utf-8")

    if not hasattr(result, "plan_text") or not hasattr(result, "meta"):
        return result

    result_meta = result.meta
    if not isinstance(result_meta, dict):
        return result
    enriched_meta = dict(result_meta)
    if remaining_budget_sec is not None:
        enriched_meta["attempt_shared_propose_deadline_enabled"] = True
        enriched_meta["attempt_shared_propose_budget_remaining_sec"] = remaining_budget_sec
        if call_budget_sec is not None:
            enriched_meta["attempt_shared_propose_call_budget_sec"] = call_budget_sec
        if budget_clamped:
            enriched_meta["attempt_shared_propose_budget_clamped"] = True
    return PatchPlanResult(
        plan_text=result.plan_text,
        raw_response=getattr(result, "raw_response", None),
        meta=as_patcher_meta(enriched_meta),
    )
