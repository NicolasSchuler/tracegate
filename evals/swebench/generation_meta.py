"""Per-prediction generation metadata kept beside official predictions files (DAT-18).

The predictions JSONL keeps the official three-key format. Facts the harness can't
carry, such as whether the completion stopped at ``max_tokens``, go to
``<condition>/generation_meta.jsonl``: one line per proposal, appended, so on resume
the last line for an instance describes the prediction in the predictions file.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals import _json as json
from evals.eval_patch_constants import patcher_meta_truncated
from evals.transport_failures import _classify_patcher_request_failure

GENERATION_META_FILENAME = "generation_meta.jsonl"
_APPEND_LOCK = threading.Lock()

_TRANSPORT_FAILURE_KINDS = frozenset({"transport", "backend_unavailable"})
"""Failure kinds that mean "no answer reached us", so there is no prediction (DAT-08)."""


def generation_meta_path(predictions_path: Path) -> Path:
    """The metadata sidecar of *predictions_path*."""
    return predictions_path.with_name(GENERATION_META_FILENAME)


def transport_request_error_from_patch_meta(meta: Mapping[str, Any] | None) -> str | None:
    """A returned transport failure, as a request error (DAT-08), or ``None``.

    ``OpenAICompatPatcher`` reports an exhausted timeout or an HTTP outage by
    *returning* ``PatchResult(diff=None, meta={"error": "request_failed", ...})``
    instead of raising, so a caller that only catches exceptions would persist the
    outage as an empty prediction and never retry it on ``--resume``. Deterministic
    model-side failures (a context-limit refusal, a reasoning-only answer) are not
    transport failures and stay empty predictions.
    """
    for candidate in _meta_chain(meta):
        failed, kind = _classify_patcher_request_failure(
            {key: value for key, value in candidate.items() if key != "fallback_meta"}
        )
        if not failed or kind not in _TRANSPORT_FAILURE_KINDS:
            continue
        error = str(candidate.get("error") or "").strip() or kind
        exc = str(candidate.get("exc") or "").strip()
        status = candidate.get("status")
        details = [part for part in (exc, f"status={status}" if status else "") if part]
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"patcher_{kind}: {error}{suffix}"
    return None


def _meta_chain(meta: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The patcher meta and every ``fallback_meta`` below it, outermost first."""
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    candidate: Any = meta
    while isinstance(candidate, Mapping) and id(candidate) not in seen:
        seen.add(id(candidate))
        chain.append(dict(candidate))
        candidate = candidate.get("fallback_meta")
    return chain


def generation_meta_from_patch_meta(
    meta: Mapping[str, Any] | None, *, request_error: str | None = None
) -> dict[str, Any]:
    """Summarize a patcher result: ``finish_reason``, truncation and the patcher error."""
    finish_reason = ""
    seen: set[int] = set()
    candidate: Any = meta
    while isinstance(candidate, Mapping) and id(candidate) not in seen:
        seen.add(id(candidate))
        value = candidate.get("finish_reason")
        if isinstance(value, str) and value:
            finish_reason = value  # the innermost call is the one that produced the answer
        candidate = candidate.get("fallback_meta")
    return {
        "finish_reason": finish_reason,
        "truncated": patcher_meta_truncated(meta) or finish_reason == "length",
        "patcher_error": str((meta or {}).get("error") or ""),
        "request_error": request_error or "",
    }


def append_generation_meta(
    predictions_path: Path, *, instance_id: str, generation_meta: Mapping[str, Any]
) -> None:
    """Append one proposal's metadata to the sidecar of *predictions_path*."""
    line = json.dumps({"instance_id": instance_id, **generation_meta}, sort_keys=True)
    path = generation_meta_path(predictions_path)
    with _APPEND_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
