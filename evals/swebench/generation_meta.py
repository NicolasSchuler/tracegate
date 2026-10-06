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

GENERATION_META_FILENAME = "generation_meta.jsonl"
_APPEND_LOCK = threading.Lock()


def generation_meta_path(predictions_path: Path) -> Path:
    """The metadata sidecar of *predictions_path*."""
    return predictions_path.with_name(GENERATION_META_FILENAME)


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
