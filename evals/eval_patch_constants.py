"""Lightweight constants shared by eval patch helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PATCH_APPLY_TIMEOUT_SEC = 10

FORCED_PATCH_MARKER = "# llmdebug-force-attempt"
"""Marker comment of runner- and patcher-generated forced (no-op) patches (LLM-13)."""

TRUNCATED_RESPONSE_ERROR = "truncated_response"
"""Patcher error and attempt note for a completion cut off at ``max_tokens`` (LLM-02)."""


def patcher_meta_truncated(meta: Mapping[str, Any] | None) -> bool:
    """Whether the patcher's last answer was truncated, following ``fallback_meta``."""
    seen: set[int] = set()
    candidate: Any = meta
    while isinstance(candidate, Mapping) and id(candidate) not in seen:
        seen.add(id(candidate))
        if candidate.get("error") == TRUNCATED_RESPONSE_ERROR:
            return True
        candidate = candidate.get("fallback_meta")
    return False
