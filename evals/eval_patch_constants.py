"""Lightweight constants shared by eval patch helpers."""

from __future__ import annotations

PATCH_APPLY_TIMEOUT_SEC = 10

FORCED_PATCH_MARKER = "# llmdebug-force-attempt"
"""Marker comment of runner- and patcher-generated forced (no-op) patches (LLM-13)."""
