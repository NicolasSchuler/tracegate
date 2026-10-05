"""Small value-coercion and path helpers shared across the eval harness (X-04)."""

from __future__ import annotations

import glob
from typing import Any


def nonnegative_int(value: Any, *, default: int = 0) -> int:
    """``max(0, int(value))``, or *default* when *value* is not an integer."""
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, coerced)


def expand_paths(items: list[str]) -> list[str]:
    """Expand shell globs, sorted per pattern; a pattern without matches is kept as is."""
    out: list[str] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        out.extend(matches if matches else [str(item)])
    return out
