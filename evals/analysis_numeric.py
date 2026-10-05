"""Small numeric and formatting helpers shared by the analysis modules."""

from __future__ import annotations

import glob
import math
from typing import Any

import numpy as np


def _increment_string_count(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, str):
        return
    normalized = value.strip()
    if not normalized:
        return
    target[normalized] = int(target.get(normalized, 0) + 1)


def _as_float_or_none(value: Any) -> float | None:
    """Convert *value* to float, returning None on failure or non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _fmt_num_with_ci(value: Any, interval: Any) -> str:
    """``value [lo, hi]`` with the case-bootstrap interval when there is one."""
    if isinstance(interval, list) and len(interval) == 2 and value is not None:
        return f"{_fmt_num(value)} [{_fmt_num(interval[0])}, {_fmt_num(interval[1])}]"
    return _fmt_num(value)


def _fmt_num(value: Any) -> str:
    """Format a numeric value as a string, or ``'n/a'`` if None."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "n/a"


def _fmt_signed_percent(value: Any) -> str:
    """Format a value as a signed percentage, or ``'n/a'`` if None."""
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:+.1%}"


def _fmt_percent(value: Any) -> str:
    """Format a value as a percentage, or ``'n/a'`` if None."""
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.1%}"


def _as_int(value: Any) -> int:
    """Coerce a value to int, returning 0 on failure."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _mean(xs: list[float]) -> float | None:
    """Arithmetic mean, or None for an empty list."""
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _stddev(xs: list[float]) -> float | None:
    """Population standard deviation, or None for an empty list."""
    if not xs:
        return None
    return float(np.std(np.asarray(xs, dtype=float), ddof=0))


def _quantile(xs: list[float], q: float) -> float | None:
    """Quantile of a float list, or None if empty."""
    if not xs:
        return None
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def _ratio(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _median(xs: list[int]) -> float | None:
    """Median of an integer list, or None if empty."""
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return (ys[mid - 1] + ys[mid]) / 2.0


def _expand_paths(items: list[str]) -> list[str]:
    """Expand shell globs in a list of path strings."""
    out: list[str] = []
    for item in items:
        matches = sorted(glob.glob(item))
        out.extend(matches if matches else [item])
    return out
