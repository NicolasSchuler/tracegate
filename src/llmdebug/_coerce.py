"""Small coercion helpers for untrusted config values and snapshot payloads."""

from __future__ import annotations

import os
from typing import Any


def coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Convert *value* to ``int`` or return *default* on invalid input.

    Args:
        value: Raw value to convert with ``int()``.
        default: Fallback returned when conversion fails or the result is out of range.
        minimum: Inclusive lower bound; ``None`` disables the check.
        maximum: Inclusive upper bound; ``None`` disables the check.

    Returns:
        The converted integer, or *default* when it is invalid or outside the bounds.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and coerced < minimum:
        return default
    if maximum is not None and coerced > maximum:
        return default
    return coerced


def read_env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read and validate an integer environment variable.

    Args:
        name: Environment variable name.
        default: Fallback used when the variable is unset, invalid, or out of range.
        minimum: Inclusive lower bound; ``None`` disables the check.
        maximum: Inclusive upper bound; ``None`` disables the check.

    Returns:
        The parsed integer, or *default* when the variable is unset or invalid.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return coerce_int(raw.strip(), default=default, minimum=minimum, maximum=maximum)
