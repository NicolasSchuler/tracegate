"""Token-bucket rate limiter for MCP tool handlers.

Provides a thread-safe token-bucket implementation that replenishes tokens at
a configurable rate and supports burst capacity.  Used by ``mcp_server`` to
prevent resource exhaustion under heavy automated IDE loops.

Environment variables (read by ``load_mcp_rate_limiter``, which ``mcp_server`` calls
once at import):
    LLMDEBUG_MCP_RATE_LIMIT  — sustained calls per second (default 10)
    LLMDEBUG_MCP_RATE_BURST  — maximum burst size (default 20)
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable


def _read_env_float(name: str, default: float) -> float:
    """Read a positive float from env var *name*, falling back to *default*."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_env_int(name: str, default: int) -> int:
    """Read a positive int from env var *name*, falling back to *default*."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter.

    The bucket starts full and refills continuously at *rate* tokens per second,
    up to *burst* tokens; each allowed call consumes one token.

    Args:
        rate: Token replenishment rate (tokens per second).
        burst: Maximum bucket capacity (allows short bursts above *rate*).
        clock: Callable returning the current time in seconds
            (defaults to :func:`time.monotonic`; injectable for testing).
    """

    __slots__ = ("_burst", "_clock", "_last_refill", "_lock", "_rate", "_tokens")

    def __init__(
        self,
        rate: float,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a full bucket.

        Raises:
            ValueError: If *rate* is not positive or *burst* is below 1.
        """
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")
        self._rate = float(rate)
        self._burst = int(burst)
        self._tokens = float(burst)  # start full
        self._last_refill = clock()
        self._lock = threading.Lock()
        self._clock = clock

    # -- properties -----------------------------------------------------------

    @property
    def rate(self) -> float:
        """Return the replenishment rate in tokens per second."""
        return self._rate

    @property
    def burst(self) -> int:
        """Return the maximum bucket capacity."""
        return self._burst

    # -- core API -------------------------------------------------------------

    def acquire(self) -> bool:
        """Consume one token if one is available.

        The bucket is first refilled for the time elapsed since the last refill,
        capped at the burst capacity.

        Returns:
            ``True`` if the call is allowed, ``False`` if rate-limited.
        """
        now = self._clock()
        with self._lock:
            elapsed = now - self._last_refill
            if elapsed > 0:
                self._tokens = min(
                    float(self._burst),
                    self._tokens + elapsed * self._rate,
                )
                self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


def load_mcp_rate_limiter(
    *,
    clock: Callable[[], float] = time.monotonic,
) -> TokenBucketRateLimiter:
    """Build a :class:`TokenBucketRateLimiter` from environment variables.

    Reads ``LLMDEBUG_MCP_RATE_LIMIT`` (calls/s, default 10) and
    ``LLMDEBUG_MCP_RATE_BURST`` (burst size, default 20). Missing, unparsable, or
    non-positive values fall back to the defaults.

    Args:
        clock: Callable returning the current time in seconds; injectable for testing.

    Returns:
        A full rate limiter configured from the environment.
    """
    rate = _read_env_float("LLMDEBUG_MCP_RATE_LIMIT", 10.0)
    burst = _read_env_int("LLMDEBUG_MCP_RATE_BURST", 20)
    return TokenBucketRateLimiter(rate, burst, clock=clock)
