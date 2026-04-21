"""Tests for MCP server token-bucket rate limiting."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from llmdebug import _json as json
from llmdebug._rate_limiter import (
    TokenBucketRateLimiter,
    _read_env_float,
    _read_env_int,
    load_mcp_rate_limiter,
)

# ---------------------------------------------------------------------------
# TokenBucketRateLimiter unit tests
# ---------------------------------------------------------------------------


class TestTokenBucketBasic:
    """Core acquire/reject behaviour."""

    def test_allows_within_burst(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=5, burst=5, clock=clock)
        results = [limiter.acquire() for _ in range(5)]
        assert results == [True] * 5

    def test_rejects_over_burst(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=5, burst=3, clock=clock)
        for _ in range(3):
            assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_rejects_immediately_after_exhaustion(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=10, burst=1, clock=clock)
        assert limiter.acquire() is True
        assert limiter.acquire() is False
        assert limiter.acquire() is False


class TestTokenBucketBurst:
    """Burst capacity semantics."""

    def test_burst_higher_than_rate(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=2, burst=10, clock=clock)
        allowed = sum(limiter.acquire() for _ in range(15))
        assert allowed == 10

    def test_burst_equals_one(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=100, burst=1, clock=clock)
        assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_burst_capped_on_refill(self) -> None:
        """After a long idle period tokens must not exceed burst."""
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=5, burst=3, clock=clock)
        # Drain all tokens
        for _ in range(3):
            limiter.acquire()
        # Advance far beyond what would refill past burst
        clock.advance(100.0)
        allowed = sum(limiter.acquire() for _ in range(5))
        assert allowed == 3  # capped at burst


class TestTokenReplenishment:
    """Token refill over simulated time."""

    def test_single_token_refill(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=10, burst=2, clock=clock)
        # Drain
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False
        # Advance 0.1s -> 10 tokens/s x 0.1s = 1 token
        clock.advance(0.1)
        assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_full_refill(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=5, burst=5, clock=clock)
        for _ in range(5):
            limiter.acquire()
        assert limiter.acquire() is False
        # 1s at rate=5 → 5 tokens
        clock.advance(1.0)
        results = [limiter.acquire() for _ in range(5)]
        assert results == [True] * 5
        assert limiter.acquire() is False

    def test_partial_refill(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=10, burst=10, clock=clock)
        for _ in range(10):
            limiter.acquire()
        # 0.3s at rate=10 → 3 tokens
        clock.advance(0.3)
        allowed = sum(limiter.acquire() for _ in range(5))
        assert allowed == 3


class TestTokenBucketThreadSafety:
    """Concurrent access must not exceed burst."""

    def test_concurrent_acquire(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=1000, burst=50, clock=clock)
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            r = limiter.acquire()
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == 50


class TestTokenBucketValidation:
    """Constructor validation."""

    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate must be > 0"):
            TokenBucketRateLimiter(rate=0, burst=5)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate must be > 0"):
            TokenBucketRateLimiter(rate=-1, burst=5)

    def test_zero_burst_raises(self) -> None:
        with pytest.raises(ValueError, match="burst must be >= 1"):
            TokenBucketRateLimiter(rate=5, burst=0)

    def test_negative_burst_raises(self) -> None:
        with pytest.raises(ValueError, match="burst must be >= 1"):
            TokenBucketRateLimiter(rate=5, burst=-3)


class TestTokenBucketProperties:
    """Read-only property access."""

    def test_rate_property(self) -> None:
        limiter = TokenBucketRateLimiter(rate=42.5, burst=10)
        assert limiter.rate == 42.5

    def test_burst_property(self) -> None:
        limiter = TokenBucketRateLimiter(rate=5, burst=99)
        assert limiter.burst == 99


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------


class TestEnvVarConfig:
    """load_mcp_rate_limiter reads env vars correctly."""

    def test_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 10.0
        assert limiter.burst == 20

    def test_custom_rate(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_LIMIT": "25"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 25.0

    def test_custom_burst(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_BURST": "50"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.burst == 50

    def test_custom_both(self) -> None:
        env = {"LLMDEBUG_MCP_RATE_LIMIT": "100", "LLMDEBUG_MCP_RATE_BURST": "200"}
        with patch.dict("os.environ", env):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 100.0
        assert limiter.burst == 200

    def test_invalid_rate_falls_back(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_LIMIT": "not_a_number"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 10.0

    def test_zero_rate_falls_back(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_LIMIT": "0"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 10.0

    def test_negative_rate_falls_back(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_LIMIT": "-5"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 10.0

    def test_invalid_burst_falls_back(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_BURST": "abc"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.burst == 20

    def test_zero_burst_falls_back(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_BURST": "0"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.burst == 20

    def test_float_rate(self) -> None:
        with patch.dict("os.environ", {"LLMDEBUG_MCP_RATE_LIMIT": "2.5"}):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 2.5

    def test_empty_string_ignored(self) -> None:
        env = {"LLMDEBUG_MCP_RATE_LIMIT": "", "LLMDEBUG_MCP_RATE_BURST": "  "}
        with patch.dict("os.environ", env):
            limiter = load_mcp_rate_limiter()
        assert limiter.rate == 10.0
        assert limiter.burst == 20


class TestReadEnvHelpers:
    """Direct tests for _read_env_float / _read_env_int."""

    def test_read_env_float_present(self) -> None:
        with patch.dict("os.environ", {"X": "3.14"}):
            assert _read_env_float("X", 1.0) == pytest.approx(3.14)

    def test_read_env_float_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _read_env_float("MISSING", 7.0) == 7.0

    def test_read_env_int_present(self) -> None:
        with patch.dict("os.environ", {"Y": "42"}):
            assert _read_env_int("Y", 1) == 42

    def test_read_env_int_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _read_env_int("MISSING", 99) == 99


# ---------------------------------------------------------------------------
# MCP server integration — _check_rate_limit
# ---------------------------------------------------------------------------


class TestMCPRateLimitIntegration:
    """Rate limiting integrated into MCP tool handlers."""

    def test_tool_returns_rate_limit_error_json(self) -> None:
        """When the limiter is exhausted, tools return a RATE_LIMITED error."""
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=5, burst=2, clock=clock)
        with patch("llmdebug.mcp_server._mcp_limiter", limiter):
            from llmdebug.mcp_server import _check_rate_limit

            # Drain the bucket
            assert _check_rate_limit("t", "json", "summary") is None
            assert _check_rate_limit("t", "json", "summary") is None
            # Third call → rate limited
            result = _check_rate_limit("t", "json", "summary")
            assert result is not None
            data = json.loads(result)
            assert data["error"]["code"] == "RATE_LIMITED"
            assert "Rate limit exceeded" in data["error"]["message"]

    def test_tool_returns_rate_limit_error_text(self) -> None:
        """Text-format returns a plain error string."""
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=5, burst=1, clock=clock)
        with patch("llmdebug.mcp_server._mcp_limiter", limiter):
            from llmdebug.mcp_server import _check_rate_limit

            assert _check_rate_limit("t", "text", "summary") is None
            result = _check_rate_limit("t", "text", "summary")
            assert result is not None
            assert "Rate limit exceeded" in result

    def test_allowed_calls_return_none(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=100, burst=100, clock=clock)
        with patch("llmdebug.mcp_server._mcp_limiter", limiter):
            from llmdebug.mcp_server import _check_rate_limit

            for _ in range(50):
                assert _check_rate_limit("t", "json", "summary") is None

    def test_recovery_after_wait(self) -> None:
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=10, burst=2, clock=clock)
        with patch("llmdebug.mcp_server._mcp_limiter", limiter):
            from llmdebug.mcp_server import _check_rate_limit

            # Exhaust
            _check_rate_limit("t", "json", "summary")
            _check_rate_limit("t", "json", "summary")
            assert _check_rate_limit("t", "json", "summary") is not None
            # Wait 0.2s → 2 tokens
            clock.advance(0.2)
            assert _check_rate_limit("t", "json", "summary") is None

    def test_shared_across_tools(self) -> None:
        """A single limiter is shared across different tool names."""
        clock = _FakeClock(0.0)
        limiter = TokenBucketRateLimiter(rate=5, burst=3, clock=clock)
        with patch("llmdebug.mcp_server._mcp_limiter", limiter):
            from llmdebug.mcp_server import _check_rate_limit

            assert _check_rate_limit("llmdebug_diagnose", "json", "summary") is None
            assert _check_rate_limit("llmdebug_show", "json", "summary") is None
            assert _check_rate_limit("llmdebug_list", "json", "summary") is None
            # Fourth call — bucket exhausted
            result = _check_rate_limit("llmdebug_frame", "json", "summary")
            assert result is not None
            data = json.loads(result)
            assert data["error"]["code"] == "RATE_LIMITED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeClock:
    """Deterministic clock for testing."""

    def __init__(self, start: float) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds
