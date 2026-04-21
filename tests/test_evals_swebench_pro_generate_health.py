"""Tests for SWE-bench Pro generation health assertions and failure thresholds."""

from __future__ import annotations

import pytest

from evals.swebench_pro.generate import _assert_generation_health


def test_generation_health_fails_when_all_requests_fail() -> None:
    with pytest.raises(RuntimeError, match="all patch requests failed"):
        _assert_generation_health(
            condition="baseline",
            attempted=10,
            total_pending=10,
            request_error_count=10,
            max_request_errors=25,
            max_request_error_rate=0.5,
            min_requests_for_error_rate_check=10,
        )


def test_generation_health_fails_on_absolute_error_threshold() -> None:
    with pytest.raises(RuntimeError, match="exceeded threshold"):
        _assert_generation_health(
            condition="baseline",
            attempted=12,
            total_pending=100,
            request_error_count=6,
            max_request_errors=5,
            max_request_error_rate=0.9,
            min_requests_for_error_rate_check=10,
        )


def test_generation_health_fails_on_error_rate_threshold() -> None:
    with pytest.raises(RuntimeError, match="error rate too high"):
        _assert_generation_health(
            condition="llmdebug",
            attempted=10,
            total_pending=100,
            request_error_count=6,
            max_request_errors=100,
            max_request_error_rate=0.5,
            min_requests_for_error_rate_check=10,
        )


def test_generation_health_allows_low_error_rate() -> None:
    _assert_generation_health(
        condition="llmdebug",
        attempted=20,
        total_pending=100,
        request_error_count=2,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
    )


def test_generation_health_passes_with_zero_errors() -> None:
    _assert_generation_health(
        condition="baseline",
        attempted=50,
        total_pending=100,
        request_error_count=0,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
    )


def test_generation_health_skips_rate_check_below_min_attempts() -> None:
    """When attempted < min_requests_for_error_rate_check, rate check is skipped."""
    # 4 errors out of 5 attempts = 80% error rate, but min_requests is 10
    # so the rate check is not triggered; only absolute threshold matters.
    _assert_generation_health(
        condition="baseline",
        attempted=5,
        total_pending=100,
        request_error_count=4,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
    )
