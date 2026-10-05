"""Retries back off with jitter and honour Retry-After (LLM-22)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

from evals.patchers.openai_transport import (
    _jittered_retry_delay_sec,
    _retry_after_header,
    _retry_after_meta,
)


def test_retry_after_parses_seconds_and_http_dates() -> None:
    assert _retry_after_meta("12") == {"retry_after_sec": 12.0}
    later = datetime.now(timezone.utc) + timedelta(seconds=30)
    parsed = _retry_after_meta(format_datetime(later, usegmt=True))["retry_after_sec"]
    assert 25.0 <= parsed <= 31.0
    assert _retry_after_meta("") == {}
    assert _retry_after_meta("soon") == {}
    assert _retry_after_meta("nan") == {}


def test_retry_after_header_tolerates_responses_without_headers() -> None:
    assert _retry_after_header(SimpleNamespace(headers={"retry-after": "3"})) == "3"
    assert _retry_after_header(object()) == ""


def test_delay_is_jittered_within_half_and_full_backoff() -> None:
    rng = random.Random(0)
    delays = {
        _jittered_retry_delay_sec(4.0, retry_after_sec=None, max_sec=8.0, rng=rng)
        for _ in range(50)
    }
    assert all(2.0 <= delay <= 4.0 for delay in delays)
    assert len(delays) > 1


def test_retry_after_is_a_floor_up_to_the_backoff_cap() -> None:
    rng = random.Random(0)
    assert _jittered_retry_delay_sec(1.0, retry_after_sec=5.0, max_sec=8.0, rng=rng) == 5.0
    assert _jittered_retry_delay_sec(1.0, retry_after_sec=60.0, max_sec=8.0, rng=rng) == 8.0
