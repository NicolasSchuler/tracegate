"""Retries back off with jitter and honour Retry-After (LLM-22)."""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

from evals.patchers.openai_transport import (
    _chat_completion,
    _jittered_retry_delay_sec,
    _retry_after_header,
    _retry_after_meta,
)

from ._openai_stub_server import OpenAIStubServer


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
    delays = {_jittered_retry_delay_sec(4.0, retry_after_sec=None, rng=rng) for _ in range(50)}
    assert all(2.0 <= delay <= 4.0 for delay in delays)
    assert len(delays) > 1


def test_retry_after_is_a_floor_above_the_backoff_cap() -> None:
    rng = random.Random(0)
    assert _jittered_retry_delay_sec(1.0, retry_after_sec=5.0, rng=rng) == 5.0
    # The cap applies to the exponential backoff, not to the server's retry floor.
    assert _jittered_retry_delay_sec(1.0, retry_after_sec=60.0, rng=rng) == 60.0
    # An untrusted header still cannot park a deadline-less run for a day.
    assert _jittered_retry_delay_sec(1.0, retry_after_sec=86_400.0, rng=rng) == 120.0


def test_retry_after_beyond_the_deadline_stops_the_loop_without_waiting() -> None:
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body={"error": "slow down"},
            status=429,
            headers={"Retry-After": "60"},
        )
        started = time.monotonic()
        raw, meta = _chat_completion(
            url=f"{server.base_url_v1}/chat/completions",
            api_key="local",
            model="dummy",
            messages=[{"role": "user", "content": "fix"}],
            temperature=0.0,
            max_tokens=8,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            deadline_monotonic=time.monotonic() + 2.0,
            transport_max_retries=3,
        )
        elapsed = time.monotonic() - started
        calls = server.requests_for(method="POST", path="/v1/chat/completions")

    assert raw is None
    assert meta["status"] == 429
    assert meta["retry_after_sec"] == 60.0
    assert meta["transport_retry_stop_reason"] == "retry_delay_exceeds_deadline"
    assert meta["transport_retry_delay_sec"] == 60.0
    assert meta["transport_retry_count"] == 0
    # The deadline ends the request, as it did before via a trimmed sleep plus one more
    # loop pass; the refused call's own error stays visible next to it.
    assert meta["error"] == "request_budget_exceeded"
    assert meta["transport_error"] == "http_error"
    assert len(calls) == 1  # no retry inside a window the server already refused
    assert elapsed < 2.0  # and no sleep until the deadline either
