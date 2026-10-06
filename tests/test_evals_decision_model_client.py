"""Tests for the SystemOne HTTP client (no network: httpx.MockTransport)."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from evals.decision_model import client as client_mod
from evals.decision_model.config import DecisionModelClientConfig

_ANSWERS = {"model": "clef-flash", "answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {}}


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    breaker: client_mod.CircuitBreakerConfig | None = None,
    clock: Callable[[], float] | None = None,
) -> client_mod.SystemOneClient:
    kwargs: dict[str, object] = {"transport": httpx.MockTransport(handler)}
    if breaker is not None:
        kwargs["breaker"] = breaker
    if clock is not None:
        kwargs["clock"] = clock
    return client_mod.SystemOneClient(
        DecisionModelClientConfig(url="http://sidecar.test", timeout_sec=1.0),
        **kwargs,  # type: ignore[arg-type]
    )


def test_query_posts_json_and_parses_answers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ANSWERS)

    result = _client(handler).query({"model": "clef-flash", "state": {}, "questions": {}})

    assert result.ok
    assert result.body == _ANSWERS
    assert seen[0].url.path == client_mod.SYSTEMONE_PATH
    assert seen[0].headers["content-type"] == "application/json"


def test_http_errors_and_invalid_bodies_become_statuses() -> None:
    assert _client(lambda _r: httpx.Response(503, text="busy")).query({}).status == "http_503"
    assert _client(lambda _r: httpx.Response(200, text="{not json")).query({}).status == (
        client_mod.STATUS_INVALID_RESPONSE
    )
    assert _client(lambda _r: httpx.Response(200, json={"model": "x"})).query({}).status == (
        client_mod.STATUS_INVALID_RESPONSE
    )


def test_transport_errors_are_retried_once() -> None:
    calls = {"count": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=_ANSWERS)

    assert _client(flaky).query({}).ok
    assert calls["count"] == 2

    def always_down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    result = _client(always_down).query({})
    assert result.status == client_mod.STATUS_TRANSPORT_ERROR
    assert "ConnectError" in result.error


def test_timeouts_are_not_retried() -> None:
    calls = {"count": 0}

    def slow(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    assert _client(slow).query({}).status == client_mod.STATUS_TIMEOUT
    assert calls["count"] == 1


def test_circuit_breaker_opens_and_recovers_after_cooldown() -> None:
    now = {"t": 0.0}
    calls = {"count": 0}

    def failing(_request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500)

    client = _client(
        failing,
        breaker=client_mod.CircuitBreakerConfig(failure_threshold=2, cooldown_sec=30.0),
        clock=lambda: now["t"],
    )
    assert client.query({}).status == "http_500"
    assert client.query({}).status == "http_500"
    assert client.query({}).status == client_mod.STATUS_CIRCUIT_OPEN
    assert calls["count"] == 2

    now["t"] = 31.0
    assert client.query({}).status == "http_500"
    assert calls["count"] == 3


def test_health_reports_body_without_answers() -> None:
    health = _client(
        lambda _r: httpx.Response(200, json={"status": "ok", "revision": "abc"})
    ).health()
    assert health.ok
    assert health.body == {"status": "ok", "revision": "abc"}

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _client(down).health().status == client_mod.STATUS_TRANSPORT_ERROR


def test_api_key_is_sent_as_bearer_token_when_configured() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ANSWERS)

    keyed = client_mod.SystemOneClient(
        DecisionModelClientConfig(url="http://sidecar.test", api_key="secret-token"),
        transport=httpx.MockTransport(handler),
    )
    assert keyed.query({}).ok
    assert keyed.health().ok
    assert all(request.headers["authorization"] == "Bearer secret-token" for request in seen)
    assert "secret-token" not in repr(keyed.config)

    seen.clear()
    _client(handler).query({})
    assert "authorization" not in seen[0].headers
