"""HTTP client for SystemOne-compatible decision-model endpoints (``POST /v1/systemone``).

Decisions sit on the attempt's critical path, so the client never raises: every failure becomes
a status string, transport errors get one retry, and a circuit breaker stops calling a sidecar
that keeps failing until a cooldown has passed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .. import _json as json
from .config import DecisionModelClientConfig

SYSTEMONE_PATH = "/v1/systemone"
HEALTH_PATH = "/health"
STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_TRANSPORT_ERROR = "transport_error"
STATUS_INVALID_RESPONSE = "invalid_response"
STATUS_CIRCUIT_OPEN = "circuit_open"
DEFAULT_CONNECT_TIMEOUT_SEC = 2.0
_ERROR_TEXT_CHARS = 200


@dataclass(frozen=True)
class SystemOneResult:
    """Outcome of one request to the decision-model endpoint."""

    status: str
    body: dict[str, Any] | None = None
    http_status: int = 0
    latency_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether a well-formed response body was received."""
        return self.status == STATUS_OK


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Consecutive failures that open the breaker, and how long it stays open."""

    failure_threshold: int = 5
    cooldown_sec: float = 60.0


class SystemOneClient:
    """Thread-safe client shared by all eval workers of a run."""

    def __init__(
        self,
        config: DecisionModelClientConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        breaker: CircuitBreakerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._breaker = breaker or CircuitBreakerConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0
        timeout_sec = max(0.001, float(config.timeout_sec))
        # trust_env=False: a proxy from the environment must not intercept a local sidecar.
        headers = {"authorization": f"Bearer {config.api_key}"} if config.api_key else None
        self._client = httpx.Client(
            base_url=str(config.url).rstrip("/"),
            timeout=httpx.Timeout(
                timeout_sec, connect=min(DEFAULT_CONNECT_TIMEOUT_SEC, timeout_sec)
            ),
            transport=transport,
            headers=headers,
            trust_env=False,
        )

    @property
    def config(self) -> DecisionModelClientConfig:
        """Endpoint configuration."""
        return self._config

    @property
    def circuit_open(self) -> bool:
        """Whether queries are short-circuited right now (cooldown after repeated failures)."""
        return self._circuit_open()

    def close(self) -> None:
        """Release pooled connections."""
        self._client.close()

    def health(self) -> SystemOneResult:
        """``GET /health``; does not count toward the circuit breaker."""
        started = time.perf_counter()
        try:
            response = self._client.get(HEALTH_PATH)
        except httpx.TimeoutException as exc:
            return SystemOneResult(STATUS_TIMEOUT, error=type(exc).__name__)
        except httpx.TransportError as exc:
            return SystemOneResult(STATUS_TRANSPORT_ERROR, error=f"{type(exc).__name__}: {exc}")
        return self._parse(response, latency_ms=_elapsed_ms(started), require_answers=False)

    def query(self, request: Mapping[str, Any]) -> SystemOneResult:
        """``POST /v1/systemone`` with one transport retry and circuit breaking."""
        if self._circuit_open():
            return SystemOneResult(STATUS_CIRCUIT_OPEN, error="circuit breaker open")
        result = self._post(request)
        self._record(result.ok)
        return result

    def _post(self, request: Mapping[str, Any]) -> SystemOneResult:
        payload = json.dumps(dict(request), ensure_ascii=False)
        headers = {"content-type": "application/json"}
        started = time.perf_counter()
        for attempt in range(2):
            started = time.perf_counter()
            try:
                response = self._client.post(SYSTEMONE_PATH, content=payload, headers=headers)
            except httpx.TimeoutException as exc:
                return SystemOneResult(
                    STATUS_TIMEOUT,
                    latency_ms=_elapsed_ms(started),
                    error=type(exc).__name__,
                )
            except httpx.TransportError as exc:
                if attempt == 0:
                    continue
                return SystemOneResult(
                    STATUS_TRANSPORT_ERROR,
                    latency_ms=_elapsed_ms(started),
                    error=f"{type(exc).__name__}: {exc}",
                )
            return self._parse(response, latency_ms=_elapsed_ms(started), require_answers=True)
        return SystemOneResult(STATUS_TRANSPORT_ERROR, latency_ms=_elapsed_ms(started))

    @staticmethod
    def _parse(
        response: httpx.Response,
        *,
        latency_ms: float,
        require_answers: bool,
    ) -> SystemOneResult:
        if response.status_code != 200:
            return SystemOneResult(
                f"http_{response.status_code}",
                http_status=response.status_code,
                latency_ms=latency_ms,
                error=response.text[:_ERROR_TEXT_CHARS],
            )
        try:
            body = json.loads(response.content)
        except Exception as exc:
            return SystemOneResult(
                STATUS_INVALID_RESPONSE,
                http_status=response.status_code,
                latency_ms=latency_ms,
                error=f"invalid JSON: {type(exc).__name__}",
            )
        if not isinstance(body, dict) or (
            require_answers and not isinstance(body.get("answers"), dict)
        ):
            return SystemOneResult(
                STATUS_INVALID_RESPONSE,
                http_status=response.status_code,
                latency_ms=latency_ms,
                error="response body has no answers object",
            )
        return SystemOneResult(
            STATUS_OK,
            body=body,
            http_status=response.status_code,
            latency_ms=latency_ms,
        )

    def _circuit_open(self) -> bool:
        with self._lock:
            return self._clock() < self._open_until

    def _record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._consecutive_failures = 0
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= max(1, int(self._breaker.failure_threshold)):
                self._open_until = self._clock() + max(0.0, float(self._breaker.cooldown_sec))
                self._consecutive_failures = 0


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)
