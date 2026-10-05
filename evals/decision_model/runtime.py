"""Glue between the attempt flow, the decision-model sidecar and the helper-policy controller.

The attempt flow calls :func:`advise_for_attempt` right before ``decide()`` (outside the
controller lock), passes the advice to the controller, and then records the rendered state with
:func:`record_advice_state`. Nothing here raises into the attempt: failures become an advice
status that the controller handles per role.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import _json as json
from .client import SystemOneClient, SystemOneResult
from .config import (
    DEFAULT_LITE_SNAPSHOT_MODE,
    LOG_STATE_FULL,
    ROLE_OFF,
    DecisionModelSettings,
)
from .policy import (
    ADVICE_STATUS_MISSING,
    ADVICE_STATUS_OK,
    DecisionModelAdvice,
    DecisionModelBinding,
    atomic_action_from_answers,
)
from .schema import build_questions, build_systemone_request, get_profile, parse_answers
from .state import DecisionStateInputs, build_decision_state

DECISION_MODEL_STATES_FILENAME = "decision_model_states.jsonl"
# The preflight may hit a cold llama-swap entry, which holds the request until the model has
# loaded instead of answering 503 like the PyTorch sidecar; give it time to finish.
DECISION_MODEL_PREFLIGHT_TIMEOUT_SEC = 300.0
# Cooldown between ``/health`` lookups of the served revision after a failed one.
DECISION_MODEL_REVISION_RETRY_SEC = 30.0
STATUS_INVALID_REQUEST = "invalid_request"
STATUS_INVALID_RESPONSE = "invalid_response"
_ERROR_CHARS = 200


@dataclass(frozen=True)
class AdviceRequest:
    """Inputs for one decision-model query."""

    state_inputs: DecisionStateInputs
    available_actions: tuple[str, ...]
    lite_snapshot_mode: str = DEFAULT_LITE_SNAPSHOT_MODE


class DecisionStateSink:
    """Thread-safe JSONL sink for rendered decision states."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Destination file."""
        return self._path

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one JSON object per line."""
        line = json.dumps(dict(record), ensure_ascii=False) + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)


class SystemOneAdvisor:
    """Turns an :class:`AdviceRequest` into :class:`DecisionModelAdvice` via the sidecar."""

    def __init__(
        self,
        settings: DecisionModelSettings,
        *,
        client: SystemOneClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._profile = get_profile(settings.policy.profile)
        self._client = client or SystemOneClient(settings.client)
        self._revision: str | None = None
        self._clock = clock
        self._revision_retry_at = 0.0
        self._lock = threading.Lock()

    @property
    def client(self) -> SystemOneClient:
        """Underlying HTTP client."""
        return self._client

    def health(self) -> SystemOneResult:
        """Query the sidecar's health endpoint."""
        return self._client.health()

    def close(self) -> None:
        """Release the HTTP client."""
        self._client.close()

    def _server_revision(self) -> str:
        """Model revision from ``/health``; cached only once the sidecar reports healthy.

        The sidecar answers 503 while the model loads, so a failed lookup is retried later
        instead of pinning an empty revision for the whole run.  Retries are rate-limited and
        skipped entirely while the query circuit breaker is open: an unresponsive sidecar used
        to get one blocking ``/health`` per decision (each up to the full timeout, under this
        shared lock) although the decision itself returned ``circuit_open`` immediately.
        """
        with self._lock:
            if self._revision is not None:
                return self._revision
            now = self._clock()
            if now < self._revision_retry_at or self._client.circuit_open:
                return ""
            result = self._client.health()
            if not result.ok or not result.body:
                self._revision_retry_at = now + DECISION_MODEL_REVISION_RETRY_SEC
                return ""
            self._revision = str(result.body.get("revision") or "")
            return self._revision

    def advise(self, request: AdviceRequest) -> DecisionModelAdvice:
        """Build the state and questions, query the sidecar, and parse the answers."""
        profile = self._profile
        state = build_decision_state(request.state_inputs, profile=profile)
        common: dict[str, Any] = {
            "profile": profile.name,
            "revision": self._server_revision(),
            "state_sha256": state.sha256,
            "state_chars": state.chars,
            "truncated_sections": state.truncated_sections,
            "state_text": state.rendered,
        }
        model = self._settings.client.model
        try:
            questions = build_questions(
                profile,
                available_actions=request.available_actions,
                reveal_mode=request.state_inputs.reveal_mode,
                lite_snapshot_mode=request.lite_snapshot_mode,
            )
        except ValueError as exc:
            return DecisionModelAdvice(
                status=STATUS_INVALID_REQUEST,
                model=model,
                error=str(exc)[:_ERROR_CHARS],
                **common,
            )
        result = self._client.query(
            build_systemone_request(model=model, state_payload=state.payload, questions=questions)
        )
        if not result.ok or result.body is None:
            return DecisionModelAdvice(
                status=result.status,
                model=model,
                latency_ms=result.latency_ms,
                error=result.error[:_ERROR_CHARS],
                **common,
            )
        try:
            answers = parse_answers(
                result.body,
                profile=profile,
                available_actions=request.available_actions,
            )
        except ValueError as exc:
            return DecisionModelAdvice(
                status=STATUS_INVALID_RESPONSE,
                model=model,
                latency_ms=result.latency_ms,
                error=str(exc)[:_ERROR_CHARS],
                **common,
            )
        atomic_action, atomic_reason = "", ""
        if profile.atomic is not None:
            atomic_action, atomic_reason = atomic_action_from_answers(
                answers,
                available_actions=request.available_actions,
                thresholds=profile.atomic,
            )
        usage = result.body.get("usage")
        input_tokens = usage.get("input_tokens", 0) if isinstance(usage, Mapping) else 0
        return DecisionModelAdvice(
            status=ADVICE_STATUS_OK,
            model=str(result.body.get("model") or model),
            answers=answers,
            atomic_action=atomic_action,
            atomic_reason=atomic_reason,
            latency_ms=result.latency_ms,
            input_tokens=int(input_tokens) if isinstance(input_tokens, (int, float)) else 0,
            **common,
        )


def build_decision_model_binding(
    settings: DecisionModelSettings,
    *,
    artifacts_root: Path | None,
    client: SystemOneClient | None = None,
) -> DecisionModelBinding | None:
    """Create the binding attached to the helper-policy controller (``None`` when off)."""
    if not settings.enabled:
        return None
    sink = None
    if artifacts_root is not None and settings.log_state == LOG_STATE_FULL:
        sink = DecisionStateSink(Path(artifacts_root) / DECISION_MODEL_STATES_FILENAME)
    return DecisionModelBinding(
        config=settings.policy,
        advisor=SystemOneAdvisor(settings, client=client),
        state_sink=sink,
    )


def binding_of(controller: Any) -> DecisionModelBinding | None:
    """Return the controller's active binding; tolerant of test doubles without one."""
    binding = getattr(controller, "decision_model", None)
    if not isinstance(binding, DecisionModelBinding):
        return None
    if binding.role == ROLE_OFF or binding.advisor is None:
        return None
    return binding


def advise_for_attempt(
    binding: DecisionModelBinding,
    request: AdviceRequest,
) -> DecisionModelAdvice:
    """Query the advisor; any exception becomes an ``error:<Type>`` status."""
    if binding.advisor is None:
        return DecisionModelAdvice(status=ADVICE_STATUS_MISSING)
    try:
        return binding.advisor.advise(request)
    except Exception as exc:  # the decision model must never break an attempt
        return DecisionModelAdvice(
            status=f"error:{type(exc).__name__}",
            error=str(exc)[:_ERROR_CHARS],
        )


def record_advice_state(
    binding: DecisionModelBinding | None,
    *,
    decision: Any,
    advice: DecisionModelAdvice | None,
    context: Mapping[str, Any],
) -> None:
    """Append the rendered state keyed by ``decision_id`` (best effort)."""
    if binding is None or binding.state_sink is None or advice is None or not advice.state_text:
        return
    record = {
        "decision_id": str(getattr(decision, "decision_id", "") or ""),
        **dict(context),
        "profile": advice.profile,
        "status": advice.status,
        "available_actions": list(getattr(decision, "available_actions", ()) or ()),
        "state_sha256": advice.state_sha256,
        "state": advice.state_text,
    }
    try:
        binding.state_sink.append(record)
    except OSError:
        return


def check_decision_model_health(
    settings: DecisionModelSettings,
    *,
    client: SystemOneClient | None = None,
) -> str:
    """Return an error message when an enabled sidecar is unreachable (empty when healthy)."""
    if not settings.enabled:
        return ""
    preflight_timeout = max(settings.client.timeout_sec, DECISION_MODEL_PREFLIGHT_TIMEOUT_SEC)
    active = client or SystemOneClient(replace(settings.client, timeout_sec=preflight_timeout))
    try:
        result = active.health()
    finally:
        if client is None:
            active.close()
    if result.ok:
        return ""
    detail = f": {result.error}" if result.error else ""
    loading = (
        " The sidecar answers 503 while the model is still loading."
        if (result.status == "http_503")
        else ""
    )
    return (
        f"decision model sidecar at {settings.client.url} is not healthy ({result.status}{detail})."
        f"{loading} Start it (just decision-model-up), wait for just decision-model-health, or use "
        "--adaptive-decision-model-role off."
    )
