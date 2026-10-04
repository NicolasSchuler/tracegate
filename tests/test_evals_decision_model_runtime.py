"""Tests for the decision-model runtime glue (advisor, binding, state sink, health)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from evals.decision_model import runtime, schema
from evals.decision_model.client import SystemOneClient
from evals.decision_model.config import (
    DecisionModelClientConfig,
    DecisionModelPolicyConfig,
    DecisionModelSettings,
)
from evals.decision_model.policy import DecisionModelAdvice, DecisionModelBinding
from evals.decision_model.state import DecisionStateInputs

ACTIONS = ("skip_helper", "invoke_helper", "frame_only")


def _answers_for(request: dict[str, Any]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    questions = request["questions"]
    if schema.QUESTION_NEXT_ACTION in questions:
        options = list(questions[schema.QUESTION_NEXT_ACTION]["criteria"])
        answers[schema.QUESTION_NEXT_ACTION] = {
            "type": "choice",
            "probabilities": {
                option: (0.7 if option == "frame_only" else 0.3 / (len(options) - 1))
                for option in options
            },
        }
    answers[schema.QUESTION_TRACEBACK_SUFFICIENT] = {"type": "noul", "noul": 0.1}
    answers[schema.QUESTION_MISSING_INFORMATION] = {
        "type": "choice",
        "probabilities": {"runtime_values": 0.8, "none": 0.2},
    }
    answers[schema.QUESTION_FIX_CONFIDENCE] = {"type": "score", "score": 1.0}
    return {"model": "clef-flash", "answers": answers, "usage": {"input_tokens": 321}}


def _sidecar(captured: list[dict[str, Any]] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "revision": "rev-123"})
        body = json.loads(request.content)
        if captured is not None:
            captured.append(body)
        return httpx.Response(200, json=_answers_for(body))

    return httpx.MockTransport(handler)


def _settings(profile: str = "evidence_v1", **policy: Any) -> DecisionModelSettings:
    return DecisionModelSettings(
        policy=DecisionModelPolicyConfig(role="shadow", profile=profile, **policy),
        client=DecisionModelClientConfig(url="http://sidecar.test"),
    )


def _advisor(
    profile: str = "evidence_v1", captured: list[dict[str, Any]] | None = None
) -> runtime.SystemOneAdvisor:
    settings = _settings(profile)
    client = SystemOneClient(settings.client, transport=_sidecar(captured))
    return runtime.SystemOneAdvisor(settings, client=client)


def _request() -> runtime.AdviceRequest:
    return runtime.AdviceRequest(
        state_inputs=DecisionStateInputs(
            stdout="E   KeyError: 'k'\n1 failed in 0.1s\n",
            redact_tokens=("secret_case_id_easy",),
        ),
        available_actions=ACTIONS,
    )


def test_advisor_round_trip_parses_answers_and_identity() -> None:
    captured: list[dict[str, Any]] = []
    advice = _advisor(captured=captured).advise(_request())

    assert advice.ok
    assert advice.profile == "evidence_v1"
    assert advice.revision == "rev-123"
    assert advice.input_tokens == 321
    assert advice.answers.action_probs["frame_only"] == 0.7
    assert advice.state_sha256 and advice.state_chars == len(advice.state_text)
    request = captured[0]
    assert request["model"] == "clef-flash"
    assert list(request["questions"][schema.QUESTION_NEXT_ACTION]["criteria"]) == list(ACTIONS)
    assert request["state"]["1_failure"]["exception"] == "KeyError"


def test_atomic_profile_maps_answers_to_an_action() -> None:
    advice = _advisor("evidence_atomic_v1").advise(_request())

    assert advice.ok
    assert advice.answers.action_probs == {}
    assert (advice.atomic_action, advice.atomic_reason) == ("frame_only", "missing_runtime_values")


def test_advisor_failures_become_statuses() -> None:
    settings = _settings()

    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(503)
        return httpx.Response(200, json={"model": "x", "answers": {}})

    advisor = runtime.SystemOneAdvisor(
        settings, client=SystemOneClient(settings.client, transport=httpx.MockTransport(broken))
    )
    advice = advisor.advise(_request())
    assert advice.status == runtime.STATUS_INVALID_RESPONSE
    assert advice.revision == ""

    too_few = runtime.AdviceRequest(
        state_inputs=DecisionStateInputs(), available_actions=("skip_helper",)
    )
    assert _advisor().advise(too_few).status == runtime.STATUS_INVALID_REQUEST


def test_build_binding_and_binding_of() -> None:
    assert (
        runtime.build_decision_model_binding(DecisionModelSettings(), artifacts_root=None) is None
    )

    binding = runtime.build_decision_model_binding(_settings(), artifacts_root=Path("/tmp/x"))
    assert isinstance(binding, DecisionModelBinding)
    assert binding.state_sink is not None
    assert binding.state_sink.path.name == runtime.DECISION_MODEL_STATES_FILENAME
    assert runtime.binding_of(SimpleNamespace(decision_model=binding)) is binding

    hashed = runtime.build_decision_model_binding(
        DecisionModelSettings(policy=DecisionModelPolicyConfig(role="shadow"), log_state="hash"),
        artifacts_root=Path("/tmp/x"),
    )
    assert hashed is not None and hashed.state_sink is None

    for controller in (
        SimpleNamespace(),
        SimpleNamespace(decision_model="not-a-binding"),
        SimpleNamespace(decision_model=DecisionModelBinding(config=DecisionModelPolicyConfig())),
    ):
        assert runtime.binding_of(controller) is None


def test_advise_for_attempt_never_raises() -> None:
    class _Exploding:
        def advise(self, request: Any) -> DecisionModelAdvice:
            raise RuntimeError("boom")

    binding = DecisionModelBinding(
        config=DecisionModelPolicyConfig(role="policy"), advisor=_Exploding()
    )
    advice = runtime.advise_for_attempt(binding, _request())
    assert advice.status == "error:RuntimeError"
    assert advice.error == "boom"

    missing = runtime.advise_for_attempt(
        DecisionModelBinding(config=DecisionModelPolicyConfig(role="policy")), _request()
    )
    assert missing.status == "missing"


def test_record_advice_state_writes_jsonl(tmp_path: Path) -> None:
    sink = runtime.DecisionStateSink(tmp_path / "nested" / "states.jsonl")
    binding = DecisionModelBinding(config=DecisionModelPolicyConfig(role="shadow"), state_sink=sink)
    advice = DecisionModelAdvice(
        status="ok", profile="evidence_v1", state_sha256="abc", state_text='{"0_note":"x"}'
    )
    decision = SimpleNamespace(decision_id="d-1", available_actions=ACTIONS)

    runtime.record_advice_state(binding, decision=decision, advice=advice, context={"attempt": 1})
    runtime.record_advice_state(binding, decision=decision, advice=None, context={})
    runtime.record_advice_state(None, decision=decision, advice=advice, context={})

    lines = sink.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["decision_id"] == "d-1"
    assert record["attempt"] == 1
    assert record["available_actions"] == list(ACTIONS)
    assert record["state"] == '{"0_note":"x"}'


def test_check_decision_model_health() -> None:
    assert runtime.check_decision_model_health(DecisionModelSettings()) == ""

    settings = _settings()
    healthy = SystemOneClient(settings.client, transport=_sidecar())
    assert runtime.check_decision_model_health(settings, client=healthy) == ""

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    unhealthy = SystemOneClient(settings.client, transport=httpx.MockTransport(down))
    message = runtime.check_decision_model_health(settings, client=unhealthy)
    assert "not healthy (transport_error" in message
    assert "--adaptive-decision-model-role off" in message


def test_revision_is_not_cached_while_the_sidecar_is_loading() -> None:
    state = {"loading": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if state["loading"]:
                return httpx.Response(503, json={"status": "loading"})
            return httpx.Response(200, json={"status": "ok", "revision": "rev-late"})
        return httpx.Response(200, json=_answers_for(json.loads(request.content)))

    settings = _settings()
    advisor = runtime.SystemOneAdvisor(
        settings, client=SystemOneClient(settings.client, transport=httpx.MockTransport(handler))
    )

    assert advisor.advise(_request()).revision == ""
    state["loading"] = False
    assert advisor.advise(_request()).revision == "rev-late"


def test_preflight_waits_for_a_cold_gateway(monkeypatch: Any) -> None:
    """llama-swap holds the first request while it loads the model; the preflight must wait."""
    timeouts: list[float] = []

    def client_factory(config: DecisionModelClientConfig, **_kwargs: Any) -> SystemOneClient:
        timeouts.append(config.timeout_sec)
        return SystemOneClient(config, transport=_sidecar())

    monkeypatch.setattr(runtime, "SystemOneClient", client_factory)

    assert runtime.check_decision_model_health(_settings()) == ""
    assert timeouts == [runtime.DECISION_MODEL_PREFLIGHT_TIMEOUT_SEC]


def test_health_message_explains_loading_sidecar() -> None:
    settings = _settings()
    loading = SystemOneClient(
        settings.client, transport=httpx.MockTransport(lambda _r: httpx.Response(503))
    )
    message = runtime.check_decision_model_health(settings, client=loading)
    assert "still loading" in message
