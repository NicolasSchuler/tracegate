"""Live integration test against a running System-1 decision-model sidecar.

Skips unless a sidecar URL is supplied, either through ``--live-decision-model-url`` or through
``TRACEGATE_DECISION_MODEL_URL``, and skips again when that URL is not healthy. Start one with
``just decision-model-up`` and check it with ``just decision-model-health``.

    uv run pytest -m integration tests/test_evals_decision_model_live.py \
        --live-decision-model-url http://127.0.0.1:8090
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Iterator

import pytest

from evals.decision_model.config import (
    DEFAULT_DECISION_MODEL_NAME,
    ROLE_SHADOW,
    DecisionModelClientConfig,
    DecisionModelPolicyConfig,
    DecisionModelSettings,
)
from evals.decision_model.policy import ADVICE_STATUS_OK
from evals.decision_model.runtime import AdviceRequest, SystemOneAdvisor
from evals.decision_model.schema import (
    DEFAULT_PROFILE_NAME,
    FILE_ONLY,
    FRAME_ONLY,
    INVOKE_HELPER,
    MISSING_INFORMATION_OPTIONS,
    SEARCH_ONLY,
    SKIP_HELPER,
)
from evals.decision_model.state import DecisionStateInputs

pytestmark = pytest.mark.integration

# A cold sidecar still has to run its first real prefill; keep the budget generous.
_TIMEOUT_SEC = 120.0
_AVAILABLE_ACTIONS = (SKIP_HELPER, INVOKE_HELPER, FRAME_ONLY, FILE_ONLY, SEARCH_ONLY)

_PYTEST_OUTPUT = textwrap.dedent("""\
    ============================= test session starts ==============================
    collected 3 items

    tests/test_window.py::test_resolve_window FAILED                         [ 33%]

    =================================== FAILURES ===================================
    _____________________________ test_resolve_window ______________________________

        def test_resolve_window():
            session = Session(window="30s")
    >       assert session.resolve_window() == 30
    E       AssertionError: assert None == 30
    E        +  where None = <Session window='30s'>.resolve_window()

    tests/test_window.py:14: AssertionError
    =========================== short test summary info ============================
    FAILED tests/test_window.py::test_resolve_window - AssertionError: assert None
    1 failed, 2 passed in 0.21s
""")


@pytest.fixture(scope="module")
def sidecar_url(pytestconfig: pytest.Config) -> str:
    """Sidecar base URL from the CLI option or the environment; skip when absent."""
    option = str(pytestconfig.getoption("--live-decision-model-url") or "").strip()
    url = option or os.environ.get("TRACEGATE_DECISION_MODEL_URL", "").strip()
    if not url:
        pytest.skip(
            "no decision-model sidecar: pass --live-decision-model-url or set "
            "TRACEGATE_DECISION_MODEL_URL"
        )
    return url.rstrip("/")


@pytest.fixture(scope="module")
def advisor(sidecar_url: str) -> Iterator[SystemOneAdvisor]:
    """Advisor bound to the live sidecar; skip when the sidecar is not healthy."""
    settings = DecisionModelSettings(
        policy=DecisionModelPolicyConfig(role=ROLE_SHADOW, profile=DEFAULT_PROFILE_NAME),
        client=DecisionModelClientConfig(
            url=sidecar_url,
            model=DEFAULT_DECISION_MODEL_NAME,
            timeout_sec=_TIMEOUT_SEC,
        ),
    )
    live_advisor = SystemOneAdvisor(settings)
    health = live_advisor.health()
    if not health.ok:
        live_advisor.close()
        pytest.skip(
            f"decision-model sidecar at {sidecar_url} is not healthy "
            f"({health.status}: {health.error})"
        )
    try:
        yield live_advisor
    finally:
        live_advisor.close()


def test_live_sidecar_health_reports_the_loaded_model(advisor: SystemOneAdvisor) -> None:
    health = advisor.health()

    assert health.ok, health
    assert health.body is not None
    assert health.body["status"] == "ok"
    for field in ("model", "revision", "device", "dtype", "max_length"):
        assert field in health.body, field


def test_live_sidecar_answers_one_attempt_with_normalized_probabilities(
    advisor: SystemOneAdvisor,
) -> None:
    request = AdviceRequest(
        state_inputs=DecisionStateInputs(
            stdout=_PYTEST_OUTPUT,
            attempt=1,
            max_attempts=2,
            helper_calls_remaining=1,
            compaction_calls_remaining=1,
            reveal_mode="full",
        ),
        available_actions=_AVAILABLE_ACTIONS,
    )

    advice = advisor.advise(request)

    assert advice.status == ADVICE_STATUS_OK, advice
    assert advice.state_sha256
    assert advice.state_chars > 0
    assert advice.latency_ms > 0.0
    assert advice.input_tokens > 0

    answers = advice.answers
    assert set(answers.action_probs) == set(_AVAILABLE_ACTIONS)
    assert sum(answers.action_probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= value <= 1.0 for value in answers.action_probs.values())

    assert set(answers.missing_information) == set(MISSING_INFORMATION_OPTIONS)
    assert sum(answers.missing_information.values()) == pytest.approx(1.0, abs=1e-6)
    assert answers.traceback_sufficient is not None
    assert 0.0 <= answers.traceback_sufficient <= 1.0
    assert answers.fix_confidence is not None
    assert 0.0 <= answers.fix_confidence <= 1.0
