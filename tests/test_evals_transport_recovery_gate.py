"""The transport-recovery gate can't stay closed or mix up outages (ORC-07)."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from evals.transport_failures import _OpenAITransportRecoveryGate


class _BrokenProbePatcher:
    """A probe whose metadata can't be read, so the leader's own code raises."""

    def probe_server_ready(self) -> tuple[bool, Any]:
        return False, "not a mapping"


def test_gate_reopens_when_the_leader_raises() -> None:
    gate = _OpenAITransportRecoveryGate(enabled=True, max_wait_sec=5.0, poll_sec=0.05)

    with pytest.raises(ValueError):
        gate.recover(patcher=_BrokenProbePatcher(), operation_name="propose_patch")

    assert gate._outage_active is False
    finished = threading.Event()
    waiter = threading.Thread(target=lambda: (gate.wait_if_open(), finished.set()), daemon=True)
    waiter.start()
    assert finished.wait(timeout=2.0)
    assert gate._outcomes[1]["error"] == "recovery_leader_failed"
    assert gate._outcomes[1]["recovered"] is False


def test_followers_get_the_outcome_of_their_own_outage() -> None:
    gate = _OpenAITransportRecoveryGate(enabled=True, max_wait_sec=5.0, poll_sec=0.02)
    with gate._condition:
        gate._outage_active = True
        gate._outage_generation = 1
    result: dict[str, Any] = {}

    def _follow() -> None:
        result.update(gate.recover(patcher=object(), operation_name="op"))

    follower = threading.Thread(target=_follow, daemon=True)
    follower.start()
    with gate._condition:
        # Outage 1 ends and outage 2 starts before the follower wakes up.
        gate._outcomes[1] = {"recovered": True, "generation": 1}
        gate._outcomes[2] = {"recovered": False, "generation": 2}
        gate._outage_generation = 2
        gate._condition.notify_all()
    follower.join(timeout=2.0)

    assert result["generation"] == 1
    assert result["recovered"] is True
    assert result["leader"] is False


def test_only_recent_outcomes_are_kept() -> None:
    class _Ready:
        def probe_server_ready(self) -> tuple[bool, dict[str, Any]]:
            return True, {}

    gate = _OpenAITransportRecoveryGate(enabled=True, max_wait_sec=1.0, poll_sec=0.1)
    for _ in range(_OpenAITransportRecoveryGate._KEPT_OUTCOMES + 4):
        gate.recover(patcher=_Ready(), operation_name="op")
    assert len(gate._outcomes) == _OpenAITransportRecoveryGate._KEPT_OUTCOMES
    assert max(gate._outcomes) == _OpenAITransportRecoveryGate._KEPT_OUTCOMES + 4
