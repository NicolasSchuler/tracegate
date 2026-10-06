"""Tests for RCA verification gate behavior."""

from __future__ import annotations

import pytest

from llmdebug.gates import GateRequest, evaluate_gate


def test_soft_mode_warns_but_allows_missing_requirements() -> None:
    feedback = evaluate_gate(
        GateRequest(
            mode="soft",
            current_state="captured",
            action="plan",
            payload={
                "evidence_refs": [],
                "hypothesis": "",
                "experiment_plan": "",
            },
        )
    )
    assert feedback.allowed is True
    assert feedback.severity == "warning"
    assert "evidence_required" in feedback.triggered_checks
    assert "hypothesis_required" in feedback.triggered_checks


def test_strict_mode_blocks_missing_requirements() -> None:
    feedback = evaluate_gate(
        GateRequest(
            mode="strict",
            current_state="captured",
            action="change",
            payload={
                "evidence_refs": [],
                "hypothesis": "",
                "experiment_plan": "",
            },
        )
    )
    assert feedback.allowed is False
    assert feedback.severity == "error"
    assert "evidence_required" in feedback.triggered_checks
    assert "hypothesis_required" in feedback.triggered_checks


def test_strict_mode_exploratory_override_allows() -> None:
    feedback = evaluate_gate(
        GateRequest(
            mode="strict",
            current_state="captured",
            action="change",
            payload={
                "evidence_refs": [],
                "hypothesis": "",
                "experiment_plan": "",
                "exploratory": True,
                "override_reason": "Need a probe to inspect runtime branch.",
            },
        )
    )
    assert feedback.allowed is True
    assert feedback.severity == "warning"
    assert any("override accepted" in warning.lower() for warning in feedback.warnings)


def test_loop_protection_warning_after_repeats() -> None:
    feedback = evaluate_gate(
        GateRequest(
            mode="soft",
            current_state="changed",
            action="change",
            payload={
                "evidence_refs": ["crash_frame:app.py:42:parse_input"],
                "hypothesis": "Input parsing fails on non-digit value.",
                "experiment_plan": "Guard int conversion with try/except.",
                "hypothesis_advanced": False,
            },
            repeated_signature_count=4,
            exploration_budget=2,
        )
    )
    assert feedback.allowed is True
    assert "loop_protection" in feedback.triggered_checks
    assert any("repeated" in warning.lower() for warning in feedback.warnings)


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 54, 56, 60, 229, 265-266
# ---------------------------------------------------------------------------


class TestNormalizeGateMode:
    """Lines 54, 56, 60: normalize_gate_mode with invalid mode."""

    def test_invalid_mode_returns_soft_and_warns_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from llmdebug.gates import normalize_gate_mode

        result = normalize_gate_mode("invalid_mode")
        assert result == "soft"
        captured = capsys.readouterr()
        assert "unrecognized gate mode" in captured.err
        assert "invalid_mode" in captured.err

    def test_none_mode_returns_soft(self) -> None:
        from llmdebug.gates import normalize_gate_mode

        assert normalize_gate_mode(None) == "soft"

    def test_valid_modes_are_accepted(self) -> None:
        from llmdebug.gates import normalize_gate_mode

        assert normalize_gate_mode("off") == "off"
        assert normalize_gate_mode("soft") == "soft"
        assert normalize_gate_mode("strict") == "strict"
        assert normalize_gate_mode("  OFF  ") == "off"


class TestEvaluateGateOffMode:
    """Line 229: gate_mode == 'off' → allowed=True, severity='info', no checks."""

    def test_off_mode_always_allows(self) -> None:
        feedback = evaluate_gate(
            GateRequest(
                mode="off",
                current_state="captured",
                action="resolve",
                payload={},
            )
        )
        assert feedback.allowed is True
        assert feedback.severity == "info"
        assert feedback.warnings == ()
        assert feedback.triggered_checks == ()
        assert feedback.mode == "off"


class TestEvaluateGateStrictNoHardFailButWarnings:
    """Lines 265-266: strict mode, no hard_failed but warnings → severity='warning', allowed=True."""

    def test_strict_with_soft_warnings_only(self) -> None:
        """plan_required is not a hard failure, so strict should allow but warn."""
        feedback = evaluate_gate(
            GateRequest(
                mode="strict",
                current_state="planned",
                action="change",
                payload={
                    "evidence_refs": ["frame:app.py:10:fn"],
                    "hypothesis": "Input parsing fails.",
                    "experiment_plan": "",  # missing plan → plan_required (not hard)
                },
            )
        )
        assert feedback.allowed is True
        assert feedback.severity == "warning"
        assert "plan_required" in feedback.triggered_checks

    def test_strict_with_no_issues_at_all(self) -> None:
        """No issues → severity='info', allowed=True."""
        feedback = evaluate_gate(
            GateRequest(
                mode="strict",
                current_state="verified",
                action="resolve",
                payload={
                    "evidence_refs": ["crash_frame:app.py:42:parse"],
                    "hypothesis": "Bug in parsing.",
                    "experiment_plan": "Guard int().",
                    "verification_result": {"status": "pass"},
                },
            )
        )
        assert feedback.allowed is True
        assert feedback.severity == "info"
        assert feedback.triggered_checks == ()


_COMPLETE_PAYLOAD = {
    "evidence_refs": ["crash_frame:app.py:42"],
    "hypothesis": "empty input",
    "experiment_plan": "guard it",
    "verification_result": {"status": "pass"},
}


def test_strict_mode_blocks_an_illegal_transition() -> None:
    """RCA-06: every prerequisite is met, but "resolve" isn't a transition from "captured"."""
    feedback = evaluate_gate(
        GateRequest(
            mode="strict", current_state="captured", action="resolve", payload=_COMPLETE_PAYLOAD
        )
    )

    assert feedback.allowed is False
    assert feedback.triggered_checks == ("illegal_transition",)
    assert "bind_evidence, capture" in feedback.next_required[0]


def test_soft_mode_warns_about_an_illegal_transition_last() -> None:
    feedback = evaluate_gate(
        GateRequest(
            mode="soft",
            current_state="captured",
            action="verify",
            payload={"evidence_refs": [], "hypothesis": "", "experiment_plan": ""},
        )
    )

    assert feedback.allowed is True
    assert feedback.triggered_checks[-1] == "illegal_transition"
    assert feedback.triggered_checks[0] == "evidence_required"


def test_illegal_transition_accepts_the_exploratory_override() -> None:
    payload = {**_COMPLETE_PAYLOAD, "exploratory": True, "override_reason": "spike"}

    feedback = evaluate_gate(
        GateRequest(mode="strict", current_state="captured", action="resolve", payload=payload)
    )

    assert feedback.allowed is True
    assert "illegal_transition" in feedback.triggered_checks


@pytest.mark.parametrize(
    ("state", "action"),
    [("verified", "resolve"), ("changed", "verify"), ("stalled", "plan"), ("resolved", "capture")],
)
def test_allowed_transitions_pass(state: str, action: str) -> None:
    feedback = evaluate_gate(
        GateRequest(mode="strict", current_state=state, action=action, payload=_COMPLETE_PAYLOAD)
    )

    assert "illegal_transition" not in feedback.triggered_checks


def test_a_first_attempt_has_no_transition_to_check() -> None:
    feedback = evaluate_gate(
        GateRequest(mode="strict", current_state=None, action="resolve", payload=_COMPLETE_PAYLOAD)
    )

    assert feedback.allowed is True
    assert feedback.triggered_checks == ()


def test_keyword_form_is_deprecated_but_still_works() -> None:
    """RCA-23: evaluate_gate takes a GateRequest; the old keyword form warns."""
    with pytest.warns(DeprecationWarning, match="pass a GateRequest"):
        feedback = evaluate_gate(
            mode="strict", current_state="verified", action="resolve", payload=_COMPLETE_PAYLOAD
        )

    assert feedback == evaluate_gate(
        GateRequest(
            action="resolve", payload=_COMPLETE_PAYLOAD, mode="strict", current_state="verified"
        )
    )


def test_request_and_keywords_together_are_rejected() -> None:
    request = GateRequest(action="plan", payload={})

    with pytest.raises(TypeError, match="not both"):
        evaluate_gate(request, mode="soft")
