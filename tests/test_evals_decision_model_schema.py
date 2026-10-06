"""Tests for decision-model question profiles and answer parsing."""

from __future__ import annotations

import math

import pytest

from evals.adaptive_online_policy import ONLINE_POLICY_ACTIONS
from evals.decision_model import schema


def _answers(**overrides: object) -> dict[str, object]:
    answers: dict[str, object] = {
        schema.QUESTION_NEXT_ACTION: {
            "type": "choice",
            "choice": "frame_only",
            "probabilities": {"skip_helper": 0.2, "frame_only": 0.6, "file_only": 0.2},
        },
        schema.QUESTION_TRACEBACK_SUFFICIENT: {"type": "noul", "noul": 0.25},
        schema.QUESTION_MISSING_INFORMATION: {
            "type": "choice",
            "probabilities": {"runtime_values": 0.7, "none": 0.1, "call_path": 0.2},
        },
        schema.QUESTION_FIX_CONFIDENCE: {"type": "score", "score": 1.5},
    }
    answers.update(overrides)
    return {"model": "clef-flash", "answers": answers}


def test_helper_actions_mirror_the_policy_action_list() -> None:
    assert schema.HELPER_ACTIONS == ONLINE_POLICY_ACTIONS


def test_every_action_has_decision_oriented_criteria() -> None:
    for action in schema.HELPER_ACTIONS:
        criteria = schema.action_criteria(action, reveal_mode="full", lite_snapshot_mode="replace")
        assert set(criteria) == {"choose_when", "not_for", "provides", "cost"}
        assert all(text.strip() for text in criteria.values())
    with pytest.raises(ValueError, match="unknown helper action"):
        schema.action_criteria("bogus", reveal_mode="full", lite_snapshot_mode="replace")


def test_criteria_follow_the_reveal_mode_and_lite_semantics() -> None:
    full_skip = schema.action_criteria(
        "skip_helper", reveal_mode="full", lite_snapshot_mode="replace"
    )
    tb_skip = schema.action_criteria(
        "skip_helper", reveal_mode="traceback_first", lite_snapshot_mode="replace"
    )
    assert "debug snapshot (crash frames" in full_skip["provides"]
    assert "no debug snapshot" in tb_skip["provides"]

    replace = schema.action_criteria(
        "frame_locals_lite", reveal_mode="full", lite_snapshot_mode="replace"
    )
    supplement = schema.action_criteria(
        "frame_locals_lite", reveal_mode="full", lite_snapshot_mode="supplement"
    )
    assert "instead of the full debug snapshot" in replace["provides"]
    assert "added to the prompt" in supplement["provides"]


def test_build_questions_offers_only_feasible_actions_in_fixed_order() -> None:
    profile = schema.get_profile("evidence_v1")
    questions = schema.build_questions(
        profile,
        available_actions=["search_only", "skip_helper", "invoke_helper"],
        reveal_mode="full",
        lite_snapshot_mode="replace",
    )

    assert list(questions) == [
        schema.QUESTION_NEXT_ACTION,
        schema.QUESTION_TRACEBACK_SUFFICIENT,
        schema.QUESTION_MISSING_INFORMATION,
        schema.QUESTION_FIX_CONFIDENCE,
    ]
    main = questions[schema.QUESTION_NEXT_ACTION]
    assert main["type"] == "choice"
    assert list(main["criteria"]) == ["skip_helper", "invoke_helper", "search_only"]
    assert questions[schema.QUESTION_TRACEBACK_SUFFICIENT]["type"] == "noul"
    assert questions[schema.QUESTION_FIX_CONFIDENCE]["criteria"] == list(
        schema.FIX_CONFIDENCE_LEVELS
    )
    assert all(question["instructions"] for question in questions.values())


def test_atomic_and_noaux_profiles_select_their_questions() -> None:
    atomic = schema.build_questions(
        schema.get_profile("evidence_atomic_v1"),
        available_actions=["skip_helper"],
        reveal_mode="full",
        lite_snapshot_mode="replace",
    )
    noaux = schema.build_questions(
        schema.get_profile("evidence_v1_noaux"),
        available_actions=["skip_helper", "invoke_helper"],
        reveal_mode="full",
        lite_snapshot_mode="replace",
    )

    assert schema.QUESTION_NEXT_ACTION not in atomic
    assert schema.QUESTION_MISSING_INFORMATION in atomic
    assert list(noaux) == [schema.QUESTION_NEXT_ACTION]


def test_main_question_needs_two_feasible_actions() -> None:
    with pytest.raises(ValueError, match="at least two feasible actions"):
        schema.build_questions(
            schema.get_profile("evidence_v1"),
            available_actions=["skip_helper"],
            reveal_mode="full",
            lite_snapshot_mode="replace",
        )


def test_get_profile_rejects_unknown_names() -> None:
    assert schema.DEFAULT_PROFILE_NAME in schema.PROFILE_NAMES
    with pytest.raises(ValueError, match="unknown decision-model profile"):
        schema.get_profile("nope")


def test_build_systemone_request_shape() -> None:
    request = schema.build_systemone_request(
        model="clef-flash", state_payload={"0_note": "x"}, questions={"q": {"type": "noul"}}
    )
    assert request == {
        "model": "clef-flash",
        "state": {"0_note": "x"},
        "questions": {"q": {"type": "noul"}},
    }


def test_parse_answers_normalizes_over_feasible_actions() -> None:
    parsed = schema.parse_answers(
        _answers(),
        profile=schema.get_profile("evidence_v1"),
        available_actions=["skip_helper", "frame_only"],
    )

    assert set(parsed.action_probs) == {"skip_helper", "frame_only"}
    assert math.isclose(sum(parsed.action_probs.values()), 1.0)
    assert math.isclose(parsed.action_probs["frame_only"], 0.75)
    assert parsed.traceback_sufficient == 0.25
    assert math.isclose(sum(parsed.missing_information.values()), 1.0)
    assert math.isclose(parsed.missing_information["runtime_values"], 0.7)
    assert parsed.missing_information["project_usage"] == 0.0
    assert math.isclose(parsed.fix_confidence or 0.0, 0.5)


@pytest.mark.parametrize(
    "response",
    [
        {"model": "x"},
        {"answers": {}},
        _answers(**{schema.QUESTION_NEXT_ACTION: {"type": "choice", "probabilities": {}}}),
        _answers(**{schema.QUESTION_TRACEBACK_SUFFICIENT: {"type": "noul"}}),
        _answers(**{schema.QUESTION_FIX_CONFIDENCE: {"type": "score", "score": "nan"}}),
    ],
)
def test_parse_answers_rejects_malformed_responses(response: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        schema.parse_answers(
            response,
            profile=schema.get_profile("evidence_v1"),
            available_actions=["skip_helper", "frame_only"],
        )


def test_parse_answers_ignores_non_finite_probabilities() -> None:
    response = _answers(
        **{
            schema.QUESTION_NEXT_ACTION: {
                "type": "choice",
                "probabilities": {"skip_helper": float("inf"), "frame_only": 0.5},
            }
        }
    )
    parsed = schema.parse_answers(
        response,
        profile=schema.get_profile("evidence_v1"),
        available_actions=["skip_helper", "frame_only"],
    )
    assert parsed.action_probs == {"skip_helper": 0.0, "frame_only": 1.0}
