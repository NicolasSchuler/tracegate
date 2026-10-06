"""Versioned question profiles for SystemOne decision models.

A profile fixes what the decision model is asked: the main evidence-action choice, the
auxiliary diagnostic questions, the option descriptions, and how the state is rendered.
Clef-style models score options by their descriptions and decide all questions jointly, so
the wording here is part of the experiment and is versioned with every logged answer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Helper actions, mirroring ``evals.adaptive_online_policy`` (kept literal to avoid an import
# cycle; a test asserts both lists match).
SKIP_HELPER = "skip_helper"
INVOKE_HELPER = "invoke_helper"
FRAME_ONLY = "frame_only"
FILE_ONLY = "file_only"
SEARCH_ONLY = "search_only"
SNAPSHOT_LITE = "snapshot_lite"
FRAME_LOCALS_LITE = "frame_locals_lite"
SNAPSHOT_FULL_LITE = "snapshot_full_lite"
COMPACT_THEN_PATCH = "compact_then_patch"
RETRY_HELPER_STRICT_JSON = "retry_helper_strict_json"
HELPER_ACTIONS = (
    SKIP_HELPER,
    INVOKE_HELPER,
    FRAME_ONLY,
    FILE_ONLY,
    SEARCH_ONLY,
    SNAPSHOT_LITE,
    FRAME_LOCALS_LITE,
    SNAPSHOT_FULL_LITE,
    COMPACT_THEN_PATCH,
    RETRY_HELPER_STRICT_JSON,
)
LITE_ACTIONS = frozenset({SNAPSHOT_LITE, FRAME_LOCALS_LITE, SNAPSHOT_FULL_LITE})

REVEAL_MODE_TRACEBACK_FIRST = "traceback_first"
LITE_SNAPSHOT_REPLACE = "replace"

QUESTION_NEXT_ACTION = "next_evidence_action"
QUESTION_TRACEBACK_SUFFICIENT = "traceback_sufficient"
QUESTION_MISSING_INFORMATION = "missing_information"
QUESTION_FIX_CONFIDENCE = "fix_confidence"

MISSING_NONE = "none"
MISSING_RUNTIME_VALUES = "runtime_values"
MISSING_CALL_PATH = "call_path"
MISSING_SOURCE_DEFINITION = "source_definition"
MISSING_PROJECT_USAGE = "project_usage"
MISSING_SEVERAL_OR_UNCLEAR = "several_or_unclear"
MISSING_INFORMATION_OPTIONS = (
    MISSING_NONE,
    MISSING_RUNTIME_VALUES,
    MISSING_CALL_PATH,
    MISSING_SOURCE_DEFINITION,
    MISSING_PROJECT_USAGE,
    MISSING_SEVERAL_OR_UNCLEAR,
)
FIX_CONFIDENCE_LEVELS = ("Very unlikely", "Unlikely", "Likely", "Very likely")

STATE_FORMAT_JSON = "json"
STATE_FORMAT_TEXT = "text"

NEXT_ACTION_INSTRUCTIONS = (
    "Which evidence plan should the next automated patch attempt use? Prefer the cheapest plan "
    "that is likely to lead to a correct fix."
)
TRACEBACK_SUFFICIENT_INSTRUCTIONS = (
    "Do the error message, crash site and test output alone identify both the faulty code and "
    "how to fix it?"
)
MISSING_INFORMATION_INSTRUCTIONS = "Which kind of information is most needed to fix this failure?"
FIX_CONFIDENCE_INSTRUCTIONS = (
    "How likely is the next patch attempt to fix the failing test if no further evidence is added?"
)
_TRACEBACK_SUFFICIENT_CRITERIA = {
    "true": "The failure output pinpoints the faulty code and the fix.",
    "false": "More evidence is needed to find the cause or the fix.",
}
_MISSING_INFORMATION_CRITERIA = {
    MISSING_NONE: "Nothing important is missing; the failure output is enough.",
    MISSING_RUNTIME_VALUES: "Values of variables at the moment of failure.",
    MISSING_CALL_PATH: "How execution reached the failing line (caller frames).",
    MISSING_SOURCE_DEFINITION: (
        "The full source of a function, class or module that the traceback does not show."
    ),
    MISSING_PROJECT_USAGE: "Where a name is defined or used elsewhere in the project.",
    MISSING_SEVERAL_OR_UNCLEAR: "Several kinds of information, or it is unclear which.",
}


@dataclass(frozen=True)
class AtomicThresholds:
    """Thresholds that map atomic diagnostic answers to a helper action in code."""

    sufficient: float = 0.7
    min_margin: float = 0.15


@dataclass(frozen=True)
class DecisionModelProfile:
    """One versioned combination of questions, option wording and state rendering."""

    name: str
    version: int
    main_question: bool
    aux_questions: bool
    state_format: str = STATE_FORMAT_JSON
    include_hints: bool = True
    max_state_chars: int = 6000
    atomic: AtomicThresholds | None = None


PROFILES: dict[str, DecisionModelProfile] = {
    profile.name: profile
    for profile in (
        DecisionModelProfile("evidence_v1", 1, main_question=True, aux_questions=True),
        DecisionModelProfile("evidence_v1_noaux", 1, main_question=True, aux_questions=False),
        DecisionModelProfile(
            "evidence_v1_text",
            1,
            main_question=True,
            aux_questions=True,
            state_format=STATE_FORMAT_TEXT,
        ),
        DecisionModelProfile(
            "evidence_v1_nohints",
            1,
            main_question=True,
            aux_questions=True,
            include_hints=False,
        ),
        DecisionModelProfile(
            "evidence_atomic_v1",
            1,
            main_question=False,
            aux_questions=True,
            atomic=AtomicThresholds(),
        ),
    )
}
DEFAULT_PROFILE_NAME = "evidence_v1"
PROFILE_NAMES = tuple(PROFILES)


def get_profile(name: str) -> DecisionModelProfile:
    """Return the registered profile *name* or raise ``ValueError``."""
    try:
        return PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown decision-model profile: {name!r}") from exc


def _criteria(choose_when: str, not_for: str, provides: str, cost: str) -> dict[str, str]:
    return {"choose_when": choose_when, "not_for": not_for, "provides": provides, "cost": cost}


_HELPER_CALL_COST = "One extra model call before patching."
_NO_CALL_COST = "No extra model call."


def action_criteria(action: str, *, reveal_mode: str, lite_snapshot_mode: str) -> dict[str, str]:
    """Describe *action* as the patch model will experience it under the given reveal mode."""
    snapshot_shown = reveal_mode != REVEAL_MODE_TRACEBACK_FIRST
    lite_replaces = snapshot_shown and lite_snapshot_mode == LITE_SNAPSHOT_REPLACE
    lite_verb = "instead of the full debug snapshot" if lite_replaces else "added to the prompt"
    if action == SKIP_HELPER:
        snapshot_part = (
            "the debug snapshot (crash frames with local values)"
            if snapshot_shown
            else "no debug snapshot unless the failure repeats"
        )
        return _criteria(
            "The error message, crash site and test output already point to the faulty code "
            "and the fix.",
            "Failures whose cause depends on runtime values, other files, or code the "
            "traceback does not show.",
            f"Patch immediately. The patch model sees the test output, {snapshot_part}, and a "
            "dump of the project's source files.",
            _NO_CALL_COST,
        )
    if action == INVOKE_HELPER:
        return _criteria(
            "The cause is unclear and the needed evidence could be any mix of source files, "
            "code search results, stack frames or variable values.",
            "Failures that are already obvious, or where one specific kind of evidence is "
            "clearly what is missing.",
            "A helper model first diagnoses the failure and fetches up to 6 targeted items "
            "(whole files, code search hits, stack frames, local variables). The patch model "
            "then sees those items instead of the project dump.",
            _HELPER_CALL_COST,
        )
    if action == FRAME_ONLY:
        return _criteria(
            "The bug depends on runtime values at one specific point in the call stack.",
            "Bugs that need other source files or project-wide usages.",
            "The helper fetches one stack frame (code and locals) plus up to two variable "
            "values. The patch model sees those instead of the project dump.",
            _HELPER_CALL_COST,
        )
    if action == FILE_ONLY:
        return _criteria(
            "The fix needs the full definition of code in one source file that the traceback "
            "does not show.",
            "Bugs that depend on runtime values or on usages spread across the project.",
            "The helper fetches exactly one whole source file. The patch model sees it instead "
            "of the project dump.",
            _HELPER_CALL_COST,
        )
    if action == SEARCH_ONLY:
        return _criteria(
            "The fix depends on where a name is defined or used elsewhere in the project.",
            "Bugs that are local to the crashing function.",
            "The helper runs up to two literal code searches and passes the matching lines on. "
            "The patch model sees those instead of the project dump.",
            _HELPER_CALL_COST,
        )
    if action == SNAPSHOT_LITE:
        return _criteria(
            "Knowing which functions the failure passed through is enough; variable values are "
            "not needed.",
            "Bugs that depend on concrete runtime values.",
            "Patch immediately with a compact index of the crashing frames (locations only) "
            f"{lite_verb}, plus the project dump.",
            _NO_CALL_COST,
        )
    if action == FRAME_LOCALS_LITE:
        return _criteria(
            "The values of a few local variables in the crashing frame explain the bug.",
            "Bugs whose cause lies in caller frames or in other files.",
            "Patch immediately with the crashing frame's code and up to 5 local values "
            f"{lite_verb}, plus the project dump.",
            _NO_CALL_COST,
        )
    if action == SNAPSHOT_FULL_LITE:
        detail = (
            "the same view the patch model gets by default in this mode"
            if lite_replaces
            else "added to the prompt"
        )
        return _criteria(
            "Runtime values across the top few frames are needed to understand the bug.",
            "Failures that are obvious from the error message.",
            "Patch immediately with up to 3 crashing frames including code and all local "
            f"values ({detail}), plus the project dump.",
            _NO_CALL_COST,
        )
    if action == COMPACT_THEN_PATCH:
        return _criteria(
            "The available evidence is large and noisy, so a short brief would help the patch "
            "model focus.",
            "Small, clear failures.",
            "A model first condenses the test output, debug snapshot and project files (up to "
            "about 120k characters) into a brief of at most about 18k characters that replaces "
            "those sections.",
            "One extra model call over a large input.",
        )
    if action == RETRY_HELPER_STRICT_JSON:
        return _criteria(
            "The helper's previous answer was malformed and a stricter, shorter retry is likely "
            "to succeed.",
            "Any situation without a previous malformed helper answer.",
            "Re-runs the helper with strict, short JSON output (at most 768 tokens).",
            "One short extra model call.",
        )
    raise ValueError(f"unknown helper action: {action!r}")


def build_questions(
    profile: DecisionModelProfile,
    *,
    available_actions: Sequence[str],
    reveal_mode: str,
    lite_snapshot_mode: str,
) -> dict[str, dict[str, Any]]:
    """Build the SystemOne ``questions`` object for *profile* (insertion order is fixed)."""
    questions: dict[str, dict[str, Any]] = {}
    if profile.main_question:
        available = set(available_actions)
        actions = [action for action in HELPER_ACTIONS if action in available]
        if len(actions) < 2:
            raise ValueError("the main question needs at least two feasible actions")
        questions[QUESTION_NEXT_ACTION] = {
            "type": "choice",
            "instructions": NEXT_ACTION_INSTRUCTIONS,
            "criteria": {
                action: action_criteria(
                    action,
                    reveal_mode=reveal_mode,
                    lite_snapshot_mode=lite_snapshot_mode,
                )
                for action in actions
            },
        }
    if profile.aux_questions:
        questions[QUESTION_TRACEBACK_SUFFICIENT] = {
            "type": "noul",
            "instructions": TRACEBACK_SUFFICIENT_INSTRUCTIONS,
            "criteria": dict(_TRACEBACK_SUFFICIENT_CRITERIA),
        }
        questions[QUESTION_MISSING_INFORMATION] = {
            "type": "choice",
            "instructions": MISSING_INFORMATION_INSTRUCTIONS,
            "criteria": dict(_MISSING_INFORMATION_CRITERIA),
        }
        questions[QUESTION_FIX_CONFIDENCE] = {
            "type": "score",
            "instructions": FIX_CONFIDENCE_INSTRUCTIONS,
            "criteria": list(FIX_CONFIDENCE_LEVELS),
        }
    return questions


def build_systemone_request(
    *,
    model: str,
    state_payload: Any,
    questions: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble a ``POST /v1/systemone`` request body."""
    return {"model": str(model), "state": state_payload, "questions": dict(questions)}


@dataclass(frozen=True)
class ParsedAnswers:
    """Answers normalized from a SystemOne response."""

    action_probs: dict[str, float] = field(default_factory=dict)
    traceback_sufficient: float | None = None
    missing_information: dict[str, float] = field(default_factory=dict)
    fix_confidence: float | None = None


def _finite_probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _answer(answers: Mapping[str, Any], question_id: str) -> Mapping[str, Any]:
    answer = answers.get(question_id)
    if not isinstance(answer, Mapping):
        raise ValueError(f"response has no answer for {question_id!r}")
    return answer


def _choice_probabilities(
    answers: Mapping[str, Any],
    question_id: str,
    *,
    options: Sequence[str],
) -> dict[str, float]:
    probabilities = _answer(answers, question_id).get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError(f"answer {question_id!r} has no probabilities")
    values = {option: _finite_probability(probabilities.get(option)) for option in options}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError(f"answer {question_id!r} has no probability mass on the options")
    return {option: value / total for option, value in values.items()}


def _noul_probability(answers: Mapping[str, Any], question_id: str) -> float:
    """Required scalar probability of a ``noul`` answer; malformed values raise ``ValueError``.

    Unlike the per-option probabilities of a choice answer (where one unusable entry is dropped
    and the remaining options still carry mass), this is the entire answer: ``null``, a
    non-numeric string, a boolean or a non-finite number must surface as an invalid response
    instead of a silent 0.0 that reads as "the traceback is not sufficient".
    """
    answer = _answer(answers, question_id)
    if "noul" not in answer:
        raise ValueError(f"answer {question_id!r} has no noul probability")
    raw = answer.get("noul")
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"answer {question_id!r} has a non-numeric noul probability")
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"answer {question_id!r} has a non-numeric noul probability") from exc
    if not math.isfinite(number):
        raise ValueError(f"answer {question_id!r} has a non-finite noul probability")
    return min(1.0, max(0.0, number))


def _score_fraction(answers: Mapping[str, Any], question_id: str, *, levels: int) -> float:
    raw_score = _answer(answers, question_id).get("score")
    if raw_score is None:
        raise ValueError(f"answer {question_id!r} has no numeric score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"answer {question_id!r} has no numeric score") from exc
    if not math.isfinite(score) or levels < 2:
        raise ValueError(f"answer {question_id!r} has an invalid score")
    return min(1.0, max(0.0, score / float(levels - 1)))


def parse_answers(
    response: Mapping[str, Any],
    *,
    profile: DecisionModelProfile,
    available_actions: Sequence[str],
) -> ParsedAnswers:
    """Normalize a SystemOne response; raise ``ValueError`` when it is malformed."""
    answers = response.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError("response has no answers object")
    action_probs: dict[str, float] = {}
    if profile.main_question:
        options = [action for action in HELPER_ACTIONS if action in set(available_actions)]
        action_probs = _choice_probabilities(answers, QUESTION_NEXT_ACTION, options=options)
    if not profile.aux_questions:
        return ParsedAnswers(action_probs=action_probs)
    return ParsedAnswers(
        action_probs=action_probs,
        traceback_sufficient=_noul_probability(answers, QUESTION_TRACEBACK_SUFFICIENT),
        missing_information=_choice_probabilities(
            answers,
            QUESTION_MISSING_INFORMATION,
            options=MISSING_INFORMATION_OPTIONS,
        ),
        fix_confidence=_score_fraction(
            answers,
            QUESTION_FIX_CONFIDENCE,
            levels=len(FIX_CONFIDENCE_LEVELS),
        ),
    )
