"""Pure role logic for using decision-model advice in the helper policy.

The controller in ``evals.adaptive_online_policy`` calls these functions to turn a decision
model's answers into LinUCB features, a fading score prior, a guided warm-up distribution, or a
direct selection. Every selection returns the full probability of the chosen action so logged
propensities stay valid for off-policy evaluation. Network access lives in ``runtime``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import (
    ROLE_FEATURES,
    ROLE_OFF,
    ROLE_POLICY,
    ROLE_PRIOR,
    ROLE_SHADOW,
    DecisionModelPolicyConfig,
)
from .schema import (
    FILE_ONLY,
    FRAME_ONLY,
    INVOKE_HELPER,
    MISSING_CALL_PATH,
    MISSING_PROJECT_USAGE,
    MISSING_RUNTIME_VALUES,
    MISSING_SOURCE_DEFINITION,
    SEARCH_ONLY,
    SKIP_HELPER,
    AtomicThresholds,
    ParsedAnswers,
)

ADVICE_STATUS_OK = "ok"
ADVICE_STATUS_MISSING = "missing"

DM_FEATURE_NAMES = (
    "dm_available",
    "dm_traceback_sufficient",
    "dm_need_runtime_values",
    "dm_need_source_definition",
    "dm_need_project_usage",
    "dm_fix_confidence",
)
_PROBABILITY_FLOOR = 1e-3
_MISSING_TO_ACTION = {
    MISSING_RUNTIME_VALUES: FRAME_ONLY,
    MISSING_CALL_PATH: FRAME_ONLY,
    MISSING_SOURCE_DEFINITION: FILE_ONLY,
    MISSING_PROJECT_USAGE: SEARCH_ONLY,
}

__all__ = [
    "ADVICE_STATUS_OK",
    "DM_FEATURE_NAMES",
    "ROLE_FEATURES",
    "ROLE_OFF",
    "ROLE_POLICY",
    "ROLE_PRIOR",
    "ROLE_SHADOW",
    "DecisionModelAdvice",
    "DecisionModelAdvisor",
    "DecisionModelBinding",
    "RoleSelection",
    "advice_feature_values",
    "advice_record",
    "atomic_action_from_answers",
    "decision_model_metadata_fields",
    "guided_warmup_selection",
    "policy_role_selection",
    "prior_score_adjustment",
]


@dataclass(frozen=True)
class DecisionModelAdvice:
    """One decision model answer for one policy decision (or why there is none)."""

    status: str
    profile: str = ""
    model: str = ""
    revision: str = ""
    answers: ParsedAnswers = field(default_factory=ParsedAnswers)
    atomic_action: str = ""
    atomic_reason: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    state_sha256: str = ""
    state_chars: int = 0
    truncated_sections: tuple[str, ...] = ()
    state_text: str = field(default="", repr=False)
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the decision model answered and the answers parsed."""
        return self.status == ADVICE_STATUS_OK


class DecisionModelAdvisor(Protocol):
    """Anything that turns an advice request into ``DecisionModelAdvice``."""

    def advise(self, request: Any) -> DecisionModelAdvice: ...


class DecisionStateRecorder(Protocol):
    """Sink for rendered decision states (one JSON object per decision)."""

    def append(self, record: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class DecisionModelBinding:
    """Decision-model configuration attached to a helper-policy controller."""

    config: DecisionModelPolicyConfig
    advisor: DecisionModelAdvisor | None = None
    state_sink: DecisionStateRecorder | None = None

    @property
    def role(self) -> str:
        """Configured role (``off`` disables the decision model)."""
        return self.config.role


@dataclass(frozen=True)
class RoleSelection:
    """An action chosen by a decision-model role, with its full selection probability."""

    action: str
    propensity: float
    strategy: str
    warmup_active: bool
    effective_epsilon: float
    effect: dict[str, Any]


def _normalized(probs: Mapping[str, float], actions: Sequence[str]) -> dict[str, float]:
    values = {action: max(0.0, float(probs.get(action, 0.0))) for action in actions}
    total = sum(values.values())
    if total <= 0.0:
        return {action: 1.0 / len(actions) for action in actions}
    return {action: value / total for action, value in values.items()}


def _mix_with_uniform(
    distribution: Mapping[str, float],
    actions: Sequence[str],
    uniform_weight: float,
) -> dict[str, float]:
    weight = min(1.0, max(0.0, float(uniform_weight)))
    share = weight / float(len(actions))
    return {
        action: (1.0 - weight) * float(distribution.get(action, 0.0)) + share for action in actions
    }


def _argmax(distribution: Mapping[str, float], actions: Sequence[str]) -> str:
    best = actions[0]
    for action in actions[1:]:
        if float(distribution.get(action, 0.0)) > float(distribution.get(best, 0.0)):
            best = action
    return best


def _tempered(
    probs: Mapping[str, float],
    actions: Sequence[str],
    *,
    temperature: float,
) -> dict[str, float]:
    base = _normalized(probs, actions)
    if temperature <= 0.0:
        best = _argmax(base, actions)
        return {action: 1.0 if action == best else 0.0 for action in actions}
    logits = {
        action: math.log(max(_PROBABILITY_FLOOR, base[action])) / temperature for action in actions
    }
    peak = max(logits.values())
    weights = {action: math.exp(value - peak) for action, value in logits.items()}
    total = sum(weights.values())
    return {action: weight / total for action, weight in weights.items()}


def _sample(distribution: Mapping[str, float], actions: Sequence[str], rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for action in actions:
        cumulative += float(distribution.get(action, 0.0))
        if draw < cumulative:
            return action
    return actions[-1]


def distribution_summary(distribution: Mapping[str, float]) -> tuple[str, float, float, float]:
    """Return ``(top_action, top_prob, margin_to_second, entropy_nats)``."""
    if not distribution:
        return "", 0.0, 0.0, 0.0
    ranked = sorted(distribution.items(), key=lambda item: (-float(item[1]), str(item[0])))
    top_action, top_prob = ranked[0]
    second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    entropy = -sum(float(p) * math.log(float(p)) for _, p in ranked if float(p) > 0.0)
    return str(top_action), float(top_prob), float(top_prob) - second, float(entropy)


def advice_feature_values(advice: DecisionModelAdvice | None) -> dict[str, float]:
    """Six diagnostic features for the ``features`` role (all zero when advice is missing)."""
    values: dict[str, float] = dict.fromkeys(DM_FEATURE_NAMES, 0.0)
    if advice is None or not advice.ok:
        return values
    answers = advice.answers
    if answers.traceback_sufficient is None or not answers.missing_information:
        return values
    missing = answers.missing_information
    values.update(
        {
            "dm_available": 1.0,
            "dm_traceback_sufficient": float(answers.traceback_sufficient),
            "dm_need_runtime_values": min(
                1.0,
                float(missing.get(MISSING_RUNTIME_VALUES, 0.0))
                + float(missing.get(MISSING_CALL_PATH, 0.0)),
            ),
            "dm_need_source_definition": float(missing.get(MISSING_SOURCE_DEFINITION, 0.0)),
            "dm_need_project_usage": float(missing.get(MISSING_PROJECT_USAGE, 0.0)),
            "dm_fix_confidence": float(answers.fix_confidence or 0.0),
        }
    )
    return values


def _feasible_fallback(target: str, reason: str, actions: Sequence[str]) -> tuple[str, str]:
    if target in actions:
        return target, reason
    for fallback in (INVOKE_HELPER, SKIP_HELPER):
        if fallback in actions:
            return fallback, f"{reason}_fallback_{fallback}"
    return str(actions[0]), f"{reason}_fallback_first_available"


def atomic_action_from_answers(
    answers: ParsedAnswers,
    *,
    available_actions: Sequence[str],
    thresholds: AtomicThresholds,
) -> tuple[str, str]:
    """Map atomic diagnostic answers to a feasible helper action in code."""
    actions = [str(action) for action in available_actions]
    if not actions:
        return "", "no_available_actions"
    sufficient = answers.traceback_sufficient
    if sufficient is not None and sufficient >= thresholds.sufficient:
        return _feasible_fallback(SKIP_HELPER, "traceback_sufficient", actions)
    if not answers.missing_information:
        return _feasible_fallback(INVOKE_HELPER, "missing_information_unavailable", actions)
    ranked = sorted(answers.missing_information.items(), key=lambda item: (-item[1], item[0]))
    top_option, top_prob = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_prob - second < thresholds.min_margin:
        return _feasible_fallback(INVOKE_HELPER, "missing_information_uncertain", actions)
    target = _MISSING_TO_ACTION.get(top_option, INVOKE_HELPER)
    return _feasible_fallback(target, f"missing_{top_option}", actions)


def policy_role_selection(
    advice: DecisionModelAdvice | None,
    *,
    available_actions: Sequence[str],
    config: DecisionModelPolicyConfig,
    epsilon: float,
    rng: random.Random,
) -> RoleSelection | None:
    """Let the decision model select (``policy`` role); ``None`` when advice is unusable."""
    if advice is None or not advice.ok or not available_actions:
        return None
    actions = [str(action) for action in available_actions]
    if advice.atomic_action in actions:
        base = {action: 1.0 if action == advice.atomic_action else 0.0 for action in actions}
        label = "decision_model_atomic"
    elif advice.answers.action_probs:
        base = _tempered(advice.answers.action_probs, actions, temperature=config.temperature)
        label = "decision_model_greedy" if config.temperature <= 0.0 else "decision_model_sample"
    else:
        return None
    distribution = _mix_with_uniform(base, actions, epsilon)
    action = _sample(distribution, actions, rng)
    if label != "decision_model_sample" and action != _argmax(base, actions):
        label = "decision_model_epsilon"
    return RoleSelection(
        action=action,
        propensity=min(1.0, max(1e-9, distribution[action])),
        strategy=label,
        warmup_active=False,
        effective_epsilon=min(1.0, max(0.0, float(epsilon))),
        effect={"distribution": distribution},
    )


def guided_warmup_selection(
    advice: DecisionModelAdvice | None,
    *,
    available_actions: Sequence[str],
    config: DecisionModelPolicyConfig,
    rng: random.Random,
) -> RoleSelection | None:
    """Sample warm-up actions from the decision model mixed with uniform (``prior`` role)."""
    if advice is None or not advice.ok or not advice.answers.action_probs or not available_actions:
        return None
    actions = [str(action) for action in available_actions]
    distribution = _mix_with_uniform(
        _normalized(advice.answers.action_probs, actions),
        actions,
        config.warmup_mix,
    )
    action = _sample(distribution, actions, rng)
    return RoleSelection(
        action=action,
        propensity=min(1.0, max(1e-9, distribution[action])),
        strategy="decision_model_guided_warmup",
        warmup_active=True,
        effective_epsilon=min(1.0, max(0.0, float(config.warmup_mix))),
        effect={"warmup_distribution": distribution},
    )


def prior_score_adjustment(
    advice: DecisionModelAdvice | None,
    *,
    available_actions: Sequence[str],
    config: DecisionModelPolicyConfig,
    updates: int,
) -> dict[str, Any] | None:
    """Centered, clipped log-probability prior whose weight halves after ``anneal`` updates."""
    if advice is None or not advice.ok or not advice.answers.action_probs or not available_actions:
        return None
    actions = [str(action) for action in available_actions]
    probs = _normalized(advice.answers.action_probs, actions)
    logs = {action: math.log(max(_PROBABILITY_FLOOR, probs[action])) for action in actions}
    mean = sum(logs.values()) / float(len(logs))
    anneal = max(1, int(config.prior_anneal_updates))
    beta = max(0.0, float(config.prior_beta)) * anneal / float(anneal + max(0, int(updates)))
    return {
        "prior_beta": beta,
        "prior_adjustment": {action: beta * (logs[action] - mean) for action in actions},
    }


def advice_record(
    advice: DecisionModelAdvice | None,
    *,
    role: str,
    effect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serializable per-decision record stored on ``PolicyDecision.decision_model``."""
    if role == ROLE_OFF:
        return {}
    if advice is None:
        return {"role": role, "status": ADVICE_STATUS_MISSING, "effect": dict(effect or {})}
    answers = advice.answers
    top_action, top_prob, margin, entropy = distribution_summary(answers.action_probs)
    return {
        "role": role,
        "status": advice.status,
        "profile": advice.profile,
        "model": advice.model,
        "revision": advice.revision,
        "error": advice.error,
        "action_probs": dict(answers.action_probs),
        "top_action": top_action,
        "top_prob": top_prob,
        "margin": margin,
        "entropy": entropy,
        "atomic_action": advice.atomic_action,
        "atomic_reason": advice.atomic_reason,
        "traceback_sufficient": answers.traceback_sufficient,
        "missing_information": dict(answers.missing_information),
        "fix_confidence": answers.fix_confidence,
        "latency_ms": float(advice.latency_ms),
        "input_tokens": int(advice.input_tokens),
        "state_sha256": advice.state_sha256,
        "state_chars": int(advice.state_chars),
        "truncated_sections": list(advice.truncated_sections),
        "effect": dict(effect or {}),
    }


_METADATA_KEYS: tuple[tuple[str, Any], ...] = (
    ("role", ROLE_OFF),
    ("status", ""),
    ("profile", ""),
    ("model", ""),
    ("error", ""),
    ("action_probs", None),
    ("top_action", ""),
    ("top_prob", 0.0),
    ("margin", 0.0),
    ("entropy", 0.0),
    ("atomic_action", ""),
    ("atomic_reason", ""),
    ("traceback_sufficient", None),
    ("missing_information", None),
    ("fix_confidence", None),
    ("latency_ms", 0.0),
    ("input_tokens", 0),
    ("state_sha256", ""),
    ("state_chars", 0),
    ("effect", None),
)
_DICT_METADATA_KEYS = frozenset({"action_probs", "missing_information", "effect"})


def decision_model_metadata_fields(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """``adaptive_online_dm_*`` result-row fields (defaults when *record* is empty)."""
    source: Mapping[str, Any] = record or {}
    fields: dict[str, Any] = {}
    for key, default in _METADATA_KEYS:
        value = source.get(key, default)
        if key in _DICT_METADATA_KEYS:
            value = dict(value) if isinstance(value, Mapping) else {}
        fields[f"adaptive_online_dm_{key}"] = value
    return fields
