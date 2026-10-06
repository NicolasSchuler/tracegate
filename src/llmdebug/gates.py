"""RCA verification gates for MCP-driven debugging workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from .rca_transitions import allowed_actions_for_state
from .rca_types import RCAState

__all__ = ["GateFeedback", "GateRequest", "RCAState", "evaluate_gate", "normalize_gate_mode"]

GateMode = Literal["off", "soft", "strict"]
GateSeverity = Literal["info", "warning", "error"]

_VALID_GATE_MODES = {"off", "soft", "strict"}
_DEFAULT_GATE_MODE: GateMode = "soft"


@dataclass(frozen=True)
class GateFeedback:
    """Result of evaluating gate requirements for a requested RCA action."""

    mode: GateMode
    allowed: bool
    severity: GateSeverity
    warnings: tuple[str, ...]
    next_required: tuple[str, ...]
    triggered_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "severity": self.severity,
            "warnings": list(self.warnings),
            "next_required": list(self.next_required),
            "triggered_checks": list(self.triggered_checks),
        }


def normalize_gate_mode(mode: str | None) -> GateMode:
    """Normalize gate mode; fallback to soft for invalid inputs."""
    if mode is None:
        return _DEFAULT_GATE_MODE
    lowered = mode.strip().lower()
    if lowered in _VALID_GATE_MODES:
        return cast(GateMode, lowered)
    import sys

    sys.stderr.write(
        f"llmdebug: warning: unrecognized gate mode {mode!r}, "
        f"defaulting to '{_DEFAULT_GATE_MODE}'. Valid: {sorted(_VALID_GATE_MODES)}\n"
    )
    return _DEFAULT_GATE_MODE


@dataclass(frozen=True)
class GateRequest:
    """Inputs for one gate evaluation; see :func:`evaluate_gate` for the fields (RCA-23)."""

    action: str
    payload: dict[str, Any]
    mode: str | None = None
    current_state: RCAState | str | None = None
    repeated_signature_count: int = 1
    exploration_budget: int = 2
    exploration_budget_remaining: int | None = None


def _collect_gate_issues(
    request: GateRequest,
) -> tuple[list[str], list[str], list[str], bool]:
    """Evaluate all gate checks and return (warnings, next_required, triggered_checks, hard_failed)."""
    action = request.action
    payload = request.payload
    current_state = request.current_state
    has_evidence = bool(payload.get("evidence_refs"))
    has_hypothesis = bool(payload.get("hypothesis"))
    has_plan = bool(payload.get("experiment_plan"))
    verification_result = payload.get("verification_result")
    hypothesis_advanced = bool(payload.get("hypothesis_advanced"))

    warnings: list[str] = []
    next_required: list[str] = []
    triggered_checks: list[str] = []
    hard_failed = False

    def _add(check_id: str, message: str, next_step: str, *, hard: bool = False) -> None:
        nonlocal hard_failed
        triggered_checks.append(check_id)
        warnings.append(message)
        if next_step not in next_required:
            next_required.append(next_step)
        hard_failed = hard_failed or hard

    if action in {"hypothesize", "plan", "change", "verify", "resolve"} and not has_evidence:
        _add(
            "evidence_required",
            "Missing evidence references. Cite crash frame, traceback, or diff evidence first.",
            "Bind evidence with llmdebug_diagnose/llmdebug_frame before continuing.",
            hard=True,
        )
    if action in {"plan", "change", "verify", "resolve"} and not has_hypothesis:
        _add(
            "hypothesis_required",
            "Missing root-cause hypothesis. Add a falsifiable causal claim before patching.",
            "Provide a hypothesis statement tied to cited evidence.",
            hard=True,
        )
    if action in {"change", "verify", "resolve"} and not has_plan:
        _add(
            "plan_required",
            "Missing experiment plan. Define minimal planned change and expected effect.",
            "Add a one-step experiment plan tied to your hypothesis.",
        )
    if action == "resolve":
        status = (
            str(cast(dict[str, Any], verification_result).get("status") or "")
            if isinstance(verification_result, dict)
            else ""
        )
        if status != "pass":
            _add(
                "verification_required",
                "Cannot mark resolved without a passing verification result.",
                "Run at least one verification check and report status=pass.",
                hard=True,
            )
    effective_budget = (
        request.exploration_budget_remaining
        if request.exploration_budget_remaining is not None
        else request.exploration_budget
    )
    if (
        action in {"change", "verify", "resolve"}
        and request.repeated_signature_count > max(effective_budget, 1)
        and not hypothesis_advanced
    ):
        _add(
            "loop_protection",
            "Failure signature repeated without hypothesis change. "
            "Exploration is still allowed, but RCA should be updated.",
            "Revise hypothesis or gather new evidence before repeating equivalent patch attempts.",
        )
    # Last, so the earlier checks keep their order in warnings (RCA-06).
    if current_state is not None:
        allowed_actions = allowed_actions_for_state(current_state)
        if action not in allowed_actions:
            _add(
                "illegal_transition",
                f"Action '{action}' is not a transition from state '{current_state}'.",
                f"Use one of the allowed actions: {', '.join(allowed_actions)}.",
                hard=True,
            )

    return warnings, next_required, triggered_checks, hard_failed


def _make_feedback(
    gate_mode: GateMode,
    warnings: list[str],
    next_required: list[str],
    triggered_checks: list[str],
    *,
    allowed: bool,
    severity: GateSeverity,
) -> GateFeedback:
    return GateFeedback(
        mode=gate_mode,
        allowed=allowed,
        severity=severity,
        warnings=tuple(warnings),
        next_required=tuple(next_required),
        triggered_checks=tuple(triggered_checks),
    )


def evaluate_gate(request: GateRequest | None = None, /, **legacy_kwargs: Any) -> GateFeedback:
    """Evaluate whether an RCA action should proceed under the active gate policy.

    Pass a :class:`GateRequest`. The keyword form
    ``evaluate_gate(mode=..., current_state=..., action=..., payload=...)`` still works
    but emits a ``DeprecationWarning``.

    Runs a series of prerequisite checks against the supplied *payload*
    (evidence, hypothesis, plan, verification status) and returns structured
    feedback.  The gate *mode* determines how check failures are surfaced:

    - **off**: All actions are allowed unconditionally; no checks run.
    - **soft**: All actions are allowed, but unmet prerequisites are
      surfaced as warnings with ``severity="warning"``.
    - **strict**: Actions with *hard* prerequisite failures (missing
      evidence, missing hypothesis, or unverified resolve) are blocked
      (``allowed=False``, ``severity="error"``).  An exploratory
      override with an ``override_reason`` in the payload can bypass
      the block.

    Prerequisite checks include:

    - ``evidence_required`` — evidence refs must be present before
      hypothesize/plan/change/verify/resolve.
    - ``hypothesis_required`` — a hypothesis is needed before
      plan/change/verify/resolve.
    - ``plan_required`` — an experiment plan is needed before
      change/verify/resolve.
    - ``verification_required`` — a passing verification result is
      required to resolve.
    - ``loop_protection`` — repeated failure signature without
      hypothesis advancement triggers a warning when the exploration
      budget is exceeded.
    - ``illegal_transition`` — the action isn't an allowed transition
      from *current_state* (hard; checked last).

    ``GateRequest`` fields:
        mode: Gate policy — ``"off"``, ``"soft"``, or ``"strict"``.
            ``None`` or unrecognized values default to ``"soft"``.
        current_state: The current RCA state, or ``None`` for a session's
            first attempt.  An action that isn't one of
            :func:`allowed_actions_for_state` for it triggers the hard
            ``illegal_transition`` check.
        action: The requested RCA action (e.g. ``"hypothesize"``,
            ``"plan"``, ``"change"``, ``"verify"``, ``"resolve"``).
        payload: Dict carrying the current attempt's data for gate
            evaluation.  Expected keys: ``evidence_refs``,
            ``hypothesis``, ``experiment_plan``,
            ``verification_result``, ``exploratory``,
            ``override_reason``, and ``hypothesis_advanced``.
        repeated_signature_count: How many consecutive attempts share the
            same failure signature.
        exploration_budget: Total exploration budget (max allowed
            same-signature attempts before loop protection fires).
        exploration_budget_remaining: Remaining budget units, if already
            partially consumed.  Falls back to *exploration_budget* when
            ``None``.

    Returns:
        A :class:`GateFeedback` instance with ``allowed``, ``severity``,
        ``warnings``, ``next_required`` steps, and ``triggered_checks``
        populated according to the gate policy and check results.
    """
    if request is None:
        import warnings as _warnings

        _warnings.warn(
            "evaluate_gate(mode=..., current_state=..., ...) is deprecated; "
            "pass a GateRequest instead",
            DeprecationWarning,
            stacklevel=2,
        )
        request = GateRequest(**legacy_kwargs)
    elif legacy_kwargs:
        raise TypeError("evaluate_gate() takes a GateRequest or keyword arguments, not both")
    gate_mode = normalize_gate_mode(request.mode)
    if gate_mode == "off":
        return _make_feedback(gate_mode, [], [], [], allowed=True, severity="info")

    warnings, next_required, triggered_checks, hard_failed = _collect_gate_issues(request)
    payload = request.payload

    if gate_mode == "soft":
        severity: GateSeverity = "warning" if warnings else "info"
        return _make_feedback(
            gate_mode, warnings, next_required, triggered_checks, allowed=True, severity=severity
        )

    # strict mode
    if hard_failed:
        exploratory = bool(payload.get("exploratory"))
        override_reason = str(payload.get("override_reason") or "").strip()
        if exploratory and override_reason:
            warnings.append(
                f"Strict gate override accepted for exploratory debugging: {override_reason}"
            )
            return _make_feedback(
                gate_mode,
                warnings,
                next_required,
                triggered_checks,
                allowed=True,
                severity="warning",
            )
        return _make_feedback(
            gate_mode, warnings, next_required, triggered_checks, allowed=False, severity="error"
        )

    severity = "warning" if warnings else "info"
    return _make_feedback(
        gate_mode, warnings, next_required, triggered_checks, allowed=True, severity=severity
    )
