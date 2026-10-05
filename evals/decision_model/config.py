"""Settings and CLI flags for the System-1 decision-model advisor of the helper policy.

Also registers the two policy-input flags introduced with it: the oracle case-label mask and the
lite-snapshot mode. Both change what the helper policy (and the decision model) can see, so they
are recorded in the run configuration next to the decision-model settings.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .schema import DEFAULT_PROFILE_NAME, LITE_SNAPSHOT_REPLACE, PROFILE_NAMES, get_profile

ROLE_OFF = "off"
ROLE_SHADOW = "shadow"
ROLE_POLICY = "policy"
ROLE_FEATURES = "features"
ROLE_PRIOR = "prior"
DECISION_MODEL_ROLES = (ROLE_OFF, ROLE_SHADOW, ROLE_POLICY, ROLE_FEATURES, ROLE_PRIOR)
ROLES_REQUIRING_POLICY_ON = frozenset({ROLE_POLICY, ROLE_FEATURES, ROLE_PRIOR})

LOG_STATE_FULL = "full"
LOG_STATE_HASH = "hash"
LOG_STATE_MODES = (LOG_STATE_FULL, LOG_STATE_HASH)

LITE_SNAPSHOT_SUPPLEMENT = "supplement"
LITE_SNAPSHOT_MODES = (LITE_SNAPSHOT_REPLACE, LITE_SNAPSHOT_SUPPLEMENT)
DEFAULT_LITE_SNAPSHOT_MODE = LITE_SNAPSHOT_REPLACE

DECISION_MODEL_URL_ENV = "TRACEGATE_DECISION_MODEL_URL"
# Optional bearer token (for example when the sidecar is reached through llama-swap). Read
# from the environment only, so it never lands in CLI history, lockfiles or run metadata.
DECISION_MODEL_API_KEY_ENV = "TRACEGATE_DECISION_MODEL_API_KEY"
DEFAULT_DECISION_MODEL_URL = "http://127.0.0.1:8090"
DEFAULT_DECISION_MODEL_NAME = "clef-flash"
DEFAULT_DECISION_MODEL_TIMEOUT_SEC = 10.0
DEFAULT_DECISION_MODEL_TEMPERATURE = 0.0
DEFAULT_DECISION_MODEL_PRIOR_BETA = 0.5
DEFAULT_DECISION_MODEL_PRIOR_ANNEAL_UPDATES = 50
DEFAULT_DECISION_MODEL_WARMUP_MIX = 0.3


@dataclass(frozen=True)
class DecisionModelPolicyConfig:
    """How the policy controller uses decision-model advice."""

    role: str = ROLE_OFF
    profile: str = DEFAULT_PROFILE_NAME
    temperature: float = DEFAULT_DECISION_MODEL_TEMPERATURE
    prior_beta: float = DEFAULT_DECISION_MODEL_PRIOR_BETA
    prior_anneal_updates: int = DEFAULT_DECISION_MODEL_PRIOR_ANNEAL_UPDATES
    warmup_mix: float = DEFAULT_DECISION_MODEL_WARMUP_MIX


@dataclass(frozen=True)
class DecisionModelClientConfig:
    """Where and how to reach the SystemOne-compatible decision-model endpoint."""

    url: str = DEFAULT_DECISION_MODEL_URL
    model: str = DEFAULT_DECISION_MODEL_NAME
    timeout_sec: float = DEFAULT_DECISION_MODEL_TIMEOUT_SEC
    api_key: str = field(default="", repr=False)


@dataclass(frozen=True)
class DecisionModelSettings:
    """All decision-model settings resolved from CLI/TOML arguments."""

    policy: DecisionModelPolicyConfig = field(default_factory=DecisionModelPolicyConfig)
    client: DecisionModelClientConfig = field(default_factory=DecisionModelClientConfig)
    log_state: str = LOG_STATE_FULL

    @property
    def enabled(self) -> bool:
        """Whether the decision model is consulted at all."""
        return self.policy.role != ROLE_OFF


def add_decision_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``--adaptive-decision-model-*`` flags on *parser*."""
    parser.add_argument(
        "--adaptive-decision-model-role",
        choices=list(DECISION_MODEL_ROLES),
        default=ROLE_OFF,
        help=(
            "How the System-1 decision model participates in helper-action selection for "
            "layered_adaptive_llm: shadow (log only), policy (it selects), features (its "
            "diagnostic answers become LinUCB features), prior (guided warm-up plus a fading "
            "score prior). policy/features/prior require --adaptive-online-policy-mode on "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-url",
        default="",
        help=(
            "Base URL of the SystemOne-compatible sidecar (POST /v1/systemone). Defaults to "
            f"${DECISION_MODEL_URL_ENV} or {DEFAULT_DECISION_MODEL_URL}."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-api-key",
        default="",
        help=(
            "Bearer token for a decision-model endpoint behind an API-key gateway such as "
            f"llama-swap (default: ${DECISION_MODEL_API_KEY_ENV}). Redacted from lockfiles and "
            "not allowed in config files."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-name",
        default=DEFAULT_DECISION_MODEL_NAME,
        help=f"Model name sent in the request body (default: {DEFAULT_DECISION_MODEL_NAME}).",
    )
    parser.add_argument(
        "--adaptive-decision-model-profile",
        choices=list(PROFILE_NAMES),
        default=DEFAULT_PROFILE_NAME,
        help=f"Versioned question profile (default: {DEFAULT_PROFILE_NAME}).",
    )
    parser.add_argument(
        "--adaptive-decision-model-timeout-sec",
        type=float,
        default=DEFAULT_DECISION_MODEL_TIMEOUT_SEC,
        help=(
            "Read timeout per decision-model request; failures fall back as documented "
            f"(default: {DEFAULT_DECISION_MODEL_TIMEOUT_SEC:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-temperature",
        type=float,
        default=DEFAULT_DECISION_MODEL_TEMPERATURE,
        help=(
            "policy role: 0 picks the most probable action (plus epsilon exploration); >0 "
            "samples from the tempered distribution (default: 0)."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-prior-beta",
        type=float,
        default=DEFAULT_DECISION_MODEL_PRIOR_BETA,
        help=(
            "prior role: initial weight of the centered log-probability score prior "
            f"(default: {DEFAULT_DECISION_MODEL_PRIOR_BETA:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-prior-anneal-updates",
        type=int,
        default=DEFAULT_DECISION_MODEL_PRIOR_ANNEAL_UPDATES,
        help=(
            "prior role: number of LinUCB updates after which the prior weight has halved "
            f"(default: {DEFAULT_DECISION_MODEL_PRIOR_ANNEAL_UPDATES})."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-warmup-mix",
        type=float,
        default=DEFAULT_DECISION_MODEL_WARMUP_MIX,
        help=(
            "prior role: uniform share mixed into the decision model's distribution during "
            f"warm-up (default: {DEFAULT_DECISION_MODEL_WARMUP_MIX:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-decision-model-log-state",
        choices=list(LOG_STATE_MODES),
        default=LOG_STATE_FULL,
        help=(
            "full: write each rendered decision state to <artifacts>/decision_model_states.jsonl; "
            "hash: log only its sha256 (default: full)."
        ),
    )


def add_policy_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the helper-policy input flags (case-label mask and lite-snapshot mode)."""
    parser.add_argument(
        "--adaptive-online-include-case-labels",
        dest="adaptive_online_include_case_labels",
        action="store_true",
        default=False,
        help=(
            "Feed case.json difficulty/snapshot-dependency/bug-source labels to the helper "
            "policy as features (pre-2026-10 behavior; off by default because the labels are "
            "not available at deployment time)."
        ),
    )
    parser.add_argument(
        "--no-adaptive-online-include-case-labels",
        dest="adaptive_online_include_case_labels",
        action="store_false",
        help="Do not feed case labels to the helper policy (default).",
    )
    parser.add_argument(
        "--adaptive-lite-snapshot-mode",
        choices=list(LITE_SNAPSHOT_MODES),
        default=DEFAULT_LITE_SNAPSHOT_MODE,
        help=(
            "With --adaptive-reveal-mode full: replace (default) makes snapshot_lite, "
            "frame_locals_lite and snapshot_full_lite replace the full snapshot section in the "
            "patch prompt; supplement keeps the full snapshot and adds the lite section "
            "(pre-2026-10 behavior)."
        ),
    )


def resolve_decision_model_api_key(
    explicit_key: str = "",
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the sidecar bearer token from the flag or the environment (empty when unset).

    Shared by the eval harness, the smoke command and the offline probe so every client
    authenticates the same way against a gateway such as llama-swap.
    """
    env = os.environ if environ is None else environ
    return str(explicit_key or "").strip() or env.get(DECISION_MODEL_API_KEY_ENV, "").strip()


def resolve_decision_model_url(
    explicit_url: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the sidecar URL from the flag, the environment, or the default."""
    env = os.environ if environ is None else environ
    for candidate in (explicit_url, env.get(DECISION_MODEL_URL_ENV, "")):
        value = str(candidate or "").strip()
        if value:
            return value.rstrip("/")
    return DEFAULT_DECISION_MODEL_URL


def _arg(args: Any, name: str, default: Any) -> Any:
    """Read a flag value; non-primitive values (e.g. MagicMock attributes) mean "not set"."""
    value = getattr(args, name, default)
    if value is None or not isinstance(value, (str, int, float, bool)):
        return default
    return value


def decision_model_settings_from_args(args: Any) -> DecisionModelSettings:
    """Build settings from parsed arguments, tolerating namespaces that lack the flags."""
    return DecisionModelSettings(
        policy=DecisionModelPolicyConfig(
            role=str(_arg(args, "adaptive_decision_model_role", ROLE_OFF)),
            profile=str(_arg(args, "adaptive_decision_model_profile", DEFAULT_PROFILE_NAME)),
            temperature=float(
                _arg(
                    args, "adaptive_decision_model_temperature", DEFAULT_DECISION_MODEL_TEMPERATURE
                )
            ),
            prior_beta=float(
                _arg(args, "adaptive_decision_model_prior_beta", DEFAULT_DECISION_MODEL_PRIOR_BETA)
            ),
            prior_anneal_updates=int(
                _arg(
                    args,
                    "adaptive_decision_model_prior_anneal_updates",
                    DEFAULT_DECISION_MODEL_PRIOR_ANNEAL_UPDATES,
                )
            ),
            warmup_mix=float(
                _arg(args, "adaptive_decision_model_warmup_mix", DEFAULT_DECISION_MODEL_WARMUP_MIX)
            ),
        ),
        client=DecisionModelClientConfig(
            url=resolve_decision_model_url(str(_arg(args, "adaptive_decision_model_url", ""))),
            model=str(_arg(args, "adaptive_decision_model_name", DEFAULT_DECISION_MODEL_NAME)),
            timeout_sec=float(
                _arg(
                    args, "adaptive_decision_model_timeout_sec", DEFAULT_DECISION_MODEL_TIMEOUT_SEC
                )
            ),
            api_key=resolve_decision_model_api_key(
                str(_arg(args, "adaptive_decision_model_api_key", ""))
            ),
        ),
        log_state=str(_arg(args, "adaptive_decision_model_log_state", LOG_STATE_FULL)),
    )


@dataclass(frozen=True)
class DecisionModelSetupCheck:
    """Outcome of cross-flag validation once the run's conditions are known."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    force_off: bool = False


def _numeric_errors(settings: DecisionModelSettings) -> list[str]:
    policy = settings.policy
    checks = (
        (settings.client.timeout_sec > 0.0, "--adaptive-decision-model-timeout-sec must be > 0"),
        (policy.temperature >= 0.0, "--adaptive-decision-model-temperature must be >= 0"),
        (policy.prior_beta >= 0.0, "--adaptive-decision-model-prior-beta must be >= 0"),
        (
            policy.prior_anneal_updates >= 1,
            "--adaptive-decision-model-prior-anneal-updates must be >= 1",
        ),
        (
            0.0 <= policy.warmup_mix <= 1.0,
            "--adaptive-decision-model-warmup-mix must be within [0, 1]",
        ),
    )
    values = (settings.client.timeout_sec, policy.temperature, policy.prior_beta, policy.warmup_mix)
    if not all(math.isfinite(float(value)) for value in values):
        return ["decision-model numeric settings must be finite"]
    return [message for ok, message in checks if not ok]


def check_decision_model_settings(
    settings: DecisionModelSettings,
    *,
    llm_discretion_condition: bool,
    online_policy_mode: str,
) -> DecisionModelSetupCheck:
    """Validate role/mode/profile combinations after conditions have been resolved."""
    role = settings.policy.role
    if role == ROLE_OFF:
        return DecisionModelSetupCheck()
    if role not in DECISION_MODEL_ROLES:
        return DecisionModelSetupCheck(errors=(f"unknown --adaptive-decision-model-role: {role}",))
    errors = _numeric_errors(settings)
    if settings.log_state not in LOG_STATE_MODES:
        errors.append(f"unknown --adaptive-decision-model-log-state: {settings.log_state}")
    try:
        profile = get_profile(settings.policy.profile)
    except ValueError as exc:
        return DecisionModelSetupCheck(errors=(*errors, str(exc)))
    if not llm_discretion_condition:
        return DecisionModelSetupCheck(
            errors=tuple(errors),
            warnings=(
                f"--adaptive-decision-model-role {role} has no effect without the "
                "layered_adaptive_llm condition; forcing it off.",
            ),
            force_off=True,
        )
    mode = str(online_policy_mode)
    if role in ROLES_REQUIRING_POLICY_ON and mode != "on":
        errors.append(
            f"--adaptive-decision-model-role {role} requires --adaptive-online-policy-mode on "
            "(shadow mode never updates or applies the helper policy)."
        )
    if role == ROLE_SHADOW and mode not in {"shadow", "on"}:
        errors.append(
            "--adaptive-decision-model-role shadow requires --adaptive-online-policy-mode "
            "shadow or on."
        )
    if role == ROLE_FEATURES and not profile.aux_questions:
        errors.append(
            f"profile {profile.name} has no diagnostic questions; the features role needs them."
        )
    if role == ROLE_PRIOR and not profile.main_question:
        errors.append(f"profile {profile.name} has no action question; the prior role needs it.")
    return DecisionModelSetupCheck(errors=tuple(errors))


def decision_model_run_config(args: Any) -> dict[str, Any]:
    """Run-configuration entries (resume fingerprint, run metadata); the URL is excluded."""
    settings = decision_model_settings_from_args(args)
    policy = settings.policy
    return {
        "adaptive_decision_model_role": policy.role,
        "adaptive_decision_model_profile": policy.profile,
        "adaptive_decision_model_name": settings.client.model,
        "adaptive_decision_model_timeout_sec": float(settings.client.timeout_sec),
        "adaptive_decision_model_temperature": float(policy.temperature),
        "adaptive_decision_model_prior_beta": float(policy.prior_beta),
        "adaptive_decision_model_prior_anneal_updates": int(policy.prior_anneal_updates),
        "adaptive_decision_model_warmup_mix": float(policy.warmup_mix),
        "adaptive_decision_model_log_state": settings.log_state,
        "adaptive_online_include_case_labels": bool(
            _arg(args, "adaptive_online_include_case_labels", False)
        ),
        "adaptive_lite_snapshot_mode": str(
            _arg(args, "adaptive_lite_snapshot_mode", DEFAULT_LITE_SNAPSHOT_MODE)
        ),
    }
