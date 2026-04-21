"""Typed extra-config schemas for each baseline adapter.

Each schema enumerates the ``config.extra`` keys an adapter consumes,
with types and defaults. ``validate()`` logs warnings for unrecognised
keys and returns a validated instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, cast

from baselines.openai_compat import normalize_compat_profile

logger = logging.getLogger(__name__)


STRICT_EXTRA_CONFIG_DEFAULT = True
_SHARED_HARNESS_EXTRA_KEYS = frozenset({"baseline_strict_extra_config"})
_PROMPT_CONTRACT_ALLOWED = frozenset({"canonical", "repair_fair_v1"})
_HYBRID_GUIDANCE_MODE_ALLOWED = frozenset({"fair_v1", "fixed_v1", "tuned_v1"})


class BaselineExtraConfigError(ValueError):
    """Raised when eval-mode config.extra validation fails."""


def _warn_unknown_keys(
    *,
    adapter_name: str,
    known: set[str],
    provided: set[str],
    strict: bool,
) -> None:
    unknown = sorted(provided - known)
    if unknown:
        if strict:
            raise BaselineExtraConfigError(
                f"{adapter_name}: unknown config.extra keys: {unknown} (known: {sorted(known)})"
            )
        logger.warning(
            "%s: ignoring unrecognised config.extra keys: %s (known: %s)",
            adapter_name,
            unknown,
            sorted(known),
        )


def _known_keys_for(cls: type[object]) -> set[str]:
    return {f.name for f in fields(cast(Any, cls))} | set(_SHARED_HARNESS_EXTRA_KEYS)


def _prompt_contract(value: Any, *, adapter_name: str, field_name: str) -> str:
    normalized = str(value or "").strip() or "canonical"
    if normalized not in _PROMPT_CONTRACT_ALLOWED:
        allowed = ", ".join(sorted(_PROMPT_CONTRACT_ALLOWED))
        raise BaselineExtraConfigError(
            f"{adapter_name}: invalid {field_name}={normalized!r}; expected one of: {allowed}"
        )
    return normalized


def _hybrid_guidance_mode(value: Any, *, adapter_name: str, field_name: str) -> str:
    normalized = str(value or "").strip() or "tuned_v1"
    if normalized not in _HYBRID_GUIDANCE_MODE_ALLOWED:
        allowed = ", ".join(sorted(_HYBRID_GUIDANCE_MODE_ALLOWED))
        raise BaselineExtraConfigError(
            f"{adapter_name}: invalid {field_name}={normalized!r}; expected one of: {allowed}"
        )
    return normalized


@dataclass(frozen=True)
class OpenAICompatExtraConfig:
    """Shared OpenAI-compatible compatibility controls."""

    baseline_openai_compat_profile: str = "auto"
    baseline_openai_allow_reasoning_fallback: bool = True
    baseline_openai_enable_thinking: bool = True
    baseline_hybrid_guidance_mode: str = "tuned_v1"


def _shared_openai_compat_kwargs(extra: dict[str, Any], *, adapter_name: str) -> dict[str, Any]:
    enable_thinking = bool(extra.get("baseline_openai_enable_thinking", True))
    allow_reasoning_fallback = bool(extra.get("baseline_openai_allow_reasoning_fallback", True))
    if enable_thinking and not allow_reasoning_fallback:
        allow_reasoning_fallback = True
    return {
        "baseline_openai_compat_profile": normalize_compat_profile(
            extra.get("baseline_openai_compat_profile")
        ),
        "baseline_openai_allow_reasoning_fallback": allow_reasoning_fallback,
        "baseline_openai_enable_thinking": enable_thinking,
        "baseline_hybrid_guidance_mode": _hybrid_guidance_mode(
            extra.get("baseline_hybrid_guidance_mode"),
            adapter_name=adapter_name,
            field_name="baseline_hybrid_guidance_mode",
        ),
    }


@dataclass(frozen=True)
class LDBExtraConfig(OpenAICompatExtraConfig):
    """Extra config keys consumed by the LDB adapter."""

    @classmethod
    def validate(
        cls,
        extra: dict[str, Any],
        *,
        adapter_name: str = "ldb",
        strict: bool = STRICT_EXTRA_CONFIG_DEFAULT,
    ) -> LDBExtraConfig:
        known = _known_keys_for(cls)
        _warn_unknown_keys(
            adapter_name=adapter_name,
            known=known,
            provided=set(extra),
            strict=strict,
        )
        return cls(**_shared_openai_compat_kwargs(extra, adapter_name=adapter_name))


@dataclass(frozen=True)
class IntervenorExtraConfig(OpenAICompatExtraConfig):
    """Extra config keys consumed by the INTERVENOR adapter."""

    baseline_intervenor_teacher_max_tokens: int = 1024
    baseline_intervenor_learner_max_tokens: int = 2048
    baseline_intervenor_teacher_guidance_max_chars: int = 4000
    baseline_intervenor_learner_guidance_max_chars: int = 6000
    baseline_intervenor_append_output_contract: bool = True

    @classmethod
    def validate(
        cls,
        extra: dict[str, Any],
        *,
        adapter_name: str = "intervenor",
        strict: bool = STRICT_EXTRA_CONFIG_DEFAULT,
    ) -> IntervenorExtraConfig:
        known = _known_keys_for(cls)
        _warn_unknown_keys(
            adapter_name=adapter_name,
            known=known,
            provided=set(extra),
            strict=strict,
        )
        return cls(
            **_shared_openai_compat_kwargs(extra, adapter_name=adapter_name),
            baseline_intervenor_teacher_max_tokens=_bounded_positive_int(
                extra.get("baseline_intervenor_teacher_max_tokens"),
                default=1024,
            ),
            baseline_intervenor_learner_max_tokens=_bounded_positive_int(
                extra.get("baseline_intervenor_learner_max_tokens"),
                default=2048,
            ),
            baseline_intervenor_teacher_guidance_max_chars=_bounded_positive_int(
                extra.get("baseline_intervenor_teacher_guidance_max_chars"),
                default=4000,
            ),
            baseline_intervenor_learner_guidance_max_chars=_bounded_positive_int(
                extra.get("baseline_intervenor_learner_guidance_max_chars"),
                default=6000,
            ),
            baseline_intervenor_append_output_contract=bool(
                extra.get("baseline_intervenor_append_output_contract", True)
            ),
        )


@dataclass(frozen=True)
class TraceCoderExtraConfig(OpenAICompatExtraConfig):
    """Extra config keys consumed by the TraceCoder adapter."""

    tracecoder_fake_responses_path: str = ""
    baseline_tracecoder_controller_guidance_max_chars: int = 2000
    baseline_tracecoder_max_no_improvement_streak: int = 2
    baseline_tracecoder_analysis_max_tokens: int = 4096
    baseline_tracecoder_implementation_max_tokens: int = 4096
    baseline_tracecoder_problem_summary_max_chars: int = 4000
    baseline_tracecoder_exact_test_context_max_chars: int = 3000
    baseline_tracecoder_failure_feedback_max_chars: int = 6000
    baseline_tracecoder_failure_feedback_tail_max_chars: int = 2000
    baseline_tracecoder_instrumentation_output_max_lines: int = 80
    baseline_tracecoder_instrumentation_output_max_chars: int = 12000

    @classmethod
    def validate(
        cls,
        extra: dict[str, Any],
        *,
        adapter_name: str = "tracecoder",
        strict: bool = STRICT_EXTRA_CONFIG_DEFAULT,
    ) -> TraceCoderExtraConfig:
        known = _known_keys_for(cls)
        _warn_unknown_keys(
            adapter_name=adapter_name,
            known=known,
            provided=set(extra),
            strict=strict,
        )
        return cls(
            **_shared_openai_compat_kwargs(extra, adapter_name=adapter_name),
            tracecoder_fake_responses_path=str(extra.get("tracecoder_fake_responses_path") or ""),
            baseline_tracecoder_controller_guidance_max_chars=_bounded_positive_int(
                extra.get("baseline_tracecoder_controller_guidance_max_chars"),
                default=2000,
            ),
            baseline_tracecoder_max_no_improvement_streak=_bounded_positive_int(
                extra.get("baseline_tracecoder_max_no_improvement_streak"),
                default=2,
            ),
            baseline_tracecoder_analysis_max_tokens=_bounded_positive_int(
                extra.get("baseline_tracecoder_analysis_max_tokens"),
                default=4096,
            ),
            baseline_tracecoder_implementation_max_tokens=_bounded_positive_int(
                extra.get("baseline_tracecoder_implementation_max_tokens"),
                default=4096,
            ),
            baseline_tracecoder_problem_summary_max_chars=_bounded_positive_int(
                extra.get("baseline_tracecoder_problem_summary_max_chars"),
                default=4000,
            ),
            baseline_tracecoder_exact_test_context_max_chars=_bounded_positive_int(
                extra.get("baseline_tracecoder_exact_test_context_max_chars"),
                default=3000,
            ),
            baseline_tracecoder_failure_feedback_max_chars=_bounded_positive_int(
                extra.get("baseline_tracecoder_failure_feedback_max_chars"),
                default=6000,
            ),
            baseline_tracecoder_failure_feedback_tail_max_chars=_bounded_positive_int(
                extra.get("baseline_tracecoder_failure_feedback_tail_max_chars"),
                default=2000,
            ),
            baseline_tracecoder_instrumentation_output_max_lines=_bounded_positive_int(
                extra.get("baseline_tracecoder_instrumentation_output_max_lines"),
                default=80,
            ),
            baseline_tracecoder_instrumentation_output_max_chars=_bounded_positive_int(
                extra.get("baseline_tracecoder_instrumentation_output_max_chars"),
                default=12000,
            ),
        )


@dataclass(frozen=True)
class LambdaRLMExtraConfig(OpenAICompatExtraConfig):
    """Extra config keys consumed by the lambda-RLM adapter."""

    baseline_lambda_context_window_chars: int = 0
    baseline_lambda_accuracy_target: float = 0.0
    baseline_lambda_a_leaf: float = 0.0
    baseline_lambda_a_compose: float = 0.0
    baseline_lambda_verbose: bool = False
    baseline_lambda_temperature: float | None = None
    baseline_lambda_top_p: float | None = None
    baseline_lambda_max_tokens: int | None = None
    baseline_lambda_stream: bool = False
    baseline_lambda_timeout_sec: float | None = None
    baseline_lambda_controller_guidance_max_chars: int = 2500
    baseline_lambda_failure_feedback_max_chars: int = 3500
    baseline_lambda_failure_feedback_meta_max_chars: int = 2000
    baseline_lambda_force_task_type: str = ""
    baseline_lambda_prompt_contract: str = "canonical"

    @classmethod
    def validate(
        cls,
        extra: dict[str, Any],
        *,
        adapter_name: str = "lambda_rlm",
        strict: bool = STRICT_EXTRA_CONFIG_DEFAULT,
    ) -> LambdaRLMExtraConfig:
        known = _known_keys_for(cls)
        _warn_unknown_keys(
            adapter_name=adapter_name,
            known=known,
            provided=set(extra),
            strict=strict,
        )
        return cls(
            **_shared_openai_compat_kwargs(extra, adapter_name=adapter_name),
            baseline_lambda_context_window_chars=int(
                extra.get("baseline_lambda_context_window_chars") or 0
            ),
            baseline_lambda_accuracy_target=float(
                extra.get("baseline_lambda_accuracy_target") or 0.0
            ),
            baseline_lambda_a_leaf=float(extra.get("baseline_lambda_a_leaf") or 0.0),
            baseline_lambda_a_compose=float(extra.get("baseline_lambda_a_compose") or 0.0),
            baseline_lambda_verbose=bool(extra.get("baseline_lambda_verbose")),
            baseline_lambda_temperature=_opt_float(extra.get("baseline_lambda_temperature")),
            baseline_lambda_top_p=_opt_float(extra.get("baseline_lambda_top_p")),
            baseline_lambda_max_tokens=_opt_int(extra.get("baseline_lambda_max_tokens")),
            baseline_lambda_stream=bool(extra.get("baseline_lambda_stream")),
            baseline_lambda_timeout_sec=_opt_float(extra.get("baseline_lambda_timeout_sec")),
            baseline_lambda_controller_guidance_max_chars=_bounded_positive_int(
                extra.get("baseline_lambda_controller_guidance_max_chars"),
                default=2500,
            ),
            baseline_lambda_failure_feedback_max_chars=_bounded_positive_int(
                extra.get("baseline_lambda_failure_feedback_max_chars"),
                default=3500,
            ),
            baseline_lambda_failure_feedback_meta_max_chars=_bounded_positive_int(
                extra.get("baseline_lambda_failure_feedback_meta_max_chars"),
                default=2000,
            ),
            baseline_lambda_force_task_type=str(extra.get("baseline_lambda_force_task_type") or ""),
            baseline_lambda_prompt_contract=_prompt_contract(
                extra.get("baseline_lambda_prompt_contract"),
                adapter_name=adapter_name,
                field_name="baseline_lambda_prompt_contract",
            ),
        )


@dataclass(frozen=True)
class RLMExtraConfig(OpenAICompatExtraConfig):
    """Extra config keys consumed by the standalone RLM adapter."""

    baseline_rlm_max_depth: int = 0
    baseline_rlm_max_iterations: int = 0
    baseline_rlm_max_errors: int = 0
    baseline_rlm_compaction: bool = False
    baseline_rlm_compaction_threshold_pct: float = 0.0
    baseline_rlm_verbose: bool = False
    baseline_rlm_max_total_tokens: int = 0
    baseline_rlm_temperature: float | None = None
    baseline_rlm_top_p: float | None = None
    baseline_rlm_max_tokens: int | None = None
    baseline_rlm_timeout_sec: float | None = None
    baseline_rlm_controller_guidance_max_chars: int = 2500
    baseline_rlm_failure_feedback_max_chars: int = 3500
    baseline_rlm_failure_feedback_meta_max_chars: int = 2000
    baseline_rlm_prompt_contract: str = "canonical"

    @classmethod
    def validate(
        cls,
        extra: dict[str, Any],
        *,
        adapter_name: str = "rlm",
        strict: bool = STRICT_EXTRA_CONFIG_DEFAULT,
    ) -> RLMExtraConfig:
        known = _known_keys_for(cls)
        _warn_unknown_keys(
            adapter_name=adapter_name,
            known=known,
            provided=set(extra),
            strict=strict,
        )
        return cls(
            **_shared_openai_compat_kwargs(extra, adapter_name=adapter_name),
            baseline_rlm_max_depth=int(extra.get("baseline_rlm_max_depth") or 0),
            baseline_rlm_max_iterations=int(extra.get("baseline_rlm_max_iterations") or 0),
            baseline_rlm_max_errors=int(extra.get("baseline_rlm_max_errors") or 0),
            baseline_rlm_compaction=bool(extra.get("baseline_rlm_compaction")),
            baseline_rlm_compaction_threshold_pct=float(
                extra.get("baseline_rlm_compaction_threshold_pct") or 0.0
            ),
            baseline_rlm_verbose=bool(extra.get("baseline_rlm_verbose")),
            baseline_rlm_max_total_tokens=int(extra.get("baseline_rlm_max_total_tokens") or 0),
            baseline_rlm_temperature=_opt_float(extra.get("baseline_rlm_temperature")),
            baseline_rlm_top_p=_opt_float(extra.get("baseline_rlm_top_p")),
            baseline_rlm_max_tokens=_opt_int(extra.get("baseline_rlm_max_tokens")),
            baseline_rlm_timeout_sec=_opt_float(extra.get("baseline_rlm_timeout_sec")),
            baseline_rlm_controller_guidance_max_chars=_bounded_positive_int(
                extra.get("baseline_rlm_controller_guidance_max_chars"),
                default=2500,
            ),
            baseline_rlm_failure_feedback_max_chars=_bounded_positive_int(
                extra.get("baseline_rlm_failure_feedback_max_chars"),
                default=3500,
            ),
            baseline_rlm_failure_feedback_meta_max_chars=_bounded_positive_int(
                extra.get("baseline_rlm_failure_feedback_meta_max_chars"),
                default=2000,
            ),
            baseline_rlm_prompt_contract=_prompt_contract(
                extra.get("baseline_rlm_prompt_contract"),
                adapter_name=adapter_name,
                field_name="baseline_rlm_prompt_contract",
            ),
        )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bounded_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
