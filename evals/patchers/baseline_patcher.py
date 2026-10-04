"""Patcher that delegates to an external baseline adapter.

Wraps a ``BaselineAdapter`` (from ``baselines.adapter_base``) into the
``Patcher`` protocol so that external baselines (LDB, INTERVENOR,
TraceCoder, lambda-RLM, and RLM) can participate in ``run_eval.py`` runs
alongside llmdebug's own conditions.

Usage from the CLI::

    python -m evals.run_eval \\
        --patcher baseline \\
        --baseline-name ldb \\
        --baseline-repo-dir baselines/repos/LLMDebugger \\
        --openai-base-url http://127.0.0.1:1234 \\
        --openai-model <model> \\
        --k 2
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineCaseIncompatibleError,
    BaselineConfig,
    BaselineGenerationSettings,
    BaselineResult,
)
from baselines.case_information import CASE_INFO_POLICY, MODEL_HIDDEN_CASE_FIELDS
from baselines.contracts import BASELINE_CONTRACT_VERSION
from evals.adaptive_controller_packet import (
    HYBRID_GUIDANCE_MODE_FAIR_V1,
    HYBRID_GUIDANCE_MODE_FIXED_V1,
    HYBRID_GUIDANCE_MODE_TUNED_V1,
    hybrid_guidance_metadata,
    parse_controller_packet,
    render_hybrid_controller_guidance,
)

from .base import ContextRequestResult, Patcher, PatchResult
from .openai_compat_patcher import (
    OpenAICompatPatcher,
    _default_api_key,
    _normalize_base_url,
    _probe_server_ready,
)
from .payload_types import PatcherMeta

logger = logging.getLogger(__name__)

# Registry of known adapters — populated lazily on first use.
_ADAPTER_REGISTRY: dict[str, type[BaselineAdapter]] | None = None

# Legacy hardcoded fallback table.  Kept only as a last resort when an
# adapter does not define ``default_model`` and no model is configured
# explicitly.  New adapters should set ``default_model`` on the class
# instead of adding entries here.
_BASELINE_DEFAULT_MODELS: dict[str, str] = {
    "ldb": "gpt-4",
    "intervenor": "gpt-4o-mini",
    "tracecoder": "gpt-4o-mini",
    "lambda_rlm": "gpt-4o-mini",
    "rlm": "gpt-4o-mini",
}


@dataclass(frozen=True)
class _BaselineCaseContextLoadResult:
    """Typed result for loading ``baseline_case_context.json``."""

    case_context: BaselineCaseContext | None
    status: str
    schema_version: str | None = None
    issues: tuple[str, ...] = ()
    error_code: str | None = None
    error_type: str | None = None
    error_detail: str | None = None

    @property
    def error_message(self) -> str | None:
        if not self.error_type or not self.error_detail:
            return None
        return f"{self.error_type}: {self.error_detail}"


def _load_baseline_case_context(
    path: Path,
    *,
    default_case_id: str,
) -> _BaselineCaseContextLoadResult:
    """Best-effort load for ``baseline_case_context.json``.

    Missing files are treated as normal and produce a ``missing`` status.
    Invalid JSON or schema problems never raise; instead the caller receives a
    typed status plus structured issue metadata.
    """

    if not path.is_file():
        return _BaselineCaseContextLoadResult(case_context=None, status="missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _BaselineCaseContextLoadResult(
            case_context=None,
            status="invalid_json",
            error_code="invalid_json",
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
    if not isinstance(payload, dict):
        return _BaselineCaseContextLoadResult(
            case_context=None,
            status="invalid_schema",
            error_code="invalid_payload_type",
            error_type="ValueError",
            error_detail=f"expected object, got {type(payload).__name__}",
        )

    issues: list[str] = []
    schema_version: str | None = None

    raw_schema_version = payload.get("schema_version")
    if raw_schema_version is None:
        issues.append("schema_version_missing")
    elif isinstance(raw_schema_version, str):
        normalized_schema_version = raw_schema_version.strip()
        if normalized_schema_version:
            schema_version = normalized_schema_version
        else:
            issues.append("schema_version_blank")
    else:
        issues.append(f"schema_version_invalid_type:{type(raw_schema_version).__name__}")

    case_id = default_case_id
    raw_case_id = payload.get("case_id")
    if isinstance(raw_case_id, str):
        if raw_case_id.strip():
            case_id = raw_case_id
    elif raw_case_id is not None:
        issues.append(f"case_id_invalid_type:{type(raw_case_id).__name__}")

    python_args: tuple[str, ...] = ()
    raw_python_args = payload.get("python_args")
    if raw_python_args is None:
        python_args = ()
    elif isinstance(raw_python_args, list):
        filtered_python_args = tuple(str(item) for item in raw_python_args if isinstance(item, str))
        dropped_items = len(raw_python_args) - len(filtered_python_args)
        if dropped_items:
            issues.append(f"python_args_dropped_items:{dropped_items}")
        python_args = filtered_python_args
    else:
        issues.append(f"python_args_invalid_type:{type(raw_python_args).__name__}")

    timeout_sec = 0
    raw_timeout_sec = payload.get("timeout_sec")
    if raw_timeout_sec is None:
        timeout_sec = 0
    elif isinstance(raw_timeout_sec, bool):
        issues.append("timeout_sec_invalid_type:bool")
    else:
        try:
            timeout_sec = max(0, int(raw_timeout_sec))
        except (TypeError, ValueError):
            issues.append(f"timeout_sec_invalid_value:{raw_timeout_sec!r}")

    # Analysis-only case labels never reach an adapter, even from older context files.
    dropped = [name for name in MODEL_HIDDEN_CASE_FIELDS if payload.get(name) is not None]
    if dropped:
        issues.append("hidden_case_fields_dropped:" + ",".join(dropped))

    case_context = BaselineCaseContext(
        case_id=case_id,
        python_args=python_args,
        timeout_sec=timeout_sec,
        schema_version=schema_version or BASELINE_CONTRACT_VERSION,
    )
    return _BaselineCaseContextLoadResult(
        case_context=case_context,
        status="loaded_with_warnings" if issues else "loaded",
        schema_version=schema_version,
        issues=tuple(issues),
    )


def _load_sidecar_mapping(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"invalid_sidecar_type:{type(payload).__name__}"
    return payload, None


def _guidance_sections_from_text(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_line in str(text or "").splitlines():
        section_name, separator, _rest = raw_line.partition(":")
        normalized = section_name.strip()
        if not separator or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _hybrid_sidecars_active(
    *,
    controller_packet: dict[str, Any] | None,
    controller_packet_error: str | None,
    evidence_bundle: dict[str, Any] | None,
    evidence_bundle_error: str | None,
    history_bundle: dict[str, Any] | None,
    history_bundle_error: str | None,
    verifier_report: dict[str, Any] | None,
    verifier_report_error: str | None,
) -> bool:
    return any(
        value
        for value in (
            controller_packet is not None,
            bool(controller_packet_error),
            evidence_bundle is not None,
            bool(evidence_bundle_error),
            history_bundle is not None,
            bool(history_bundle_error),
            verifier_report is not None,
            bool(verifier_report_error),
        )
    )


def _baseline_hybrid_guidance_metadata(
    *,
    adapter: BaselineAdapter,
    config: BaselineConfig,
    controller_packet_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    mode = str(config.extra.get("baseline_hybrid_guidance_mode") or "").strip()
    if not mode:
        mode = HYBRID_GUIDANCE_MODE_TUNED_V1
    controller_packet = parse_controller_packet(controller_packet_raw)

    if adapter.name == "intervenor":
        from baselines.extra_configs import IntervenorExtraConfig

        extra_config = IntervenorExtraConfig.validate(
            config.extra,
            adapter_name=adapter.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        )
        if mode in {HYBRID_GUIDANCE_MODE_FAIR_V1, HYBRID_GUIDANCE_MODE_FIXED_V1}:
            render_kwargs: dict[str, Any] = {
                "mode": mode,
                "include_history": mode != HYBRID_GUIDANCE_MODE_FIXED_V1,
                "include_retry_plan": mode != HYBRID_GUIDANCE_MODE_FIXED_V1,
            }
            if mode == HYBRID_GUIDANCE_MODE_FIXED_V1:
                render_kwargs["max_chars"] = min(
                    extra_config.baseline_intervenor_teacher_guidance_max_chars,
                    extra_config.baseline_intervenor_learner_guidance_max_chars,
                )
            rendered = render_hybrid_controller_guidance(
                controller_packet,
                **render_kwargs,
            )
            metadata = hybrid_guidance_metadata(rendered)
            metadata["baseline_hybrid_guidance_chars"] = rendered.char_count * 2
            return metadata
        rich_renderer = getattr(adapter, "_render_controller_guidance", None)
        if callable(rich_renderer):
            teacher_guidance = str(
                rich_renderer(
                    controller_packet=controller_packet,
                    focus="teacher",
                    max_chars=extra_config.baseline_intervenor_teacher_guidance_max_chars,
                )
                or ""
            )
            learner_guidance = str(
                rich_renderer(
                    controller_packet=controller_packet,
                    focus="learner",
                    max_chars=extra_config.baseline_intervenor_learner_guidance_max_chars,
                )
                or ""
            )
            sections = _guidance_sections_from_text(
                "\n".join(part for part in (teacher_guidance, learner_guidance) if part)
            )
            return {
                "baseline_hybrid_guidance_mode": mode,
                "baseline_hybrid_guidance_chars": len(teacher_guidance) + len(learner_guidance),
                "baseline_hybrid_guidance_sections": list(sections),
            }
        return {
            "baseline_hybrid_guidance_mode": mode,
            "baseline_hybrid_guidance_chars": 0,
            "baseline_hybrid_guidance_sections": [],
        }

    max_chars = 2_000
    if adapter.name == "tracecoder":
        from baselines.extra_configs import TraceCoderExtraConfig

        max_chars = TraceCoderExtraConfig.validate(
            config.extra,
            adapter_name=adapter.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        ).baseline_tracecoder_controller_guidance_max_chars
    elif adapter.name == "lambda_rlm":
        from baselines.extra_configs import LambdaRLMExtraConfig

        max_chars = LambdaRLMExtraConfig.validate(
            config.extra,
            adapter_name=adapter.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        ).baseline_lambda_controller_guidance_max_chars
    elif adapter.name == "rlm":
        from baselines.extra_configs import RLMExtraConfig

        max_chars = RLMExtraConfig.validate(
            config.extra,
            adapter_name=adapter.name,
            strict=bool(config.extra.get("baseline_strict_extra_config", True)),
        ).baseline_rlm_controller_guidance_max_chars

    return hybrid_guidance_metadata(
        render_hybrid_controller_guidance(
            controller_packet,
            mode=mode,
            include_history=True,
            include_retry_plan=True,
            max_chars=max_chars,
        )
    )


def _normalize_baseline_metrics(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    metrics_raw = cast(dict[object, object], raw)
    out: dict[str, Any] = {}
    int_keys = (
        "llm_call_count",
        "llm_timeout_count",
        "usage_reported_call_count",
        "usage_missing_call_count",
        "reported_tokens_prompt",
        "reported_tokens_completion",
        "reported_tokens_total",
        "reported_tokens_reasoning",
        "prompt_chars_total",
        "completion_chars_total",
        "chars_input",
        "chars_output",
        "chars_total",
        "reported_chars_prompt",
        "reported_chars_completion",
        "reported_chars_total",
        "learner_empty_response_count",
        "attempts_with_postcheck_count",
        "test_feedback_steps",
        "test_feedback_nonempty_steps",
        "test_feedback_improvement_steps",
        "test_feedback_regression_steps",
        "test_feedback_stagnation_steps",
        "test_feedback_final_fail_count",
        "internal_task_detection_call_count",
        "internal_repl_llm_query_call_count",
        "internal_length_finish_count",
        "internal_compat_recovery_count",
        "internal_planned_decomposition_attempt_count",
        "internal_executed_decomposition_attempt_count",
        "internal_guard_stop_count",
        "internal_search_exhausted_attempt_count",
        "internal_workspace_write_success_count",
        "internal_salvage_success_count",
    )
    float_keys = (
        "llm_duration_sec",
        "eval_duration_sec",
        "internal_total_duration_sec",
        "orchestration_duration_sec",
        "teacher_duration_sec_total",
        "learner_duration_sec_total",
        "postcheck_duration_sec_total",
    )
    bool_keys = (
        "usage_available",
        "direct_gen_passed",
        "debug_added_value",
        "method_instrumentation",
        "method_two_step_repair",
        "is_passing",
    )
    string_keys = (
        "usage_source",
        "stopped_reason",
        "last_llm_failure_kind",
        "last_llm_failure_stage",
    )
    for key in int_keys:
        value = metrics_raw.get(key)
        if isinstance(value, int):
            out[key] = max(0, value)
    for key in float_keys:
        value = metrics_raw.get(key)
        if isinstance(value, (int, float)):
            out[key] = max(0.0, float(value))
    for key in bool_keys:
        value = metrics_raw.get(key)
        if isinstance(value, bool):
            out[key] = value
    for key in string_keys:
        value = metrics_raw.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out or None


def _exact_baseline_char_totals(metrics: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(metrics, dict):
        return None
    prompt_chars = metrics.get("prompt_chars_total")
    if not isinstance(prompt_chars, int):
        prompt_chars = metrics.get("chars_input")
    if not isinstance(prompt_chars, int):
        prompt_chars = metrics.get("reported_chars_prompt")
    completion_chars = metrics.get("completion_chars_total")
    if not isinstance(completion_chars, int):
        completion_chars = metrics.get("chars_output")
    if not isinstance(completion_chars, int):
        completion_chars = metrics.get("reported_chars_completion")
    if not isinstance(prompt_chars, int) or not isinstance(completion_chars, int):
        return None
    return max(0, prompt_chars), max(0, completion_chars)


def _build_adapter_registry() -> dict[str, type[BaselineAdapter]]:
    """Lazily import and register all known baseline adapters."""
    registry: dict[str, type[BaselineAdapter]] = {}
    try:
        from baselines.ldb.adapter import LDBAdapter

        registry["ldb"] = LDBAdapter
    except ImportError:
        pass
    try:
        from baselines.intervenor.adapter import IntervenorAdapter

        registry["intervenor"] = IntervenorAdapter
    except ImportError:
        pass
    try:
        from baselines.tracecoder.adapter import TraceCoderAdapter

        registry["tracecoder"] = TraceCoderAdapter
    except ImportError:
        pass
    try:
        from baselines.lambda_rlm.adapter import LambdaRLMAdapter

        registry["lambda_rlm"] = LambdaRLMAdapter
    except ImportError:
        pass
    try:
        from baselines.rlm.adapter import RLMAdapter

        registry["rlm"] = RLMAdapter
    except ImportError:
        pass
    return registry


def get_adapter_registry() -> dict[str, type[BaselineAdapter]]:
    """Return the (lazily built) adapter registry."""
    global _ADAPTER_REGISTRY
    if _ADAPTER_REGISTRY is None:
        _ADAPTER_REGISTRY = _build_adapter_registry()
    return _ADAPTER_REGISTRY


def available_baselines() -> list[str]:
    """Return names of all importable baseline adapters."""
    return sorted(get_adapter_registry().keys())


def resolve_effective_baseline_model(
    *,
    baseline_name: str,
    configured_model: str,
    adapter: BaselineAdapter | None = None,
) -> str:
    """Return the model a baseline run will effectively use.

    Resolution order:
    1. ``configured_model`` if non-empty.
    2. ``adapter.default_model`` if an adapter instance is provided.
    3. Legacy ``_BASELINE_DEFAULT_MODELS`` lookup table (kept for compat).
    """
    model = str(configured_model or "").strip()
    if model:
        return model
    if adapter is not None:
        adapter_default = str(getattr(adapter, "default_model", "") or "").strip()
        if adapter_default:
            return adapter_default
    return _BASELINE_DEFAULT_MODELS.get(str(baseline_name or "").strip(), "")


def _infer_repair_contract(result: BaselineResult) -> tuple[str, list[str]]:
    """Infer reporting metadata for the adapter result shape.

    Reporting precedence at the patcher boundary is intentionally distinct from
    diff extraction:

    1. explicit ``result.meta["repair_targets"]`` / ``repair_scope``
    2. structured ``result.file_edits``
    3. adapter-supplied ``result.patch_diff``
    4. legacy single-file ``result.target_path``

    This preserves adapter-owned metadata when supplied, while still recording
    a reasonable repair scope for legacy or partially structured results.
    """

    raw_targets = result.meta.get("repair_targets")
    if isinstance(raw_targets, list):
        targets = sorted({str(item).strip() for item in raw_targets if str(item).strip()})
        if targets:
            raw_scope = result.meta.get("repair_scope")
            scope = (
                str(raw_scope).strip()
                if isinstance(raw_scope, str) and str(raw_scope).strip()
                else ("single_file" if len(targets) == 1 else "multi_file")
            )
            return scope, targets

    if result.file_edits:
        targets = sorted(
            {str(edit.path).strip() for edit in result.file_edits if str(edit.path).strip()}
        )
        if targets:
            return ("single_file" if len(targets) == 1 else "multi_file"), targets

    if isinstance(result.patch_diff, str) and result.patch_diff.strip():
        return "patch_diff", []

    target_path = str(result.target_path or "").strip()
    if target_path:
        return "single_file", [target_path]

    return "unknown", []


def baseline_case_support(
    adapter: BaselineAdapter, case_dirs: Sequence[tuple[str, Path]]
) -> dict[str, Any]:
    """Count the cases *adapter* can convert (BAS-01, D3).

    Recorded in run metadata so a baseline's comparison states its N explicitly; the
    incompatible cases are excluded from its success rate and paired statistics.
    """
    reasons: dict[str, int] = {}
    convertible = 0
    for case_id, case_dir in case_dirs:
        try:
            adapter.convert_case(
                case_dir,
                case_context=BaselineCaseContext(
                    case_id=case_id, python_args=("-m", "pytest", "-q"), timeout_sec=0
                ),
            )
        except BaselineCaseIncompatibleError as exc:
            reasons[exc.reason] = reasons.get(exc.reason, 0) + 1
        except Exception as exc:  # converter bug: count it, don't hide it
            key = f"conversion_error:{type(exc).__name__}"
            reasons[key] = reasons.get(key, 0) + 1
        else:
            convertible += 1
    return {
        "baseline": adapter.name,
        "total_cases": len(case_dirs),
        "convertible_cases": convertible,
        "incompatible_reasons": dict(sorted(reasons.items())),
    }


class BaselinePatcher(Patcher):
    """Patcher that delegates to an external baseline adapter.

    The patcher wraps a single ``BaselineAdapter`` instance and forwards
    each ``propose_patch()`` call through the adapter's convert → repair →
    diff pipeline.

    Parameters:
        adapter: An initialised ``BaselineAdapter`` (already ``setup()``-ed).
        config: Shared ``BaselineConfig`` for this run.
    """

    def __init__(self, adapter: BaselineAdapter, config: BaselineConfig) -> None:
        self.adapter = adapter
        self.config = config
        self.name = adapter.name
        self._helper_delegate: OpenAICompatPatcher | None = None

    def _helper_backend(self) -> OpenAICompatPatcher:
        if self._helper_delegate is None:
            resolved_model = resolve_effective_baseline_model(
                baseline_name=self.adapter.name,
                configured_model=self.config.model,
                adapter=self.adapter,
            )
            self._helper_delegate = OpenAICompatPatcher(
                base_url=str(self.config.api_base_url),
                model=resolved_model,
                api_key=(self.config.api_key or "").strip() or None,
                timeout_sec=float(self.config.api_timeout_sec or 120.0),
                compat_profile=str(
                    self.config.extra.get("baseline_openai_compat_profile") or "auto"
                ),
                allow_reasoning_fallback=bool(
                    self.config.extra.get("baseline_openai_allow_reasoning_fallback", True)
                ),
                enable_thinking=bool(
                    self.config.extra.get("baseline_openai_enable_thinking", True)
                ),
                name=f"{self.adapter.name}_stage_a_helper",
            )
        return self._helper_delegate

    def supports_request_context(self) -> bool:
        """Baseline-backed adaptive runs use a shared OpenAI-compatible Stage-A helper."""
        return True

    def supports_strict_json_first(self) -> bool:
        """The shared Stage-A helper supports strict-JSON-first retries."""
        return True

    def request_context(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        strict_json_first: bool = False,
    ) -> ContextRequestResult | None:
        """Delegate Stage-A helper requests through a shared OpenAI-compatible backend."""
        return self._helper_backend().request_context(
            prompt=prompt,
            context_dir=context_dir,
            cwd=cwd,
            strict_json_first=strict_json_first,
        )

    def probe_server_ready(self) -> tuple[bool, PatcherMeta]:
        """Probe the OpenAI-compatible backend used by the baseline adapter."""
        base_url = _normalize_base_url(self.config.api_base_url)
        api_key = self.config.api_key or _default_api_key()
        ready, meta = _probe_server_ready(
            base_url=base_url,
            api_key=api_key,
            timeout_sec=self.config.api_timeout_sec,
            http2=False,
        )
        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault(
            "model",
            resolve_effective_baseline_model(
                baseline_name=self.adapter.name,
                configured_model=self.config.model,
                adapter=self.adapter,
            ),
        )
        return ready, result_meta

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Run the baseline on the case at *cwd* and return a PatchResult."""
        meta: dict[str, Any] = {
            "baseline": self.adapter.name,
            "model": resolve_effective_baseline_model(
                baseline_name=self.adapter.name,
                configured_model=self.config.model,
                adapter=self.adapter,
            ),
            "base_url": self.config.api_base_url,
            "max_attempts": self.config.max_attempts,
            "api_timeout_sec": self.config.api_timeout_sec,
            "process_timeout_sec": self.config.process_timeout_sec,
            "config_validation_mode": (
                "strict" if self.config.extra.get("baseline_strict_extra_config", True) else "warn"
            ),
        }
        meta["case_info_policy"] = CASE_INFO_POLICY
        meta["baseline_generation"] = self.config.generation.record(
            enforced=bool(getattr(self.adapter, "honours_generation_contract", False))
        )
        contract = self.adapter.contract()
        meta.update(contract.to_dict())
        if not meta.get("contract_version"):
            meta["contract_version"] = BASELINE_CONTRACT_VERSION
        case_context_load = _load_baseline_case_context(
            context_dir / "baseline_case_context.json",
            default_case_id=cwd.name,
        )
        case_context = case_context_load.case_context
        if case_context_load.status not in {"missing", "loaded"}:
            meta["baseline_case_context_status"] = case_context_load.status
        if case_context_load.issues:
            meta["baseline_case_context_issues"] = list(case_context_load.issues)
        if case_context_load.schema_version:
            meta["baseline_case_context_schema_version"] = case_context_load.schema_version
        if case_context_load.error_code:
            meta["baseline_case_context_error_code"] = case_context_load.error_code
        if case_context_load.error_type:
            meta["baseline_case_context_error_type"] = case_context_load.error_type
        if case_context_load.error_detail:
            meta["baseline_case_context_error_detail"] = case_context_load.error_detail
        if case_context_load.error_message:
            meta["baseline_case_context_error"] = case_context_load.error_message

        # 1. Convert the case.
        try:
            case_conversion_started = time.perf_counter()
            raw_case = self.adapter.convert_case(cwd, case_context=case_context)
            meta["baseline_case_conversion_sec"] = time.perf_counter() - case_conversion_started
            # Typed schemas expose .to_dict(); plain dicts (legacy stubs) pass through.
            if hasattr(raw_case, "to_dict"):
                case_input: dict[str, Any] = raw_case.to_dict()
            elif isinstance(raw_case, dict):
                case_input = raw_case
            else:
                raise TypeError(
                    f"convert_case() must return a BaseCaseInput or dict, got {type(raw_case).__name__}"
                )
            case_information = case_input.get("case_information")
            if isinstance(case_information, dict):
                meta["case_information"] = dict(case_information)
        except BaselineCaseIncompatibleError as exc:
            meta["error"] = "case_incompatible"
            meta["baseline_supported"] = False
            meta["compatibility_reason"] = exc.reason
            if exc.detail:
                meta["compatibility_detail"] = exc.detail
            return PatchResult(diff=None, meta=cast(PatcherMeta, meta))
        except Exception as exc:
            meta["error"] = "case_conversion_failed"
            meta["exc"] = f"{type(exc).__name__}: {exc}"
            return PatchResult(diff=None, meta=cast(PatcherMeta, meta))

        sidecar_load_started = time.perf_counter()
        controller_packet, controller_packet_error = _load_sidecar_mapping(
            context_dir / "llmdebug_controller_packet.json"
        )
        evidence_bundle, evidence_bundle_error = _load_sidecar_mapping(
            context_dir / "llmdebug_evidence_bundle.json"
        )
        history_bundle, history_bundle_error = _load_sidecar_mapping(
            context_dir / "llmdebug_history_bundle.json"
        )
        verifier_report, verifier_report_error = _load_sidecar_mapping(
            context_dir / "llmdebug_verifier_report.json"
        )
        if controller_packet is not None:
            case_input["llmdebug_controller_packet"] = controller_packet
            meta["adaptive_controller_packet_loaded"] = True
            meta["adaptive_controller_schema_version"] = str(
                controller_packet.get("schema_version") or ""
            )
            sidecar_version = str(controller_packet.get("_sidecar_schema_version") or "")
            if sidecar_version:
                meta["adaptive_sidecar_schema_version"] = sidecar_version
        elif controller_packet_error:
            meta["adaptive_controller_packet_error"] = controller_packet_error
        if evidence_bundle is not None:
            case_input["llmdebug_evidence"] = evidence_bundle
            meta["adaptive_evidence_bundle_loaded"] = True
        elif evidence_bundle_error:
            meta["adaptive_evidence_bundle_error"] = evidence_bundle_error
        if history_bundle is not None:
            case_input["llmdebug_history"] = history_bundle
            meta["adaptive_history_bundle_loaded"] = True
        elif history_bundle_error:
            meta["adaptive_history_bundle_error"] = history_bundle_error
        if verifier_report is not None:
            case_input["llmdebug_verifier_report"] = verifier_report
            meta["adaptive_verifier_report_loaded"] = True
        elif verifier_report_error:
            meta["adaptive_verifier_report_error"] = verifier_report_error
        hybrid_sidecars_active = _hybrid_sidecars_active(
            controller_packet=controller_packet,
            controller_packet_error=controller_packet_error,
            evidence_bundle=evidence_bundle,
            evidence_bundle_error=evidence_bundle_error,
            history_bundle=history_bundle,
            history_bundle_error=history_bundle_error,
            verifier_report=verifier_report,
            verifier_report_error=verifier_report_error,
        )
        if hybrid_sidecars_active:
            meta.update(
                _baseline_hybrid_guidance_metadata(
                    adapter=self.adapter,
                    config=self.config,
                    controller_packet_raw=controller_packet,
                )
            )
        meta["baseline_sidecar_load_sec"] = time.perf_counter() - sidecar_load_started

        # 2. Run the baseline's repair loop.
        result: BaselineResult
        repair_started = time.perf_counter()
        deadline_sec = float(self.config.process_timeout_sec or 0.0)
        meta["baseline_deadline_sec"] = deadline_sec
        try:
            self.adapter.set_progress_callback(self._current_progress_callback())
            result = self.adapter.run_repair(case_input, self.config)
        except Exception as exc:
            meta["error"] = "repair_failed"
            meta["exc"] = f"{type(exc).__name__}: {exc}"
            return PatchResult(diff=None, meta=cast(PatcherMeta, meta))
        finally:
            self.adapter.set_progress_callback(None)
            # BAS-08: one observed wall clock per run_repair, comparable across adapters.
            repair_wall_sec = time.perf_counter() - repair_started
            meta["baseline_run_repair_wall_sec"] = repair_wall_sec
            meta["baseline_deadline_exceeded"] = bool(
                deadline_sec > 0 and repair_wall_sec > deadline_sec
            )

        meta["iterations_used"] = result.iterations_used
        meta["baseline_success"] = result.success
        repair_scope, repair_targets = _infer_repair_contract(result)
        meta["repair_scope"] = repair_scope
        if repair_targets:
            meta["repair_targets"] = repair_targets
        if result.token_usage:
            meta["usage"] = result.token_usage
        baseline_metrics = _normalize_baseline_metrics(result.meta.get("metrics"))
        if baseline_metrics is not None:
            meta["baseline_metrics"] = baseline_metrics
            llm_call_count = baseline_metrics.get("llm_call_count")
            if isinstance(llm_call_count, int):
                meta["llm_call_count"] = llm_call_count
            exact_char_totals = _exact_baseline_char_totals(baseline_metrics)
            if exact_char_totals is not None:
                prompt_chars_total, completion_chars_total = exact_char_totals
                meta["prompt_chars_total"] = prompt_chars_total
                meta["completion_chars_total"] = completion_chars_total
                meta["chars_input"] = prompt_chars_total
                meta["chars_output"] = completion_chars_total
                meta["chars_total"] = prompt_chars_total + completion_chars_total
        if result.meta:
            meta["baseline_meta"] = result.meta

        # 3. Extract unified diff.
        # Diff extraction precedence is adapter-owned and intentionally differs
        # from ``_infer_repair_contract()``:
        # patch_diff -> file_edits -> repaired_code.
        try:
            diff = self.adapter.extract_diff(cwd, result)
        except Exception as exc:
            meta["error"] = "diff_extraction_failed"
            meta["exc"] = f"{type(exc).__name__}: {exc}"
            return PatchResult(diff=None, meta=cast(PatcherMeta, meta))

        return PatchResult(diff=diff, meta=cast(PatcherMeta, meta))


def build_baseline_patcher(
    *,
    baseline_name: str,
    repo_dir: str | Path,
    python_executable: str | Path | None = None,
    model: str = "",
    api_base_url: str = "http://127.0.0.1:1234",
    api_key: str = "",
    max_attempts: int = 2,
    api_timeout_sec: float | None = None,
    process_timeout_sec: float | None = None,
    generation: BaselineGenerationSettings | None = None,
    extra: dict[str, Any] | None = None,
    strict_extra_config: bool = True,
) -> BaselinePatcher:
    """Factory: build a ready-to-use ``BaselinePatcher``.

    Looks up the adapter by *baseline_name*, creates a ``BaselineConfig``,
    calls ``adapter.setup()``, and returns a configured ``BaselinePatcher``.

    Raises:
        ValueError: If *baseline_name* is not found in the registry.
        RuntimeError: If ``adapter.setup()`` fails.
    """
    registry = get_adapter_registry()
    if baseline_name not in registry:
        available = ", ".join(sorted(registry.keys())) or "(none)"
        msg = f"Unknown baseline {baseline_name!r}. Available: {available}"
        raise ValueError(msg)

    adapter = registry[baseline_name]()
    config = BaselineConfig(
        model=model,
        api_base_url=api_base_url,
        api_key=api_key,
        max_attempts=max_attempts,
        api_timeout_sec=api_timeout_sec,
        process_timeout_sec=process_timeout_sec,
        repo_dir=Path(repo_dir),
        generation=generation if generation is not None else BaselineGenerationSettings(),
        python_executable=Path(python_executable).expanduser() if python_executable else None,
        extra={**(extra or {}), "baseline_strict_extra_config": bool(strict_extra_config)},
    )
    adapter.setup(config)
    adapter.validate_extra_config(config)
    return BaselinePatcher(adapter=adapter, config=config)
