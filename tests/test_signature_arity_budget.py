"""Repository-wide ratchet for named-parameter signature debt."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIGNATURE_BUDGET = 6
_SOURCE_ROOTS = ("src/llmdebug", "evals", "baselines", "scripts")
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        "_generated",
        "artifacts",
        "cases",
        "envs",
        "generated",
        "repos",
        "results",
        "swebench_pro_assets",
        "swebench_pro_runs",
        "swebench_runs",
    }
)


@dataclass(frozen=True)
class _FileArityDebt:
    reason: str
    ceilings: dict[str, int]


# This inventory records existing debt, not desired API design. Each ceiling is
# the current named-parameter count and each file carries an explicit migration
# reason. New entries and increases fail; reductions and removals pass.
_EXISTING_ARITY_DEBT: dict[str, _FileArityDebt] = {
    "baselines/intervenor/adapter.py": _FileArityDebt(
        "Boundary adapter still flattens upstream outcome metrics.",
        {
            "IntervenorAdapter._build_metrics": 10,
            "IntervenorAdapter._compatibility_failure_result": 14,
            "IntervenorAdapter._final_metrics": 11,
        },
    ),
    "baselines/lambda_rlm/adapter.py": _FileArityDebt(
        "Boundary adapter still flattens upstream outcome metrics.",
        {"LambdaRLMAdapter._build_metrics": 20},
    ),
    "baselines/rlm/adapter.py": _FileArityDebt(
        "Boundary adapter still flattens invocation and outcome state.",
        {"RLMAdapter._build_metrics": 24, "RLMAdapter._invoke_rlm": 7},
    ),
    "evals/adaptive_budget_online_policy.py": _FileArityDebt(
        "Research controller APIs still mirror the persisted policy event schema.",
        {
            "AdaptiveBudgetPolicyController.record_outcome": 38,
            "BudgetPolicyConfig.normalize": 15,
        },
    ),
    "evals/adaptive_controller_packet.py": _FileArityDebt(
        "Controller sidecar assembly still receives flattened run telemetry.",
        {"build_adaptive_controller_sidecars": 45},
    ),
    "evals/adaptive_controller_runtime.py": _FileArityDebt(
        "Controller runtime sidecar helpers still expose transitional flat inputs.",
        {"_build_sidecars": 8, "write_controller_sidecars_for_cycle": 7},
    ),
    "evals/adaptive_online_policy.py": _FileArityDebt(
        "Research controller APIs still mirror the persisted policy event schema.",
        {
            "AdaptiveOnlinePolicyController.record_outcome": 34,
            "OnlinePolicyConfig.normalize": 14,
        },
    ),
    "evals/adaptive_tep.py": _FileArityDebt(
        "TEP research plumbing has not yet been grouped into phase request objects.",
        {
            "AdaptiveTepController.__init__": 8,
            "AdaptiveTepController.record_free_phase": 8,
            "_render_critic_prompt": 7,
        },
    ),
    "evals/analyze_results.py": _FileArityDebt(
        "The OPE analysis helper directly exposes its statistical inputs.",
        {"_deterministic_target_ope": 7},
    ),
    "evals/attempt_metadata.py": _FileArityDebt(
        "The transitional grouped builder still mirrors AttemptMetadataRequest fields.",
        {"build_attempt_metadata_request": 26},
    ),
    "evals/adaptive_progress.py": _FileArityDebt(
        "Adaptive progress helpers (from run_eval, X-02) still take flattened attempt state.",
        {
            "_adaptive_progress_signal": 8,
            "_apply_no_progress_token_cap_candidate": 7,
            "_recompute_adaptive_progress_for_attempt_outcome": 11,
        },
    ),
    "evals/attempt_proposal.py": _FileArityDebt(
        "Proposal helpers (from run_eval, X-02) still take separate budget and context inputs.",
        {
            "_ensure_deterministic_seed_ack": 7,
            "_plan_patch_with_snapshot_context": 7,
            "_propose_patch_with_snapshot_context": 8,
        },
    ),
    "evals/attempt_records.py": _FileArityDebt(
        "Record builders (from run_eval, X-02) still mirror the flat JSONL record schema.",
        {"_build_attempt_outcome_record": 16, "_build_infra_event_record": 13},
    ),
    "evals/attempt_usage.py": _FileArityDebt(
        "Usage sync (from run_eval, X-02) still writes flattened patch metadata.",
        {"_apply_usage_totals_to_patch_meta": 7},
    ),
    "evals/attempt_verification.py": _FileArityDebt(
        "Patch recovery and metadata sync still receive separate verifier and recovery controls.",
        {
            "apply_diff_with_optional_recovery": 8,
            "_sync_patch_sanitize_metadata": 7,
            "_sync_patch_verifier_metadata": 8,
            "_sync_patch_verifier_recovery_metadata": 14,
            "_sync_syntax_preflight_metadata": 8,
        },
    ),
    "evals/eval_attempt_flow.py": _FileArityDebt(
        "The attempt orchestrator still has local seams over flattened cycle state.",
        {
            "_controller_runtime_evidence_inputs": 11,
            "_decide_adaptive_helper_recovery_topup": 16,
            "_resolve_stage_b_reveal_policy": 7,
            "_update_aux_signals_from_attempt": 8,
            "_run_one_condition_attempts._append_infra_event": 7,
            "_run_one_condition_attempts._build_stage_b_sections_for_attempt": 25,
            "_run_one_condition_attempts._capture_attempt_history": 7,
            "_run_one_condition_attempts._run_openai_operation_with_transport_recovery": 9,
            "_run_one_condition_attempts._run_retry_plan_patch": 11,
        },
    ),
    "evals/eval_evidence.py": _FileArityDebt(
        "Evidence assembly still receives separate limits and source handles.",
        {"build_fallback_context_bundle": 9, "execute_context_requests": 8},
    ),
    "evals/eval_policy_budget.py": _FileArityDebt(
        "Budget finalization still mirrors the full research telemetry record.",
        {"_finalize_adaptive_budget_policy_attempt": 55},
    ),
    "evals/eval_policy_features.py": _FileArityDebt(
        "Feature construction intentionally exposes the current research feature vector.",
        {
            "_adaptive_budget_family_availability": 17,
            "_adaptive_budget_feature_vector": 43,
            "_adaptive_budget_local_reward_components": 21,
            "_adaptive_budget_profile_for_family_step": 17,
            "_adaptive_online_feature_vector": 33,
            "_adaptive_online_local_reward_components": 25,
            "_build_adaptive_budget_control_profile": 13,
            "_build_adaptive_budget_family_profile": 15,
        },
    ),
    "evals/eval_policy_online.py": _FileArityDebt(
        "Online-policy accounting still mirrors the complete decision event schema.",
        {
            "_adaptive_online_reward_breakdown": 30,
            "_adaptive_online_should_defer_credit": 7,
            "_finalize_adaptive_online_attempt": 50,
            "_record_adaptive_online_policy_outcome": 19,
        },
    ),
    "evals/eval_prompts.py": _FileArityDebt(
        "Prompt renderers still accept separate sections and budget controls.",
        {
            "_build_context_compaction_bundle": 7,
            "_build_stage_b_prompt_sections": 27,
            "_render_retry_packet": 11,
            "_should_invoke_context_compaction": 8,
            "render_retry_plan_prompt": 20,
            "render_stage_b_patch_prompt": 23,
        },
    ),
    "evals/eval_recording.py": _FileArityDebt(
        "Base-record assembly still mirrors the stable JSONL record schema.",
        {"_build_base_record": 9},
    ),
    "evals/eval_types.py": _FileArityDebt(
        "Compatibility constructors and policy helpers still expose flat inputs.",
        {
            "RunConditionInputs.from_components": 13,
            "_helper_action_availability": 8,
            "_helper_action_fallback_reason": 9,
        },
    ),
    "evals/importers/bugsinpy_importer.py": _FileArityDebt(
        "Importer CLI seam still exposes individual materialization options.",
        {"import_bugsinpy": 7},
    ),
    "evals/importers/pybughive_importer.py": _FileArityDebt(
        "Importer CLI seam still exposes individual materialization options.",
        {"import_pybughive": 8},
    ),
    "evals/importers/top3_bootstrap.py": _FileArityDebt(
        "Bootstrap orchestration still passes importer paths and limits separately.",
        {
            "_materialize_bugsinpy": 8,
            "_materialize_pybughive": 8,
            "_materialize_trickybugs": 8,
            "materialize_top3": 8,
        },
    ),
    "evals/importers/trickybugs_importer.py": _FileArityDebt(
        "Importer CLI seam still exposes individual materialization options.",
        {"import_trickybugs": 7},
    ),
    "evals/lambda_rlm_backend.py": _FileArityDebt(
        "The backend boundary still receives provider and decomposition controls separately.",
        {"invoke_lambda_rlm_decomposition": 8},
    ),
    "evals/native_openai_attempt_flow.py": _FileArityDebt(
        "Response-mode telemetry synchronization still mirrors patch metadata fields.",
        {"sync_openai_response_mode_autoswitch_metadata": 8},
    ),
    "evals/native_stage_b_flow.py": _FileArityDebt(
        "Native Stage-B helpers still expose separate prompt and retry controls.",
        {"build_stage_b_prompt_with_budget": 7, "maybe_run_retry_plan_then_patch": 13},
    ),
    "evals/patch_relevance.py": _FileArityDebt(
        "The verifier scorer (from run_eval, X-02) still takes separate scope inputs.",
        {"_score_patch_relevance": 8},
    ),
    "evals/patchers/baseline_patcher.py": _FileArityDebt(
        "Baseline construction still exposes adapter-specific compatibility controls.",
        {"_hybrid_sidecars_active": 8, "build_baseline_patcher": 12},
    ),
    "evals/patchers/openai_compat_patcher.py": _FileArityDebt(
        "Provider compatibility helpers still mirror transport and response controls.",
        {
            "OpenAICompatPatcher.generate_json": 9,
            "OpenAICompatPatcher.generate_text": 7,
            "_chat_completion": 17,
            "_chat_completion_async": 17,
            "_compact_context_via_json": 17,
            "_compact_context_via_json_async": 17,
            "_propose_patch_via_diff": 17,
            "_propose_patch_via_diff_async": 16,
            "_propose_patch_via_edits": 19,
            "_propose_patch_via_edits_async": 18,
            "_propose_patch_via_full_file": 18,
            "_propose_patch_via_full_file_async": 17,
            "_request_context_via_json": 16,
            "_request_context_via_json_async": 16,
        },
    ),
    "evals/progress_display.py": _FileArityDebt(
        "Startup rendering still accepts independent run-state sections.",
        {"build_startup_sections": 12},
    ),
    "evals/pytest_runner.py": _FileArityDebt(
        "The pytest runner (from run_eval, X-02) mirrors run_pytest's public arguments.",
        {"_run_pytest_local": 7, "_run_pytest_pooled": 7, "run_pytest": 8},
    ),
    "evals/report_run_metrics.py": _FileArityDebt(
        "Metrics reporting still accepts independent run aggregates.",
        {"_print_report": 16},
    ),
    "evals/run_condition_builder.py": _FileArityDebt(
        "Run-condition assembly still receives separate subsystem configs.",
        {"build_run_condition_inputs": 12},
    ),
    "evals/subprocess_monitor.py": _FileArityDebt(
        "Subprocess monitoring still exposes independent resource and callback controls.",
        {"run_monitored_subprocess": 10},
    ),
    "evals/transport_failures.py": _FileArityDebt(
        "Request-failure sync (from run_eval, X-02) still writes flattened patch metadata.",
        {"_sync_patcher_request_failure_metadata": 9},
    ),
    "evals/swebench/generate.py": _FileArityDebt(
        "SWE-Bench generation still exposes per-run backend and prompt controls.",
        {
            "_assert_generation_health": 7,
            "_generate_for_condition": 13,
            "_generate_for_condition_async": 18,
            "_propose_prediction": 7,
            "_propose_prediction_async": 7,
            "generate_lite_predictions": 23,
        },
    ),
    "evals/swebench/score_official.py": _FileArityDebt(
        "Official scorer integration still mirrors command-line options.",
        {"build_run_evaluation_command": 9, "run_official_scoring": 10},
    ),
    "evals/swebench_pro/generate.py": _FileArityDebt(
        "SWE-Bench Pro generation still exposes per-run backend and prompt controls.",
        {
            "_assert_generation_health": 7,
            "_generate_for_condition": 13,
            "_generate_for_condition_async": 18,
            "_propose_prediction": 7,
            "_propose_prediction_async": 7,
            "generate_pro_predictions": 23,
        },
    ),
    "evals/swebench_pro/score_docker.py": _FileArityDebt(
        "Docker scorer integration still mirrors runtime and artifact options.",
        {"evaluate_instance_docker": 10, "run_docker_scoring": 12},
    ),
    "src/llmdebug/_debug_session.py": _FileArityDebt(
        "Debug-session construction still mirrors the serialized envelope.",
        {"build_debug_session": 7},
    ),
    "src/llmdebug/capture.py": _FileArityDebt(
        "Capture internals still expose separate serialization and exception controls.",
        {"_serialize_frame": 7, "_summarize_exception": 8},
    ),
    "src/llmdebug/cli.py": _FileArityDebt(
        "The Typer command signature intentionally mirrors public CLI flags.",
        {"export": 8},
    ),
    "src/llmdebug/export_artifact.py": _FileArityDebt(
        "Artifact rendering still exposes independent payload and presentation inputs.",
        {"_build_export_html_cards": 9, "build_export_payload": 10},
    ),
    "src/llmdebug/hooks.py": _FileArityDebt(
        "Hook setup still mirrors public installation and production config options.",
        {"_build_production_config": 7, "install_hooks": 9},
    ),
    "src/llmdebug/jupyter_plugin.py": _FileArityDebt(
        "Jupyter installation still mirrors public extension options.",
        {"install": 9},
    ),
    "src/llmdebug/mcp_contract.py": _FileArityDebt(
        "MCP response helpers still mirror protocol envelope fields.",
        {"build_json_response": 8, "render_tool_error": 7, "render_tool_response": 9},
    ),
    "src/llmdebug/mcp_diff_tools.py": _FileArityDebt(
        "MCP diff dispatch still receives individual protocol parameters.",
        {"handle_diff": 8},
    ),
    "src/llmdebug/mcp_server.py": _FileArityDebt(
        "MCP tool signatures intentionally mirror their published JSON contracts.",
        {
            "llmdebug_diagnose": 7,
            "llmdebug_diff": 8,
            "llmdebug_frame": 8,
            "llmdebug_git_context": 8,
            "llmdebug_rca_advance": 16,
            "llmdebug_show": 9,
        },
    ),
    "src/llmdebug/mcp_snapshot_tools.py": _FileArityDebt(
        "MCP snapshot dispatch still receives individual protocol parameters.",
        {"handle_frame": 8},
    ),
    "src/llmdebug/redaction_policy.py": _FileArityDebt(
        "Redaction resolution still mirrors the stable public policy options.",
        {"resolve_redaction_policy": 9},
    ),
}

_REMOVED_WIDE_SIGNATURES = frozenset(
    {
        "evals/attempt_metadata.py::build_attempt_metadata_request_from_legacy_kwargs",
        "evals/eval_recording.py::_materialize_attempt_metadata",
    }
)


def _named_parameter_count(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    parameters = (
        list(function_node.args.posonlyargs)
        + list(function_node.args.args)
        + list(function_node.args.kwonlyargs)
    )
    return sum(arg.arg not in {"self", "cls"} for arg in parameters)


class _FunctionArityVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.arities: dict[str, int] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified_name = ".".join((*self.scope, node.name))
        key = f"{self.relative_path}::{qualified_name}"
        self.arities[key] = max(self.arities.get(key, 0), _named_parameter_count(node))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def _iter_source_files() -> list[Path]:
    source_files: list[Path] = []
    for source_root in _SOURCE_ROOTS:
        root = _REPO_ROOT / source_root
        for module_path in root.rglob("*.py"):
            relative_parts = module_path.relative_to(root).parts
            if any(part in _EXCLUDED_DIRECTORY_NAMES for part in relative_parts[:-1]):
                continue
            source_files.append(module_path)
    return sorted(source_files)


def _collect_signature_arities() -> dict[str, int]:
    arities: dict[str, int] = {}
    for module_path in _iter_source_files():
        relative_path = module_path.relative_to(_REPO_ROOT).as_posix()
        module = ast.parse(module_path.read_text(encoding="utf-8"), filename=relative_path)
        visitor = _FunctionArityVisitor(relative_path)
        visitor.visit(module)
        arities.update(visitor.arities)
    return arities


def _flatten_existing_debt() -> dict[str, tuple[int, str]]:
    debt: dict[str, tuple[int, str]] = {}
    for relative_path, allowance in _EXISTING_ARITY_DEBT.items():
        assert allowance.reason.strip(), f"Missing arity-debt reason for {relative_path}"
        for qualified_name, ceiling in allowance.ceilings.items():
            key = f"{relative_path}::{qualified_name}"
            assert key not in debt, f"Duplicate arity-debt entry: {key}"
            assert ceiling > _SIGNATURE_BUDGET, f"Non-debt entry in arity baseline: {key}"
            debt[key] = (ceiling, allowance.reason)
    return debt


def test_named_parameter_arity_debt_does_not_grow() -> None:
    current_arities = _collect_signature_arities()
    current_debt = {
        key: count for key, count in current_arities.items() if count > _SIGNATURE_BUDGET
    }
    baseline = _flatten_existing_debt()

    assert not (_REMOVED_WIDE_SIGNATURES & baseline.keys()), (
        "Removed 120-parameter compatibility definitions must not remain in the debt baseline"
    )
    assert not (_REMOVED_WIDE_SIGNATURES & current_debt.keys()), (
        "The deprecated compatibility adapters must stay free of named-parameter debt"
    )

    failures: list[str] = []
    for key, count in sorted(current_debt.items()):
        allowance = baseline.get(key)
        if allowance is None:
            failures.append(
                f"new debt: {key} has {count} named parameters (budget {_SIGNATURE_BUDGET})"
            )
            continue
        ceiling, reason = allowance
        if count > ceiling:
            failures.append(f"increased debt: {key} grew {ceiling} -> {count}; reason: {reason}")

    assert not failures, "Signature arity debt grew:\n" + "\n".join(failures)
