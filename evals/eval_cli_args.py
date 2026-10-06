"""Command-line parser for ``python -m evals.run_eval``."""

from __future__ import annotations

import argparse
import os

from baselines.openai_compat import COMPAT_PROFILE_CHOICES
from evals.baseline_runtime import (
    _BASELINE_SHORTHAND_CHOICES,
    DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT,
    DEFAULT_BASELINE_RLM_MAX_DEPTH,
    DEFAULT_BASELINE_RLM_MAX_ERRORS,
    DEFAULT_BASELINE_RLM_MAX_ITERATIONS,
)
from evals.decision_model.config import add_decision_model_arguments, add_policy_input_arguments
from evals.eval_cli_args_adaptive import (
    add_adaptive_arguments,
)
from evals.eval_config import add_execution_arguments, resolve_openai_api_key_default
from evals.eval_recording import DEFAULT_REPO_OPENAI_COMPAT_PROFILE
from evals.eval_types import (
    DEFAULT_EXECUTOR,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_MAX_WAIT_SEC,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_POLL_SEC,
    DEFAULT_PROGRESS_HEARTBEAT_SEC,
    DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
    DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
    DEFAULT_TRANSPORT_FAIL_FAST_STREAK,
)
from evals.lambda_rlm_backend import (
    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
    DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
)


def build_parser() -> argparse.ArgumentParser:
    """The ``run_eval`` command-line parser, with config-file and environment defaults.

    The groups are added in a fixed order: ``set_defaults`` in the OpenAI group and the
    ``--help`` layout both depend on it.
    """
    parser = argparse.ArgumentParser(
        description="Run the llmdebug multi-condition repair benchmark harness.",
        # ORC-06: config-file values must not override abbreviated CLI flags.
        allow_abbrev=False,
    )
    _add_run_selection_arguments(parser)
    _add_patcher_and_baseline_arguments(parser)
    _add_openai_arguments(parser)
    _add_context_and_prompt_arguments(parser)
    add_adaptive_arguments(parser)
    add_execution_arguments(parser, default_executor=DEFAULT_EXECUTOR)
    add_decision_model_arguments(parser)
    add_policy_input_arguments(parser)
    return parser


def _add_run_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Case selection, scheduling, run identity and the patch strategy."""
    parser.add_argument("--cases-dir", default="evals/cases", help="Cases directory.")
    parser.add_argument("--case", action="append", default=[], help="Case id(s) to run.")
    parser.add_argument(
        "--case-list-file",
        action="append",
        default=[],
        help=(
            "Path to newline-delimited case ids to include (repeatable). "
            "Blank lines and '#' comments are ignored."
        ),
    )
    parser.add_argument(
        "--category", action="append", default=[], help="Category filter (repeatable)."
    )
    parser.add_argument(
        "--difficulty", action="append", default=[], help="Difficulty filter (repeatable)."
    )
    parser.add_argument(
        "--snapshot-dependency",
        action="append",
        default=[],
        help="Snapshot dependency filter (repeatable): required|helpful|none.",
    )
    parser.add_argument(
        "--bug-source", action="append", default=[], help="Bug source filter (repeatable)."
    )
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Max number of cases to run (0 = all)."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Number of case/condition units to execute concurrently.",
    )
    parser.add_argument(
        "--progress-verbosity",
        default="basic",
        help="Progress verbosity level for runtime status output (basic|detailed|trace).",
    )
    parser.add_argument(
        "--progress-heartbeat-sec",
        type=float,
        default=DEFAULT_PROGRESS_HEARTBEAT_SEC,
        help="Heartbeat interval in seconds for in-flight unit progress (0 disables).",
    )
    parser.add_argument(
        "--transport-fail-fast-streak",
        type=int,
        default=DEFAULT_TRANSPORT_FAIL_FAST_STREAK,
        help=(
            "Abort early after this many consecutive terminal transport request failures "
            "(0 disables; default: 5)."
        ),
    )
    parser.add_argument(
        "--case-fail-fast-on-request-budget-exceeded",
        dest="case_fail_fast_on_request_budget_exceeded",
        action="store_true",
        default=True,
        help=(
            "Stop remaining attempts for the current case/condition when an attempt reports "
            "`request_budget_exceeded` (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-case-fail-fast-on-request-budget-exceeded",
        dest="case_fail_fast_on_request_budget_exceeded",
        action="store_false",
        help=(
            "Disable per-case fail-fast on `request_budget_exceeded` and continue retry attempts."
        ),
    )
    parser.add_argument(
        "--conditions",
        default="base_repair",
        help=(
            "Comma-separated list: "
            "base_repair,cot_repair,"
            "layered_adaptive_gated,layered_adaptive_llm,"
            "ldb,intervenor,tracecoder,lambda_rlm,rlm"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier. If the matching results file exists, unfinished units are resumed.",
    )
    parser.add_argument(
        "--allow-settings-drift",
        action="store_true",
        help=(
            "Resume even when the run settings fingerprint differs from the existing results; "
            "new records then carry their own run metadata (ORC-04)."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Path to TOML config file with run settings. "
            "Top-level keys or [run_eval] keys map to CLI destinations."
        ),
    )
    parser.add_argument("--k", type=int, default=2, help="Max patch attempts per condition.")
    parser.add_argument(
        "--patcher",
        default="command",
        choices=["command", "heuristic", "null", "openai", "baseline"],
        help="Patch strategy.",
    )


def _add_patcher_and_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    """Command patcher, external baselines and the lambda-RLM controller."""
    parser.add_argument(
        "--baseline",
        default="",
        choices=_BASELINE_SHORTHAND_CHOICES,
        help=(
            "One-switch baseline selector. External baselines resolve to "
            "--patcher baseline --baseline-name NAME and auto-fill the standard "
            "baselines/repos/<repo> and baselines/envs/<name>/bin/python paths when present. "
            "Native shorthand 'cot_repair' resolves onto the native OpenAI path instead."
        ),
    )
    parser.add_argument(
        "--patch-cmd",
        default="",
        help="Command to run for command patcher (overrides LLMDEBUG_EVAL_PATCH_CMD).",
    )
    parser.add_argument(
        "--patch-cmd-timeout-sec",
        type=float,
        default=30.0,
        help="Timeout in seconds for command patcher subprocess execution.",
    )
    parser.add_argument(
        "--patch-cmd-shell",
        action="store_true",
        help="Run command patcher with shell=True (default: disabled).",
    )
    parser.add_argument(
        "--baseline-name",
        default="",
        help=(
            "External baseline adapter name (for --patcher baseline). "
            "E.g. ldb, intervenor, tracecoder, lambda_rlm, rlm."
        ),
    )
    parser.add_argument(
        "--baseline-repo-dir",
        default="",
        help="Path to the cloned external baseline repository (for --patcher baseline).",
    )
    parser.add_argument(
        "--baseline-python",
        default="",
        help=(
            "Optional Python executable for the selected baseline runtime env. "
            "If omitted, run_eval requires baselines/envs/<baseline>/bin/python."
        ),
    )
    parser.add_argument(
        "--baseline-process-timeout-sec",
        type=float,
        default=None,
        help=(
            "Optional timeout for one baseline-owned subprocess attempt. Defaults to "
            "--openai-timeout-sec when unset."
        ),
    )
    parser.add_argument(
        "--baseline-max-iterations",
        type=int,
        default=None,
        help=(
            "Max internal iterations per baseline invocation. Controls the adapter's "
            "debug/repair loop count independently of --k (which sets the harness-level "
            "retry count). Defaults to 2 when omitted."
        ),
    )
    parser.add_argument(
        "--baseline-openai-compat-profile",
        default=DEFAULT_REPO_OPENAI_COMPAT_PROFILE,
        choices=["auto", "qwen_openai_raw", "qwen_dashscope", "lmstudio_model_yaml", "off"],
        help=(
            "Shared OpenAI-compatible request-shaping profile for external baselines. "
            "Use qwen_openai_raw for vLLM/SGLang-style Qwen endpoints, "
            "qwen_dashscope for DashScope-compatible Qwen endpoints, "
            "lmstudio_model_yaml for LM Studio / llama.cpp model-template setups, "
            "or off to disable request shaping."
        ),
    )
    parser.add_argument(
        "--baseline-openai-allow-reasoning-fallback",
        dest="baseline_openai_allow_reasoning_fallback",
        action="store_true",
        default=True,
        help=(
            "Allow external baselines to consume reasoning-only response fields when "
            "message.content is empty (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-baseline-openai-allow-reasoning-fallback",
        dest="baseline_openai_allow_reasoning_fallback",
        action="store_false",
        help="Disable reasoning-only response fallback for external baselines.",
    )
    parser.add_argument(
        "--baseline-openai-enable-thinking",
        dest="baseline_openai_enable_thinking",
        action="store_true",
        default=True,
        help=(
            "Enable thinking/reasoning mode for external baselines "
            "(e.g. Qwen enable_thinking=True). Also auto-enables "
            "--baseline-openai-allow-reasoning-fallback (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-baseline-openai-enable-thinking",
        dest="baseline_openai_enable_thinking",
        action="store_false",
        help="Disable thinking/reasoning mode for external baselines.",
    )
    parser.add_argument(
        "--baseline-hybrid-guidance-mode",
        default="tuned_v1",
        choices=["tuned_v1", "fair_v1", "fixed_v1"],
        help=(
            "How llmdebug controller guidance is exposed to hybrid adaptive+baseline runs: "
            "'tuned_v1' preserves baseline-specific prompt integration, 'fair_v1' "
            "uses one shared canonical guidance block across all baselines, and "
            "'fixed_v1' uses a deterministic evidence block with full-snapshot "
            "injection intended for INTERVENOR fixed-evidence studies."
        ),
    )
    parser.add_argument(
        "--baseline-intervenor-teacher-max-tokens",
        type=int,
        default=1024,
        help="Teacher-stage max_tokens for the INTERVENOR adapter (default: 1024).",
    )
    parser.add_argument(
        "--baseline-intervenor-learner-max-tokens",
        type=int,
        default=2048,
        help="Learner-stage max_tokens for the INTERVENOR adapter (default: 2048).",
    )
    parser.add_argument(
        "--baseline-intervenor-teacher-guidance-max-chars",
        type=int,
        default=4000,
        help="Teacher-stage controller-guidance budget for INTERVENOR (default: 4000).",
    )
    parser.add_argument(
        "--baseline-intervenor-learner-guidance-max-chars",
        type=int,
        default=6000,
        help="Learner-stage controller-guidance budget for INTERVENOR (default: 6000).",
    )
    parser.add_argument(
        "--baseline-intervenor-append-output-contract",
        dest="baseline_intervenor_append_output_contract",
        action="store_true",
        default=True,
        help="Append the explicit single-file output suffix to INTERVENOR learner prompts (default: enabled).",
    )
    parser.add_argument(
        "--no-baseline-intervenor-append-output-contract",
        dest="baseline_intervenor_append_output_contract",
        action="store_false",
        help="Disable the explicit output-contract suffix on INTERVENOR learner prompts.",
    )
    parser.add_argument(
        "--baseline-tracecoder-controller-guidance-max-chars",
        type=int,
        default=2000,
        help="Controller-guidance budget appended to TraceCoder prompts (default: 2000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-max-no-improvement-streak",
        type=int,
        default=2,
        help="TraceCoder --max-no-improvement-streak value (default: 2).",
    )
    parser.add_argument(
        "--baseline-tracecoder-analysis-max-tokens",
        type=int,
        default=4096,
        help="TraceCoder max_tokens for instrumentation/analysis-stage calls (default: 4096).",
    )
    parser.add_argument(
        "--baseline-tracecoder-implementation-max-tokens",
        type=int,
        default=4096,
        help="TraceCoder max_tokens for direct/code-implementation calls (default: 4096).",
    )
    parser.add_argument(
        "--baseline-tracecoder-problem-summary-max-chars",
        type=int,
        default=4000,
        help="Max chars kept in the compact TraceCoder task/problem summary (default: 4000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-exact-test-context-max-chars",
        type=int,
        default=3000,
        help="Max chars of verbatim failing-test context injected into TraceCoder analysis prompts (default: 3000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-failure-feedback-max-chars",
        type=int,
        default=6000,
        help="Max chars retained in TraceCoder failure feedback summaries (default: 6000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-failure-feedback-tail-max-chars",
        type=int,
        default=2000,
        help="Max chars reserved for the bounded tail excerpt in TraceCoder failure feedback (default: 2000).",
    )
    parser.add_argument(
        "--baseline-tracecoder-instrumentation-output-max-lines",
        type=int,
        default=80,
        help="Max lines retained from llmdebug instrumentation output before TraceCoder prompts (default: 80).",
    )
    parser.add_argument(
        "--baseline-tracecoder-instrumentation-output-max-chars",
        type=int,
        default=12000,
        help="Max chars retained from llmdebug instrumentation output before TraceCoder prompts (default: 12000).",
    )
    parser.add_argument(
        "--baseline-lambda-context-window-chars",
        type=int,
        default=DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
        help=(
            "Context window budget in chars passed to the lambda_rlm baseline "
            f"(default: {DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS})."
        ),
    )
    parser.add_argument(
        "--baseline-lambda-accuracy-target",
        type=float,
        default=0.80,
        help="Target accuracy passed to the lambda_rlm planner (default: 0.80).",
    )
    parser.add_argument(
        "--baseline-lambda-a-leaf",
        type=float,
        default=0.95,
        help="Leaf-call accuracy prior for the lambda_rlm planner (default: 0.95).",
    )
    parser.add_argument(
        "--baseline-lambda-a-compose",
        type=float,
        default=0.90,
        help="Composition accuracy prior for the lambda_rlm planner (default: 0.90).",
    )
    parser.add_argument(
        "--baseline-lambda-verbose",
        dest="baseline_lambda_verbose",
        action="store_true",
        default=False,
        help="Enable verbose native lambda_rlm plan logging inside the isolated env.",
    )
    parser.add_argument(
        "--baseline-lambda-temperature",
        type=float,
        default=None,
        help="Optional temperature override passed to lambda_rlm's OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-top-p",
        type=float,
        default=None,
        help="Optional top-p override passed to lambda_rlm's OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override passed to lambda_rlm's OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-stream",
        dest="baseline_lambda_stream",
        action="store_true",
        default=False,
        help="Enable streaming responses inside the lambda_rlm OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-lambda-timeout-sec",
        type=int,
        default=None,
        help=(
            "Optional per-attempt timeout cap inside the lambda_rlm isolated env. "
            "Defaults to the effective baseline process timeout when unset."
        ),
    )
    parser.add_argument(
        "--baseline-lambda-force-task-type",
        default="",
        help=(
            "Optional lambda-RLM task-type override forwarded into the isolated env "
            '(for example "qa") when small models misclassify repair prompts.'
        ),
    )
    parser.add_argument(
        "--baseline-lambda-prompt-contract",
        default="",
        help=(
            "Optional lambda-RLM prompt contract override: 'canonical' or "
            "'repair_fair_v1'. Defaults to canonical when unset."
        ),
    )
    parser.add_argument(
        "--baseline-lambda-controller-guidance-max-chars",
        type=int,
        default=2500,
        help="Controller-guidance budget appended to lambda-RLM prompts (default: 2500).",
    )
    parser.add_argument(
        "--baseline-lambda-failure-feedback-max-chars",
        type=int,
        default=3500,
        help="Failure-feedback budget injected into lambda-RLM prompts (default: 3500).",
    )
    parser.add_argument(
        "--baseline-lambda-failure-feedback-meta-max-chars",
        type=int,
        default=2000,
        help="Failure-feedback truncation budget stored in lambda-RLM attempt metadata (default: 2000).",
    )
    parser.add_argument(
        "--baseline-rlm-max-depth",
        type=int,
        default=DEFAULT_BASELINE_RLM_MAX_DEPTH,
        help=(
            "Maximum recursive depth for the standalone rlm baseline "
            f"(default: {DEFAULT_BASELINE_RLM_MAX_DEPTH})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-max-iterations",
        type=int,
        default=DEFAULT_BASELINE_RLM_MAX_ITERATIONS,
        help=(
            "Maximum native RLM iterations per outer attempt "
            f"(default: {DEFAULT_BASELINE_RLM_MAX_ITERATIONS})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-timeout-sec",
        type=int,
        default=None,
        help=(
            "Optional per-attempt timeout cap inside the standalone rlm env. "
            "Defaults to the effective baseline process timeout when unset."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-controller-guidance-max-chars",
        type=int,
        default=2500,
        help="Controller-guidance budget appended to RLM root prompts (default: 2500).",
    )
    parser.add_argument(
        "--baseline-rlm-prompt-contract",
        default="",
        help=(
            "Optional RLM root prompt contract override: 'canonical' or "
            "'repair_fair_v1'. Defaults to canonical when unset."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-failure-feedback-max-chars",
        type=int,
        default=3500,
        help="Failure-feedback budget injected into RLM root prompts (default: 3500).",
    )
    parser.add_argument(
        "--baseline-rlm-failure-feedback-meta-max-chars",
        type=int,
        default=2000,
        help="Failure-feedback truncation budget stored in RLM attempt metadata (default: 2000).",
    )
    parser.add_argument(
        "--baseline-rlm-max-errors",
        type=int,
        default=DEFAULT_BASELINE_RLM_MAX_ERRORS,
        help=(
            "Maximum consecutive REPL errors before aborting an rlm attempt "
            f"(default: {DEFAULT_BASELINE_RLM_MAX_ERRORS})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-max-tokens",
        type=int,
        default=None,
        help="Optional OpenAI-compatible max_tokens override passed to the standalone rlm backend.",
    )
    parser.add_argument(
        "--baseline-rlm-max-total-tokens",
        type=int,
        default=None,
        help="Optional total token ceiling enforced by the standalone rlm loop.",
    )
    parser.add_argument(
        "--baseline-rlm-compaction",
        dest="baseline_rlm_compaction",
        action="store_true",
        default=False,
        help="Enable native trajectory compaction inside the standalone rlm loop.",
    )
    parser.add_argument(
        "--baseline-rlm-compaction-threshold-pct",
        type=float,
        default=DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT,
        help=(
            "Compaction trigger threshold for the standalone rlm loop "
            f"(default: {DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT})."
        ),
    )
    parser.add_argument(
        "--baseline-rlm-temperature",
        type=float,
        default=None,
        help="Optional temperature override passed to the standalone rlm OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-rlm-top-p",
        type=float,
        default=None,
        help="Optional top-p override passed to the standalone rlm OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--baseline-rlm-verbose",
        dest="baseline_rlm_verbose",
        action="store_true",
        default=False,
        help="Enable verbose native rlm logging inside the isolated env.",
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm",
        dest="adaptive_controller_lambda_rlm",
        action="store_true",
        default=False,
        help=(
            "Enable lambda-RLM as a controller-owned decomposition backend for adaptive "
            "hybrid baseline attempts (default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-adaptive-controller-lambda-rlm",
        dest="adaptive_controller_lambda_rlm",
        action="store_false",
        help="Disable the lambda-RLM controller backend.",
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-repo-dir",
        default="",
        help=(
            "Path to the cloned lambda-RLM repository. Defaults to "
            "baselines/repos/lambda-RLM when --adaptive-controller-lambda-rlm is enabled."
        ),
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-python",
        default="",
        help=(
            "Optional Python executable for the lambda-RLM env. Defaults to "
            "baselines/envs/lambda_rlm/bin/python when available, else the current interpreter."
        ),
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-context-window-chars",
        type=int,
        default=DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
        help=(
            "Context window budget in chars passed to lambda-RLM "
            f"(default: {DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS})."
        ),
    )
    parser.add_argument(
        "--adaptive-controller-lambda-rlm-timeout-sec",
        type=int,
        default=DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
        help=(
            "Per-attempt timeout in seconds for lambda-RLM controller calls "
            f"(default: {DEFAULT_LAMBDA_RLM_TIMEOUT_SEC})."
        ),
    )
    parser.add_argument(
        "--output-format", default="json_compact", choices=["json", "json_compact", "toon"]
    )


def _add_openai_arguments(parser: argparse.ArgumentParser) -> None:
    """OpenAI-compatible backend, transport and response controls."""
    parser.add_argument("--keep-workdirs", action="store_true", help="Keep copied case workdirs.")
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:12000/v1"),
    )
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", ""))
    parser.add_argument(
        "--openai-compat-profile",
        default=os.getenv("OPENAI_COMPAT_PROFILE", DEFAULT_REPO_OPENAI_COMPAT_PROFILE),
        choices=COMPAT_PROFILE_CHOICES,
        help=(
            "OpenAI-compatible request-shaping profile for the internal OpenAI patcher. "
            "Defaults to lmstudio_model_yaml for local llama.cpp / model-template "
            "servers. Use qwen_openai_raw for raw Qwen endpoints that need "
            "chat_template_kwargs.enable_thinking=false."
        ),
    )
    parser.add_argument(
        "--openai-allow-reasoning-fallback",
        dest="openai_allow_reasoning_fallback",
        action="store_true",
        default=True,
        help=(
            "Allow the internal OpenAI patcher to consume reasoning-only response fields "
            "when message.content is empty (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-openai-allow-reasoning-fallback",
        dest="openai_allow_reasoning_fallback",
        action="store_false",
        help="Disable reasoning-only response fallback for the internal OpenAI patcher.",
    )
    parser.add_argument(
        "--openai-enable-thinking",
        dest="openai_enable_thinking",
        action="store_true",
        default=True,
        help=(
            "Enable thinking/reasoning mode for the internal OpenAI patcher "
            "(e.g. Qwen enable_thinking=True). Also auto-enables "
            "--openai-allow-reasoning-fallback (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-openai-enable-thinking",
        dest="openai_enable_thinking",
        action="store_false",
        help="Disable thinking/reasoning mode for the internal OpenAI patcher.",
    )
    resolved_openai_api_key = resolve_openai_api_key_default()
    parser.set_defaults(
        _openai_api_key_default_source=resolved_openai_api_key.source,
        _openai_api_key_default_env_key=resolved_openai_api_key.env_key,
    )
    parser.add_argument("--openai-api-key", default=resolved_openai_api_key.value)
    parser.add_argument(
        "--openai-http2",
        dest="openai_http2",
        action="store_true",
        default=True,
        help="Enable HTTP/2 for OpenAI-compatible transport (default: enabled).",
    )
    parser.add_argument(
        "--no-openai-http2",
        dest="openai_http2",
        action="store_false",
        help="Disable HTTP/2 and force HTTP/1.1 for OpenAI-compatible transport.",
    )
    parser.add_argument("--openai-timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--openai-transport-max-retries",
        type=int,
        default=6,
        help=(
            "Max immediate retries for retryable OpenAI transport failures inside one request "
            "(default: 6)."
        ),
    )
    parser.add_argument(
        "--openai-transport-retry-backoff-base-sec",
        type=float,
        default=1.0,
        help="Base exponential backoff seconds for OpenAI transport retries (default: 1.0).",
    )
    parser.add_argument(
        "--openai-transport-retry-backoff-max-sec",
        type=float,
        default=8.0,
        help="Max backoff seconds for OpenAI transport retries (default: 8.0).",
    )
    parser.add_argument(
        "--openai-transport-recovery",
        dest="openai_transport_recovery",
        action="store_true",
        default=DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED,
        help=(
            "Wait for OpenAI-compatible backends to recover after terminal transport failures "
            "and replay the same high-level operation (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-openai-transport-recovery",
        dest="openai_transport_recovery",
        action="store_false",
        help="Disable runner-managed OpenAI transport recovery.",
    )
    parser.add_argument(
        "--openai-transport-recovery-max-wait-sec",
        type=float,
        default=DEFAULT_OPENAI_TRANSPORT_RECOVERY_MAX_WAIT_SEC,
        help="Max seconds to wait for OpenAI backend recovery after an outage (default: 60).",
    )
    parser.add_argument(
        "--openai-transport-recovery-poll-sec",
        type=float,
        default=DEFAULT_OPENAI_TRANSPORT_RECOVERY_POLL_SEC,
        help="Polling interval in seconds for OpenAI backend recovery probes (default: 2.0).",
    )
    parser.add_argument(
        "--openai-max-propose-sec",
        type=float,
        default=300.0,
        help=(
            "Maximum wall-clock budget per patch attempt for OpenAI propose calls "
            "(initial + recovery proposals), including per-call retries/fallback calls "
            "(0 disables cap)."
        ),
    )
    parser.add_argument("--openai-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--openai-plan-patch-max-tokens",
        type=int,
        default=768,
        help="Max tokens for native cot_repair planning calls before patch generation (default: 768).",
    )
    parser.add_argument(
        "--openai-context-window-tokens",
        type=int,
        default=0,
        help=(
            "Optional OpenAI-compatible context window for Stage-B prompt preflight capping "
            "(0 disables guard)."
        ),
    )
    parser.add_argument(
        "--openai-auto-context-window",
        action="store_true",
        default=False,
        help=(
            "Probe OpenAI-compatible model metadata for context window tokens before run start "
            "(manual --openai-context-window-tokens overrides)."
        ),
    )
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument(
        "--openai-response-mode",
        choices=["edits", "diff", "full_file"],
        default="edits",
    )
    parser.add_argument(
        "--openai-response-mode-autoswitch",
        choices=["off", "retry_to_diff"],
        default="off",
        help=(
            "OpenAI retry policy for internal syntax/sanitize retries: off|retry_to_diff "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--openai-force-patch-attempt",
        action="store_true",
        help="Force a minimal patch attempt when model output cannot be parsed into a diff.",
    )
    parser.add_argument(
        "--force-patch-attempt",
        dest="force_patch_attempt",
        action="store_true",
        default=False,
        help=(
            "Try a runner-side marker patch when the model's diff is missing or fails to apply "
            "(default: off). The attempt is still recorded as patch_apply_failed (ATT-07)."
        ),
    )
    parser.add_argument(
        "--no-force-patch-attempt",
        dest="force_patch_attempt",
        action="store_false",
        help="Disable runner-side forced patch attempts when no diff is proposed.",
    )
    parser.add_argument(
        "--no-files-context", action="store_true", help="Do not include full file text in prompt."
    )


def _add_context_and_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    """Context protocol, retry prompts, prompt budgets and compaction."""
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=200_000,
        help="Max characters for project file context in prompt (default: 200000).",
    )
    parser.add_argument(
        "--context-protocol",
        choices=["monolithic", "staged"],
        default="monolithic",
        help="Context protocol for prompting: monolithic|staged (default: monolithic).",
    )
    parser.add_argument(
        "--context-stage-a-max-requests",
        type=int,
        default=6,
        help="Max Stage-A context requests when --context-protocol=staged (default: 6).",
    )
    parser.add_argument(
        "--context-stage-a-max-chars",
        type=int,
        default=120_000,
        help="Max characters returned from Stage-A context retrieval (default: 120000).",
    )
    parser.add_argument(
        "--retry-prompt-mode",
        choices=["full", "delta", "auto"],
        default="auto",
        help="Retry prompt protocol: full|delta|auto (default: auto).",
    )
    parser.add_argument(
        "--retry-context-mode",
        choices=["fresh", "history"],
        default="history",
        help="Retry context mode: fresh|history (default: history).",
    )
    parser.add_argument(
        "--retry-history-max-chars",
        type=int,
        default=48_000,
        help="Max characters of retry history transcript to include when --retry-context-mode=history (default: 48000).",
    )
    parser.add_argument(
        "--prompt-retry-reflection",
        dest="prompt_retry_reflection",
        action="store_true",
        default=False,
        help=(
            "Enable retry-only reflection-on-failure guidance in Stage-B prompts "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-retry-reflection",
        dest="prompt_retry_reflection",
        action="store_false",
        help="Disable retry-only reflection guidance in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--prompt-adaptive-stage-b-cot",
        dest="prompt_adaptive_stage_b_cot",
        action="store_true",
        default=False,
        help=(
            "Enable CoT-style Stage-B reasoning scaffolding for adaptive conditions "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-adaptive-stage-b-cot",
        dest="prompt_adaptive_stage_b_cot",
        action="store_false",
        help="Disable CoT-style Stage-B scaffolding in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--prompt-repair-method-scaffold",
        dest="prompt_repair_method_scaffold",
        action="store_true",
        default=False,
        help=(
            "Enable teacher-style modification-method scaffolding in Stage-B prompts "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-repair-method-scaffold",
        dest="prompt_repair_method_scaffold",
        action="store_false",
        help="Disable repair-method scaffolding in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--prompt-retry-plan-then-patch",
        dest="prompt_retry_plan_then_patch",
        action="store_true",
        default=False,
        help=(
            "Enable retry-only plan-then-patch orchestration before the final Stage-B patch call "
            "(legacy native OpenAI path only; default: disabled)."
        ),
    )
    parser.add_argument(
        "--no-prompt-retry-plan-then-patch",
        dest="prompt_retry_plan_then_patch",
        action="store_false",
        help="Disable retry-only plan-then-patch orchestration in the legacy native OpenAI path.",
    )
    parser.add_argument(
        "--pytest-output-tail-lines",
        type=int,
        default=DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
        help=(
            "Persist at most the latest N lines in pytest stdout/stderr artifact files "
            f"(default: {DEFAULT_PYTEST_OUTPUT_TAIL_LINES}, 0 disables line truncation)."
        ),
    )
    parser.add_argument(
        "--snapshot-artifact-max-string-chars",
        type=int,
        default=DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
        help=(
            "Per-string cap for persisted snapshot.json artifacts "
            f"(default: {DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS}, 0 disables)."
        ),
    )
    parser.add_argument(
        "--prompt-budget-total-chars",
        type=int,
        default=180_000,
        help="Stage-B prompt total character budget (0 disables total budget cap).",
    )
    parser.add_argument(
        "--prompt-budget-retry-delta-chars",
        type=int,
        default=16_000,
        help="Per-section character cap for retry delta packet in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-stdout-chars",
        type=int,
        default=32_000,
        help="Per-section character cap for pytest stdout/stderr in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-snapshot-chars",
        type=int,
        default=32_000,
        help="Per-section character cap for snapshot summary in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-evidence-chars",
        type=int,
        default=64_000,
        help="Per-section character cap for Stage-A evidence blocks in Stage-B prompts.",
    )
    parser.add_argument(
        "--prompt-budget-retry-history-chars",
        type=int,
        default=12_000,
        help="Per-section character cap for retry-history transcript in Stage-B prompts.",
    )
    parser.add_argument(
        "--context-compaction-mode",
        choices=["off", "auto_overflow", "always"],
        default="auto_overflow",
        help=(
            "Delegated context compaction policy for Stage-B prompts: "
            "off|auto_overflow|always (default: auto_overflow)."
        ),
    )
    parser.add_argument(
        "--context-compaction-scope",
        default="snapshot,pytest,files",
        help=(
            "Comma-separated raw context sources eligible for delegated compaction "
            "(snapshot,pytest,files)."
        ),
    )
    parser.add_argument(
        "--context-compaction-max-input-chars",
        type=int,
        default=120_000,
        help="Max characters sent to delegated context compaction calls (default: 120000).",
    )
    parser.add_argument(
        "--context-compaction-target-chars",
        type=int,
        default=18_000,
        help="Target compacted context size in characters (default: 18000).",
    )
    parser.add_argument(
        "--context-compaction-max-calls",
        type=int,
        default=1,
        help="Max delegated context compaction calls per case/condition (default: 1, 0 disables).",
    )
    parser.add_argument(
        "--prompt-order-mode",
        choices=["current", "crash_first", "evidence_last"],
        default="current",
        help="Stable Stage-B section ordering mode (default: current).",
    )
    parser.add_argument(
        "--adaptive-reveal-mode",
        choices=["full", "traceback_first"],
        default="full",
        help="Adaptive Stage-B reveal policy for adaptive conditions (default: full).",
    )
    parser.add_argument(
        "--snapshot-frame-pruning-mode",
        choices=["off", "heuristic"],
        default="off",
        help="Snapshot-frame pruning mode applied before Stage-B/TOON assembly (default: off).",
    )
    parser.add_argument(
        "--adaptive-gate-policy",
        choices=["aggressive", "balanced", "conservative"],
        default="balanced",
        help="Gate policy for layered_adaptive_gated helper invocation.",
    )
