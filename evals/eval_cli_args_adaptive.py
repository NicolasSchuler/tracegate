"""Adaptive-condition, TEP, patch-check, determinism and recording flags of ``run_eval``."""

from __future__ import annotations

import argparse

from evals.adaptive_budget_online_policy import (
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_ARCHITECTURES,
)
from evals.eval_types import (
    ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
    ADAPTIVE_FEATURE_SCALING_PROFILES,
    ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
    ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
    ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
    ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
    ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
    ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
    ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
    ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
    DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
    DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
    DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
    DEFAULT_ADAPTIVE_ONLINE_ALPHA,
    DEFAULT_ADAPTIVE_ONLINE_DECAY,
    DEFAULT_ADAPTIVE_ONLINE_EPSILON,
    DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS,
    DEFAULT_ADAPTIVE_POLICY_L2,
    DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
    DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
    PATCH_VERIFIER_MIN_SCORE,
    PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
    PATCH_VERIFIER_SCOPE_ALL,
    PATCH_VERIFIER_SCOPE_DISABLED,
)
from evals.policy_feature_scaling import DEFAULT_POLICY_FEATURE_SCALING_PROFILE


def add_adaptive_arguments(parser: argparse.ArgumentParser) -> None:
    """Adaptive-condition, TEP, patch-check, determinism and recording flags."""
    _add_adaptive_policy_arguments(parser)
    _add_tep_patch_and_recording_arguments(parser)


def _add_adaptive_policy_arguments(parser: argparse.ArgumentParser) -> None:
    """Adaptive gating, the helper and budget policies and their rewards."""
    parser.add_argument(
        "--adaptive-max-helper-calls",
        type=int,
        default=1,
        help="Max helper invocations per case/condition for adaptive modes (default: 1, 0 = unlimited).",
    )
    parser.add_argument(
        "--adaptive-budget-topup-max-calls",
        type=int,
        default=DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
        help=(
            "Extra one-time helper budget grants per case/condition when adaptive policies are "
            "stagnating (default: 1, 0 disables)."
        ),
    )
    parser.add_argument(
        "--adaptive-stagnation-window",
        type=int,
        default=1,
        help="Consecutive repeated-failure window for adaptive gate decisions.",
    )
    parser.add_argument(
        "--adaptive-saturation-threshold",
        type=float,
        default=0.75,
        help="Disable further helper calls when low-yield helper ratio reaches this threshold.",
    )
    parser.add_argument(
        "--adaptive-no-progress-token-cap",
        type=int,
        default=DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
        help=(
            "Stop adaptive loops when repeated no-progress attempts accumulate at least this many "
            "tokens (default: 150000, 0 disables)."
        ),
    )
    parser.add_argument(
        "--adaptive-helper-budget-profile",
        choices=["default", "compact"],
        default="default",
        help="Prompt-budget profile applied when adaptive helper is invoked.",
    )
    parser.add_argument(
        "--adaptive-openai-helper-response-mode",
        choices=["inherit", "edits", "diff", "full_file"],
        default="inherit",
        help="OpenAI response-mode override for adaptive helper-invoked attempts (default: inherit).",
    )
    parser.add_argument(
        "--adaptive-informativeness-pregate",
        action="store_true",
        default=False,
        help="Enable rule-based traceback informativeness pre-gate for adaptive conditions.",
    )
    parser.add_argument(
        "--adaptive-informativeness-pregate-threshold",
        type=float,
        default=0.7,
        help="Score threshold to skip Stage-A (0.0-1.0, default: 0.7).",
    )
    parser.add_argument(
        "--adaptive-online-policy-mode",
        choices=["off", "shadow", "on"],
        default="off",
        help="Online adaptive policy mode for layered_adaptive_llm (default: off).",
    )
    parser.add_argument(
        "--adaptive-online-policy-algo",
        choices=["linucb", "thompson"],
        default="linucb",
        help="Online adaptive policy algorithm (default: linucb).",
    )
    parser.add_argument(
        "--adaptive-online-alpha",
        type=float,
        default=DEFAULT_ADAPTIVE_ONLINE_ALPHA,
        help=(
            "LinUCB exploration coefficient alpha for the helper policy "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_ALPHA:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-epsilon",
        type=float,
        default=DEFAULT_ADAPTIVE_ONLINE_EPSILON,
        help=(
            "Epsilon-greedy exploration probability for the helper policy "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_EPSILON:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-l2",
        type=float,
        default=DEFAULT_ADAPTIVE_POLICY_L2,
        help=f"L2 regularization for LinUCB matrices (default: {DEFAULT_ADAPTIVE_POLICY_L2}).",
    )
    parser.add_argument(
        "--adaptive-online-warmup-decisions",
        type=int,
        default=DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS,
        help=(
            "Uniform warmup decisions before LinUCB greedy/explore policy "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-save-every",
        type=int,
        default=DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
        help=(
            "Persist online policy state every N updates "
            f"(default: {DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-decay",
        type=float,
        default=DEFAULT_ADAPTIVE_ONLINE_DECAY,
        help=(
            "Optional LinUCB recency decay in (0,1], applied to the updated arm only; "
            f"1.0 disables forgetting (default: {DEFAULT_ADAPTIVE_ONLINE_DECAY:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-feature-scaling-profile",
        choices=sorted(ADAPTIVE_FEATURE_SCALING_PROFILES),
        default=DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        help="Feature scaling profile for adaptive online policy (default: v1).",
    )
    parser.add_argument(
        "--adaptive-online-action-score-biases",
        default=DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
        help=(
            "Comma-separated additive helper-action score priors as action=bias pairs, "
            "each within +/-0.1 of the success reward "
            f"(default: {DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES})."
        ),
    )
    parser.add_argument(
        "--adaptive-online-state-path",
        default="",
        help="Optional path for adaptive online policy state JSON.",
    )
    parser.add_argument(
        "--adaptive-online-init-state-path",
        default="",
        help="Optional warm-start path to load adaptive online policy state without reusing save path.",
    )
    parser.add_argument(
        "--adaptive-online-events-path",
        default="",
        help="Optional path for adaptive online policy JSONL event log.",
    )
    parser.add_argument(
        "--adaptive-budget-policy-mode",
        choices=["off", "shadow", "on"],
        default="on",
        help="Online adaptive budget policy mode for adaptive conditions (default: on).",
    )
    parser.add_argument(
        "--no-adaptive-budget-policy",
        dest="adaptive_budget_policy_mode",
        action="store_const",
        const="off",
        help="Disable adaptive budget policy.",
    )
    parser.add_argument(
        "--adaptive-budget-policy-algo",
        choices=["linucb", "thompson"],
        default="linucb",
        help="Online adaptive budget policy algorithm (default: linucb).",
    )
    parser.add_argument(
        "--adaptive-budget-policy-architecture",
        choices=sorted(BUDGET_POLICY_ARCHITECTURES),
        default=BUDGET_POLICY_ARCH_TIER_BINARY,
        help=(
            "Adaptive budget policy architecture: legacy bundled tiers or hierarchical "
            "family escalation (default: tier_binary)."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-policy-tiers",
        default=DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
        help=(
            "Comma-separated budget multipliers for attempt propose budget tiers "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS}). Must include 1.0."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-policy-max-ups-per-case",
        type=int,
        default=DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
        help=(
            "Max applied budget increases per case/condition "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE}, 0 disables)."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-alpha",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA,
        help=(
            "LinUCB exploration coefficient alpha for adaptive budget policy "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-epsilon",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON,
        help=(
            "Epsilon-greedy exploration probability for budget policy "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-l2",
        type=float,
        default=DEFAULT_ADAPTIVE_POLICY_L2,
        help=(
            "L2 regularization for adaptive budget LinUCB matrices "
            f"(default: {DEFAULT_ADAPTIVE_POLICY_L2})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-warmup-decisions",
        type=int,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS,
        help=(
            "Uniform warmup decisions before budget policy exploitation "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-save-every",
        type=int,
        default=DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
        help=(
            "Persist adaptive budget policy state every N updates "
            f"(default: {DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-online-decay",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
        help=(
            "Optional LinUCB recency decay in (0,1] for the budget policy, applied to the "
            "updated arm only; 1.0 disables forgetting "
            f"(default: {DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY:g})."
        ),
    )
    parser.add_argument(
        "--adaptive-budget-feature-scaling-profile",
        choices=sorted(ADAPTIVE_FEATURE_SCALING_PROFILES),
        default=DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        help="Feature scaling profile for adaptive budget policy (default: v1).",
    )
    parser.add_argument(
        "--adaptive-budget-online-state-path",
        default="",
        help="Optional path for adaptive budget policy state JSON.",
    )
    parser.add_argument(
        "--adaptive-budget-init-state-path",
        default="",
        help="Optional warm-start path to load adaptive budget policy state without reusing save path.",
    )
    parser.add_argument(
        "--adaptive-budget-online-events-path",
        default="",
        help="Optional path for adaptive budget policy JSONL event log.",
    )
    parser.add_argument(
        "--adaptive-budget-bwk-run-token-budget",
        type=int,
        default=0,
        help="Explicit run-level token budget required by adaptive budget architecture bwk_v1.",
    )
    parser.add_argument(
        "--adaptive-budget-reward-rho-budget-exceeded",
        type=float,
        default=ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
        help="Reward penalty coefficient when attempt ends with request_budget_exceeded.",
    )
    parser.add_argument(
        "--adaptive-reward-lambda-tokens",
        type=float,
        default=ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
        help=(
            "Reward penalty coefficient for token usage: "
            "reward = success - lambda*(tokens/token_scale) - mu*(latency/latency_scale)."
        ),
    )
    parser.add_argument(
        "--adaptive-reward-mu-latency",
        type=float,
        default=ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
        help="Reward penalty coefficient for latency term (default: 0.05).",
    )
    parser.add_argument(
        "--adaptive-reward-token-scale",
        type=float,
        default=ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
        help="Token normalization scale in adaptive reward (default: 1000).",
    )
    parser.add_argument(
        "--adaptive-reward-latency-scale",
        type=float,
        default=ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
        help="Latency normalization scale in adaptive reward seconds (default: 10).",
    )
    parser.add_argument(
        "--adaptive-reward-eta-progress",
        type=float,
        default=ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
        help="Reward bonus coefficient for adaptive progress signal (default: 0.15).",
    )
    parser.add_argument(
        "--adaptive-reward-kappa-stagnation",
        type=float,
        default=ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
        help="Reward penalty coefficient for adaptive no-progress streak (default: 0.10).",
    )
    parser.add_argument(
        "--adaptive-reward-success-hint",
        type=float,
        default=ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
        help="Reward bonus for a successful attempt (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-reward-progress-scale",
        type=float,
        default=ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
        help="Scale multiplier for the progress-hint reward component (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-reward-stagnation-scale",
        type=float,
        default=ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
        help="Scale multiplier for the stagnation penalty component (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-reward-materialization-scale",
        type=float,
        default=ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
        help="Scale multiplier for the helper-context materialization bonus (default: 0.5).",
    )
    parser.add_argument(
        "--adaptive-toon-encoding",
        action="store_true",
        default=False,
        help=(
            "Use TOON encoding for LLM-bound snapshot/evidence payloads in adaptive conditions "
            "(layered_adaptive_gated, layered_adaptive_llm). Stage-B snapshot payloads stay "
            "within the legacy prompt snapshot surface and are only substituted when shorter "
            "than the standard snapshot section. Silently ignored for non-adaptive conditions. "
            "Requires 'pip install llmdebug[toon]' for full TOON; falls back to compact JSON "
            "when toons is unavailable."
        ),
    )
    parser.add_argument(
        "--adaptive-aux-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable auxiliary reward signals (pytest reduction, evidence novelty, compaction, informativeness). On by default; use --no-adaptive-aux-reward to disable.",
    )
    parser.add_argument(
        "--adaptive-aux-pytest-reduction-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
        help=f"Weight for pytest fail reduction auxiliary reward (default: {ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT}).",
    )
    parser.add_argument(
        "--adaptive-aux-evidence-novelty-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
        help=f"Weight for evidence novelty auxiliary reward (default: {ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT}).",
    )
    parser.add_argument(
        "--adaptive-aux-compaction-efficiency-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
        help=f"Weight for compaction efficiency auxiliary reward (default: {ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT}).",
    )
    parser.add_argument(
        "--adaptive-aux-informativeness-alignment-weight",
        type=float,
        default=ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
        help=f"Weight for informativeness alignment auxiliary reward (default: {ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT}).",
    )
    # TEP configuration.
    parser.add_argument(
        "--adaptive-tep",
        action="store_true",
        default=False,
        help="Enable TEP-inspired stagnation nudges in the retry loop.",
    )


def _add_tep_patch_and_recording_arguments(parser: argparse.ArgumentParser) -> None:
    """TEP, syntax preflight, patch checks, determinism and recording."""
    parser.add_argument(
        "--adaptive-tep-equilibrium",
        action="store_true",
        default=False,
        help="Enable TEP-inspired multi-iteration retry-plan stability checks (implies --adaptive-tep).",
    )
    parser.add_argument(
        "--adaptive-tep-equilibrium-max-iterations",
        type=int,
        default=2,
        help="Max free-phase iterations for TEP equilibrium check (default: 2).",
    )
    parser.add_argument(
        "--adaptive-tep-equilibrium-similarity-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for TEP plan convergence (default: 0.85).",
    )
    parser.add_argument(
        "--adaptive-tep-mode",
        choices=("off", "legacy", "full"),
        default="off",
        help=(
            "Select the adaptive TEP implementation. "
            "'off' disables TEP, 'legacy' keeps the current nudge/stability loop, "
            "and 'full' enables the native-graph prompt-state optimizer."
        ),
    )
    parser.add_argument(
        "--adaptive-tep-state-path",
        default="",
        help=(
            "Path to the persisted full-mode TEP prompt state. "
            "Defaults to evals/artifacts/policies/<run_id>.adaptive_tep_state.json."
        ),
    )
    parser.add_argument(
        "--adaptive-tep-init-state-path",
        default="",
        help="Optional initial prompt-state artifact used to warm-start full-mode TEP.",
    )
    parser.add_argument(
        "--adaptive-tep-beta-init",
        type=float,
        default=1.0,
        help="Initial nudge strength for full-mode TEP (default: 1.0).",
    )
    parser.add_argument(
        "--adaptive-tep-beta-decay",
        type=float,
        default=0.9,
        help="Decay applied to beta after each accepted full-mode TEP update cycle (default: 0.9).",
    )
    parser.add_argument(
        "--adaptive-tep-full-equilibrium-max-iters",
        type=int,
        default=20,
        help="Maximum free-phase iterations per full-mode TEP node (default: 20).",
    )
    parser.add_argument(
        "--adaptive-tep-full-equilibrium-window",
        type=int,
        default=3,
        help="Rolling score window used for full-mode TEP equilibrium checks (default: 3).",
    )
    parser.add_argument(
        "--adaptive-tep-full-equilibrium-variance-threshold",
        type=float,
        default=0.5,
        help="Variance threshold for full-mode TEP score stabilization (default: 0.5).",
    )
    parser.add_argument(
        "--adaptive-tep-validation-buffer-size",
        type=int,
        default=4,
        help="Rolling validation-buffer size for full-mode TEP prompt updates (default: 4).",
    )
    parser.add_argument(
        "--syntax-preflight-enabled",
        dest="syntax_preflight_enabled",
        action="store_true",
        default=True,
        help=(
            "Validate patched Python files with py_compile before test execution "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-syntax-preflight",
        dest="syntax_preflight_enabled",
        action="store_false",
        help="Disable py_compile syntax preflight before test execution.",
    )
    parser.add_argument(
        "--syntax-preflight-max-cycles",
        type=int,
        default=DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
        help=(
            "Maximum internal syntax-repair cycles per logical attempt "
            "(default: 2, does not consume additional retry attempts)."
        ),
    )
    parser.add_argument(
        "--syntax-preflight-fallthrough",
        dest="syntax_preflight_fallthrough",
        action="store_true",
        default=True,
        help=(
            "When syntax preflight exhausts all cycles, fall through to test execution "
            "instead of terminating the attempt. Normalizes comparison with baselines that "
            "self-validate internally (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-syntax-preflight-fallthrough",
        dest="syntax_preflight_fallthrough",
        action="store_false",
        help="Disable syntax preflight fallthrough (old behavior: exhaustion terminates attempt).",
    )
    parser.add_argument(
        "--patch-sanitize-enabled",
        dest="patch_sanitize_enabled",
        action="store_true",
        default=True,
        help=(
            "Reject malformed patch text patterns before patch apply/syntax preflight "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-patch-sanitize",
        dest="patch_sanitize_enabled",
        action="store_false",
        help="Disable patch text sanitation guard before syntax preflight.",
    )
    parser.add_argument(
        "--patch-verifier-scope",
        choices=[
            PATCH_VERIFIER_SCOPE_DISABLED,
            PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
            PATCH_VERIFIER_SCOPE_ALL,
        ],
        default=PATCH_VERIFIER_SCOPE_ALL,
        help=(
            "Patch verifier execution scope: disabled|adaptive_only|all (default: all, so every "
            "condition faces the same verifier; adaptive_only is an ablation)."
        ),
    )
    parser.add_argument(
        "--patch-verifier-min-score",
        type=int,
        default=PATCH_VERIFIER_MIN_SCORE,
        help=(
            "Minimum patch verifier relevance score required to pass "
            f"(default: {PATCH_VERIFIER_MIN_SCORE})."
        ),
    )
    parser.add_argument(
        "--patch-verifier-infer-test-source-scope",
        dest="patch_verifier_infer_test_source_scope",
        action="store_true",
        default=True,
        help=(
            "Allow patch verifier crash-scope matching via source imports inferred from "
            "failing/crash test files (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-patch-verifier-infer-test-source-scope",
        dest="patch_verifier_infer_test_source_scope",
        action="store_false",
        help="Disable patch verifier test-to-source crash-scope inference.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (requires --seed; fails fast if guarantees are missing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic seed used for patch generation and subprocess hash randomization.",
    )
    parser.add_argument(
        "--record-prompt",
        choices=["full", "hash", "none"],
        default="hash",
        help="How to persist prompt material into attempt metadata.",
    )
    parser.add_argument(
        "--record-evidence",
        choices=["full", "hash", "none"],
        default="hash",
        help="How to persist evidence material into attempt metadata.",
    )
    parser.add_argument(
        "--record-env",
        action="append",
        default=[],
        help="Environment variable key to include in run_metadata (repeatable).",
    )
    parser.add_argument(
        "--record-deps",
        choices=["none", "minimal", "runtime"],
        default="minimal",
        help="Dependency version capture mode for run_metadata.",
    )
