"""Golden eval scenarios: flag sets that together exercise the main harness paths.

``{stub}`` in an argument is replaced by the stub server's origin (``http://host:port``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    argv: tuple[str, ...]
    # Every n-th chat completion fails once with HTTP 503 (0 = never).
    transport_fault_every: int = 0


_NATIVE = "base_repair,cot_repair,layered_adaptive_gated,layered_adaptive_llm"
_ADAPTIVE = "layered_adaptive_gated,layered_adaptive_llm"

# Latency enters the adaptive rewards; zero it so rewards depend only on tokens and outcomes.
_NO_LATENCY_REWARD = ("--adaptive-reward-mu-latency", "0")
_LEARN_FAST = (
    "--adaptive-online-warmup-decisions",
    "2",
    "--adaptive-online-save-every",
    "1",
    "--adaptive-budget-online-warmup-decisions",
    "2",
    "--adaptive-budget-online-save-every",
    "1",
)

SCENARIOS: dict[str, Scenario] = {
    # Defaults: every native condition, online helper policy off, budget policy on.
    "native_default": Scenario(
        argv=("--conditions", _NATIVE, "--k", "3", *_NO_LATENCY_REWARD),
    ),
    # The template configuration: LinUCB helper and budget policies learning online,
    # reflection and plan-then-patch prompts, response-mode autoswitch, forced patches.
    "adaptive_linucb": Scenario(
        argv=(
            "--conditions",
            _ADAPTIVE,
            "--k",
            "3",
            *_NO_LATENCY_REWARD,
            *_LEARN_FAST,
            "--adaptive-online-policy-mode",
            "on",
            "--adaptive-budget-policy-mode",
            "on",
            "--prompt-retry-reflection",
            "--prompt-retry-plan-then-patch",
            "--openai-response-mode-autoswitch",
            "retry_to_diff",
            "--force-patch-attempt",
            "--patch-verifier-scope",
            "all",
            "--adaptive-toon-encoding",
            "--context-protocol",
            "staged",
            "--retry-prompt-mode",
            "full",
            "--adaptive-tep",
        ),
    ),
    # Thompson sampling, the BwK budget architecture, full TEP with equilibrium re-plans,
    # traceback-first reveal, frame pruning and the alternative prompt orderings.
    "adaptive_thompson_tep": Scenario(
        argv=(
            "--conditions",
            "layered_adaptive_llm",
            "--k",
            "3",
            *_NO_LATENCY_REWARD,
            *_LEARN_FAST,
            "--adaptive-online-policy-mode",
            "on",
            "--adaptive-online-policy-algo",
            "thompson",
            "--adaptive-online-feature-scaling-profile",
            "percentile_v1",
            "--adaptive-budget-policy-algo",
            "thompson",
            "--adaptive-budget-policy-architecture",
            "bwk_v1",
            "--adaptive-budget-bwk-run-token-budget",
            "50000",
            "--adaptive-tep-mode",
            "full",
            "--adaptive-tep-equilibrium",
            "--adaptive-reveal-mode",
            "traceback_first",
            "--snapshot-frame-pruning-mode",
            "heuristic",
            "--prompt-order-mode",
            "crash_first",
            "--prompt-adaptive-stage-b-cot",
            "--prompt-repair-method-scaffold",
            "--adaptive-helper-budget-profile",
            "compact",
            "--context-compaction-mode",
            "always",
        ),
    ),
    # The System-1 decision model as a prior over a hierarchical budget policy.
    "adaptive_decision_model": Scenario(
        argv=(
            "--conditions",
            "layered_adaptive_llm",
            "--k",
            "2",
            *_NO_LATENCY_REWARD,
            *_LEARN_FAST,
            "--adaptive-online-policy-mode",
            "on",
            "--adaptive-online-feature-scaling-profile",
            "v1_interactions",
            "--adaptive-budget-policy-architecture",
            "hierarchical_v1",
            "--adaptive-decision-model-role",
            "prior",
            "--adaptive-decision-model-url",
            "{stub}",
            "--prompt-order-mode",
            "evidence_last",
        ),
    ),
    # Diff and full-file response modes, strict syntax preflight, shadow policies,
    # tight prompt budgets, full prompt/evidence recording and transport faults.
    "native_diff_faults": Scenario(
        argv=(
            "--conditions",
            "base_repair,cot_repair,layered_adaptive_llm",
            "--k",
            "3",
            *_NO_LATENCY_REWARD,
            "--openai-response-mode",
            "diff",
            "--no-syntax-preflight-fallthrough",
            "--syntax-preflight-max-cycles",
            "1",
            "--force-patch-attempt",
            "--retry-prompt-mode",
            "delta",
            "--retry-context-mode",
            "fresh",
            "--record-prompt",
            "full",
            "--record-evidence",
            "full",
            "--prompt-budget-total-chars",
            "6000",
            "--adaptive-online-policy-mode",
            "shadow",
            "--adaptive-budget-policy-mode",
            "shadow",
            "--adaptive-gate-policy",
            "aggressive",
            "--openai-transport-max-retries",
            "0",
            "--openai-transport-recovery",
            "--openai-transport-recovery-poll-sec",
            "0.01",
            "--progress-verbosity",
            "detailed",
        ),
        transport_fault_every=5,
    ),
    "native_full_file": Scenario(
        argv=(
            "--conditions",
            "base_repair,layered_adaptive_gated",
            "--k",
            "2",
            *_NO_LATENCY_REWARD,
            "--openai-response-mode",
            "full_file",
            "--adaptive-budget-policy-mode",
            "off",
            "--adaptive-gate-policy",
            "conservative",
            "--patch-verifier-scope",
            "disabled",
            "--no-patch-sanitize",
        ),
    ),
}
