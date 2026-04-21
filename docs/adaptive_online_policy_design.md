# Adaptive Online Policy Design (P2.2 bootstrap)

> **Date:** Unknown | **Status:** Current

## Goal
Learn helper-invocation decisions online during `layered_adaptive_llm` runs, instead of relying on fixed heuristics.

`invoke_helper` is defined as: consult Stage A via `request_context`.

This is intentionally separate from whether Stage A materializes extra context or a snapshot
for Stage B. Result records expose:
- `adaptive_helper_invoked`: Stage A was actually consulted.
- `adaptive_helper_materialized_context`: Stage A yielded context and/or an approved snapshot for Stage B.

## Objective
Helper-policy reward is local to Stage A consultation:

`reward = success_hint + progress_hint + helper_materialization_bonus - token_penalty - latency_penalty - stagnation_penalty`

With:
- `success_hint = 0.25` on final success
- `progress_hint = 0.5 * eta_progress` when downstream progress is made
- `helper_materialization_bonus = 0.25 * eta_progress` when Stage A yields context/snapshot for Stage B
- `token_penalty = lambda_tokens * (helper_local_tokens_total / token_scale)`
- `latency_penalty = mu_latency * (helper_local_wall_sec / latency_scale)`
- `stagnation_penalty = 0.5 * kappa_stagnation * min(1, streak / 3)`

Default coefficients:
- `lambda_tokens = 0.08`
- `mu_latency = 0.05`
- `token_scale = 1000`
- `latency_scale = 10`
- `success_hint = 0.25`
- `progress_hint scale = 0.5`
- `helper materialization scale = 0.25`
- `stagnation scale = 0.5`

## Budget-policy extension (adaptive tooling only)
The online policy now has a companion **adaptive budget policy** for adaptive conditions
(`layered_adaptive_gated`, `layered_adaptive_llm`).

It supports two architectures:

- `tier_binary`
  - legacy behavior
  - action space:
    - `keep_budget`
    - `increase_budget`
  - shared defaults use the conservative ladder `1.0,1.1,1.2,1.35,1.5`
  - on that dense ladder, `increase_budget` follows coarse promotion targets:
    - `1.0 -> 1.2`
    - `1.2 -> 1.5`
    - `1.5 -> max tier`
  - if a target rung is absent in a custom ladder, the controller falls back to
    the next larger configured tier
- `hierarchical_v1`
  - a two-stage controller
  - gate action space:
    - `keep_budget`
    - `increase_budget`
  - family action space (only when the gate escalates):
    - `time`
    - `tokens`
    - `context`
  - exactly one family can advance on an attempt
  - learned families intentionally exclude `response_mode`

Budget increases are constrained by:
- max tier ceiling
- `max_ups_per_case` (default `1`)
- adaptive-only eligibility (no effect on `base_repair`/plain snapshot conditions)
- family feasibility masking in `hierarchical_v1`

Current `hierarchical_v1` family definitions:

- `time`
  - `propose_budget_multiplier`
  - `propose_budget_sec`
- `tokens`
  - `openai_max_tokens`
- `context`
  - `helper_max_calls`
  - `stage_a_max_requests`
  - `stage_a_max_chars`
  - `retry_history_max_chars`
  - `context_compaction_max_calls` via the same capped discrete-widening envelope

Budget-policy reward is local to the patch-proposal family for the attempt:

`reward = success_hint + progress_hint - token_penalty - latency_penalty - stagnation_penalty - rho_budget_exceeded*I(request_budget_exceeded)`

Default:
- `rho_budget_exceeded = 0.5`

Budget local accounting records the proposal-family usage/latency for the attempt, while
`keep_budget` is still treated as having zero additional budget-cost penalty in the reward update.

The budget policy uses the same runtime modes as helper policy:
- `off`
- `shadow`
- `on`

## Runtime modes
- `off`: no policy decisions, baseline behavior.
- `shadow`: policy produces decisions/logs, but applied action remains the feasible baseline action
  (normally `invoke_helper`, or `skip_helper` when Stage A is unavailable).
- `on`: policy decisions are applied and used for online updates.

## Current algorithm
- Contextual bandit with LinUCB + epsilon exploration.
- Action space:
  - `skip_helper`
  - `invoke_helper`
  - `frame_only`
  - `file_only`
  - `search_only`
  - `snapshot_lite`
  - `compact_then_patch`
  - `retry_helper_strict_json`
- Features are attempt-level signals (pregate score, stagnation, helper usage, etc.) from `run_eval`.
- Restricted Stage-A actions now use bounded mini-budgets instead of a single surviving request:
  - `frame_only`: one `frame` plus up to two `local_var` requests
  - `file_only`: one `file` request
  - `search_only`: up to two `search` requests
- Greedy helper selection can include additive score priors. Shared adaptive
  defaults down-weight `search_only` by `-0.50` and `frame_only` by `-0.25`
  without removing either action from exploration or feasibility masks.
- Helper arms are feasibility-masked before selection. Actions that cannot run because of missing
  snapshot state, missing `request_context`, disabled/unsupported compaction, or strict-JSON
  preconditions are excluded from the LinUCB choice set and logged as unavailable.
- Feasibility is now based on explicit patcher capabilities (`supports_request_context()`,
  `supports_context_compaction()`, `supports_strict_json_first()`), not on whether a patcher
  merely inherits the base no-op helper methods.
- `retry_helper_strict_json` is available whenever the active Stage-A backend exposes
  strict-JSON-capable request-context support. That includes baseline-backed adaptive runs,
  because `BaselinePatcher` now delegates Stage A through a shared OpenAI-compatible helper while
  keeping Stage B under the baseline adapter's control.
- Adaptive helper budgeting keeps the default single base helper call plus a single top-up, but
  the top-up now acts as a deterministic recovery chance on strong prior signals:
  invalid helper payload, `no_progress_v2`, `patch_verifier_rejected`,
  `syntax_preflight_failed`, `retry_gate_no_new_evidence`, or the existing
  `need_more_context` recovery case.
- Optional recency decay is supported via multiplicative discounting of `A`/`b` before each update.
- Feature scaling supports:
  - `v1`: fixed log-scaled caps
  - `none`: raw features
  - `percentile_v1`: runtime percentile-fitted caps with safe fallback to `v1` until enough samples exist

## Persistence and observability
- State JSON:
  - helper policy: `evals/artifacts/policies/<run_id>.json` by default
  - budget policy: `evals/artifacts/policies/<run_id>.adaptive_budget_policy.json` by default
  - cross-run reuse only happens when an explicit state path is provided
- Event log JSONL remains run-scoped by default:
  - helper policy: `evals/results/<run_id>.adaptive_policy.jsonl`
  - budget policy: `evals/results/<run_id>.adaptive_budget_policy.jsonl`
- Event schema versions:
  - helper policy events: `3`
  - budget policy events: `4`
- Every decision/outcome emits one event with:
  - action selected/applied
  - propensity
  - feature vector
  - available vs unavailable helper actions (helper policy)
  - feature-scaling caps/history sizes/fallback flag
  - `reward_scope`
  - local reward decomposition
  - local LLM-call / token / wall accounting
  - update status/reason
- Budget-policy events additionally log:
  - architecture
  - selected/applied/default multipliers
  - tier index before/after
  - selected/applied family
  - available vs unavailable families
  - family propensity / family decision strategy
  - request-budget-exceeded penalty contribution

## Offline training handoff
- `python -m evals.export_policy_dataset` flattens event JSONL into CSV with:
  - metadata (`run_id`, `case_id`, `attempt`, etc.)
  - behavior policy fields (`selected_action`, `applied_action`, `propensity`)
  - local reward targets and components
  - local accounting columns (`local_*`)
  - feature columns (`feature_*`)
  - importance weights (`1 / propensity_clipped`)
- The export path accepts both current helper-policy and budget-policy event schema versions.
- Budget-policy exports now include architecture/family telemetry (`architecture`,
  `selected_family`, `applied_family`, `family_propensity`, `family_score_*`,
  `family_feature_*`, and gate/family update flags).
- This dataset is intended for future LR/MLP policy training and off-policy evaluation.

## Safety notes
- `shadow` mode is the recommended first rollout for calibration.
- Hard constraints still apply before policy decisions (pregate skip, budget exhaustion,
  saturation, missing `request_context`).
