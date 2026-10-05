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
- `success_hint = success_hint_scale` on final success
- `progress_hint = progress_hint_scale * eta_progress * progress_score` (score in [0, 1])
- `helper_materialization_bonus = materialization_scale * eta_progress` when Stage A yields
  context/snapshot for Stage B
- `token_penalty = lambda_tokens * (helper_local_tokens_total / token_scale)`
- `latency_penalty = mu_latency * (helper_local_wall_sec / latency_scale)`
- `stagnation_penalty = stagnation_scale * kappa_stagnation * min(1, streak / 3)`
- plus the delayed rescue bonus when a later success re-sends the helper's evidence (POL-11)

## Defaults

The tables are generated from the code (POL-25); the hand-written values that used to be
here had drifted (8 actions, hint scales 0.25/0.5/0.25/0.5, event schema 3).

<!-- BEGIN AUTO-GENERATED: POLICY_DEFAULTS -->
<!-- Rendered from the code by `python -m evals.doc_sync --write`; do not edit. -->

| Setting | Helper policy | Budget policy |
|---------|---------------|---------------|
| Algorithms | `linucb`, `thompson` | `linucb`, `thompson` |
| Modes | `off`, `on`, `shadow` | `off`, `on`, `shadow` |
| Actions | 10: `skip_helper`, `invoke_helper`, `frame_only`, `file_only`, `search_only`, `snapshot_lite`, `frame_locals_lite`, `snapshot_full_lite`, `compact_then_patch`, `retry_helper_strict_json` | `keep_budget`, `increase_budget`; families `time`, `tokens`, `context` |
| Architectures | - | `bwk_v1`, `hierarchical_v1`, `tier_binary` |
| `alpha` (CLI / Python API) | 0.6 / 0.8 | 0.4 / 0.6 |
| `epsilon` (CLI / Python API) | 0.1 / 0.15 | 0.05 / 0.08 |
| Warm-up decisions (CLI / Python API) | 50 / 50 | 20 / 20 |
| Recency decay | 1.0 (per arm when < 1) | 1.0 (per arm when < 1) |
| `l2` | 1.0 | 1.0 |
| Save every (updates) | 50 | 50 |
| Feature scaling | `v1` (of `none`, `percentile_v1`, `v1`, `v1_interactions`) | `v1` |
| Score biases | `frame_only=-0.05,search_only=-0.10`, each within ±0.1 | - |
| Budget ladder | - | `1.0,1.1,1.2,1.35,1.5`, max 1 up(s) per case |
| Event / state schema | 7 / 4 | 7 / 4 |

| Reward coefficient (CLI default) | Value |
|----------------------------------|-------|
| `lambda_tokens` (token penalty weight) | 0.08 |
| `token_scale` | 1000.0 |
| `mu_latency` (latency penalty weight) | 0.05 |
| `latency_scale` (seconds) | 10.0 |
| `eta_progress` | 0.15 |
| `kappa_stagnation` | 0.1 |
| Success hint scale | 1.0 |
| Progress hint scale | 1.0 |
| Stagnation penalty scale | 1.0 |
| Helper materialization scale | 0.5 |
| Budget `rho_budget_exceeded` | 0.5 |

Runs take the CLI defaults; the Python API defaults apply only to configs built in code. Percentile caps freeze after 30 observations per feature. Run metadata records `policy_defaults_version: 2`.
<!-- END AUTO-GENERATED: POLICY_DEFAULTS -->

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

The discrete limits (`helper_max_calls`, `stage_a_max_requests`, `context_compaction_max_calls`)
widen by `floor(2 · steps_above_base / ladder_span)`, at least 1 and at most 2 for any level
above base (POL-10). The first step on the default five-rung ladder used to floor to 0, so with
`max_ups_per_case = 1` the `context` family changed only the char budgets.

Budget-policy reward is local to the patch-proposal family for the attempt:

`reward = success_hint + progress_hint - token_penalty - latency_penalty - stagnation_penalty - rho_budget_exceeded*I(request_budget_exceeded)`

Default:
- `rho_budget_exceeded = 0.5`

Budget local accounting records the proposal-family usage/latency for the attempt. The
proposal is paid under either arm, so the token and latency penalties charge only what the
escalation added (POL-08): `keep_budget` pays nothing extra, and `increase_budget` pays
`1 - default_multiplier / applied_multiplier` of the attempt's tokens and time (cost treated as
proportional to the budget multiplier). It used to be charged the whole attempt, about 0.40 of
a 1.0 success reward at `lambda_tokens = 0.08` and 5k tokens. Events still log the attempt's
full `local_tokens_*`; `reward_token_penalty` is the charged part.

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
- Contextual bandit with LinUCB + epsilon exploration (`--adaptive-online-policy-algo thompson`
  switches to Thompson sampling over the same A/b state; its greedy picks are logged as
  `thompson_greedy`, and its logged propensity uses the epsilon-greedy formula, so it is only
  approximate for off-policy evaluation).
- Action space (10 actions):
  - `skip_helper`
  - `invoke_helper`
  - `frame_only`
  - `file_only`
  - `search_only`
  - `snapshot_lite`
  - `frame_locals_lite`
  - `snapshot_full_lite`
  - `compact_then_patch`
  - `retry_helper_strict_json`
- Features are attempt-level signals (pregate score, stagnation, helper usage, etc.) from `run_eval`.
- Restricted Stage-A actions now use bounded mini-budgets instead of a single surviving request:
  - `frame_only`: one `frame` plus up to two `local_var` requests
  - `file_only`: one `file` request
  - `search_only`: up to two `search` requests
- Greedy helper selection can include additive score priors. The defaults down-weight
  `search_only` by `-0.10` and `frame_only` by `-0.05` without removing either action from
  exploration or feasibility masks. They come from one constant
  (`DEFAULT_ONLINE_ACTION_SCORE_BIAS_OVERRIDES`), and every bias must stay within ±0.1 of the
  success reward of 1.0 (decision D6, POL-07); the CLI default used to be `-1.00`/`-0.50`,
  which acted as a veto.
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
- Scores use each arm's cached `A^-1`, updated by Sherman-Morrison after every rank-1 update
  and recomputed exactly every 64 updates; replacing a matrix (decay, feature expansion, arm
  reset, load) drops its cache entry (POL-20).
- Optional recency decay (`decay < 1`; the default is 1.0) discounts the updated arm's `A`/`b` toward the prior before its update. Decaying every arm on every update left about 20 observations of evidence in total and is gone (POL-01). Outcome events carry `effective_evidence` (`trace(A_k) - d·λ` per arm).
- Feature scaling supports:
  - `v1`: fixed log-scaled caps
  - `none`: raw features
  - `percentile_v1`: each count-like feature's cap is fitted once from its first 30
    observations and then frozen (POL-09); until then it uses the `v1` cap
  - `v1_interactions`: `v1` plus three `ix_*` interaction features built from `v1`-scaled
    factors (POL-19)

## Persistence and observability
- State JSON:
  - helper policy: `evals/artifacts/policies/<run_id>.json` by default
  - budget policy: `evals/artifacts/policies/<run_id>.adaptive_budget_policy.json` by default
  - cross-run reuse only happens when an explicit state path is provided
- Event log JSONL remains run-scoped by default:
  - helper policy: `evals/results/<run_id>.adaptive_policy.jsonl`
  - budget policy: `evals/results/<run_id>.adaptive_budget_policy.jsonl`
- Event schema versions:
  - helper policy events: `7` (v5 adds the nested `decision_model` block, v6 `effective_evidence`,
    v7 `state_resets`; the record classes live in `evals/adaptive_policy_records.py`)
  - budget policy events: `7` (v5 `effective_evidence`, v6 `state_resets`, v7 the `bwk_v1`
    pacing fields)
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

## Policy inputs and evidence semantics (2026-10 defaults)

Three changes make the helper policy's inputs and logs trustworthy; each has an opt-out so
historical runs can be reproduced, and all are recorded in the run configuration.

- **No oracle case labels.** `case_difficulty_ord`, `case_snapshot_dependency_ord` and
  `case_bug_source_imported` come from `case.json` and are not observable at deployment time.
  They are dropped from the helper feature vector by default (`drop_case_label_features`,
  applied in the attempt flow so the raw, scaled and logged copies agree);
  `--adaptive-online-include-case-labels` restores them. `bootstrap_policy_state` strips them
  from replayed logs as well unless `--keep-case-label-features` is given.
- **Lite actions replace the snapshot.** With `--adaptive-reveal-mode full` the full snapshot
  section is always in the Stage-B prompt, so `snapshot_lite`, `frame_locals_lite` and
  `snapshot_full_lite` used to add redundant text while still earning the materialization
  bonus. Under `--adaptive-lite-snapshot-mode replace` (default) their compact section replaces
  the full snapshot section; `supplement` restores the old behavior. Rows record
  `adaptive_lite_replaced_snapshot`. `traceback_first` keeps its own reveal logic.
- **The policy decides the snapshot (2026-10-04 review, ATT-06, D5).** Under
  `--adaptive-reveal-mode full` the full snapshot reaches Stage-B only when the executed action
  grants it: Stage-A asked for it (with a specific justification) under `invoke_helper` or
  `retry_helper_strict_json`, the action is `compact_then_patch`, or a lite action runs in
  `supplement` mode. Rows record `adaptive_snapshot_requested` (the Stage-A request),
  `stage_b_snapshot_in_prompt`, and `adaptive_snapshot_included` (now equal to the latter).
- **Independent controllers and seeds (POL-05, POL-23).** When a run includes both adaptive
  conditions, each gets its own budget-policy controller and files
  (`<stem>.<condition><suffix>`); the helper policy only serves `layered_adaptive_llm`. Each
  controller's seed is derived from `--seed` (`evals.seed_derivation.derive_seed`), and every
  decision draws from a stream keyed by (case, condition, attempt), so exploration does not
  depend on how `--jobs` workers interleave. The model update order still does, so use
  `--jobs 1` for reproducible learning curves.
- **Restricted actions stay restricted (ATT-03, ATT-04).** When the request filter (layer 3)
  empties a `frame_only`/`file_only`/`search_only` request list, or Stage-A returns an invalid
  payload, those actions get an empty Stage-A context (reason
  `restricted_action_no_valid_requests`) instead of the whole-file fallback bundle, and the
  whole-file seed bundle is only added for `invoke_helper` and `retry_helper_strict_json`.
  `attempt_metadata.seed_bundle_chars` and `evidence_chars_by_type` (snapshot, snapshot_lite,
  seed, file, search, frame, local_var) record what reached the Stage-B prompt.
- **Shadow propensities.** In shadow mode the executed baseline action is deterministic, so the
  logged behavior propensity is 1.0 (helper and budget policies). OPE corrects older shadow
  rows at read time.
- **Greedy ties (POL-24).** When several feasible actions share the top score (always the case
  at cold start), the greedy action is drawn uniformly among them with the policy RNG; ties used
  to go to `invoke_helper` (helper policy) or the default action (budget gate, which made the
  `time` family win budget-family ties). The logged propensity is
  `ε/K + (1−ε)/|ties|` for a tied action and `ε/K` otherwise.
- **Selected and applied counts (POL-24).** Persisted state keeps `action_counts` (the selected
  action) and `applied_action_counts` (the action that ran) for every controller; they differ in
  shadow mode. State schema 4 adds the applied counts; schema 3 states still load with applied
  counts at zero.

## Decision-model advice (System-1)

`evals/decision_model/` lets a System-1 decision model (a SystemOne-compatible endpoint such
as Cloudflare clef-flash: one forward pass, a probability for every allowed option) take part
in helper-action selection. It acts only at layer 2 (policy action selection): it runs after the
informativeness pregate and the budget/saturation/recovery constraints, only sees the feasible
actions, and never changes Stage-A request filtering.

Roles (`--adaptive-decision-model-role`, requires `layered_adaptive_llm`):

| Role | Effect | Policy mode |
|------|--------|-------------|
| `off` | not consulted (default) | any |
| `shadow` | queried and logged; selection unchanged | `shadow` or `on` |
| `policy` | the decision model selects: argmax (`--adaptive-decision-model-temperature 0`) or tempered sampling, mixed with epsilon-uniform; atomic profiles map diagnostic answers to an action in code; no warm-up | `on` |
| `features` | six diagnostic answers become LinUCB features (`dm_available`, `dm_traceback_sufficient`, `dm_need_runtime_values`, `dm_need_source_definition`, `dm_need_project_usage`, `dm_fix_confidence`) | `on` |
| `prior` | warm-up samples from (1−ρ)·p + ρ/K (`--adaptive-decision-model-warmup-mix`), then a centered log-probability prior with weight β·n₀/(n₀+updates) is added to the greedy ranking (logged scores stay LinUCB's) | `on` |

Every selection logs the full probability of the chosen action, so propensities stay valid for
off-policy evaluation. LinUCB keeps updating in `on` mode under every role. If the sidecar
fails, `policy` falls back to the baseline action (`decision_model_unavailable`), `features`
sets `dm_available=0`, and `prior` falls back to the plain bandit.

How the decision model is asked (versioned profiles in `evals/decision_model/schema.py`):

- `evidence_v1` (default): a `next_evidence_action` choice over the feasible actions, with
  option descriptions as `{choose_when, not_for, provides, cost}` written for the active reveal
  mode, plus three joint diagnostic questions (`traceback_sufficient`, `missing_information`,
  `fix_confidence`).
- `evidence_atomic_v1`: only the diagnostic questions; `atomic_action_from_answers` maps them to
  an action in code (sufficient ≥ 0.7 → `skip_helper`; otherwise the most likely missing
  information routes to `frame_only` / `file_only` / `search_only`; uncertain → `invoke_helper`).
- Ablations: `evidence_v1_noaux`, `evidence_v1_text`, `evidence_v1_nohints`.

What it sees (`evals/decision_model/state.py`): a bounded state (about 6000 characters) with
numeric-prefixed sections in priority order — failure, crash site, assertion lines, call path,
crash-frame locals (only under full reveal), hints (pregate tier, top llmdebug hypotheses),
history (attempt ≥ 2), budget, and the test-output tail last because Clef-style models cut
over-long states from the end. It never receives `CaseSpec`; workdir paths (host and
container), case id, condition and run id are redacted, and home/interpreter prefixes are
normalized so state hashes are portable.

Logging: result rows carry `adaptive_online_dm_*` fields; policy events (schema v5) carry the
`decision_model` block (answers, applied effect, latency, state hash); with
`--adaptive-decision-model-log-state full` (default) the rendered states go to
`<artifacts_root>/decision_model_states.jsonl` keyed by `decision_id`. The sidecar is checked
before the first unit runs; an unhealthy sidecar stops the run with a clear error.

## Offline training handoff
- `python -m evals.export_policy_dataset` flattens event JSONL into CSV with:
  - metadata (`run_id`, `case_id`, `attempt`, etc.)
  - behavior policy fields (`selected_action`, `applied_action`, `propensity`)
  - local reward targets and components
  - local accounting columns (`local_*`)
  - feature columns (`feature_*`)
  - importance weights (`1 / propensity_clipped`)
- The export path accepts helper-policy and budget-policy event schema versions 4 to 7
  (explicit sets, `SUPPORTED_*_EVENT_SCHEMA_VERSIONS`); fields added after v4 are optional.
  Earlier versions predate the published repository and are counted as
  `records_unsupported_schema`.
- Export and bootstrap replay share one loader (`evals/policy_event_log.py`, POL-18): events
  are ordered by `(created_at_utc, run_id, log, line)`, the first occurrence of a duplicate
  wins, and events with a non-finite reward, propensity, score or feature value are rejected
  and counted as `records_non_finite` (POL-13). Export used to sort by case id and keep the
  last duplicate; replay followed glob order, and `inf` values reached the CSV.
- Decision-model columns (v5 events): `dm_role`, `dm_status`, `dm_profile`, `dm_model`,
  `dm_top_action`, `dm_top_prob`, `dm_margin`, `dm_entropy`, `dm_atomic_action`,
  `dm_traceback_sufficient`, `dm_fix_confidence`, `dm_latency_ms`, `dm_input_tokens`,
  `dm_state_sha256`, `dm_prior_beta`, plus `dm_p_<action>` and `dm_missing_<option>`
  expansions. Events without a decision-model block leave these cells empty, so "no advice" is
  distinguishable from a probability of 0.
- Budget-policy exports now include architecture/family telemetry (`architecture`,
  `selected_family`, `applied_family`, `family_propensity`, `family_score_*`,
  `family_feature_*`, and gate/family update flags).
- This dataset is intended for future LR/MLP policy training and off-policy evaluation.
- `analyze_results` OPE adds three per-sample targets whenever rows carry decision-model advice:
  `decision_model_greedy` (argmax of its action probabilities), `decision_model_stochastic`
  (its distribution over the feasible actions) and `decision_model_atomic` (the code-mapped
  action of `evidence_atomic_v1`; for other profiles the same thresholds are applied to the
  logged `traceback_sufficient`/`missing_information` answers, an approximation because Clef
  answers a profile's questions jointly). Missing or failed advice puts all target mass on the
  baseline action. Stochastic targets use IPS/SNIPS/ESS with w = π(a|x)/μ and a doubly robust
  estimate Σₐ π(a|x)·q̂ₐ(x) + w·(r − q̂(x, a)) with the same ridge reward models;
  `shadow_corrected_samples` counts old shadow rows whose propensity was corrected to 1.0.
  Meaningful estimates need a stochastic behavior policy, i.e. LinUCB `on` with the decision
  model in `shadow` role.
- Each stochastic target therefore also reports how much of its probability mass the behavior
  policy can identify: `behavior_supported_target_mass`, `unsupported_target_mass`,
  `unsupported_mass_samples`, `deterministic_behavior_samples` and `estimates_identifiable`.
  Mass counts as supported when the sample took that action, or when the sample's behavior was
  stochastic (propensity < 1) and the action occurs somewhere in the data. When unsupported
  mass remains, `support_warning` is `unsupported_target_mass`: `matched_samples` /
  `support_rate` alone read as full support as soon as the executed action keeps any target
  probability, so a deterministic shadow run with 10% of the target mass on the baseline action
  used to report IPS = SNIPS = 0 with `support_rate` 1 and no warning at all. The point
  estimates are kept (they estimate the supported part of the target), but they must not be
  read as estimates of the full target policy.
- Bootstrap intervals follow the point estimates: with no importance-weighted sample,
  `ope_ci.ips` / `ope_ci.snips` are `null` instead of a seemingly certain `[0, 0]`, and
  `ope_ci.available` is true only when at least one interval exists.

## Safety notes
- `shadow` mode is the recommended first rollout for calibration.
- Hard constraints still apply before policy decisions (pregate skip, budget exhaustion,
  saturation, missing `request_context`).
