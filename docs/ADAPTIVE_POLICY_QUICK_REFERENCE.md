# Adaptive Online Learning Policy - Quick Reference

## 🎯 Core Concept

A **contextual-bandit (LinUCB)** controller that learns which helper-invocation **action** (out of 10) to take given the current **context** (50+ features), by observing **rewards** from past attempts.

## 📊 The Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                      DECISION-OUTCOME CYCLE                      │
└─────────────────────────────────────────────────────────────────┘

1. EXTRACT FEATURES
   • Attempt progress, helper budget, failure signals
   • Context request state, difficulty metadata
   • 50+ raw features from eval state
   ▼
2. SCALE FEATURES (profile "v1" / "percentile_v1" / "none")
   • Count features: log1p normalization
   • `percentile_v1`: fit runtime caps from recent history, else fall back to `v1`
   • Clip all to [-4, 4]
   ▼
3. DECIDE ACTION
   • Feasibility-mask unavailable helper arms before scoring
   • Warm-up phase (first 200): Uniform random over feasible actions
   • Then: LinUCB greedy (95%) + ε-greedy (5%)
   • Score = exploit + α × explore
   ▼
4. EXECUTE ATTEMPT
   • Use selected action to route helper
   • Track tokens, latency, success
   ▼
5. OBSERVE REWARD
   • Success hint + progress bonus
   - Token penalty - latency penalty
   - Stagnation penalty + delayed rescue bonus
   ▼
6. UPDATE MODEL (if mode="on")
   • A_k ← A_k + x·x^T
   • b_k ← b_k + reward·x
   • Every 50 updates: save to disk
```

## 🎮 10 Actions the Policy Can Choose

| Action | Meaning |
|--------|---------|
| `skip_helper` | Don't invoke helper at all |
| `invoke_helper` | Standard invocation (baseline) |
| `frame_only` | Stack frames only (no code/search) |
| `file_only` | Only requested files |
| `search_only` | Only search results |
| `snapshot_lite` | Compact frame index (locations only) |
| `frame_locals_lite` | Crash frame code plus up to 5 locals |
| `snapshot_full_lite` | Up to 3 frames with code and locals |
| `compact_then_patch` | Compress context first |
| `retry_helper_strict_json` | Stricter JSON validation |

The three `*_lite` actions call no model. Under `--adaptive-reveal-mode full` they replace the
full snapshot section in the patch prompt (`--adaptive-lite-snapshot-mode replace`, default);
`supplement` restores the pre-2026-10 behavior of adding them on top. Under full reveal the
full snapshot itself is only in the prompt when the executed action grants it (ATT-06, D5);
restricted actions never fall back to whole files (ATT-03, ATT-04). Check
`attempt_metadata.evidence_chars_by_type` for what each attempt actually received.

## 🧠 What Gets Learned

**Per-action matrices** A_k and b_k maintain:
- **A_k**: Accumulated feature covariance (outer products of contexts)
- **b_k**: Reward-weighted feature accumulation
- **θ_k = A_k^{-1} b_k**: Estimated reward coefficients for action k

**Scoring formula**:
```
score_k = θ_k^T · x  +  α × √(x^T · A_k^{-1} · x)
          └─────┬─────┘     └──────────┬──────────┘
          exploit           exploration bonus
```

## 📈 50+ Input Features

### Attempt Progress (4)
- `attempt_frac`: current / max attempts (`attempts_remaining_frac = 1 - attempt_frac` was
  dropped as redundant with `bias`, POL-06)
- `timed_out`: 1 if timeout exceeded
- (others)

### Helper Budget (8)
- `helper_calls_used`: absolute count
- `helper_budget_remaining_frac`: fraction left
- `helper_topup_used_frac`, `...remaining_frac`
- `helper_budget_exhausted`: binary
- `helper_saturation_guarded`: binary
- (others)

### Failure Signals (4)
- `failure_signature_repeat`: same test failure?
- `stagnation_streak`: attempts with no progress
- `adaptive_no_progress_streak`: no new evidence
- (others)

### Context Requests (10)
- `last_helper_need_more_context`: binary
- `last_helper_invalid_response`: binary
- `last_helper_declined`: binary
- `supports_context_request`: patcher capability
- `last_helper_action_*`: one flag per action except `skip_helper`, the reference level (POL-06)
- (others)

### Context Compaction (8)
- `context_compaction_enabled`: binary
- `supports_context_compaction`: binary
- `context_compaction_calls_used_frac`
- `context_compaction_calls_remaining_frac`
- `last_context_compaction_success`: binary
- `last_context_compaction_ratio`: 0-1 compression
- `last_context_compaction_trigger_*`: categorical
- (others)

### Case Metadata (opt-in, off by default)
- `case_difficulty_ord`: 0 (easy) / 0.5 / 1 (hard)
- `case_snapshot_dependency_ord`: 0 / 0.5 / 1
- `case_bug_source_imported`: binary

These come from `case.json` labels, which are not observable at deployment time, so they are
dropped unless `--adaptive-online-include-case-labels` is set.
- (others)

### Pre-gate Signals (3)
- `pregate_score`: 0-1 confidence
- `pregate_fired`: should skip?
- `pregate_type_specificity`: exception type precision

## 💰 Reward Formula

```
reward = success_hint (if success)
       + η_progress × progress_score × SCALE
       + materialization_bonus (if context used)
       - λ_tokens × (tokens_total / 1000)
       - μ_latency × (wall_sec / 10)
       - κ_stagnation × min(1, no_progress_streak/3) × SCALE
       + delayed_rescue_bonus (if resolved later)
```

**Default parameters** (eval_types.py):
- λ_tokens = 0.08
- μ_latency = 0.05
- η_progress = 0.15
- κ_stagnation = 0.10
- token_scale = 1000, latency_scale = 10

## 🔧 Configuration

```python
OnlinePolicyConfig(
    mode: "off" | "shadow" | "on"        # off=no policy, shadow=eval only, on=active
    algorithm: "linucb"                  # or "thompson"
    alpha: 0.6                           # Exploration bonus strength
    epsilon: 0.10                        # ε-greedy probability
    l2: 1.0                              # Regularization for the LinUCB solve
    warmup_decisions: 50                 # Uniform sampling for first N
    save_every: 50                       # Save state after N updates
    decay: 1.0                           # < 1 discounts the updated arm's older data (opt-in)
    feature_scaling_profile: "v1"        # "v1", "percentile_v1", or "none"
    state_path: Path | None              # Default: evals/artifacts/policies/<run_id>.json
    events_path: Path | None             # Log all decisions+outcomes as JSONL
    random_seed: int | None              # RNG seed
)
```

## 🚀 Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `off` | Always use default action | Baseline condition |
| `shadow` | Policy selects action but default is executed; rewards observed under default | Offline evaluation without risk |
| `on` | Policy selects and executes action; actual rewards observed | Active learning (production) |

## 📊 LinUCB Algorithm Overview

1. **Maintain** A_k (d×d matrix) and b_k (d vector) for each of K=10 actions
2. **On decision** with context x:
    - Compute θ̂_k = A_k^{-1} b_k for each action
    - Score each action: UCB_k = θ̂_k^T x + α √(x^T A_k^{-1} x)
    - Select argmax_k UCB_k (greedy; ties drawn uniformly) or random (ε-greedy) from the feasible action set
3. **On reward** r for action a with context x:
    - Optionally (decay < 1) decay A_a / b_a of the updated arm toward the L2 prior
    - A_a ← A_a + x x^T
    - b_a ← b_a + r·x
4. **Save state** every 50 updates to enable resuming

## 📝 Key Data Structures

```python
# Decision made by policy
PolicyDecision:
  - selected_action: action chosen by LinUCB
  - applied_action: action actually executed (selected if mode="on", else default)
  - propensity: P(selected_action) — for off-policy corrections
  - scores: dict of UCB scores for all 10 actions
  - configured_action_score_biases: additive score priors applied before greedy selection
  - feature_values: the scaled features used
  - available_actions / unavailable_action_reasons: feasibility mask telemetry
  - decision_id: UUID to link to outcome

# Outcome recorded after attempt
PolicyEvent:
  - paired with PolicyDecision (via decision_id)
  - reward: observed reward signal
  - behavior_action: what actually happened (for off-policy eval)
  - behavior_propensity: P(behavior_action) — for weighting
  - model_updated: whether A/b matrices were updated
  - [all reward components broken down]
```

## 🧩 Decision-Model Advice (System-1)

A SystemOne-compatible decision model (e.g. Cloudflare clef-flash in a GPU sidecar) can take
part in helper-action selection for `layered_adaptive_llm`. Details:
`docs/adaptive_online_policy_design.md` ("Decision-model advice").

| Flag | Default | Meaning |
|------|---------|---------|
| `--adaptive-decision-model-role` | `off` | `shadow` (log only), `policy` (it selects), `features` (6 diagnostic LinUCB features), `prior` (guided warm-up + fading score prior) |
| `--adaptive-decision-model-url` | `$TRACEGATE_DECISION_MODEL_URL` or `http://127.0.0.1:8090` | Endpoint base URL (sidecar, or `.../upstream/<model>` on llama-swap) |
| `--adaptive-decision-model-api-key` | `$TRACEGATE_DECISION_MODEL_API_KEY` | Bearer token for an API-key gateway such as llama-swap; redacted in lockfiles, rejected in config files |
| `--adaptive-decision-model-profile` | `evidence_v1` | Question profile (`evidence_atomic_v1` maps diagnostic answers to an action in code) |
| `--adaptive-decision-model-temperature` | `0` | `policy` role: 0 = argmax plus epsilon exploration, >0 = tempered sampling |
| `--adaptive-decision-model-prior-beta` / `-prior-anneal-updates` / `-warmup-mix` | `0.5` / `50` / `0.3` | `prior` role weights |
| `--adaptive-decision-model-log-state` | `full` | Write rendered states to `<artifacts>/decision_model_states.jsonl` |

`policy`, `features` and `prior` need `--adaptive-online-policy-mode on`; `shadow` works with
`shadow` or `on`. New decision strategies: `decision_model_greedy`, `decision_model_sample`,
`decision_model_atomic`, `decision_model_epsilon`, `decision_model_unavailable`,
`decision_model_guided_warmup`.

## 🎯 Budget Policy (Secondary)

A companion budget controller supports two architectures:

- `tier_binary`
  - classic 2-action LinUCB:
    - `keep_budget`
    - `increase_budget`
  - shared defaults use `1.0,1.1,1.2,1.35,1.5`
  - on that dense ladder, `increase_budget` advances via coarse targets:
    - `1.0 -> 1.2`
    - `1.2 -> 1.5`
    - `1.5 -> max tier`
- `hierarchical_v1`
  - gate controller:
    - `keep_budget`
    - `increase_budget`
  - family controller (only when escalating):
    - `time`
    - `tokens`
    - `context`
  - exactly one family step per attempt

Current `hierarchical_v1` family meanings:
- `time`: `propose_budget_multiplier`, `propose_budget_sec`
- `tokens`: `openai_max_tokens`
- `context`: `helper_max_calls`, `stage_a_max_requests`, `stage_a_max_chars`,
  `retry_history_max_chars`, `context_compaction_max_calls`

Not learned in `hierarchical_v1`:
- `response_mode`

**Configuration**:
```python
BudgetPolicyConfig(
    architecture: "tier_binary" | "hierarchical_v1"
    alpha: 0.4        # Lower than the helper policy (0.6)
    epsilon: 0.05     # Lower exploration for the 2-action controller
    warmup: 20        # Short uniform warm-up before budget learning
    decay: 1.0        # Optional per-arm recency decay, same semantics as helper policy
    max_ups_per_case: 1
    feature_scaling_profile: "v1" | "percentile_v1" | "none"
)
```

Shared helper defaults also apply additive score priors (one source,
`DEFAULT_ONLINE_ACTION_SCORE_BIAS_OVERRIDES`; each bias at most ±0.1 of the success reward,
POL-07 / D6):
- `search_only = -0.10`
- `frame_only = -0.05`

`hierarchical_v1` also exposes/logs:
- family availability masks
- per-family levels
- family propensities / family scores
- family-scaled feature vectors

## 📂 Key Files

| File | Purpose |
|------|---------|
| `adaptive_online_policy.py` | Core LinUCB controller |
| `adaptive_policy_records.py` | Decision/outcome records (`PolicyDecision`, `PolicyEvent`, event schema v5) |
| `decision_model/` | System-1 decision-model advice (config, schema, state, policy roles, client, runtime, probe) |
| `adaptive_budget_online_policy.py` | Budget escalation policy (secondary) |
| `policy_feature_scaling.py` | Feature normalization (log1p for counts, clipping) |
| `run_eval.py` | Orchestration: feature extraction, decision calling, reward calculation |
| `stage_a_contract.py` | JSON schema for helper requests (file/search/frame) |
| `analyze_results.py` | Off-policy evaluation (OPE) of target policies |

## 🔄 Integration in run_eval.py

```python
# Initialize before condition loop
online_policy_controller = AdaptiveOnlinePolicyController(config)

# Per attempt
decision = online_policy_controller.decide(
    feature_values=scaled_features,
    default_action=ACTION_INVOKE_HELPER
)
helper_action = decision.applied_action  # Use this to route helper

# After attempt, observe reward
online_policy_controller.record_outcome(
    decision=decision,
    reward=reward_value,
    success=success,
    # ... + many other parameters
)

# Before exiting
online_policy_controller.close()  # Flush state to disk
```

## 🎓 Learning Dynamics

- **Warm-up (decisions 1-200)**: Uniform random → explores all actions equally
- **Early phase**: ε-greedy with high uncertainty → balanced exploration/exploitation
- **Steady state**: Greedy arm selection (high confidence actions) with 5% random exploration
- **Uncertainty bonus**: √(x^T A^{-1} x) is **larger** when context x is new → encourages exploration of unfamiliar situations

## 💡 Key Insights

1. **Feature scaling is critical**: Prevents large features from dominating the regression; log1p handles skewed distributions (streaks, budgets)

2. **Reward decomposition helps interpretability**: Each component (success, progress, token cost, etc.) can be analyzed independently

3. **Deferred credit**: If helper materializes context but attempt fails, credit is deferred to later attempt that resolves case → encourages long-term thinking

4. **Off-policy evaluation**: Shadow mode enables safe experimentation; OPE in `analyze_results.py` estimates expected reward of alternative policies with analysis-only propensity trimming and reward diagnostics

5. **State persistence**: State defaults are run-scoped (`<run_id>.json`,
   `<run_id>.adaptive_budget_policy.json`); reuse only happens when you set an explicit state path

## 🛠️ Practical Tuning Guide

### Tuning Workflow

How to A/B test policy changes:

1. **Run the same cases with two different configs** using distinct run IDs:
   ```bash
   # Baseline
   python -m evals.run_eval \
     --run-id baseline-policy-001 \
     --conditions layered_adaptive_llm \
     --adaptive-online-policy-mode on \
     --adaptive-online-alpha 0.8 \
     --adaptive-reward-lambda-tokens 0.08

   # Experiment
   python -m evals.run_eval \
     --run-id experiment-policy-001 \
     --conditions layered_adaptive_llm \
     --adaptive-online-policy-mode on \
     --adaptive-online-alpha 1.2 \
     --adaptive-reward-lambda-tokens 0.04
   ```

   `--conditions layered_adaptive_llm` is required here. If you leave the default
   `base_repair` condition in place, the eval harness forces adaptive-only policy
   settings off and the run will not exercise the online policy.

2. **Compare pass rates, token usage, and helper call rates**:
   ```bash
   uv run python -m evals.analyze_results \
     evals/results/baseline-policy-001.jsonl \
     evals/results/experiment-policy-001.jsonl \
     --stats --out evals/results/comparison.json
   ```

3. **Use `just eval-two-pass-summary` for combined metrics** if running a two-pass (base + rescue) workflow:
   ```bash
   just eval-two-pass-summary baseline-policy-001 experiment-policy-001
   ```

### Hyperparameter Sensitivity

Which parameters matter most and suggested sweep ranges:

| Parameter | CLI Flag | Default | Sweep Range | Effect |
|-----------|----------|---------|-------------|--------|
| `alpha` | `--adaptive-online-alpha` | 0.8 | 0.3–2.0 | Higher = more exploration via UCB bonus |
| `epsilon` | `--adaptive-online-epsilon` | 0.05 | 0.01–0.15 | Higher = more random exploration |
| `lambda_tokens` | `--adaptive-reward-lambda-tokens` | 0.08 | 0.02–0.15 | Higher = heavier penalty for token usage |
| `mu_latency` | `--adaptive-reward-mu-latency` | 0.05 | 0.01–0.10 | Higher = heavier penalty for wall-clock time |
| `eta_progress` | `--adaptive-reward-eta-progress` | 0.15 | 0.05–0.30 | Higher = larger bonus for progress signals |
| `kappa_stagnation` | `--adaptive-reward-kappa-stagnation` | 0.10 | 0.05–0.20 | Higher = larger penalty for no-progress streaks |
| `pregate_threshold` | `--adaptive-informativeness-pregate-threshold` | 0.7 | 0.3–0.9 | Higher = skip helper more often |
| `warmup_decisions` | `--adaptive-online-warmup-decisions` | 50 | 25–500 | Uniform random sampling before LinUCB kicks in |
| `decay` | `--adaptive-online-decay` | 1.0 | 0.90–1.0 | Below 1, the updated arm forgets older data faster (opt-in, for non-stationary runs; POL-01) |

**Budget policy** has its own parameter set with different defaults (lower `alpha=0.4`, lower `epsilon=0.05`, shorter `warmup=20`). Use the `--adaptive-budget-*` family of flags to tune those independently.

### Common Tuning Scenarios

#### "Helper is called too often"

**Symptoms**: High token spend, many helper calls that don't improve pass rate.

→ Increase the pre-gate threshold to skip helper on self-explanatory tracebacks:
```bash
--adaptive-informativeness-pregate-threshold 0.85   # default 0.7
```
→ Increase the token cost penalty so the reward signal discourages expensive calls:
```bash
--adaptive-reward-lambda-tokens 0.12                # default 0.08
```
→ Keep the hard cap at one helper invocation per case (`0` removes the cap and means unlimited):
```bash
--adaptive-max-helper-calls 1
```

#### "Helper is called too rarely"

**Symptoms**: Low helper call rate, policy learns to always `skip_helper`, poor pass rate on hard cases.

→ Lower the pre-gate threshold so fewer tracebacks are considered "self-explanatory":
```bash
--adaptive-informativeness-pregate-threshold 0.4    # default 0.7
```
→ Reduce the token cost penalty so helper calls are less punished:
```bash
--adaptive-reward-lambda-tokens 0.03                # default 0.08
```
→ Increase exploration to let the policy try helper more often:
```bash
--adaptive-online-alpha 1.0                         # default 0.6
--adaptive-online-epsilon 0.15                      # default 0.10
```

#### "Token costs are too high"

**Symptoms**: High per-case token usage without proportional pass rate improvement.

→ Increase the token penalty weight:
```bash
--adaptive-reward-lambda-tokens 0.12                # default 0.08
```
→ Reduce the helper budget so less context is fetched:
```bash
--adaptive-max-helper-calls 1
```
→ Use budget policy tiers to control escalation:
```bash
--adaptive-budget-policy-tiers "1.0,1.25"           # default "1.0,1.1,1.2,1.35,1.5"
--adaptive-budget-policy-max-ups-per-case 0         # disable budget increases
```

#### "Policy makes bad decisions early"

**Symptoms**: Poor action selection in the first ~50 helper decisions (expected during warmup).

This is normal warmup behavior — the first `warmup_decisions` (default 50) use uniform random exploration. To adjust:

→ Increase exploration bonus for faster learning after warmup:
```bash
--adaptive-online-alpha 1.0                         # default 0.6
```
→ Shorten warmup if you have a smaller eval set:
```bash
--adaptive-online-warmup-decisions 25               # default 50
```
→ Warm-start from a prior run to skip cold-start entirely (see next section).

### Warm-Start from Prior Policy

Load a previous policy state to continue learning instead of starting from scratch:

```bash
python -m evals.run_eval \
  --run-id continued-run-002 \
  --conditions layered_adaptive_llm \
  --adaptive-online-policy-mode on \
  --adaptive-online-state-path evals/artifacts/policies/prior-run-001.json
```

This resumes the LinUCB A_k/b_k weight matrices from where the prior run left
off. If the saved state already contains at least `warmup_decisions` prior
decisions, the policy immediately resumes its normal greedy/epsilon behavior.
If not, it continues warmup for the remaining decisions.

For the budget policy, use the corresponding flag:
```bash
--adaptive-budget-online-state-path evals/artifacts/policies/prior-run-001.adaptive_budget_policy.json
```

**Caveat**: Feature dimensions must match between runs.  If you changed the feature set or scaling profile, start fresh.

### Reward Debugging

How to diagnose low or negative rewards:

1. **Locate the event log** for your run:
   ```
   evals/results/<run_id>.adaptive_policy.jsonl
   ```

2. **Inspect individual events**.  Each JSONL line contains:
   - `feature_values`: the scaled feature vector used for the decision
   - `selected_action` / `applied_action`: what the policy chose vs. what executed
   - `reward`: the final scalar reward
   - Reward components: `reward_success_hint`, `reward_progress_hint`,
     `reward_progress_score`, `reward_token_penalty`, `reward_latency_penalty`,
     `reward_stagnation_penalty`, `reward_helper_materialization_bonus`

3. **Interpret negative rewards**:
   - A failed attempt with even modest token usage produces a negative reward
     (`reward_success_hint=0.0` minus `reward_token_penalty` and
     `reward_latency_penalty`)
   - Large `reward_stagnation_penalty` means the `adaptive_no_progress_streak`
     is climbing and the policy is repeating attempts without new evidence
   - Missing `reward_helper_materialization_bonus` means the helper did not
     provide useful context

4. **Export for bulk analysis**:
   ```bash
   uv run python -m evals.export_policy_dataset \
     --events "evals/results/<run_id>.adaptive_policy.jsonl" \
     --output-csv evals/results/policy_analysis.csv
   ```
   The output CSV has one row per decision with all features, actions, and reward components denormalized for filtering and plotting.

### Feature Importance

Which feature categories tend to drive action selection the most:

| Category | Key Features | Why They Matter |
|----------|-------------|-----------------|
| **Budget signals** | `helper_budget_remaining_frac`, `helper_budget_exhausted` | Policy learns to skip helper when budget is nearly spent |
| **Attempt progress** | `attempt_frac` | Late attempts get more conservative; early attempts explore |
| **Failure history** | `stagnation_streak`, `adaptive_no_progress_streak`, `failure_signature_repeat` | Repeated failures signal that current strategy isn't working |
| **Pre-gate signals** | `pregate_score`, `pregate_fired`, `pregate_type_specificity` | High informativeness score → traceback is self-explanatory → skip helper |
| **Context state** | `last_helper_need_more_context`, `last_helper_declined` | If helper asked for more context, `compact_then_patch` or `file_only` may help next |
| **Compaction state** | `context_compaction_calls_remaining_frac`, `last_context_compaction_ratio` | Compaction headroom drives `compact_then_patch` selection |

**Tip**: To see which features the learned policy weights most heavily, inspect the θ̂_k vectors in the saved state file (`evals/artifacts/policies/<run_id>.json`).  The `b_k` arrays encode reward-weighted feature accumulation per action — features with large absolute b_k entries relative to their A_k diagonal are the strongest drivers.
