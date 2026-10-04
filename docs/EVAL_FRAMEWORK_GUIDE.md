# LLMDebug Evaluation Framework & Adaptive Online Learning

## 1. EVALUATION FRAMEWORK OVERVIEW

### High-Level Architecture
The eval harness is split across a small set of modules:
- `eval_main_setup.py` normalizes CLI/config state and resolves the run plan
- `eval_main_flow.py` owns run-level orchestration and resume handling
- `eval_attempt_flow.py` executes per-condition, per-case attempts
- `eval_policy_features.py` builds adaptive features and reward components
- `eval_policy_online.py` syncs adaptive policy metadata and finalizes updates
- `run_eval.py` remains the compatibility facade and CLI entry point

The harness covers 9 supported conditions:
- 4 native conditions: `base_repair`, `cot_repair`, `layered_adaptive_gated`,
  `layered_adaptive_llm`
- 5 external baselines: `ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, `rlm`

Common responsibilities:
- **Case discovery** from the `evals/cases/` directory
- **Patch proposal** via pluggable patcher backends (command, heuristic, null,
  openai, baseline)
- **Pytest verification** of patches
- **Result recording** to JSONL with full attempt-level metadata

Key entry point: `run_eval.py::main()` → `eval_main_flow.run_main_flow()` →
`eval_attempt_flow.run_one_condition_impl()` → per-case patch attempts

### Conditions

| Condition | Description |
|-----------|-------------|
| `base_repair` | Baseline: no context enrichment |
| `cot_repair` | Base repair + chain-of-thought reasoning strategy |
| **`layered_adaptive_gated`** | Gate policy (conservative/balanced/aggressive) controls helper invocation |
| **`layered_adaptive_llm`** | Online LinUCB contextual-bandit decides helper actions per attempt |

---

## 2. ADAPTIVE GATED CONDITION

### Gate Policy Decision
**Function**: `_adaptive_gate_should_invoke_helper()` in `eval_attempt_flow.py`

```python
def _adaptive_gate_should_invoke_helper(
    *,
    policy: str,  # "aggressive" | "balanced" | "conservative"
    attempt: int,
    timed_out: bool,
    signature_repeated: bool,  # failure signature is repeated
    stagnation_streak: int,
    stagnation_window: int,  # window for signature repeatedness
) -> tuple[bool, str]:
```

**Decision Rules**:
- **aggressive**: Always invoke helper
- **conservative**: Invoke only if signature_repeated AND stagnation_streak ≥ (window + 1)
- **balanced** (default): Invoke if attempt ≥ 2 AND signature_repeated AND stagnation_streak ≥ window

**State tracked**:
- Failure signature tracking (test names, exception types)
- Stagnation streak (consecutive attempts with no new evidence/progress)
- Evidence count per attempt

---

## 3. ADAPTIVE ONLINE LEARNING (Critical)

### Core Components

#### 3.1 Policy Configuration
**Class**: `OnlinePolicyConfig` (adaptive_online_policy.py:67-131)

```python
@dataclass(frozen=True)
class OnlinePolicyConfig:
    mode: str = "off"              # "off" | "shadow" | "on"
    algorithm: str = "linucb"      # LinUCB contextual bandit
    alpha: float = 0.6             # Exploration-exploitation tradeoff (UCB bonus)
    epsilon: float = 0.10          # Epsilon-greedy probability
    l2: float = 1.0                # L2 regularization for matrix inversion
    warmup_decisions: int = 50     # Uniform sampling phase
    save_every: int = 50           # Save state after N updates
    decay: float = 1.0             # < 1: per-arm recency weighting for non-stationary runs
    feature_scaling_profile: str = "v1"  # Feature normalization
    state_path: Path | None = None # Persist model to disk
    events_path: Path | None = None # Log decisions/outcomes as JSONL
    random_seed: int | None = None
```

**Modes**:
- **off**: Always use default action (base_repair behavior)
- **shadow**: Select action via policy but apply default_action (offline evaluation)
- **on**: Select and apply policy actions (active learning)

#### 3.2 What Actions Does the Policy Choose?

**Actions** (8 options in `ONLINE_POLICY_ACTIONS`):
1. `skip_helper` — Don't invoke helper at all
2. `invoke_helper` — Standard helper invocation
3. `frame_only` — Helper gets only stack frames (no code/search)
4. `file_only` — Helper gets only requested files
5. `search_only` — Helper gets only search results
6. `snapshot_lite` — Minimal snapshot context
7. `compact_then_patch` — Compact context before invoking
8. `retry_helper_strict_json` — Retry with stricter JSON validation

**Selection Mechanism**: 
- Warm-up phase (first `warmup_decisions`, default 50): Uniform random across the feasible actions (up to 10)
- Then: epsilon-greedy (5% random, 95% LinUCB greedy)
- LinUCB score = exploit + alpha × explore

#### 3.3 What Is Being Learned?

**State Machine**: `LinUCBState` (adaptive_online_policy.py:250-262)

```python
@dataclass
class LinUCBState:
    feature_names: list[str]           # Feature ordering (sorted)
    decisions: int                     # Total decisions made
    updates: int                       # Total model updates
    action_counts: dict[str, int]      # Count per action
    A: dict[str, np.ndarray]           # A matrices (d×d per action)
    b: dict[str, np.ndarray]           # b vectors (d× per action)
```

**What is learned**:
- **Per-action A matrix**: A_k = Σ_t x_t x_t^T (outer product of feature vectors)
- **Per-action b vector**: b_k = Σ_t reward_t × x_t (reward-weighted features)
- **Parameter estimate**: θ_k = A_k^{-1} b_k (regression coefficients for action k)

**Update Rule** (rank-1 update):
```
A_k ← A_k + x x^T        # Accumulate feature covariance
b_k ← b_k + reward × x   # Accumulate reward-weighted features
θ_k ← A_k^{-1} b_k       # Least-squares solution
```

**Scoring** (UCB arm selection):
```
score_k = θ_k^T x + alpha × sqrt(x^T A_k^{-1} x)
        = exploit + explore
```
The second term is the prediction confidence interval width.

#### 3.4 Features (Input Signals)

**Function**: `_adaptive_online_feature_vector()` in `eval_policy_features.py`

**50+ features grouped by category**:

**Attempt Progress**:
- `attempt_frac`: current attempt / max attempts
- `attempts_remaining_frac`: fraction of attempts left
- `timed_out`: 1 if attempt timeout exceeded

**Helper Budget**:
- `helper_calls_used`: absolute count
- `helper_budget_remaining_frac`: fraction of budget left
- `helper_topup_used_frac` / `helper_topup_remaining_frac`
- `helper_budget_exhausted`: binary flag
- `helper_saturation_guarded`: rate-limiting activated

**Failure Signals**:
- `failure_signature_repeat`: binary (test failed same way as before)
- `stagnation_streak`: count of attempts with no progress
- `adaptive_no_progress_streak`: count attempts with no new evidence

**Helper Context Requests**:
- `last_helper_need_more_context`: binary
- `last_helper_invalid_response`: binary
- `last_helper_declined`: binary
- `supports_context_request`: patcher capability
- `last_helper_action_executed`: one of 10 actions (encoded as binary flags)

**Context Compaction**:
- `context_compaction_enabled`: binary
- `supports_context_compaction`: binary
- `context_compaction_calls_used_frac`
- `last_context_compaction_success`: binary
- `last_context_compaction_ratio`: 0-1 compression ratio
- `last_context_compaction_trigger_*`: categorical (stage_a_requested, overflow, mode_always)

**Case Metadata** (opt-in via `--adaptive-online-include-case-labels`; off by default because `case.json` labels are not observable at deployment time):
- `case_difficulty_ord`: 0 (easy) / 0.5 (medium) / 1.0 (hard)
- `case_snapshot_dependency_ord`: 0 (none) / 0.5 (helpful) / 1.0 (required)
- `case_bug_source_imported`: 1 if from dataset, 0 if hand-crafted

**Pre-gate Diagnostics**:
- `pregate_score`: 0-1 pre-gate confidence score
- `pregate_fired`: binary (pre-gate suggested skip)
- `pregate_type_specificity`: 0-1 how specific the exception type is

**Feature Scaling** (`policy_feature_scaling.py`):
- Profile "v1" (default):
  - Count-like features (streaks, budget): log1p-normalized then clipped to [-4, 4]
  - Bounded 0-1 features: kept as-is
  - Other features: clipped to [-4, 4]
- Profile "none": pass-through

#### 3.5 Reward Signal

**Function**: `_adaptive_online_local_reward_components()` in `eval_policy_features.py`

**Reward Breakdown**:
```
reward = success_hint 
       + progress_hint 
       + helper_materialization_bonus
       - token_penalty 
       - latency_penalty 
       - stagnation_penalty
       + delayed_rescue_bonus (if resolved later)
```

**Components**:

| Component | Formula | Note |
|-----------|---------|------|
| success_hint | HINT_VALUE if success else 0 | Binary signal |
| progress_hint | η_progress × progress_score × SCALE | Continuous progress assessment |
| helper_bonus | η_progress × SCALE if context materialized | Reward for materializing context |
| token_penalty | λ_tokens × (tokens / token_scale) | Discourage token usage |
| latency_penalty | μ_latency × (wall_sec / latency_scale) | Discourage slow calls |
| stagnation_penalty | κ_stagnation × min(1, no_progress_streak/3) × SCALE | Discourage repeated failures |
| delayed_rescue_bonus | Deferred credit from later rescue | Exploit-later reward |

**Default Parameters** (eval_types.py):
- λ_tokens = 0.08 (token scaling)
- μ_latency = 0.05 (latency scaling)
- η_progress = 0.15 (progress weight)
- κ_stagnation = 0.10 (stagnation weight)
- token_scale = 1000 tokens
- latency_scale = 10 seconds

**Credit Assignment**:
- **Scope**: "helper_credit_v2" assigns credit based on which decision caused the outcome
- **Deferred**: If helper materialized context but attempt failed, credit is deferred to next attempt that resolves the case
- **Horizon**: Credit horizon configurable (up to N attempts ahead)

#### 3.6 Decision-Outcome Cycle

**Function**: `decide()` → **action selection** → **patch attempt** → `record_outcome()` → **model update**

**Step 1: Feature Extraction**
```python
features_raw = _adaptive_online_feature_vector(...)  # ~50 features
features_scaled = scale_feature_mapping(features_raw, profile="v1")  # Normalize
```

**Step 2: Policy Decision**
```python
decision = online_policy_controller.decide(
    feature_values=features_scaled,
    default_action=ACTION_INVOKE_HELPER
)
# Returns: PolicyDecision(
#   selected_action,  # From LinUCB arm selection
#   applied_action,   # selected if mode="on", else default
#   propensity,       # Sampling probability (for off-policy correction)
#   scores,           # UCB scores for all 10 actions
#   feature_values    # The scaled features
# )
```

**Step 3: Execute Attempt**
- Use `applied_action` to route helper or skip it
- Track helper request type (file/search/frame/snapshot/compaction/retry)
- Execute patch and validate

**Step 4: Observe Reward**
```python
reward = _adaptive_online_local_reward_components(
    success=success,
    helper_materialized_context=...,
    local_tokens_total=tokens_used,
    local_wall_sec=latency,
    adaptive_progress_score=progress_state.score,
    adaptive_no_progress_streak=streak_count,
    # ... + 8 parameters
)
```

**Step 5: Update Model** (if mode="on")
```python
updated = controller._update_model(
    action=applied_action,  # Which action was actually executed
    reward=reward,          # Observed reward
    feature_values=features_scaled
)
# A += x x^T, b += reward * x
# (every 50 updates, state is persisted to disk)
```

#### 3.7 Key Data Structures

**PolicyDecision** (adaptive_online_policy.py:134-149):
```python
@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str              # UUID for this decision
    mode: str                     # "off", "shadow", "on"
    algorithm: str                # "linucb"
    decision_index: int           # Sequential decision number
    created_at_utc: str
    selected_action: str          # Arm selected by LinUCB
    applied_action: str           # Action actually executed
    default_action: str           # Fallback action
    propensity: float             # P(selected_action)
    decision_strategy: str        # "warmup_uniform", "epsilon_uniform", "linucb_greedy", "thompson_greedy", "policy_off", or "decision_model_*"
    scores: dict[str, float]      # UCB scores for all 10 actions
    feature_values: dict[str, float]  # Scaled feature dict
```

**PolicyEvent** (adaptive_online_policy.py:152-199):
```python
@dataclass(frozen=True)
class PolicyEvent:
    # Pairing of decision + observed outcome
    decision_id: str
    selected_action: str
    applied_action: str
    behavior_action: str          # What actually happened (offline correction)
    reward: float                 # Observed reward
    reward_scope: str             # "helper_credit_v2"
    success: bool
    # Reward breakdown components:
    reward_success_hint: float
    reward_progress_hint: float
    reward_progress_score: float
    reward_token_penalty: float
    reward_latency_penalty: float
    reward_stagnation_penalty: float
    reward_helper_materialization_bonus: float
    reward_delayed_rescue_bonus: float
    # Credit tracking:
    reward_credit_resolved_attempt: int
    reward_credit_horizon_attempts: int
    reward_credit_resolution_reason: str
    # Local accounting:
    local_llm_call_count: int
    local_tokens_prompt: int
    local_tokens_completion: int
    local_tokens_total: int
    local_wall_sec: float
    # Update tracking:
    model_updated: bool
    update_reason: str            # "on_policy_update", "shadow_no_update", etc.
    nan_replacement_counts: dict[str, int]
```

---

## 4. BUDGET-AWARE VARIANT

### Overview
**File**: `adaptive_budget_online_policy.py`
**Purpose**: Decides whether to increase helper request budget per attempt

### Actions (2 options)
- `keep_budget` — Use current budget cap
- `increase_budget` — Escalate to next tier

### Configuration
```python
@dataclass(frozen=True)
class BudgetPolicyConfig:
    mode: str = "off"
    algorithm: str = "linucb"
    alpha: float = 0.4            # Lower than online policy (0.4 vs 0.6)
    epsilon: float = 0.05         # Lower exploration for the 2-action controller
    l2: float = 1.0
    warmup_decisions: int = 20    # Short warm-up before budget learning
    save_every: int = 50
    decay: float = 1.0
    feature_scaling_profile: str = "v1"
    state_path: Path | None = None
    events_path: Path | None = None
    random_seed: int | None = None
```

### Features
**Function**: `_adaptive_budget_feature_vector()` in `eval_policy_features.py`

Includes all helper-budget related signals:
- Attempt progress, helper calls used, tier/multiplier state
- Context compaction status and success rate
- Budget exhaustion and saturation flags
- Last compaction trigger type and compression ratio

### Reward
**Function**: `_adaptive_budget_local_reward_components()` (similar structure)

```
reward = success_bonus 
       - budget_exceeded_penalty
       + progress_bonus
       - stagnation_penalty
```

---

## 5. FEATURE SCALING

### Purpose
Normalize raw features into bounded range [-4, 4] suitable for LinUCB (prevents numerical instability).

### Profile "v1" (Default)
**File**: `policy_feature_scaling.py:179-244`

```python
def scale_feature_mapping(
    feature_values: dict[str, float],
    profile: str = "v1",
    cap_overrides: dict[str, float] | None = None
) -> dict[str, float]:
    # For count-like features (streaks, budgets, totals):
    #   normalized = log1p(value) / log1p(cap)
    #   output = clip(normalized, -4, 4)
    
    # For 0-1 bounded features: keep unchanged
    # For others: clip to [-4, 4]
```

**Hardcoded caps for count features**:
```python
_V1_COUNT_CAPS = {
    "stagnation_streak": 10.0,
    "adaptive_no_progress_streak": 10.0,
    "helper_calls_used": 8.0,
    "adaptive_no_progress_token_sum_k": 200.0,
    "base_propose_budget_sec": 300.0,
}
_V1_DEFAULT_STREAK_CAP = 10.0
_V1_FINAL_CLIP = 4.0
```

### Percentile Scaler
**Class**: `PercentileScaler` (policy_feature_scaling.py:140-195)

Optionally learns caps from historical data:
```python
scaler = PercentileScaler(percentile=95.0, min_samples=30)
scaler.fit({"stagnation_streak": [1, 2, 3, ...], ...})
scaled = scaler.scale(raw_features, profile="v1")
```

---

## 6. STAGE A CONTRACT

**File**: `stage_a_contract.py`

Defines the JSON schema for helper context requests and responses:

```python
STAGE_A_REQUEST_SCHEMA_COMPACT = (
    '{"diagnosis":"one-sentence root-cause hypothesis",'
    '"missing_fact":"specific fact needed, or null",'
    '"need_more_context":true,'
    '"need_snapshot":false,'
    '"requests":[{"type":"file","path":"relative/path.py"},'
    '{"type":"search","path":"...","pattern":"..."},'
    '{"type":"frame","index":1}],'
    '"sufficiency_rationale":"..."}' 
)
```

**Request types**: "file", "search", "frame"

---

## 7. PATCHERS

**Location**: `evals/patchers/`

### Available Patchers

| Patcher | Description |
|---------|-------------|
| `openai_compat_patcher.py` | Main patcher using OpenAI-compatible API (LLMs via llama.cpp, etc.) |
| `command_patcher.py` | Runs external command (for testing) |
| `heuristic_patcher.py` | Simple rule-based patching |
| `null_patcher.py` | No-op patcher |
| `base.py` | Abstract base class |

### Patcher Interface
Each patcher receives:
- **Evidence**: Traceback, error message, test failure
- **Context Request**: JSON payload specifying file/search/frame requests
- **Snapshot** (optional): Full execution context
- **Budget**: Token/time limits

Returns:
- **Patch**: Proposed code modification
- **Metadata**: Success/failure, tokens used, requests made

---

## 8. IMPORTERS

**Location**: `evals/importers/`

### Available Importers

| Importer | Dataset |
|----------|---------|
| `bugsinpy_importer.py` | BugsinPy (Python bugs) |
| `condefects_importer.py` | ConDefects |
| `debugbench_importer.py` | DebugBench |
| `humanevalpack_importer.py` | HumanEval-Pack |
| `mutmut_generator.py` | Mutation testing (MutMut) |
| `pybughive_importer.py` | PyBugHive |
| `pyresbugs_importer.py` | PyResBugs |
| `trickybugs_importer.py` | TrickyBugs |
| `native_repro_helpers.py` | Reproduction helpers |

### Import Pipeline
1. **Extract** cases from dataset
2. **Generate** buggy + fixed code versions
3. **Create** test harness
4. **Compute** metadata (category, difficulty, snapshot_dependency)
5. **Write** to `evals/cases/` directory

---

## 9. RESULTS ANALYSIS

### `analyze_results.py`
**Purpose**: Load JSONL records, validate schema, derive per-case-condition summaries, aggregate metrics.

**Key Functions**:
- `_load_and_group_records()` — Load JSONL + validate schema v5
- `_derive_case_condition_rows()` — Extract terminal attempt per case-condition
- `_summarize_bucket()` — Aggregate success rate, token usage, latency
- `_adaptive_ope_for_actions()` — Off-policy evaluation of target policies

**Off-Policy Evaluation (OPE)**:
Uses RidgeCV regression to estimate expected reward under target policy:
```
# Given historical (context, action, reward, propensity) tuples,
# estimate E[reward | context, target_action] via regression
```

### `eval_summary.py`
**Purpose**: Print concise single-run summary with condition comparison.

**Output includes**:
- Condition effectiveness (success rates)
- Pairwise uplifts
- Adaptive signal diagnostics
- Guardrail statistics

### `eval_two_pass_summary.py`
**Purpose**: Summarize baseline run + rescue run two-pass workflow.

**Output includes**:
- Baseline success rate
- Rescue effectiveness on baseline failures
- Cumulative success after both passes, with each pass's own cost (a compute effect, not a treatment effect; STA-22)
- Coverage diagnostics

---

## 10. INTEGRATION ACROSS THE EVAL MODULES

### How Conditions Are Orchestrated
1. `eval_main_setup.py` parses `--conditions`, validates the run shape, and
   resolves baseline-specific defaults.
2. `eval_main_flow.py` owns run-level orchestration, resume handling, and the
   shared execution palette.
3. `eval_attempt_flow.py` handles per-condition execution and branches into the
   gated or online adaptive logic when needed.
4. `eval_policy_features.py` builds the feature vector and local reward
   components for online learning.
5. `eval_policy_online.py` turns those features and rewards into controller
   metadata updates and persistent policy state.
6. `run_eval.py` stays as the compatibility facade for older imports and CLI
   entry points.

### Adaptive Control Flow
```python
conditions = [Condition(c) for c in args.conditions.split(",")]

is_adaptive_gated = condition == Condition.LAYERED_ADAPTIVE_GATED
is_adaptive_llm_discretion = condition == Condition.LAYERED_ADAPTIVE_LLM
is_adaptive = is_adaptive_gated or is_adaptive_llm_discretion

online_policy_config = OnlinePolicyConfig.normalize(...)
adaptive_online_policy_controller = AdaptiveOnlinePolicyController(online_policy_config)

budget_policy_config = BudgetPolicyConfig.normalize(...)
adaptive_budget_policy_controller = AdaptiveBudgetPolicyController(budget_policy_config)
```

### Per-Attempt Decision Loop
```python
if is_adaptive_llm_discretion:
    online_policy_decision = adaptive_online_policy_controller.decide(
        feature_values=online_policy_features_raw,
        default_action=helper_action_default,
    )
    helper_action_executed = online_policy_decision.applied_action

_record_adaptive_online_policy_outcome(
    adaptive_online_policy_controller=controller,
    decision=online_policy_decision,
    reward_breakdown=reward_breakdown,
    local_accounting=local_accounting,
    allow_model_update=True,
)

adaptive_online_policy_controller.close()
```

---

## 11. COMMAND-LINE PARAMETERS

### Online Policy Control
```bash
--adaptive-online-policy-mode {off,shadow,on}
  Default: off
  
--adaptive-online-policy-alpha FLOAT
  Default: 0.8
  
--adaptive-online-policy-epsilon FLOAT
  Default: 0.05
  
--adaptive-online-policy-l2 FLOAT
  Default: 1.0
  
--adaptive-online-policy-warmup INT
  Default: 200
  
--adaptive-online-policy-save-every INT
  Default: 50
  
--adaptive-online-policy-feature-scaling-profile {none,v1}
  Default: v1
  
--adaptive-online-policy-state-path PATH
  Persist model state to disk
  
--adaptive-online-policy-events-path PATH
  Log decisions/outcomes as JSONL
```

### Budget Policy Control
```bash
--adaptive-budget-policy-mode {off,shadow,on}
  Default: off
  
--adaptive-budget-policy-alpha FLOAT
  Default: 0.6
```

### Gate Policy (layered_adaptive_gated)
```bash
--adaptive-gate-policy {aggressive,balanced,conservative}
  Default: balanced
```

---

## 12. KEY INSIGHTS & DESIGN PATTERNS

### Exploration vs. Exploitation
- **Warm-up**: First `warmup_decisions` (default 50) decisions uniformly random across the feasible actions
- **Steady-state**: 95% LinUCB greedy (select max UCB score) + 5% random
- **UCB bonus**: α × √(x^T A^{-1} x) encourages exploration of uncertain actions

### Credit Assignment
- **Deferred credit**: If helper materializes context but attempt fails, reward is deferred to later attempt that resolves
- **Horizon**: Credit can span up to K attempts forward
- **Scope**: "helper_credit_v2" tracks which decision led to which outcome

### Off-Policy Learning (Shadow Mode)
- Policy selects action but baseline (default) action is executed
- Reward observed under baseline action behavior
- Enables offline evaluation before going live

### State Persistence
- State saved every N model updates (default 50)
- Enables resuming training across eval runs
- Includes feature names, A/b matrices, action counts, metadata

### Reward Decomposition
- Explicit components (success, progress, token cost, latency cost, stagnation penalty)
- Each tuned independently
- Combined to produce scalar reward for gradient updates

### Budget Escalation Policy
- Separate LinUCB instance with only 2 actions (keep / increase)
- Lower exploration (α=0.6, ε=0.03) because fewer actions
- No warm-up phase
- Complements online policy (which decides what helper requests to make)

---

## 13. FILE CROSS-REFERENCE

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `run_eval.py` | Compatibility CLI facade | `main()`, legacy re-exports |
| `eval_main_flow.py` | Run orchestration | `run_main_flow()`, shared scheduling/resume logic |
| `eval_main_setup.py` | CLI/config normalization | `prepare_main_setup()`, run-plan resolution |
| `eval_attempt_flow.py` | Per-condition execution | `run_one_condition_impl()`, `_adaptive_gate_should_invoke_helper()` |
| `eval_policy_features.py` | Adaptive features and local reward | `_adaptive_online_feature_vector()`, `_adaptive_online_local_reward_components()` |
| `eval_policy_online.py` | Policy metadata and finalization | `_adaptive_online_reward_breakdown()`, `_sync_adaptive_online_metadata()` |
| `adaptive_online_policy.py` | LinUCB contextual-bandit | `AdaptiveOnlinePolicyController`, `decide()`, `record_outcome()`, `_update_model()` |
| `adaptive_budget_online_policy.py` | Budget escalation LinUCB | `AdaptiveBudgetPolicyController` |
| `policy_feature_scaling.py` | Feature normalization | `scale_feature_mapping()`, `PercentileScaler` |
| `stage_a_contract.py` | Context request schema | `STAGE_A_REQUEST_SCHEMA_*` constants |
| `analyze_results.py` | Results aggregation & OPE | `_adaptive_ope_for_actions()`, `_summarize_bucket()` |
| `eval_summary.py` | Single-run summary | `_print_core_summary()` |
| `eval_two_pass_summary.py` | Baseline + rescue summary | Two-pass workflow analysis |
| `select_cases.py` | Rerun case selection | `_select_case_ids()` modes (failed, attempt-ge, etc.) |
| `eval_types.py` | Constants & types | `Condition` enum, policy parameter defaults |
| `patchers/openai_compat_patcher.py` | Main patcher | Handles helper invocation & patch generation |
| `importers/*.py` | Dataset importers | Load bugs from external datasets |
