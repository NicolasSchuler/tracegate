# llmdebug — Architecture Diagrams

Visual reference for the llmdebug codebase: core capture pipeline, RCA workflow,
eval framework, and adaptive online learning.

---

## 1. Core Tool — Capture Pipeline & Presentation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        llmdebug — How It Works                             │
└─────────────────────────────────────────────────────────────────────────────┘

  YOUR CODE crashes                     ENTRY POINTS
  ════════════════                      ══════════════════════════════════════
  │                                     ┌──────────┐ ┌────────┐ ┌──────────┐
  │  raise ValueError("x != y")         │  pytest   │ │ hooks  │ │middleware│
  │                                     │  plugin   │ │(prod)  │ │WSGI/ASGI│
  ▼                                     └────┬─────┘ └───┬────┘ └────┬─────┘
  Exception + Traceback                      │           │           │
                                             │   Rate    │           │
                                             │  Limiter  │           │
                                             ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CAPTURE PIPELINE  (capture.py)                                            │
│                                                                            │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
│  │ Walk exception   │──▶│ Collect frames   │──▶│ Serialize locals       │  │
│  │ chain (__cause__, │   │ (module filter,  │   │ (arrays→shape/stats,   │  │
│  │  __context__,     │   │  frame limit,    │   │  DataFrames→summary,   │  │
│  │  ExceptionGroups) │   │  crash frame)    │   │  redaction, safe_repr) │  │
│  └──────────────────┘   └──────────────────┘   └────────────────────────┘  │
│           │                                               │                │
│           ▼                                               ▼                │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────────┐  │
│  │ Classify error   │   │ Collect context  │   │ Extract function args  │  │
│  │ (12 categories   │   │ • git commit/diff│   │ (bytecode → arg names  │  │
│  │  + suggestions)  │   │ • coverage data  │   │  + values)             │  │
│  │ error_categories │   │ • async tasks    │   └────────────────────────┘  │
│  └──────────────────┘   │ • recent logs    │                               │
│                         │ • pytest context │                               │
│                         └──────────────────┘                               │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SNAPSHOT ENVELOPE  (output.py + _debug_session.py)                        │
│                                                                            │
│  DebugSession {                                                            │
│    schema_version: "2.1",                                                  │
│    session: {                                                              │
│      snapshot:  { exception, frames[], traceback, crash_frame_index }      │
│      context:   { env, git, async, recent_logs, pytest, coverage }         │
│      trace:     { ... optional ... }                                       │
│      eval:      { ... optional ... }                                       │
│    }                                                                       │
│  }                                                                         │
│                                                                            │
│  Write: atomic temp → rename │ Formats: JSON │ json_compact │ TOON         │
│  Index: SQLite .state.db     │ Symlink: latest.json → newest snapshot      │
│  Cleanup: prune beyond max_snapshots                                       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
│ DETAIL FILTER    │  │ HYPOTHESES       │  │ SNAPSHOT DIFF            │
│ (detail_filter)  │  │ (hypotheses.py)  │  │ (snapshot_diff.py)       │
│                  │  │                  │  │                          │
│ crash  ≈2K tok   │  │ 10 detectors:   │  │ Compare two snapshots:   │
│  └ exc + 1 frame │  │ • empty_array   │  │ • exception changes      │
│ full   ≈5K tok   │  │ • shape_mismatch│  │ • frame stack delta      │
│  └ + all frames  │  │ • none_deref    │  │ • locals delta           │
│ context ≈10K tok │  │ • off_by_one    │  │ • config changes         │
│  └ + git/cov/env │  │ • nan_inf       │  └──────────────────────────┘
└──────────────────┘  │ • type_mismatch │
                      │ • key_not_found │
                      │ • import_fail   │
                      │ • file_not_found│
                      │ • assert_fail   │
                      │                 │
                      │ → Ranked by     │
                      │   confidence    │
                      └────────┬────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│     CLI          │  │   MCP Server     │  │  Jupyter Plugin  │
│  (cli.py)        │  │ (mcp_server.py)  │  │ (jupyter_plugin) │
│                  │  │                  │  │                  │
│ • show [--full]  │  │ 9 tools:         │  │ Auto-capture in  │
│ • list           │  │ • diagnose       │  │ notebook cells   │
│ • frames         │  │ • show/list      │  │ on exceptions    │
│ • diff           │  │ • frame          │  │                  │
│ • git-context    │  │ • git_context    │  └──────────────────┘
│ • clean          │  │ • diff           │
│                  │  │ • rca_status     │
│ Rich panels if   │  │ • rca_history    │
│ TTY + rich       │  │ • rca_advance    │
└──────────────────┘  └──────────────────┘
```

---

## 2. RCA State Machine & Gate Logic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         RCA STATE MACHINE  (rca_engine.py + rca_transitions.py)            │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌───────────┐  bind_evidence  ┌────────────────┐  hypothesize  ┌──────────────┐
  │ captured  │────────────────▶│ evidence_bound │─────────────▶│ hypothesized │
  │           │                 │                │              │              │
  │ Snapshot  │                 │ Crash frame,   │              │ Root cause   │
  │ loaded    │                 │ exception cited│              │ proposed     │
  └───────────┘                 └────────────────┘              └──────┬───────┘
       ▲                                                              │
       │ capture                                                plan  │
       │ (new failure)                                                ▼
  ┌────┴──────┐                                                ┌───────────┐
  │ resolved  │◀── resolve ── ┌────────────┐◀── verify ──────│  planned  │
  │           │               │  verified  │                  │           │
  │ Fixed!    │               │            │                  │ Patch     │
  │ ✓         │               │ Test result│                  │ plan      │
  └───────────┘               │ captured   │                  └─────┬─────┘
                              └────────────┘                        │
                                    ▲                         change│
                                    │                               ▼
                              ┌─────┴──────┐                  ┌───────────┐
                              │            │◀─────────────────│  changed  │
                              │  (verify)  │     verify       │           │
                              │            │                  │ Patch     │
                              └────────────┘                  │ applied   │
                                                              └───────────┘

  STALL DETECTION (escape hatch):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  IF same failure signature AND repeat_count ≥ stall_threshold           │
  │     AND NOT hypothesis_advanced AND NOT plan_advanced                   │
  │  THEN → state forced to "stalled"                                      │
  │                                                                        │
  │  ┌──────────┐                                                          │
  │  │ stalled  │──▶ bind_evidence / hypothesize / plan  (escape routes)   │
  │  │          │    Must bring NEW evidence or NEW hypothesis to proceed   │
  │  └──────────┘                                                          │
  └──────────────────────────────────────────────────────────────────────────┘

  GATE LOGIC (gates.py):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  Mode: off │ soft │ strict                                             │
  │                                                                        │
  │  Checks enforced per action:                                           │
  │  ┌──────────────────────────────────────────────────────────────────┐   │
  │  │ hypothesize/plan/change/verify/resolve → evidence_refs required │   │
  │  │ plan/change/verify/resolve             → hypothesis required    │   │
  │  │ change/verify/resolve                  → experiment_plan req.   │   │
  │  │ resolve                                → verification pass req. │   │
  │  │ repeated signature w/o progress        → loop_protection warn   │   │
  │  │ action not allowed from current state  → illegal_transition     │   │
  │  └──────────────────────────────────────────────────────────────────┘   │
  │                                                                        │
  │  off:    allowed=True always, severity=info                            │
  │  soft:   allowed=True always, severity=warning if checks fail          │
  │  strict: allowed=False if hard_failed (unless exploratory override)    │
  │                                                                        │
  │  Exploration budget: consumed on repeated signature w/o hypothesis      │
  │  advancement; REFILLS when hypothesis changes                          │
  │                                                                        │
  └──────────────────────────────────────────────────────────────────────────┘

  PERSISTENCE:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  SQLite: .llmdebug/llmdebug_state.sqlite3                              │
  │  Table:  rca_attempts (attempt_id, session_id, failure_signature,      │
  │          state, payload_json, created_at, updated_at)                   │
  │  Index:  (session_id, updated_at DESC) for fast session history         │
  │  Prune:  max 5000 records, oldest dropped                              │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Eval Framework — A/B Debugging Experiment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EVAL HARNESS  (run_eval.py)                            │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │  CASE DISCOVERY          │
                    │  evals/cases/            │
                    │  + importers/            │
                    │    ├─ DebugBench         │
                    │    ├─ HumanEvalPack      │
                    │    └─ ConDefects         │
                    │  + validate_cases.py     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  SELECT & CONFIGURE      │
                    │  select_cases.py         │
                    │  TOML config files       │
                    │  Resume from partial run │
                    └────────────┬─────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
  ┌─────────────────┐  ┌─────────────────────┐  ┌──────────────────────────┐
  │ CONDITION:      │  │ CONDITION:          │  │ CONDITION:               │
  │ base_repair     │  │ cot_repair          │  │ layered_adaptive_gated / │
  │                 │  │                     │  │ layered_adaptive_llm     │
  │ LLM gets only:  │  │                     │  │                          │
  │ • Exception msg │  │ LLM gets:           │  │ LLM gets snapshot BUT    │
  │ • Traceback text│  │ • Exception msg     │  │ controlled by adaptive   │
  │ • Source code   │  │ • Traceback text    │  │ online policy (LinUCB)   │
  │                 │  │ • Source code       │  │ which decides per-attempt │
  │ (Baseline)      │  │ • CoT reasoning     │  │ what strategy to use     │
  │                 │  │                     │  │                          │
  │                 │  │                     │  │ (see Section 4 below)    │
  └────────┬────────┘  └────────┬────────────┘  └────────────┬─────────────┘
           │                    │                             │
           └────────────────────┼─────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PER-CASE EXECUTION LOOP (up to max_attempts per case)                     │
│                                                                            │
│  For each attempt:                                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 1. INJECT BUG        Patcher applies known bug to source             │  │
│  │ 2. RUN FAILING TEST  Capture exception / traceback                   │  │
│  │ 3. PREPARE CONTEXT   Apply condition's context (traceback vs snap)   │  │
│  │ 4. LLM PROPOSES FIX  Via pluggable patcher backend                   │  │
│  │ 5. APPLY PATCH       Write proposed fix to source                    │  │
│  │ 6. RUN PYTEST        Verify if fix resolves the bug                  │  │
│  │ 7. RECORD RESULT     → results.jsonl (pass/fail + metadata)          │  │
│  │ 8. COMPUTE PROGRESS  Track stagnation, signature repeats, tokens     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  PATCHER BACKENDS (evals/patchers/):                                       │
│  ┌────────────┐ ┌────────────┐ ┌─────────────┐ ┌────────────┐             │
│  │  command   │ │  openai    │ │  heuristic  │ │    null    │             │
│  │ (CLI tool) │ │ (API call) │ │ (rule-based)│ │ (no-op)    │             │
│  └────────────┘ └────────────┘ └─────────────┘ └────────────┘             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ANALYSIS (analyze_results.py + eval_summary.py + stats.py)                │
│                                                                            │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐  │
│  │ Per-condition    │   │ Statistical      │   │ Two-pass summary        │  │
│  │ success rates    │   │ significance:    │   │ (baseline + rescue):    │  │
│  │ by category,     │   │ • McNemar's test │   │ eval_two_pass_summary   │  │
│  │ difficulty,      │   │ • Cohen's h      │   │                         │  │
│  │ source dataset   │   │ • Bootstrap CI   │   │ What % of baseline      │  │
│  │                  │   │ • Holm-Bonferroni│   │ failures were rescued   │  │
│  │                  │   │   correction     │   │ by snapshot condition?  │  │
│  └─────────────────┘   └──────────────────┘   └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Adaptive Online Learning — LinUCB Contextual Bandit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          ADAPTIVE ONLINE POLICY — LinUCB Contextual Bandit                 │
│          (adaptive_online_policy.py + adaptive_budget_online_policy.py)     │
└─────────────────────────────────────────────────────────────────────────────┘

 TWO PARALLEL BANDITS run per eval case:

 ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐
 │  HELPER INVOCATION POLICY       │  │  BUDGET ESCALATION POLICY           │
 │  "Should I use the helper, and  │  │  "Should I increase the resource    │
 │   if so, which strategy?"       │  │   budget for this attempt?"         │
 │                                 │  │                                     │
 │  10 Arms (actions):             │  │  2 Arms (actions):                  │
 │  ├─ skip_helper                 │  │  ├─ keep_budget                     │
 │  ├─ invoke_helper               │  │  └─ increase_budget                 │
 │  ├─ frame_only                  │  │                                     │
 │  ├─ file_only                   │  │  Controls: timeout multiplier,      │
 │  ├─ search_only                 │  │  max tokens, helper call limit,     │
 │  ├─ snapshot_lite               │  │  stage A request limits             │
 │  ├─ frame_locals_lite           │  │                                     │
 │  ├─ snapshot_full_lite          │  │                                     │
 │  ├─ compact_then_patch          │  │                                     │
 │  └─ retry_helper_strict_json    │  │                                     │
 └────────────────┬────────────────┘  └───────────────────┬─────────────────┘
                  │                                       │
                  └──────────────┬─────────────────────────┘
                                 │
                                 ▼

 ════════════════════════════════════════════════════════════════════════════
                          DECISION FLOW (per attempt)
 ════════════════════════════════════════════════════════════════════════════

 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STEP 1: EXTRACT CONTEXT FEATURES  (~45 dimensions)                    │
 │                                                                        │
 │  Attempt progress:           Helper state:                             │
 │  ├─ bias (=1.0, intercept)   ├─ helper_calls_used                     │
 │  ├─ attempt_frac             ├─ helper_low_yield_ratio                │
 │  ├─ timed_out                ├─ helper_budget_remaining_frac          │
 │  └─ failure_signature_repeat ├─ helper_budget_exhausted               │
 │                              ├─ helper_saturation_guarded             │
 │                              ├─ helper_topup_used/remaining_frac      │
 │  Stagnation signals:         └─ last_helper_need_more_context         │
 │  ├─ stagnation_streak                                                 │
 │  └─ adaptive_no_progress_    Case metadata (opt-in, off by default):  │
 │       streak                 ├─ case_difficulty_ord (0/0.5/1)         │
 │                              ├─ case_snapshot_dependency_ord           │
 │  Context compaction:         └─ case_bug_source_imported              │
 │  ├─ compaction_calls_frac                                             │
 │  ├─ compaction_remaining     Pre-gate signals:                        │
 │  ├─ last_compaction_ratio    ├─ pregate_score                         │
 │  └─ last_compaction_trigger  ├─ pregate_type_specificity              │
 │       (stage_a/overflow/     ├─ pregate_crash_in_user_code            │
 │        always)               ├─ pregate_assertion_quality             │
 │                              └─ pregate_exception_is_*                │
 │  Last action one-hot (skip_helper is the reference level):            │
 │  └─ last_helper_action_{invoke|frame|file|search|lite|...}            │
 └───────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STEP 2: FEATURE SCALING  (policy_feature_scaling.py, profile "v1")    │
 │                                                                        │
 │  Count-like features (streaks, call counts):                           │
 │       scaled = log1p(value) / log1p(cap)    cap from _V1_COUNT_CAPS   │
 │  All features:                                                         │
 │       clipped to [-4.0, +4.0]               _V1_FINAL_CLIP            │
 │  NaN/Inf → 0.0 (tracked per feature for diagnostics)                  │
 └───────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STEP 3: LinUCB ARM SELECTION                                          │
 │                                                                        │
 │  For each arm a ∈ {skip, invoke, frame_only, ...}:                     │
 │                                                                        │
 │     θ_a = A_a⁻¹ · b_a              (learned weight vector)            │
 │                                                                        │
 │     exploit_a = θ_aᵀ · x           (expected reward given context)     │
 │     explore_a = √(xᵀ · A_a⁻¹ · x) (uncertainty bonus)               │
 │                                                                        │
 │     UCB_a = exploit_a + α · explore_a          α = 0.8                 │
 │                                                                        │
 │  Selection strategy:                                                   │
 │  ┌───────────────────────────────────────────────────────────────────┐  │
 │  │  IF decision_index ≤ warmup (200):                               │  │
 │  │      action ← uniform random         (warmup_uniform)            │  │
 │  │  ELIF random() < ε (0.05):                                       │  │
 │  │      action ← uniform random         (epsilon_uniform)           │  │
 │  │  ELSE:                                                           │  │
 │  │      action ← argmax(UCB_a)          (linucb_greedy)             │  │
 │  └───────────────────────────────────────────────────────────────────┘  │
 │                                                                        │
 │  Mode behavior:                                                        │
 │  ├─ "off":    always use default_action (no learning)                  │
 │  ├─ "shadow": select + log action, but apply default (learn offline)   │
 │  └─ "on":     select + apply + learn (full online learning)            │
 └───────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STEP 4: EXECUTE SELECTED STRATEGY                                     │
 │                                                                        │
 │  ┌────────────────────────┬────────────────────────────────────────┐    │
 │  │ skip_helper            │ No snapshot/helper; traceback only     │    │
 │  │ invoke_helper          │ Full snapshot + helper call            │    │
 │  │ frame_only             │ Only crash frame info to LLM          │    │
 │  │ file_only              │ Only file-level info to LLM           │    │
 │  │ search_only            │ Only search-based context             │    │
 │  │ snapshot_lite          │ Compressed snapshot (fewer tokens)     │    │
 │  │ compact_then_patch     │ Compact representation → direct patch │    │
 │  │ retry_helper_strict_json│ Retry with strict JSON parsing       │    │
 │  └────────────────────────┴────────────────────────────────────────┘    │
 └───────────────────────────────────┬─────────────────────────────────────┘
                                     │
                          ┌──────────┤
                          ▼          │
               [LLM proposes fix]    │
               [Pytest verifies]     │
                          │          │
                          ▼          │
 ┌────────────────────────────────────────────────────────────────────────┐
 │  STEP 5: COMPUTE REWARD (multi-component signal)                       │
 │                                                                        │
 │  reward = success_hint                                                 │
 │         + progress_hint                                                │
 │         + helper_materialization_bonus                                  │
 │         − token_penalty                                                │
 │         − latency_penalty                                              │
 │         − stagnation_penalty                                           │
 │                                                                        │
 │  ┌────────────────────────────────────────────────────────────────────┐ │
 │  │ success_hint  = 0.25 if test passed, else 0                      │ │
 │  │ progress_hint = 0.5 · η · progress_score                        │ │
 │  │   (progress_score: how much closer to fix vs previous attempt)   │ │
 │  │ helper_materialization_bonus = 0.25 · η if helper produced ctx   │ │
 │  │ token_penalty = λ · (tokens_used / token_scale)                  │ │
 │  │ latency_penalty = μ · (wall_seconds / latency_scale)             │ │
 │  │ stagnation_penalty = 0.5 · κ · min(1, no_progress_streak / 3)   │ │
 │  └────────────────────────────────────────────────────────────────────┘ │
 │                                                                        │
 │  + Delayed credit assignment:                                          │
 │    If a later attempt resolves, earlier helper calls within a          │
 │    credit horizon get a delayed_rescue_bonus (reward back-propagation) │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STEP 6: MODEL UPDATE (LinUCB rank-1 update)                           │
 │                                                                        │
 │  For the arm a that was executed:                                       │
 │                                                                        │
 │     A_a ← A_a + x · xᵀ           (update precision matrix)           │
 │     b_a ← b_a + reward · x       (update reward accumulator)          │
 │                                                                        │
 │  Single-sample incremental update — no batch needed.                   │
 │                                                                        │
 │  What the model learns over time:                                      │
 │  ┌──────────────────────────────────────────────────────────────────┐   │
 │  │  θ_a = A_a⁻¹ · b_a    (weight vector per arm)                  │   │
 │  │                                                                  │   │
 │  │  θ_a maps context features → expected reward for arm a.         │   │
 │  │  After many updates, θ encodes patterns such as:                │   │
 │  │                                                                  │   │
 │  │  "When stagnation_streak is high and helper_budget is low,      │   │
 │  │   skip_helper gets better rewards than invoke_helper"           │   │
 │  │                                                                  │   │
 │  │  "When case_difficulty is high and we're early in attempts,     │   │
 │  │   snapshot_lite outperforms frame_only"                         │   │
 │  │                                                                  │   │
 │  │  A_a captures feature covariance → controls exploration bonus.  │   │
 │  │  Less-explored feature regions get higher UCB → more sampling.  │   │
 │  └──────────────────────────────────────────────────────────────────┘   │
 │                                                                        │
 │  Persistence:                                                          │
 │  ├─ State (A, b matrices): saved every 50 updates → JSON on disk      │
 │  ├─ Events (decision+reward): appended as JSONL → offline analysis    │
 │  └─ Feature names: auto-expand when new features appear (zero-pad)     │
 └────────────────────────────────────────────────────────────────────────┘


 ════════════════════════════════════════════════════════════════════════════
                EXAMPLE: FULL LIFECYCLE WITHIN ONE EVAL CASE
 ════════════════════════════════════════════════════════════════════════════

  Case: "shape_mismatch_numpy_001", difficulty=hard, max_attempts=5

  Attempt 1                    Attempt 2                    Attempt 3
  ┌────────────────────┐       ┌────────────────────┐       ┌────────────────┐
  │ Features:          │       │ Features:          │       │ Features:      │
  │  attempt_frac=0.2  │       │  attempt_frac=0.4  │       │  attempt=0.6   │
  │  stagnation=0      │       │  stagnation=1      │       │  stagnation=0  │
  │  difficulty=hard   │       │  sig_repeat=1      │       │  (new sig)     │
  │                    │       │                    │       │                │
  │ Decision:          │       │ Decision:          │       │ Decision:      │
  │  warmup_uniform    │       │  warmup_uniform    │       │  linucb_greedy │
  │  → invoke_helper   │       │  → frame_only      │       │  → snapshot_   │
  │                    │       │                    │       │     lite       │
  │ Outcome:           │       │ Outcome:           │       │                │
  │  FAIL, progress=0.3│       │  FAIL, progress=0  │       │ Outcome:       │
  │  reward = +0.07    │       │  reward = −0.12    │       │  PASS! ✓       │
  │                    │       │  (stagnation pen.) │       │  reward = +0.32│
  │ Update A,b for     │       │                    │       │                │
  │  invoke_helper     │       │ Update A,b for     │       │ Update A,b for │
  └────────────────────┘       │  frame_only        │       │  snapshot_lite │
                               └────────────────────┘       └────────────────┘
                                                              ▲
                                                              │
                                                     Model learned:
                                                     snapshot_lite works
                                                     well for hard cases
                                                     after stagnation
```

With a System-1 decision model attached (`--adaptive-decision-model-role`), the `policy` role
replaces the selection above (`decision_model_greedy`, `decision_model_sample`,
`decision_model_atomic`, `decision_model_epsilon`, or `decision_model_unavailable` when the
sidecar fails), and the `prior` role replaces warm-up sampling (`decision_model_guided_warmup`)
before adding a fading score prior to the greedy ranking. Thompson sampling's greedy picks are
labeled `thompson_greedy`. See `docs/adaptive_online_policy_design.md`.

---

## 5. Module Map

```
src/llmdebug/
├── CAPTURE & SERIALIZATION ─────────────────────────────────────────────────
│   ├── capture.py              Exception → structured snapshot payload
│   ├── config.py               SnapshotConfig (all settings as dataclass)
│   ├── serialize.py            Locals → safe JSON (arrays, DFs, redaction)
│   ├── output.py               Atomic write, index, symlink, format select
│   ├── compact_keys.py         Key abbreviation for json_compact format
│   └── toon_encoder.py         Binary TOON format encoder
│
├── ANALYSIS & PRESENTATION ─────────────────────────────────────────────────
│   ├── detail_filter.py        crash/full/context layered disclosure
│   ├── error_categories.py     12-category auto-classifier + suggestions
│   ├── hypotheses.py           10 pattern detectors → ranked hypotheses
│   ├── snapshot_diff.py        Structural comparison of two snapshots
│   ├── snapshot_types.py       TypedDict definitions for strict typing
│   ├── _formatting.py          Shared text formatting helpers
│   ├── _presentation.py        HTML/text rendering helpers
│   ├── _debug_session.py       DebugSession envelope normalization
│   └── _snapshot_loader.py     Load + normalize + filter snapshots
│
├── RCA WORKFLOW ─────────────────────────────────────────────────────────────
│   ├── rca_state.py            State machine (8 states) + tracking
│   ├── rca_transitions.py      Valid transitions + transition matrix
│   ├── rca_engine.py           Pure domain logic: build_rca_attempt_record
│   ├── rca_identity.py         Failure signature + session ID derivation
│   ├── gates.py                Gate evaluation (strict/soft/off)
│   ├── rca_backend.py          Backend selection (SQLite vs JSONL)
│   ├── rca_store.py            Persistence queries (read/write)
│   └── state_db.py             SQLite schema + CRUD operations
│
├── CONTEXT COLLECTORS ──────────────────────────────────────────────────────
│   ├── git_context.py          Git commit, branch, dirty files, blame
│   ├── coverage_context.py     pytest-cov integration
│   ├── async_context.py        asyncio task info capture
│   └── log_capture.py          Recent log records
│
├── INTERFACES (kept thin) ──────────────────────────────────────────────────
│   ├── cli.py                  Click CLI: show, list, frames, diff, clean
│   ├── mcp_server.py           MCP server: 9 tools for IDE agents
│   ├── mcp_rca.py              MCP RCA tools
│   ├── mcp_contract.py         MCP contract definitions
│   ├── mcp_git_context.py      MCP git context tool
│   ├── mcp_types.py            MCP type definitions
│   ├── pytest_plugin.py        Auto-capture on test failures
│   ├── jupyter_plugin.py       Notebook exception capture
│   └── middleware.py           WSGI/ASGI middleware
│
├── PRODUCTION SAFETY ───────────────────────────────────────────────────────
│   ├── hooks.py                sys.excepthook + threading + asyncio hooks
│   ├── redaction_policy.py     PII redaction policies
│   └── _warnings.py            Warning utilities
│
evals/
├── ORCHESTRATION ───────────────────────────────────────────────────────────
│   ├── run_eval.py             Multi-condition eval harness (all conditions)
│   ├── select_cases.py         Case selection + filtering
│   ├── validate_cases.py       Schema validation
│   └── stage_a_contract.py     Stage A contract enforcement
│
├── ADAPTIVE LEARNING ───────────────────────────────────────────────────────
│   ├── adaptive_online_policy.py        LinUCB: 8-arm helper invocation
│   ├── adaptive_budget_online_policy.py LinUCB: 2-arm budget escalation
│   └── policy_feature_scaling.py        v1 scaling: log1p + clip
│
├── ANALYSIS ────────────────────────────────────────────────────────────────
│   ├── analyze_results.py      Category/source slicing
│   ├── eval_summary.py         Single-run summary
│   ├── eval_two_pass_summary.py Baseline+rescue summary
│   └── stats.py                McNemar, Cohen's h, bootstrap CI
│
├── PATCHERS ────────────────────────────────────────────────────────────────
│   └── patchers/               command │ openai │ heuristic │ null
│
└── IMPORTERS ───────────────────────────────────────────────────────────────
    └── importers/              DebugBench │ HumanEvalPack │ ConDefects
```
