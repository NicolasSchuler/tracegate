# LLMDebug Evaluation Framework - Comprehensive Guide

This directory contains extensive documentation on the llmdebug evaluation framework and adaptive online learning system.
For the canonical day-to-day quickstart and run contract, start with
[`evals/README.md`](../evals/README.md); this document is the deep-dive companion index.

## 📚 Documentation Files

### 1. **EVAL_FRAMEWORK_GUIDE.md** (Main Reference)
Comprehensive guide covering:
- Complete evaluation framework architecture
- 4 native conditions plus 5 external baselines
- Detailed adaptive online learning mechanism
- Feature extraction (50+ features)
- Reward formulation with all components
- Budget policy variant
- Feature scaling strategy
- Analysis framework (results analysis, summaries, selection)
- Patcher and importer systems
- Integration across eval_main_flow.py / eval_attempt_flow.py / eval_policy_features.py / eval_policy_online.py
- Complete file cross-reference

**Best for**: Deep understanding, implementation details, complete reference

### 2. **ADAPTIVE_POLICY_QUICK_REFERENCE.md** (Visual Guide)
Quick-reference with:
- Visual decision-outcome cycle diagram
- 10 actions matrix table
- What gets learned (A/b matrices, LinUCB formula)
- 50+ features grouped by category
- Reward formula breakdown
- Configuration options
- Integration code snippets
- Learning dynamics explanation
- Key insights and patterns

**Best for**: Quick lookup, understanding the big picture, implementation examples

### 3. **eval_baseline_callgraph.md** (Callgraph + Baseline Routing)
Targeted guide covering:
- the shared eval trunk across `eval_main_flow.py`, `eval_main_setup.py`, and `eval_attempt_flow.py`
- where `base_repair`, `cot_repair`, `layered_adaptive_gated`, and `layered_adaptive_llm` diverge
- how `ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, and `rlm` route through `BaselinePatcher`
- the main convergence points back into diff apply and pytest verification
- copy-paste `pueue` commands for 100-case Docker smoke runs per external baseline

**Best for**: Comparing control flow across native conditions and external baselines

## 🎯 Quick Navigation

### For Understanding the Online Policy
1. Start: **ADAPTIVE_POLICY_QUICK_REFERENCE.md** - "🎯 Core Concept" and "📊 The Learning Loop"
2. Deep dive: **EVAL_FRAMEWORK_GUIDE.md** - Section 3 "ADAPTIVE ONLINE LEARNING (Critical)"

### For Understanding Conditions
- **EVAL_FRAMEWORK_GUIDE.md** - Section 1 "Evaluation Framework Overview"
- Adaptive Gated: Section 2
- Adaptive LLM (`layered_adaptive_llm`): Section 3

### For Implementation Details
- Feature extraction: Section 3.4 in EVAL_FRAMEWORK_GUIDE.md
- Reward calculation: Section 3.5
- Integration across the eval modules: Section 10
- Policy files: Section 13 "File Cross-Reference"

### For Analysis & Results
- Results analysis: Section 9 "Results Analysis"
- Off-policy evaluation: analyze_results.py documentation
- Summary generation: eval_summary.py and eval_two_pass_summary.py

## 🔑 Key Concepts at a Glance

### What Is the Adaptive Online Learning Policy?
A **contextual-bandit (LinUCB) controller** that learns to select the best helper-invocation action (out of 8) based on the current evaluation state (50+ features). It observes rewards from past attempts and updates per-action prediction models.

### 10 Actions the Policy Can Choose
```
1. skip_helper              — Don't invoke helper
2. invoke_helper            — Standard invocation (baseline)
3. frame_only               — Only stack frames
4. file_only                — Only requested files
5. search_only              — Only search results
6. snapshot_lite            — Compact frame index (no model call)
7. frame_locals_lite        — Crash frame code + up to 5 locals (no model call)
8. snapshot_full_lite       — Up to 3 frames with locals (no model call)
9. compact_then_patch       — Compress context first
10. retry_helper_strict_json — Stricter JSON validation
```

### 50+ Features the Policy Uses
**Categories**: Attempt progress, helper budget, failure signals, context requests, context compaction, case metadata, pre-gate diagnostics

**Example features**:
- `attempt_frac`: current_attempt / max_attempts
- `stagnation_streak`: count of attempts with no progress
- `helper_calls_used`: total calls made so far
- `case_difficulty_ord`: 0 (easy) / 0.5 / 1 (hard) — opt-in via `--adaptive-online-include-case-labels` (off by default)

### What Gets Learned?
**Per-action matrices** (one set for each of the 10 actions):
- **A_k**: Accumulated feature covariance matrix (d×d)
- **b_k**: Reward-weighted feature accumulation (d×1)
- **θ_k = A_k^{-1} b_k**: Estimated reward coefficients

**Update rule** (rank-1 update):
```
A_k ← A_k + x·x^T           (accumulate feature covariance)
b_k ← b_k + reward·x        (accumulate reward-weighted features)
```

### How Does It Make Decisions?
**LinUCB scoring**:
```
score_k = θ_k^T·x + α·√(x^T·A_k^{-1}·x)
        = exploit + exploration_bonus
```
Select action with highest UCB score (95%) or random action (5%)

### What Is the Reward?
Decomposed reward signal:
```
reward = success_hint (if success)
       + η_progress × progress_score
       + materialization_bonus (if context used)
       - λ_tokens × (tokens_total / 1000)
       - μ_latency × (wall_sec / 10)
       - κ_stagnation × no_progress_penalty
```

## 📖 Reading Path by Role

### For Researchers
1. ADAPTIVE_POLICY_QUICK_REFERENCE.md (all sections)
2. EVAL_FRAMEWORK_GUIDE.md sections 3, 5, 10, 12
3. Source code: adaptive_online_policy.py, eval_main_flow.py, eval_attempt_flow.py

### For Engineers
1. ADAPTIVE_POLICY_QUICK_REFERENCE.md sections "🔧 Configuration", "🔄 Integration across the eval modules"
2. EVAL_FRAMEWORK_GUIDE.md sections 1, 10, 13
3. eval_main_flow.py, eval_main_setup.py, eval_attempt_flow.py, eval_policy_features.py, eval_policy_online.py

### For Data Scientists / Analysts
1. EVAL_FRAMEWORK_GUIDE.md section 9 "Results Analysis"
2. EVAL_FRAMEWORK_GUIDE.md section 4 "Budget-Aware Variant"
3. analyze_results.py (_adaptive_ope_for_actions function)

### For Newcomers
1. ADAPTIVE_POLICY_QUICK_REFERENCE.md (skip math sections)
2. EVAL_FRAMEWORK_GUIDE.md section 3.1-3.2
3. Then jump to specific sections as needed

## 🧮 Key Equations & Formulas

### LinUCB Score
```
UCB(a, x) = θ̂_a^T·x + α·√(x^T·(A_a)^{-1}·x)
```
where:
- θ̂_a = (A_a)^{-1}·b_a is estimated parameter vector
- x is feature vector
- α controls exploration-exploitation tradeoff

### Reward Decomposition
See ADAPTIVE_POLICY_QUICK_REFERENCE.md "💰 Reward Formula" section

## 🔗 Important File Locations

| File | Lines | Purpose |
|------|-------|---------|
| adaptive_online_policy.py | 1-780 | Core LinUCB controller |
| adaptive_budget_online_policy.py | 1-798 | Budget escalation policy |
| policy_feature_scaling.py | All | Feature normalization |
| eval_policy_features.py | All | Feature vector construction and local reward terms |
| eval_policy_online.py | All | Policy integration and metadata sync |
| eval_attempt_flow.py | All | Per-attempt adaptive routing |
| analyze_results.py | All | Results analysis & OPE |

## 🚀 Quick Start

### Running Adaptive LLM Discretion Condition
```bash
python evals/run_eval.py \
  --conditions layered_adaptive_llm \
  --patcher openai \
  --adaptive-online-policy-mode on \
  --adaptive-online-policy-state-path state.json \
  --adaptive-online-policy-events-path events.jsonl
```

### Analyzing Results
```bash
python evals/analyze_results.py \
  --results evals/results/run_id.jsonl \
  --condition layered_adaptive_llm
```

### Summarizing a Run
```bash
python evals/eval_summary.py run_id
```

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│      EVAL FRAMEWORK (eval_main_flow.py)             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         CONDITION ORCHESTRATION              │  │
│  │  • 4 native conditions                       │  │
│  │  • 5 external baselines                      │  │
│  │  • eval_main_setup.py resolves the run plan   │  │
│  └──────────────────────────────────────────────┘  │
│                      ▲                              │
│                      │                              │
│  ┌──────────────────────────────────────────────┐  │
│  │    ADAPTIVE ONLINE POLICY CONTROLLER         │  │
│  │    (adaptive_online_policy.py)               │  │
│  │                                              │  │
│  │  ┌─────────────────────────────────────┐    │  │
│  │  │ decide(features) → PolicyDecision   │    │  │
│  │  │  • Extract 50+ features             │    │  │
│  │  │  • Scale features (log1p, clip)     │    │  │
│  │  │  • Compute LinUCB scores            │    │  │
│  │  │  • Select action via ε-greedy       │    │  │
│  │  └─────────────────────────────────────┘    │  │
│  │                                              │  │
│  │  ┌─────────────────────────────────────┐    │  │
│  │  │ record_outcome(decision, reward)    │    │  │
│  │  │  • Observe reward signal            │    │  │
│  │  │  • Update A/b matrices (A←A+xx^T)   │    │  │
│  │  │  • Log PolicyEvent to JSONL         │    │  │
│  │  │  • Save state every 50 updates      │    │  │
│  │  └─────────────────────────────────────┘    │  │
│  │                                              │  │
│  │  ┌─────────────────────────────────────┐    │  │
│  │  │ close()                             │    │  │
│  │  │  • Flush state to disk              │    │  │
│  │  └─────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────┘  │
│                      ▲                              │
│                      │                              │
│  ┌──────────────────────────────────────────────┐  │
│  │         PATCHER SYSTEM                       │  │
│  │  • openai_compat_patcher (main)              │  │
│  │  • command_patcher                           │  │
│  │  • heuristic_patcher                         │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│    RESULTS (JSONL) → ANALYSIS FRAMEWORK             │
│    • analyze_results.py (aggregation, OPE)          │
│    • eval_summary.py (single-run summary)           │
│    • eval_two_pass_summary.py (baseline + rescue)   │
└─────────────────────────────────────────────────────┘
```

## 🎓 Learning Objectives After Reading

After completing these guides, you should understand:
- ✅ What a contextual bandit (LinUCB) is and how it works
- ✅ What 50+ features are used and why
- ✅ How actions are selected (warm-up, ε-greedy, UCB scoring)
- ✅ How the reward signal is composed and why each component matters
- ✅ How A/b matrices are updated and why matrix inversion matters
- ✅ How feature scaling prevents numerical issues
- ✅ What modes (off/shadow/on) mean and when to use each
- ✅ How to interpret policy decisions and events
- ✅ How to implement the policy in practice
- ✅ How to analyze results and perform off-policy evaluation

## 📞 Questions?

For specific code questions, refer to:
1. **adaptive_online_policy.py** for LinUCB implementation
2. **eval_main_flow.py**, **eval_attempt_flow.py**, and **eval_policy_online.py** for integration examples
3. **analyze_results.py** for result analysis patterns

For conceptual questions, refer to:
1. **ADAPTIVE_POLICY_QUICK_REFERENCE.md** for intuitive explanations
2. **EVAL_FRAMEWORK_GUIDE.md** for detailed technical descriptions
