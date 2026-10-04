# Hierarchical Bandit Architecture for Adaptive Debugging

This document describes the hierarchical bandit architecture used in llmdebug's
adaptive eval conditions (`layered_adaptive_gated`, `layered_adaptive_llm`) and
its relationship to BOAD (Bandit Optimization for Autonomous Debugging).

## Architecture Overview

The adaptive policy stack uses a **3-level hierarchical bandit** to control how
the LLM-based repair pipeline allocates resources and selects evidence:

```
┌─────────────────────────────────────────────────┐
│  Level 1: Budget Policy                         │
│  ─────────────────────                          │
│  Algorithm: LinUCB / Thompson Sampling          │
│  Actions:   keep_budget | increase_budget       │
│  Decides:   Whether to spend more tokens/time   │
│  File:      adaptive_budget_online_policy.py    │
├─────────────────────────────────────────────────┤
│  Level 2: Evidence Policy                       │
│  ─────────────────────                          │
│  Algorithm: LinUCB / Thompson Sampling          │
│  Actions:   10 evidence-type combinations       │
│  Decides:   Which evidence to show the LLM      │
│  File:      adaptive_online_policy.py           │
├─────────────────────────────────────────────────┤
│  Level 3: Strategy Policy (future extension)    │
│  ─────────────────────                          │
│  Algorithm: [not yet implemented]               │
│  Actions:   Prompt configuration combinations   │
│  Decides:   How to frame the repair task        │
│  See:       OPTS (ACL 2025) for design model    │
└─────────────────────────────────────────────────┘
```

### Decision flow per attempt

1. **Budget policy** observes case features → decides resource envelope
2. **Evidence policy** observes case + budget features → selects evidence action
3. Evidence action determines which types of context the Stage-A LLM may request
4. Request filtering silently discards LLM requests outside the allowed types
5. Repair pipeline executes with filtered evidence
6. Outcome (success, progress, token cost) feeds reward back to both levels

## Policy-Controlled Evidence Selection

A key design principle: **the policy — not the LLM — controls what evidence the
patching model receives**. This is enforced through 3 layers:

1. **Informativeness pregate**: Scores traceback diagnostic value before any
   bandit decision. If the traceback is already sufficient (score > threshold),
   the helper call is skipped entirely, saving tokens.

2. **Policy action selection**: A LinUCB (or Thompson Sampling) contextual
   bandit picks one of 10 actions that constrain which *types* of evidence the
   Stage-A helper LLM may request (e.g., "snapshot only", "files + frames",
   "compact context").

3. **Request filtering**: Any Stage-A LLM requests for evidence types outside
   the policy's allowed set are silently discarded before execution.

This architecture is motivated by empirical observation: when given free choice,
LLMs consistently over-request context (snapshots, files, frames), inflating
token usage without proportional accuracy gains.

## Mapping to BOAD

BOAD (Bandit Optimization for Autonomous Debugging, ICLR 2026) proposes
hierarchical SE agents that decompose debugging into bandit-controlled sub-tasks.
Our architecture was developed independently but shares structural parallels:

| BOAD Concept | llmdebug Implementation | Notes |
|--------------|------------------------|-------|
| Resource allocation level | Budget policy (Level 1) | 2 arms: keep/increase |
| Tool/evidence selection level | Evidence policy (Level 2) | 10 arms: evidence combos |
| Strategy selection level | [Level 3, future] | Prompt config bandits |
| Static allocation | **Online learning** (LinUCB/TS) | Our extension: learns during run |
| Per-task optimization | Per-attempt contextual bandit | Features include case difficulty, history |
| Reward signal | Multi-component reward | Success + progress + cost penalties |

### Key differences from BOAD

1. **Online learning**: BOAD uses static or pre-trained allocation. Our bandits
   learn during the eval run via LinUCB or Thompson Sampling with recency decay.
   This allows the policy to adapt to the specific model and dataset.

2. **3-layer enforcement**: BOAD delegates tool selection to agents. We enforce
   evidence selection through the policy layer, preventing the LLM from
   overriding the bandit's decision.

3. **Contextual features**: Our bandits condition on 15+ features per decision
   (informativeness score, prior attempt outcomes, token budget, error category,
   etc.), enabling fine-grained context-dependent policies.

4. **Auxiliary reward signals**: Beyond binary success, we use progress signals
   (pytest fail reduction, evidence novelty, failure signature changes) to
   provide denser learning signal.

## Reward Structure

The reward function combines multiple signals:

```
reward = success_hint × I(success)
       + η × progress_signal
       − κ × stagnation_penalty
       + materialization_bonus
       − λ × token_cost
       − μ × latency_cost
       + auxiliary_rewards (when enabled)
```

Where auxiliary rewards include:
- `aux_pytest_fail_reduction`: Reward for reducing test failures
- `aux_evidence_novelty`: Reward for gathering new (non-redundant) evidence
- `aux_compaction_efficiency`: Reward for efficient context compression
- `aux_informativeness_alignment`: Reward for actions aligned with pregate

All signals are attempt-derived (no ground-truth case labels in reward) to
maintain fairness and avoid data leakage.

## Extension Points

### Thompson Sampling (implemented)

Thompson Sampling is available as a feature-gated alternative to LinUCB at both
the budget and evidence levels. Enable via:
```bash
--adaptive-online-policy-algo thompson
--adaptive-budget-policy-algo thompson
```

TS provides natural exploration through posterior sampling, eliminating the need
for explicit epsilon-greedy exploration. It reuses the same A/b matrix
infrastructure as LinUCB, including state persistence and event-log bootstrap.

### System-1 Decision-Model Advice (implemented)

A pretrained System-1 decision model (SystemOne API, e.g. Cloudflare clef-flash in a GPU
sidecar) can advise the evidence-level policy without changing the 3-layer structure: it runs
at layer 2, after the pregate and hard constraints, sees only the feasible actions, and leaves
Stage-A request filtering untouched. It reads a redacted, bounded rendering of the failure that
the numeric bandit features cannot express, and returns a probability per action plus
diagnostic answers.
```bash
--adaptive-decision-model-role shadow|policy|features|prior
```

- `shadow` logs its answers next to the bandit's decisions (for OPE).
- `policy` lets it select (an argmax or tempered distribution mixed with epsilon, or an atomic
  mapping of diagnostic answers to an action in code).
- `features` and `prior` keep the bandit as the learner: diagnostic answers become features, or
  the decision model's distribution guides warm-up and a fading score prior. This targets the
  bandit's cold start, which matters because the shipped template makes roughly one helper
  decision per case.

See `docs/adaptive_online_policy_design.md` ("Decision-model advice").

### Level 3: Strategy Policy (future)

Following the OPTS framework (ACL 2025, arxiv:2503.01163), a third bandit level
could select prompt configurations:
- CoT mode (plan-then-patch vs direct patch)
- Calibration example style
- RCA checklist emphasis
- Closing instruction framing

The existing 9 prompt flags yield ~252 feasible combinations per condition,
naturally mapping to bandit arms. The infrastructure (state persistence, event
logging, feature scaling) is reusable.

### TEP Modes

The harness now exposes two distinct TEP paths:
- `legacy`: the original TEP-inspired retry mechanism with a stagnation-sensitive prompt nudge (`loop_breaker`) plus an optional retry-plan stability loop that reruns planning until the plan text stabilizes.
- `full`: a native-only prompt-state controller that persists mutable prompt surfaces for Stage A, context compaction, retry planning, and patch proposal under `--adaptive-tep-mode full`.

`full` mode is the research path for adapting Textual Equilibrium Propagation to
the native repair graph. It persists stage-local prompt state, makes the
Stage-B instruction blocks explicit optimization targets, and records per-node
state/metadata separately from the legacy retry nudges.

After each attempt, every invoked node runs a paired comparison (POL-17): the critic
scores the current prompt surface, the updater proposes a change from that critique,
and the same critic prompt scores the candidate surface. The update is accepted only
if both critiques parsed and the candidate scores strictly higher (and not below the
node's validation floor). The two critiques used to score the same surface under a
"free"/"nudged" label, so about half of all updates were accepted on noise; a missing
critic payload also read as `stable`, i.e. convergence, and string scores raised.
Unparsed critiques now skip the update (`reason: critic_unparsed`). The state is saved
after every learning pass; a failed save is recorded as `adaptive_tep_save_error` on
the attempt instead of being swallowed (ATT-17).

### Optimization Layers: Bandits vs TEP

The adaptive stack now has two distinct learning layers:

1. **Bandit controllers** optimize **action selection and resource allocation**.
   They decide whether to invoke helper behavior, which evidence families are
   allowed, and whether to escalate budget.
2. **TEP full mode** optimizes **stage-local textual parameters** in the native
   repair graph. It rewrites prompt surfaces for Stage A request generation,
   context compaction, retry planning, and patch proposal.

These mechanisms are complementary rather than interchangeable.

- The bandit layers learn **what to do next**.
- TEP learns **how each native textual stage should behave once that path is
  chosen**.
- TEP can therefore improve the adaptive pipeline indirectly, but it does not
  directly update the LinUCB or Thompson controller state, reward weights, or
  feature maps.

This distinction matters when interpreting results: a TEP gain should be read as
an improvement in prompt-conditioned stage behavior, not as evidence that the
helper or budget policies themselves have become better action selectors.

### Related Optimization Methods and Roadmap

Other prompt or agent optimization methods are relevant, but they fit different
layers of the stack:

- **PromptWizard / APO** are best treated as offline prompt-optimization
  baselines or initialization methods for a small number of prompt surfaces.
- **Trace** is the more natural future research target for end-to-end
  trajectory optimization because it can optimize heterogeneous workflow
  parameters from execution traces and rich feedback.

Current roadmap:

1. Keep the shipped bandit controllers as the online learners for helper,
   evidence, and budget allocation.
2. Use TEP full mode as the native prompt-state adaptation layer for Stage A,
   compaction, retry planning, and patch proposal.
3. Evaluate PromptWizard or APO as offline prompt bootstrapping baselines for
   those native prompt surfaces.
4. Prototype Trace-style end-to-end optimization in a separate research branch,
   with explicit trainable workflow parameters and trajectory-level feedback,
   rather than folding it into the shipped adaptive controllers immediately.

## References

- **BOAD**: Bandit Optimization for Autonomous Debugging. ICLR 2026.
- **OPTS**: Online Prompt Strategy Selection via Thompson Sampling. ACL 2025. arxiv:2503.01163
- **COBALT**: Contextual Bandits for Multi-turn Code LLM Generation. arxiv:2602.03806
- **TEP**: Textual Equilibrium Propagation. ICLR 2026. arxiv:2601.21064
- **Trace**: Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs. arxiv:2406.16218
- **PromptWizard**: PromptWizard: Task-Aware Prompt Optimization Framework. arxiv:2405.18369
- **APO**: Automatic Prompt Optimization with "Gradient Descent" and Beam Search. arxiv:2305.03495
- **BAMDP Shaping**: All Pseudo-rewards Are Valid Reward Shaping. ICLR 2025. arxiv:2409.05358
- **LinUCB**: Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation." WWW 2010.
