# Experiment Protocol: Cross-Baseline Comparison

This document specifies the fair comparison protocol for ISSRE paper
experiments.  All conditions — llmdebug's own and external baselines —
run under identical constraints to ensure a valid head-to-head evaluation.

## 1. Experimental Design

### Conditions

| ID | Condition | Source | Description |
|----|-----------|--------|-------------|
| `base_repair` | Base repair | llmdebug | Standard Python traceback + buggy code |
| `cot_repair` | CoT repair | llmdebug | Base repair + chain-of-thought reasoning strategy |
| `layered_adaptive_llm` | Layered adaptive LLM | llmdebug | Structured debug snapshot with online LinUCB policy |
| `ldb` | LDB | External | Block-level runtime tracing + iterative debug |
| `intervenor` | INTERVENOR | External | Teacher-learner chain-of-repair |

### Optional Conditions

| ID | Condition | Source | Description |
|----|-----------|--------|-------------|
| `layered_adaptive_gated` | Adaptive gated | llmdebug | Two-stage adaptive pipeline |
| `tracecoder` | TraceCoder | External | Repair-seeded trace-driven iterative repair |

## 2. Controlled Variables

These are held constant across all conditions:

| Variable | Value | Rationale |
|----------|-------|-----------|
| **LLM endpoint** | Same OpenAI-compatible server | Eliminates model variation |
| **Model** | Same model for all conditions | Fair prompt-to-output comparison |
| **Temperature** | 0.0 | Deterministic generation |
| **Max repair attempts** | k=2 | Budget fairness; no multi-candidate |
| **Execution sandbox** | Docker isolation | Identical security and environment |
| **Test framework** | pytest via eval harness | Same scoring for all |
| **Case set** | Full `evals/cases/` directory | No cherry-picking |
| **Timeout** | 300s per case | Prevents runaway cost |

### What Differs (Independent Variable)

The only intentional difference is **what debugging information the LLM
receives** (or how it processes the buggy code):

- **base_repair**: Python traceback text
- **cot_repair**: Python traceback text + chain-of-thought reasoning prompt
- **layered_adaptive_llm**: Structured llmdebug snapshot (frames, locals, etc.) with online policy
- **ldb**: LDB's block-level runtime traces (printed variable values)
- **intervenor**: INTERVENOR's teacher analysis → learner repair chain

## 3. Case Selection

All cases from `evals/cases/` are included. No filtering by difficulty,
category, or snapshot dependency is required at recording time; baseline
incompatibilities are preserved as explicit failures in the result logs.

Cases are tagged with metadata for subgroup analysis:
- `category`: Bug type (logic, type_confusion, off_by_one, etc.)
- `difficulty`: easy / medium / hard
- `snapshot_dependency`: required / helpful / none
- `bug_source`: hand_crafted / debugbench / humanevalpack / condefects / mutmut

### Baseline Compatibility Notes

Some cases are incompatible with external baselines and are recorded as
explicit baseline failures rather than being shimmed into a different task:

- **LDB** only accepts a strict callable subset: synchronous `from buggy
  import <function>` cases that can be lowered into upstream-compatible
  `check(candidate)` and `given_tests` artifacts.
- **INTERVENOR** has broader coverage because the adapter copies the full case
  workspace and runs the case's native test command locally, but it still
  depends on a runnable benchmark workspace and command.

These compatibility boundaries are part of the comparison and should be
reported, not hidden.

- **TraceCoder** is repair-seeded on llmdebug: it starts from the supplied
  `buggy.py`, then runs the patched upstream trace/analyze/repair loop on a
  copied workspace via the case's native pytest command.

Baseline-only runs do not inject llmdebug controller/evidence/history/verifier
sidecars. Hybrid adaptive/controller-backed runs may inject additive
`llmdebug_*` sidecars into baseline adapter inputs, but that is a distinct
hybrid condition rather than the default external-baseline path.

## 4. Prompt Fairness

Each condition receives the information it would normally get:

| Condition | Prompt contains |
|-----------|----------------|
| base_repair | Buggy code + traceback text |
| cot_repair | Buggy code + traceback text + CoT reasoning prompt |
| layered_adaptive_llm | Buggy code + structured snapshot JSON |
| ldb | (LDB manages its own prompts internally) |
| intervenor | (INTERVENOR manages teacher/learner prompts) |

External baselines manage their own prompt construction.  This is fair
because the experimental question is "does the *system* (including its
prompt strategy) produce better repairs?"

Timeout-safe telemetry sidecars are also limited to the patched llmdebug
support for **LDB** and **TraceCoder**. They are a compatibility/recovery layer
for those patched upstream integrations, not a general sidecar behavior shared
by every baseline.

Baseline integrations now emit explicit contract metadata (`contract_version`,
`fidelity_level`, `repair_scope_policy`, `prompt_contract_kind`,
`upstream_commit`) so analysis can distinguish upstream-patched baselines from
prompt ports and generic API adaptations. See `docs/baseline_contracts.md`.

The first adapter-owned behavioral knobs are also explicit now rather than
hardcoded: INTERVENOR exposes teacher/learner token limits, controller-guidance
budgets, and output-contract suffix control; TraceCoder exposes controller
guidance budget and max no-improvement streak. Defaults preserve current
behavior and are documented in `docs/baseline_contracts.md` and the baseline
template configs under `evals/configs/`.

The same now applies to the generic recursive baselines: lambda-RLM and RLM
expose controller-guidance and failure-feedback truncation budgets explicitly,
so their prompt framing is versioned and reviewable rather than being hidden in
adapter-local constants.

### Secondary Tuned-Baseline Track

Canonical baseline runs remain the paper-primary comparison. Secondary
`repair_fair_v1` variants for lambda-RLM and RLM are reported separately as
tuned-baseline runs because they intentionally change baseline-owned prompt
framing and, for RLM, its native inner search budget. These tuned variants are
useful for practical timeout/churn analysis, but they do not replace the
canonical baseline tables.

## 5. Scoring

All conditions are scored identically:

1. The condition produces a unified diff (patch)
2. The patch is applied to `buggy.py` in a fresh Docker container
3. `pytest test_buggy.py -q` is run in the container
4. **Pass**: all tests pass → `success = true`
5. **Fail**: any test fails, patch doesn't apply, or timeout → `success = false`

No partial credit.  Binary pass/fail on the hidden test suite.

## 6. Statistical Analysis

### Primary Endpoint

Success rate (proportion of cases where the repaired code passes all tests).

### Pairwise Tests

For each pair (reference, treatment):
- **McNemar's test** (exact binomial when discordant < 25, χ² otherwise)
- **Cohen's h** effect size with magnitude labels
- **Bootstrap 95% CI** for the success-rate uplift (10,000 resamples)
- **Holm-Bonferroni correction** across all pairwise tests

### Subgroup Analysis

Per-category and per-difficulty breakdowns with per-category
Holm-Bonferroni correction.

### Efficiency Metrics (Secondary)

- Token usage (prompt + completion)
- Wall-clock time per case
- Iterations/attempts used

## 7. Reproduction Commands

### Setup

```bash
# Clone external baseline repos (pinned commits).
bash baselines/setup_repos.sh

# Create isolated baseline runtime envs.
bash baselines/setup_envs.sh

# Install dev dependencies for the main repo.
uv sync --group dev
```

### Run All Conditions

```bash
# Reference: base-repair
python -m evals.run_eval \
  --patcher command \
  --conditions base_repair \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <MODEL> \
  --k 2 --run-id issre_run_001

# Treatment: Layered adaptive LLM
python -m evals.run_eval \
  --patcher command \
  --conditions layered_adaptive_llm \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <MODEL> \
  --k 2 --run-id issre_run_001

# Baseline: LDB
python -m evals.run_eval \
  --baseline ldb \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <MODEL> \
  --k 2 --run-id issre_run_001

# Baseline: INTERVENOR
python -m evals.run_eval \
  --baseline intervenor \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <MODEL> \
  --k 2 --run-id issre_run_001

# Baseline: TraceCoder
python -m evals.run_eval \
  --baseline tracecoder \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <MODEL> \
  --k 2 --run-id issre_run_001
```

The shorthand `--baseline <name>` is the preferred invocation for all external
baselines. It resolves the dedicated baseline condition automatically and uses
the standard cloned repo and isolated env locations when available.

### Analysis

```bash
# Per-condition summary.
python -m evals.eval_summary issre_run_001

# Pairwise report (llmdebug vs base-repair).
python -m evals.report_run_metrics \
  --jsonl 'evals/results/issre_run_001*.jsonl' \
  --a base_repair --b layered_adaptive_llm

# Cross-baseline comparison.
python -m baselines.compare_results \
  --jsonl 'evals/results/issre_run_001*.jsonl' \
  --conditions base_repair cot_repair layered_adaptive_llm ldb intervenor tracecoder \
  --reference base_repair \
  --primary layered_adaptive_llm \
  --bootstrap 10000 --seed 42 \
  --out reports/issre_comparison.json \
  --latex reports/issre_table.tex
```

## 8. Reporting

### Figures and Tables

1. **Table 1**: Overall success rates for all conditions (with McNemar p, Cohen's h)
2. **Table 2**: Per-category breakdown (categories with ≥5 cases)
3. **Figure 1**: Success-rate bar chart with bootstrap 95% CI error bars
4. **Table 3**: Efficiency metrics (median tokens, median time, median attempts)

### What to Report in the Paper

- Number of cases per condition (should be identical for all)
- Number of cases where each baseline produced no output (conversion failures)
- Exact model name and endpoint used
- Pinned commit hashes for external baselines
- Any patches applied to external baseline code

## 9. Ethical Considerations

### Licensing

- **LDB**: MIT license ✅ — can be freely used and reported
- **INTERVENOR**: No license file ⚠️ — code not redistributed; used for
  evaluation only under fair-use research exemption
- **TraceCoder**: No license file ⚠️ — same as INTERVENOR

External repos are **not** committed to this repository.  They are cloned
at experiment time from their public GitHub repositories.

### Reproducibility

All external baselines are pinned to specific commit hashes in
`baselines/setup_repos.sh`.  The comparison protocol, analysis code, and
result JSONL files are committed to this repository for full reproducibility.
