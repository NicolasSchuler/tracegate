# Evaluation Framework Metrics - Quick Reference

## All Current Metrics Tracked

### Per-Attempt Timing (in JSONL records)

| Metric | Path | When | Granularity | Unit |
|--------|------|------|-------------|------|
| `before_duration_sec` | `run_eval.py:3671` | Always | Per attempt | seconds |
| `after_duration_sec` | `run_eval.py:3678` | If patched | Per attempt | seconds |
| `attempt_wall_clock_sec` | `run_eval.py:3673` | Always | Per attempt | seconds |
| `propose_duration_sec` | `run_eval.py:8969` | If proposed | Per patch call | seconds |
| `request_context_duration_sec` | `openai_compat_patcher.py:737` | If called | Per stage A | seconds |

### Timing Breakdown Semantics (in `attempt_metadata.timing_breakdown`)

| Metric | Meaning |
|--------|---------|
| `stage_b_llm_sec` | Pure Stage-B LLM time. For baseline patchers, this is the baseline-reported internal LLM time, not outer subprocess wall time. |
| `baseline_internal_total_sec` | Total time spent inside the baseline repair loop. |
| `baseline_eval_sec` | Time the baseline spent running its own internal checks/tests. |
| `baseline_orchestration_sec` | Baseline-internal non-LLM, non-eval orchestration time. |
| `llm_total_sec` | Pure LLM time across stage A, stage B, and context compaction. |
| `overhead_sec` | Residual framework overhead after subtracting pytest, pure LLM, and baseline-internal eval/orchestration. |
| `adaptive_controller_prepare_sec` | Time spent building the adaptive controller runtime for hybrid baseline runs before proposal starts. |
| `adaptive_controller_summarizer_sec` | Time spent generating the controller strategy summary used by hybrid baseline sidecars. |
| `adaptive_controller_sidecar_build_sec` | Time spent materializing hybrid baseline sidecar payloads for a syntax cycle. |
| `adaptive_controller_sidecar_write_sec` | Time spent writing hybrid baseline sidecar files for a syntax cycle. |
| `baseline_case_conversion_sec` | Time spent converting an eval case into the baseline adapter input. |
| `baseline_sidecar_load_sec` | Time spent loading hybrid sidecars and wiring hybrid guidance inside the baseline patcher. |

### Token Usage (in attempt_metadata)

| Metric | Path | When | Granularity | Type |
|--------|------|------|-------------|------|
| `llm_call_count` | `eval_recording.py:552` | Always | Cumulative | int |
| `tokens_prompt` | `eval_recording.py:553` | Always | Cumulative | int |
| `tokens_completion` | `eval_recording.py:554` | Always | Cumulative | int |
| `tokens_total` | `eval_recording.py:555` | Always | Cumulative | int |

### Adaptive Policy Metrics (when active)

| Metric | Path | Policy | Type |
|--------|------|--------|------|
| `adaptive_online_local_llm_call_count` | `run_eval.py:4836` | Online | int |
| `adaptive_online_local_tokens_prompt` | `run_eval.py:4837` | Online | int |
| `adaptive_online_local_tokens_completion` | `run_eval.py:4838` | Online | int |
| `adaptive_online_local_tokens_total` | `run_eval.py:4839` | Online | int |
| `adaptive_online_local_wall_sec` | `run_eval.py:4840` | Online | float |
| `adaptive_online_reward` | `run_eval.py:4840+` | Online | float |
| `adaptive_budget_local_llm_call_count` | `run_eval.py:5542` | Budget | int |
| `adaptive_budget_local_tokens_prompt` | `run_eval.py:5543` | Budget | int |
| `adaptive_budget_local_tokens_completion` | `run_eval.py:5544` | Budget | int |
| `adaptive_budget_local_tokens_total` | `run_eval.py:5545` | Budget | int |
| `adaptive_budget_local_wall_sec` | `run_eval.py:5546` | Budget | float |
| `adaptive_budget_reward` | `run_eval.py:5546+` | Budget | float |

### Stage A / Helper Metrics

| Metric | Path | Tracking |
|--------|------|----------|
| `stage_a_prompt_chars` | `eval_recording.py:546` | Prompt size |
| `stage_a_requests_count` | `eval_recording.py:547` | # of context requests |
| `stage_a_chars_returned` | `eval_recording.py:548` | Returned context size |
| `request_context_duration_sec` | `openai_compat_patcher.py:737` | Call duration |
| `request_context_retry_count` | `openai_compat_patcher.py:739` | Retry attempts |
| `request_context_transport_retry_count` | `openai_compat_patcher.py:740` | HTTP retries |
| `request_context_strict_retry_count` | `openai_compat_patcher.py:741` | JSON parse retries |

### Context Compaction Metrics (if used)

| Metric | Path |
|--------|------|
| `context_compaction_invoked` | `eval_recording.py:658` |
| `context_compaction_success` | `eval_recording.py:659` |
| `context_compaction_calls_used` | `eval_recording.py:661` |
| `context_compaction_input_chars` | `eval_recording.py:666` |
| `context_compaction_output_chars` | `eval_recording.py:668` |

---

## Key Functions for Metrics

### Token Extraction

**File**: `openai_compat_patcher.py`

```python
def _usage_counts_from_payload(usage: Any) -> tuple[int, int, int]:
    # Lines 333-345
    # Returns: (prompt_tokens, completion_tokens, total_tokens)
    # Extracts from OpenAI response usage object
```

**File**: `run_eval.py`

```python
def _sync_tokens(attempt_meta, llm_call_count, tokens_prompt, tokens_completion, tokens_total):
    # Lines 3247-3259
    # Updates attempt_metadata with token counts
```

### Timing Measurement

**File**: `run_eval.py`

```python
def _measure_attempt_wall_clock_sec(attempt_start_monotonic: float) -> float:
    # Lines 3691-3692
    # return max(0.0, float(time.perf_counter() - attempt_start_monotonic))
```

### Reward Computation (Includes Timing)

**File**: `run_eval.py`

```python
def _adaptive_online_reward_signal(
    success: bool,
    progress_score: float,
    local_tokens_total: int,
    local_wall_sec: float,
    lambda_tokens: float = 0.08,
    mu_latency: float = 0.05,
    token_scale: float = 1000.0,
    latency_scale: float = 10.0,
) -> float:
    # Lines 4430-4500
    # reward = success - lambda*(tokens/token_scale) - mu*(latency/latency_scale)
```

---

## Where Metrics Are Computed

### Before Stage A
```
attempt_start = time.perf_counter()  # line 8741, 9453
↓
before_pytest_run() → before_duration_sec
↓
Evidence collection → stage_a_prompt_chars
```

### Stage A (Context Request)
```
helper_local_started_at = time.perf_counter()  # line 7383
↓
request_context() calls openai_compat_patcher
  ├─ request_context_duration_sec (line 737)
  ├─ request_context_retry_count (line 739)
  └─ tokens extracted from response usage (line 333-345)
↓
execute_context_requests()
  └─ stage_a_chars_returned
↓
helper_local_wall_sec = time.perf_counter() - helper_local_started_at  # line 7544
```

### Patch Proposal
```
patch_start = time.perf_counter()  # line 8861
↓
propose_patch() via openai_compat_patcher
  ├─ propose_duration_sec = time.perf_counter() - patch_start  # line 8871
  └─ tokens extracted from response usage
↓
Accumulated: llm_call_count, tokens_*
```

For external baselines, `propose_duration_sec` remains the outer patcher wall time for audit/debugging, but `stage_b_llm_sec` is sourced from baseline-internal telemetry when available.

### After Stage B
```
after_pytest_run() → after_duration_sec
↓
attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
↓
All metrics → attempt_metadata
```

---

## Statistical Conventions

One implementation in `evals/stats.py` serves `analyze_results --stats`,
`report_run_metrics` and `baselines/compare_results` (2026-10-04 review, Phase 2).

| Topic | Convention |
|-------|------------|
| Case outcome | `evals/case_outcome.py` is the one definition, used by `analyze_results`, `report_run_metrics`, `select_cases` and `extract_failed_cases`: `infra_event` rows aren't attempts, one row per attempt number (a real outcome beats a resume-eligible `unit_runtime_error` row, otherwise the later row), a case succeeded when any scored attempt succeeded, and the terminal row is the highest attempt (STA-19, DAT-24). |
| Condition names | `normalize_condition_name` strips names; `condition_family` (`native`, `adaptive`, `external_baseline`, `unknown`) comes from the `CONDITION_FAMILIES` registry in `evals/eval_types.py`, not from substrings. |
| Token coverage | An attempt with `token_usage_available: false` is unmeasured, not 0 tokens. Case rows carry `token_usage = None` when no attempt was measured (STA-01), `token_covered_attempts`, `token_covered_fraction`, and `token_usage_per_attempt_mean` = tokens / covered attempts (STA-09); conditions report `token_covered_cases`, `token_covered_attempts` and `token_covered_fraction`. Means skip unmeasured cases. |
| Unit of analysis | The case. Each case contributes one paired observation however many runs it has (STA-02). McNemar uses one decided outcome per case: `majority_vote` (default in `analyze_paired_results`; a tie on an even number of runs leaves the case out and is counted in `tied_cases`, STA-14) or `any_success`; `per_run` refuses inputs with several runs of a case. Bootstrap CIs resample cases with their run-averaged success rates. `report_run_metrics` averages token, runtime and success per case across runs and uses the majority outcome for kappa. |
| Paired success test | Exact McNemar: two-sided `scipy.stats.binomtest` on the discordant pairs at every sample size; significant when `p <= alpha` (default 0.05, `alpha` recorded in the result). The chi-square approximation above 25 discordant pairs is gone (STA-23). |
| Paired sign test | The same exact binomial helper, `exact_binomial_p_value`. |
| Matched case sets | Per-condition success rates stay over each condition's own cases, but every uplift (`pairwise_uplifts`, `uplift_success_rate`, `uplift_layered_*_vs_base`) is the mean per-case difference over the cases both conditions ran, with run-averaged success per case. Entries carry `n_matched_cases` and `case_sets_match`; conditions carry `cases_expected` (cases any condition ran in the bucket) and `cases_missing` (STA-03). |
| Share of runtime | `report_run_metrics` "% of runtime" is `sum(component) / sum(runtime_sec)` over the rows that have both values, zeros included, with the number of rows used (`pct_of_runtime_rows`); it used to divide a mean over some rows by a mean over others (STA-08). Timing fields are accumulated and displayed from one list, so the baseline internal/eval/orchestration times appear (STA-17). |
| Bucket adaptive rates | Bucket-level (overall, per-category, …) adaptive helper, online-policy and budget-policy rates count only rows with adaptive telemetry (an adaptive condition, or any condition run with an adaptive mode); `adaptive_rollup_conditions` and `adaptive_rollup_cases` name them (STA-16). |
| Common attempt budget | Conditions with different `max_attempts` are compared at `common_attempt_k`, the smallest attempt budget in the bucket: `success_within_common_k` is the share of cases solved within the first K attempts. `success_rate` stays each condition's own-budget rate. `evals.stats.pass_at_k(n, c, k)` is the unbiased estimator (Chen et al.) for independent samples such as repeated runs (STA-13, D12). |
| OPE samples | A sample without a logged behaviour propensity is dropped from IPS, SNIPS and DR and counted in `propensity_missing_dropped` (STA-12); shadow-mode samples logged before the POL-03 fix are read with propensity 1.0 and counted in `shadow_corrected_samples`. Both counts are printed before the estimates. |
| Power | `power_report(n_cases, n_01, n_10)`: normal approximation to the exact McNemar test at the observed discordance, `Phi(|n_01 - n_10| / sqrt(n_01 + n_10) - z_{alpha/2})`; within 0.02 of simulation at n=200 (STA-04). |
| Agreement | Cohen's kappa is `None` with magnitude `undefined` when it is not finite (both sides constant), never "almost_perfect" (STA-10). |
| Multiple comparisons | Holm step-down: adjusted p is the running max of `p * (m - rank + 1)`, capped at 1; rejected when adjusted p `<= alpha` (STA-24). Families: every condition pair in `analyze_results --stats` (`mcnemar_p_adjusted`, `holm_rejected`; STA-06), the per-category tests of one pair, and in `report_run_metrics` the three exploratory slices (All, Required-only, Controls), while the `--primary-slice` test is the pre-registered confirmatory endpoint (STA-20). |
| Confidence intervals | Percentile bootstrap, 10,000 resamples, seeded `random.Random`; default seed `DEFAULT_BOOTSTRAP_SEED = 42`. Results carry `method` and `seed`. `analyze_results --bootstrap-seed` sets it and records `bootstrap_seed`/`bootstrap_method` in `analysis_config`; `report_run_metrics --seed` does the same for its report (STA-05, STA-07). |
| p-value display | `format_p_value`: three significant digits, `<0.0001` below that, `n/a` when missing (STA-15). |

## Schema Validation

**File**: `analyze_results.py`

**Required token fields** (lines 65-79):
```python
REQUIRED_ATTEMPT_METADATA_FIELDS = (
    "attempt",
    "max_attempts",
    "attempt_ordinal_label",
    "llm_call_count",
    "tokens_prompt",
    "tokens_completion",
    "tokens_total",
    "retry_history_budget_chars",
    "retry_history_budget_applied",
    "prompt_xxh3_128",
    "prompt_path",
    "evidence_xxh3_128",
    "evidence_path",
)
```

---

## Gaps & Opportunities

### Currently NOT Tracked

1. **API Call Breakdown**:
   - Request body preparation time
   - Network latency
   - Response parsing time

2. **Helper Operations**:
   - File read time
   - Snapshot extraction time
   - Search/grep time
   - Index lookup time

3. **Policy Computation**:
   - LinUCB matrix computation
   - Feature scaling
   - Propensity calculation

4. **Test Operations**:
   - pytest collection time
   - Test setup/teardown time
   - Per-test execution time

5. **Context Compaction**:
   - Compaction LLM call duration
   - Compression ratio tracking

### Where to Add

**High Impact (Most Value)**:
1. `openai_compat_patcher.py` - API call breakdown
2. `run_eval.py:7357-7544` - Helper operation breakdown
3. `run_eval.py:10535` - Test execution details

**Medium Impact**:
4. `adaptive_online_policy.py` - Decision computation time
5. Context compaction duration

**Implementation Strategy**:
- Add to `PatcherMeta` (payload_types.py)
- Add to `AdaptivePolicyLocalAccounting` (eval_types.py)
- Add to `attempt_metadata` template (eval_recording.py)
- Update `EVAL_RECORD_SCHEMA_VERSION` when adding fields

---

## Example: Reading Results

```python
import json
from pathlib import Path

results_path = Path("evals/results/<run_id>.jsonl")
for line in results_path.read_text().splitlines():
    record = json.loads(line)
    
    # Top-level timing
    wall_sec = record["attempt_wall_clock_sec"]
    before_sec = record["before_duration_sec"]
    after_sec = record.get("after_duration_sec", 0)
    
    # Tokens
    attempt_meta = record["attempt_metadata"]
    tokens = attempt_meta["tokens_total"]
    calls = attempt_meta["llm_call_count"]
    
    # If adaptive
    if record["condition"] in ["layered_adaptive_gated", "layered_adaptive_llm"]:
        policy_tokens = attempt_meta.get("adaptive_online_local_tokens_total", 0)
        policy_wall = attempt_meta.get("adaptive_online_local_wall_sec", 0)
        reward = attempt_meta.get("adaptive_online_reward")
        print(f"Attempt {record['attempt']}: "
              f"tokens={tokens}, calls={calls}, wall={wall_sec:.2f}s, "
              f"policy_tokens={policy_tokens}, policy_wall={policy_wall:.2f}s, "
              f"reward={reward}")
```

---

## Files to Review

1. **Main Analysis**: `EVAL_FRAMEWORK_ANALYSIS.md` (706 lines)
2. **Core Orchestrator**: `/evals/run_eval.py` (11,384 lines)
3. **Metadata Builder**: `/evals/eval_recording.py` (926 lines)
4. **Patchers**: `/evals/patchers/openai_compat_patcher.py` (3,963 lines)
5. **Results Analysis**: `/evals/analyze_results.py` (1000+ lines)
6. **Schema**: `/evals/eval_types.py` (especially lines 447-457, 72)
