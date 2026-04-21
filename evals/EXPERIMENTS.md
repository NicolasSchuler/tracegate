# Experiment Log

Chronological record of A/B evaluation runs comparing **traceback-only** (baseline) vs **with-snapshot** (treatment) conditions.

Each entry contains enough information to reproduce the run. Raw data is in `evals/results/<run_id>.jsonl` and artifacts in `evals/artifacts/<run_id>/`.

Historical experiment entries below keep the case counts from when each run was executed.
For the current repository-wide case inventory, see `evals/README.md`.

---

## Experiment 1: Qwen 2.5 Coder 3B (Local)

**Date:** 2025-02-12
**Run ID:** `20260212T103935-ea51e6cf`
**Results file:** `evals/results/20260212T103935-ea51e6cf.jsonl`

### Setup

| Parameter | Value |
|-----------|-------|
| Model | `qwen2.5-coder-3b-instruct-mlx` |
| Serving | LM Studio (local), `http://127.0.0.1:1234` |
| Context window | 32,768 tokens (LM Studio setting) |
| Response mode | `edits` (JSON find/replace) |
| Temperature | 0.0 |
| Max output tokens | 1,024 |
| Max context chars | 30,000 |
| Patch attempts (k) | 1 |
| Cases | 87 (all — 46 hand-crafted, 41 DebugBench) |
| Structured output | Disabled (LM Studio errors with it enabled) |
| llmdebug version | 2.5.1 |
| Python | 3.13.5 |
| Platform | macOS 26.2, Apple Silicon (arm64) |

### Reproduction command

```bash
PYTHONPATH=. uv run python evals/run_eval.py \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model qwen2.5-coder-3b-instruct-mlx \
  --openai-response-mode edits \
  --openai-max-tokens 1024 \
  --max-context-chars 30000 \
  --k 1 \
  --record-prompt full
```

### Results: Overall

| Condition | Successes | Rate |
|-----------|-----------|------|
| Traceback-only | 21/87 | 24.1% |
| With-snapshot | 20/87 | 23.0% |
| **Uplift** | **-1** | **-1.1pp** |

### Results: Statistical analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| McNemar's p-value | 1.0000 | No significant difference |
| Discordant pairs | 9 (5 TB wins, 4 WS wins) | Nearly even split |
| Cohen's h | -0.027 | Negligible effect size |
| Bootstrap 95% CI | [-8.0%, +5.7%] | Interval includes zero |
| Power | 0.03 | Severely underpowered |

### Results: By category

| Category | n | TB rate | WS rate | Uplift |
|----------|---|---------|---------|--------|
| async_temporal | 7 | 0.0% | 14.3% | +14.3pp |
| cross_file_contract | 4 | 50.0% | 25.0% | -25.0pp |
| data_shape_runtime | 6 | 50.0% | 50.0% | 0.0pp |
| hidden_state | 4 | 25.0% | 25.0% | 0.0pp |
| logic_error | 10 | 10.0% | 20.0% | +10.0pp |
| mutability_aliasing | 4 | 0.0% | 0.0% | 0.0pp |
| parse_value | 5 | 80.0% | 40.0% | -40.0pp |
| path_import_env | 4 | 0.0% | 25.0% | +25.0pp |
| reference_error | 15 | 40.0% | 40.0% | 0.0pp |
| syntax_error | 19 | 5.3% | 5.3% | 0.0pp |
| traceback_controls | 4 | 50.0% | 25.0% | -25.0pp |
| type_coercion | 5 | 20.0% | 20.0% | 0.0pp |

### Results: By snapshot dependency

| Dependency | n | TB rate | WS rate | Uplift |
|------------|---|---------|---------|--------|
| required | 26 | 26.9% | 23.1% | -3.8pp |
| helpful | 23 | 21.7% | 26.1% | +4.3pp |
| none | 38 | 23.7% | 21.1% | -2.6pp |

### Results: By difficulty

| Difficulty | n | TB rate | WS rate | Uplift |
|------------|---|---------|---------|--------|
| easy | 43 | 25.6% | 16.3% | -9.3pp |
| medium | 30 | 23.3% | 33.3% | +10.0pp |
| hard | 14 | 21.4% | 21.4% | 0.0pp |

### Token usage

| Metric | Value |
|--------|-------|
| Total prompt tokens | 170,491 |
| Total completion tokens | 11,708 |
| Mean prompt tokens/case | ~982 |
| Mean completion tokens/case | ~67 |

### Observations

1. **No detectable effect.** The overall uplift is -1.1pp with p=1.0 — the snapshot provides no measurable benefit with this model.

2. **Model is too weak for the task.** 24% baseline success rate with only 9 discordant pairs out of 87 means the model fails regardless of what information it receives. The `find_not_found` error (model hallucinates code strings that don't exist in the file) accounts for 11/87 traceback-only and 9/87 with-snapshot failures — a fundamental capability limitation.

3. **Suggestive patterns in medium difficulty.** Medium cases show +10pp uplift (23.3% → 33.3%). These are cases where the model sometimes can and sometimes can't fix the bug, so extra context might help. However, with n=30 this is not statistically significant.

4. **Per-category numbers are noise.** With n=4-19 per category, the per-category uplifts (ranging from -40pp to +25pp) are dominated by sampling variance.

5. **Hypothesis for next experiment.** A model with 40-60% baseline success rate is needed to detect a 10-15% uplift with adequate power. This requires a 14B+ parameter model or a frontier API model.

---

## Experiment 2: GLM-4.5-Air (z.ai Coding Plan)

**Date:** 2025-02-12
**Run ID:** `20260212T111409-1d6dc749`
**Results file:** `evals/results/20260212T111409-1d6dc749.jsonl`

### Setup

| Parameter | Value |
|-----------|-------|
| Model | `glm-4.5-air` |
| Serving | z.ai Coding Plan API, `https://api.z.ai/api/coding/paas/v4` |
| Response mode | `edits` (JSON find/replace) |
| Temperature | 0.0 |
| Max output tokens | 2,048 |
| Max context chars | 60,000 |
| Patch attempts (k) | 1 |
| Cases | 87 (all — 46 hand-crafted, 41 DebugBench) |
| llmdebug version | 2.5.1 |
| Python | 3.13.5 |
| Platform | macOS 26.2, Apple Silicon (arm64) |

### Reproduction command

```bash
PYTHONPATH=. uv run python evals/run_eval.py \
  --patcher openai \
  --openai-base-url https://api.z.ai/api/coding/paas/v4 \
  --openai-model glm-4.5-air \
  --openai-api-key "$Z_API" \
  --openai-response-mode edits \
  --openai-max-tokens 2048 \
  --openai-timeout-sec 60 \
  --max-context-chars 60000 \
  --k 1 \
  --record-prompt full
```

### Results: Overall

| Condition | Successes | Rate |
|-----------|-----------|------|
| Traceback-only | 61/87 | 70.1% |
| With-snapshot | 60/87 | 69.0% |
| **Uplift** | **-1** | **-1.1pp** |

### Results: Statistical analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| McNemar's p-value | 1.0000 | No significant difference overall |
| Discordant pairs | 17 (9 TB wins, 8 WS wins) | Nearly even split |
| Cohen's h | -0.025 | Negligible effect size |
| Bootstrap 95% CI | [-10.3%, +8.0%] | Interval includes zero |
| Power | 0.03 | Underpowered for overall effect |

### Results: By category

| Category | n | TB rate | WS rate | Uplift | Cohen's h |
|----------|---|---------|---------|--------|-----------|
| **async_temporal** | **7** | **42.9%** | **85.7%** | **+42.9pp** | **0.939** |
| cross_file_contract | 4 | 100.0% | 100.0% | 0.0pp | 0.000 |
| data_shape_runtime | 6 | 100.0% | 100.0% | 0.0pp | 0.000 |
| hidden_state | 4 | 100.0% | 100.0% | 0.0pp | 0.000 |
| logic_error | 10 | 70.0% | 50.0% | -20.0pp | -0.412 |
| mutability_aliasing | 4 | 100.0% | 100.0% | 0.0pp | 0.000 |
| parse_value | 5 | 100.0% | 100.0% | 0.0pp | 0.000 |
| path_import_env | 4 | 0.0% | 0.0% | 0.0pp | 0.000 |
| reference_error | 15 | 80.0% | 80.0% | 0.0pp | 0.000 |
| syntax_error | 19 | 42.1% | 31.6% | -10.5pp | -0.219 |
| traceback_controls | 4 | 75.0% | 75.0% | 0.0pp | 0.000 |
| type_coercion | 5 | 100.0% | 100.0% | 0.0pp | 0.000 |

### Results: By snapshot dependency

| Dependency | n | TB rate | WS rate | Uplift |
|------------|---|---------|---------|--------|
| **required** | **26** | **73.1%** | **84.6%** | **+11.5pp** |
| helpful | 23 | 82.6% | 73.9% | -8.7pp |
| none | 38 | 60.5% | 55.3% | -5.3pp |

### Results: By difficulty

| Difficulty | n | TB rate | WS rate | Uplift |
|------------|---|---------|---------|--------|
| easy | 43 | 67.4% | 62.8% | -4.7pp |
| medium | 30 | 80.0% | 76.7% | -3.3pp |
| **hard** | **14** | **57.1%** | **71.4%** | **+14.3pp** |

### Results: By bug source

| Source | n | TB rate | WS rate | Uplift |
|--------|---|---------|---------|--------|
| hand_crafted | 46 | 80.4% | 84.8% | +4.3pp |
| debugbench | 41 | 58.5% | 51.2% | -7.3pp |

### Token usage

| Metric | Value |
|--------|-------|
| Total prompt tokens | 165,937 |
| Total completion tokens | 112,707 |
| Mean tokens/record | ~1,600 prompt, ~648 completion |

### Discordant pair analysis

All 17 cases where the two conditions disagreed:

| Case | Winner | Category | Snap Dep |
|------|--------|----------|----------|
| async_lock_reentrant_medium | **WS** | async_temporal | required |
| async_missing_await_chain_easy | **WS** | async_temporal | required |
| async_task_order_race_hard | **WS** | async_temporal | required |
| db_2862_easy | **WS** | syntax_error | none |
| db_2870_easy | **WS** | syntax_error | none |
| db_2887_hard | **WS** | logic_error | helpful |
| db_2908_easy | **WS** | reference_error | none |
| db_2936_hard | **WS** | reference_error | none |
| db_2841_easy | TB | syntax_error | none |
| db_2859_easy | TB | syntax_error | none |
| db_2860_easy | TB | syntax_error | none |
| db_2876_easy | TB | syntax_error | none |
| db_2889_hard | TB | logic_error | helpful |
| db_2897_medium | TB | logic_error | helpful |
| db_2923_easy | TB | reference_error | none |
| db_2939_easy | TB | reference_error | none |
| logic_negative_modulo_medium | TB | logic_error | helpful |
| parse_decimal_comma_easy | TB | parse_value | helpful |

### Observations

1. **No overall effect, but strong subgroup signal.** The aggregate -1.1pp masks a striking category-level pattern: the snapshot produces a **+42.9pp uplift on async_temporal cases** (Cohen's h = 0.939, a large effect). The model goes from 3/7 to 6/7 on async bugs when given runtime state.

2. **Snapshot dependency predicts uplift direction.** Cases labeled `snapshot_dependency=required` show +11.5pp uplift (73.1% → 84.6%). Cases labeled `none` show -5.3pp. This validates the case labeling scheme — the snapshot helps precisely where we designed it to help.

3. **Hard cases benefit most.** The +14.3pp uplift on hard cases (57.1% → 71.4%) suggests snapshots help when the traceback alone is insufficient for diagnosis, which is more likely for harder bugs.

4. **Ceiling effect on 7 categories.** The model achieves 100% on both conditions for data_shape_runtime, hidden_state, mutability_aliasing, parse_value, cross_file_contract, and type_coercion. These categories can't show uplift because there's no room to improve. This inflates the concordant pair count and dilutes the overall statistic.

5. **DebugBench cases show regression.** The -7.3pp on DebugBench cases may reflect the snapshot adding noise for algorithm-level bugs (LeetCode-style) where runtime locals don't illuminate the logic error. The hand-crafted cases (+4.3pp) were specifically designed to test runtime-state-dependent bugs.

6. **Discordant pair pattern.** Of the 8 WS wins, 3 are async (snap_dep=required) and the rest are scattered. Of the 9 TB wins, 4 are syntax errors (snap_dep=none) and the rest are logic/reference errors from DebugBench. The snapshot helps on runtime-state bugs and adds marginal noise on syntactic/algorithmic bugs.

7. **Statistical power remains low.** With only 17 discordant pairs, McNemar's test is underpowered. A per-category McNemar on async_temporal alone (3 discordant, all WS wins) yields p=0.25 — suggestive but not significant at n=7. More async/runtime-state cases are needed.

### Hypothesis for next experiment

The overall null result hides a meaningful category-specific effect. To isolate and power the signal:
- Add more cases in `async_temporal`, `hidden_state`, and `mutability_aliasing` categories (currently only 7, 4, and 4 cases respectively)
- Run with k=3 to increase discordant pair count through repeated sampling
- Consider a model with a lower baseline (50-60%) to avoid ceiling effects on easy categories
- Test with a second model to confirm the pattern generalizes
