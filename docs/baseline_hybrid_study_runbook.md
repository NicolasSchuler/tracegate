# Baseline Hybrid Study Runbook

Last verified: March 29, 2026

> **Looking for a quick command?** See the [Master Runbook](runbook.md) for all commands across setup, quality, eval ops, and analysis in one place.

This is the exact execution path for the paper-facing three-arm baseline study:

- `raw baseline`
- `hybrid-canonical` using `baseline_hybrid_guidance_mode = "fair_v1"`
- `hybrid-tuned` using `baseline_hybrid_guidance_mode = "tuned_v1"`

The runbook assumes the shipped `study_*.template.toml` presets under
[`evals/configs`](../evals/configs).

## Guardrails

- Use the `study_*.template.toml` configs only.
- Keep the same endpoint, model, and case list across all three arms for a given baseline.
- Keep the pinned study budget: `k = 2`, `jobs = 2`.
- Keep hybrid runs on `conditions = "layered_adaptive_gated"`.
- Keep `adaptive_controller_lambda_rlm = false`.
- Do not treat this runbook as the right path for online-policy benchmarking.
  Learned `layered_adaptive_llm` controller studies should still run separately
  with `jobs = 1`.

## Prerequisites

- Baseline repos cloned:

  ```bash
  chmod +x baselines/setup_repos.sh
  ./baselines/setup_repos.sh
  ```

- Baseline envs created:

  ```bash
  bash baselines/setup_envs.sh
  ```

- Local OpenAI-compatible endpoint ready:

  ```bash
  curl http://127.0.0.1:12000/v1/models
  ```

- Case list prepared for the slice you want to run.

For paper-facing runs, prefer an explicit case list file over `--max-cases`.

## Config Matrix

| Baseline | Raw | Hybrid canonical | Hybrid tuned |
|----------|-----|------------------|--------------|
| `ldb` | [`study_ldb_raw.template.toml`](../evals/configs/study_ldb_raw.template.toml) | [`study_ldb_hybrid_canonical.template.toml`](../evals/configs/study_ldb_hybrid_canonical.template.toml) | [`study_ldb_hybrid_tuned.template.toml`](../evals/configs/study_ldb_hybrid_tuned.template.toml) |
| `intervenor` | [`study_intervenor_raw.template.toml`](../evals/configs/study_intervenor_raw.template.toml) | [`study_intervenor_hybrid_canonical.template.toml`](../evals/configs/study_intervenor_hybrid_canonical.template.toml) | [`study_intervenor_hybrid_tuned.template.toml`](../evals/configs/study_intervenor_hybrid_tuned.template.toml) |
| `tracecoder` | [`study_tracecoder_raw.template.toml`](../evals/configs/study_tracecoder_raw.template.toml) | [`study_tracecoder_hybrid_canonical.template.toml`](../evals/configs/study_tracecoder_hybrid_canonical.template.toml) | [`study_tracecoder_hybrid_tuned.template.toml`](../evals/configs/study_tracecoder_hybrid_tuned.template.toml) |
| `lambda_rlm` | [`study_lambda_rlm_raw.template.toml`](../evals/configs/study_lambda_rlm_raw.template.toml) | [`study_lambda_rlm_hybrid_canonical.template.toml`](../evals/configs/study_lambda_rlm_hybrid_canonical.template.toml) | [`study_lambda_rlm_hybrid_tuned.template.toml`](../evals/configs/study_lambda_rlm_hybrid_tuned.template.toml) |
| `rlm` | [`study_rlm_raw.template.toml`](../evals/configs/study_rlm_raw.template.toml) | [`study_rlm_hybrid_canonical.template.toml`](../evals/configs/study_rlm_hybrid_canonical.template.toml) | [`study_rlm_hybrid_tuned.template.toml`](../evals/configs/study_rlm_hybrid_tuned.template.toml) |

## Shared Variables

Set these once before queueing runs:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:12000/v1"
export OPENAI_MODEL="<model-name>"
export CASE_LIST="evals/artifacts/case_lists/<slice>.txt"
export RUN_TAG="<slice>"
```

Suggested `RUN_TAG` values:

- `failed10`
- `matched157`

Example:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:12000/v1"
export OPENAI_MODEL="Qwen3.5-35B-A3B-UD-Q4_K_M"
export CASE_LIST="evals/artifacts/case_lists/failed10.txt"
export RUN_TAG="failed10"
```

## Run One Config

Single foreground run:

```bash
uv run python -m evals.run_eval \
  --config evals/configs/study_ldb_hybrid_canonical.template.toml \
  --run-id "${RUN_TAG}-ldb-hybrid-canonical" \
  --case-list-file "$CASE_LIST" \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL"
```

The config already pins the study defaults. Do not override `--k`, `--jobs`,
or adaptive controller flags unless you are intentionally leaving the exact
study protocol.

## Queue the Full 15-Run Matrix

Use `pueue` to queue the matrix without opening 15 terminals:

```bash
BASELINES=(ldb intervenor tracecoder lambda_rlm rlm)
ARMS=(raw hybrid_canonical hybrid_tuned)

for baseline in "${BASELINES[@]}"; do
  for arm in "${ARMS[@]}"; do
    cfg="evals/configs/study_${baseline}_${arm}.template.toml"
    run_id="${RUN_TAG}-${baseline}-${arm//_/-}"

    pueue add -- uv run python -m evals.run_eval \
      --config "$cfg" \
      --run-id "$run_id" \
      --case-list-file "$CASE_LIST" \
      --openai-base-url "$OPENAI_BASE_URL" \
      --openai-model "$OPENAI_MODEL"
  done
done
```

Monitor the queue:

```bash
pueue status
pueue log <task_id>
pueue follow <task_id>
```

If you want a smaller smoke matrix, queue only one baseline first:

```bash
for arm in raw hybrid_canonical hybrid_tuned; do
  cfg="evals/configs/study_ldb_${arm}.template.toml"
  run_id="${RUN_TAG}-ldb-${arm//_/-}"

  pueue add -- uv run python -m evals.run_eval \
    --config "$cfg" \
    --run-id "$run_id" \
    --case-list-file "$CASE_LIST" \
    --openai-base-url "$OPENAI_BASE_URL" \
    --openai-model "$OPENAI_MODEL"
done
```

## Summarize Results

Per-run summaries:

```bash
just eval-summary "${RUN_TAG}-ldb-raw"
just eval-summary "${RUN_TAG}-ldb-hybrid-canonical"
just eval-summary "${RUN_TAG}-ldb-hybrid-tuned"
```

Machine-readable aggregation over the slice:

```bash
uv run python -m evals.analyze_results \
  "evals/results/${RUN_TAG}-*.jsonl" \
  --out "evals/results/${RUN_TAG}.summary.json"
```

## Important Comparison Note

The two hybrid arms both run under `conditions = "layered_adaptive_gated"`.
They are separated by:

- `run_id`
- `baseline_hybrid_guidance_mode`

That means:

- per-run summaries are unambiguous
- mixed-file condition-only comparisons are not sufficient to distinguish
  `hybrid-canonical` from `hybrid-tuned`

If you are building paper tables, compare runs by explicit run ID or by
`baseline_hybrid_guidance_mode`, not by condition label alone.
