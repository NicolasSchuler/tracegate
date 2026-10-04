# Eval Configs

TOML files that configure eval runs — model endpoints, parameters, conditions,
and budget. Pass one to the eval harness with `--config`.

See [`evals/README.md`](../README.md) for full eval documentation.

## Available Configs

| Config | Condition | Use case |
|--------|-----------|----------|
| `openai_base_repair.latest.toml` | `base_repair` | Baseline: pytest output only, no reasoning chain, no adaptive helpers |
| `openai_base_repair_fair_v2.template.toml` | `base_repair` | Fairness-tuned native base_repair lane: keeps thinking on, raises Stage-B max tokens to `4096` |
| `openai_cot_repair.template.toml` | `cot_repair` | Native plan-then-patch: first-attempt repair planning transcript, then structured patching; also reachable via `baseline = "cot_repair"` |
| `openai_cot_repair_fair_v2.template.toml` | `cot_repair` | Fairness-tuned native cot_repair lane: keeps thinking on, raises planner tokens to `2048` and Stage-B max tokens to `4096` |
| `openai_adaptive_gated.template.toml` | `layered_adaptive_gated` | Adaptive control: rule-based helper gate with no learned controller |
| `openai_adaptive.template.toml` | `layered_adaptive_llm` | Adaptive: LLM-driven helper/budget policies with LinUCB online learning |
| `openai_adaptive_tuned.template.toml` | `layered_adaptive_llm` | Adaptive preset: tuned exploration, warmup, and decay while keeping baseline reward scaling |
| `openai_adaptive_gated_aggressive.template.toml` | `layered_adaptive_gated` | Adaptive control variant: aggressive rule-based helper gate |
| `openai_adaptive_gated_conservative.template.toml` | `layered_adaptive_gated` | Adaptive control variant: conservative rule-based helper gate |
| `baseline_ldb.template.toml` | `ldb` | External baseline: LLMDebugger via the baseline shorthand |
| `baseline_intervenor.template.toml` | `intervenor` | External baseline: INTERVENOR via the baseline shorthand |
| `baseline_intervenor_fair_v2.template.toml` | `intervenor` | Fairness-tuned INTERVENOR lane: preserves prompt assets/loop, raises teacher/learner caps to `3072` / `6144` |
| `baseline_tracecoder.template.toml` | `tracecoder` | External baseline: TraceCoder via the baseline shorthand |
| `baseline_tracecoder_fair_v2.template.toml` | `tracecoder` | Fairness-tuned TraceCoder lane: explicit seam compaction plus `2048` / `4096` analysis-vs-implementation caps |
| `baseline_lambda_rlm.template.toml` | `lambda_rlm` | External baseline: lambda-RLM via the baseline shorthand |
| `baseline_lambda_rlm_repair_fair.template.toml` | `lambda_rlm` | Secondary tuned baseline: lambda-RLM with solve-first prompt framing |
| `baseline_rlm.template.toml` | `rlm` | External baseline: upstream RLM via the baseline shorthand |
| `baseline_rlm_repair_fair.template.toml` | `rlm` | Secondary tuned baseline: RLM with solve-first prompt framing and shallower search |
| `openai_base_repair_retry_reflection.template.toml` | `base_repair` | Prompt ablation: enable retry reflection only |
| `openai_adaptive_stage_b_cot.template.toml` | `layered_adaptive_llm` | Prompt ablation: enable adaptive Stage-B CoT only; this is not equivalent to the native `baseline = "cot_repair"` adaptive shorthand |
| `openai_base_repair_repair_method_scaffold.template.toml` | `base_repair` | Prompt ablation: enable repair-method scaffold only |
| `openai_base_repair_retry_plan_then_patch.template.toml` | `base_repair` | Prompt ablation: enable retry-only extra planning on top of direct base repair |
| `openai_adaptive_probe_decay_098.template.toml` | `layered_adaptive_llm` | Probe preset: helper/budget recency decay candidate (`0.98`) |
| `openai_adaptive_probe_decay_0995.template.toml` | `layered_adaptive_llm` | Probe preset: gentler decay candidate (`0.995`) |
| `openai_adaptive_probe_prompt_order_crash_first.template.toml` | `layered_adaptive_llm` | Probe preset: Stage-B ordering ablation (`crash_first`) |
| `openai_adaptive_probe_prompt_order_evidence_last.template.toml` | `layered_adaptive_llm` | Probe preset: Stage-B ordering ablation (`evidence_last`) |
| `openai_adaptive_probe_current_knobs_compact.template.toml` | `layered_adaptive_llm` | Probe preset: tighten existing Stage-A/prompt/compaction knobs only |
| `openai_adaptive_probe_warm_start.template.toml` | `layered_adaptive_llm` | Probe preset: reuse prior controller state via init-state paths |
| `openai_adaptive_probe_bwk_v1.template.toml` | `layered_adaptive_llm` | Probe preset: explicit-token-budget `bwk_v1` smoke |
| `openai_adaptive_probe_hierarchical_v1.template.toml` | `layered_adaptive_llm` | Probe preset: hierarchical budget-family controller instead of bundled tier escalation |
| `openai_adaptive_probe_thompson.template.toml` | `layered_adaptive_llm` | Probe preset: Thompson Sampling for both helper and budget controllers |
| `openai_adaptive_probe_feature_scaling_interactions.template.toml` | `layered_adaptive_llm` | Probe preset: helper-side `v1_interactions` feature scaling |
| `openai_adaptive_probe_feature_scaling_percentile.template.toml` | `layered_adaptive_llm` | Probe preset: percentile-fitted feature scaling for both adaptive controllers |
| `openai_adaptive_probe_traceback_first.template.toml` | `layered_adaptive_llm` | Probe preset: adaptive traceback-first reveal mode |
| `openai_adaptive_probe_snapshot_pruning_heuristic.template.toml` | `layered_adaptive_llm` | Probe preset: heuristic prompt-facing snapshot pruning |
| `openai_adaptive_probe_tep_full.template.toml` | `layered_adaptive_llm` | Probe preset: full-mode TEP prompt-state adaptation |
| `openai_adaptive_probe_decision_model_shadow.template.toml` | `layered_adaptive_llm` | Probe preset: System-1 decision model logged next to LinUCB (run first; feeds the decision-model OPE targets) |
| `openai_adaptive_probe_decision_model_policy.template.toml` | `layered_adaptive_llm` | Probe preset: decision model selects the helper action (`evidence_v1`) |
| `openai_adaptive_probe_decision_model_policy_atomic.template.toml` | `layered_adaptive_llm` | Probe preset: decision model answers diagnostic questions, code maps them to an action (`evidence_atomic_v1`) |
| `openai_adaptive_probe_decision_model_features.template.toml` | `layered_adaptive_llm` | Probe preset: six decision-model diagnostic answers as LinUCB features |
| `openai_adaptive_probe_decision_model_prior.template.toml` | `layered_adaptive_llm` | Probe preset: decision-model-guided warm-up plus a fading score prior on LinUCB |

Files named `*.template.toml` must be copied and customised before use (set
`run_id`, policy state paths, etc.). The `*.latest.toml` file is ready to use
after setting `run_id`.

Shared adaptive templates now ship with two paper-facing defaults:
- helper score priors: templates don't set `online_action_score_biases`, so they use the one
  runtime default, `frame_only=-0.05,search_only=-0.10` (POL-07, decision D6). Each bias must
  stay within ±0.1 of the success reward; the templates used to set `-1.00`/`-0.50`, which
  vetoed those arms.
- conservative budget ladder: `budget_policy_tiers = "1.0,1.1,1.2,1.35,1.5"` with `budget_policy_max_ups_per_case = 1`

Two policy-input defaults changed in 2026-10 and are recorded in the run configuration:
- `online_include_case_labels = false`: `case.json` labels are no longer helper-policy features.
- `lite_snapshot_mode = "replace"`: under `adaptive_reveal_mode = "full"` the lite snapshot
  actions replace the full snapshot section (`"supplement"` restores the old behavior).

The decision-model presets need the GPU sidecar (`just decision-model-up`, see
`docs/local_docker_eval_runbook.md`) and set, under `[run_eval.adaptive]`:

```toml
[run_eval.adaptive]
online_policy_mode = "on"          # policy/features/prior roles need "on"
decision_model_role = "policy"     # off | shadow | policy | features | prior
decision_model_profile = "evidence_v1"
decision_model_url = "http://127.0.0.1:8090"  # or $TRACEGATE_DECISION_MODEL_URL
```

For `tier_binary`, that dense ladder does not advance via literal `+1` rung steps. The effective promotion path is `1.0 -> 1.2 -> 1.5 -> max tier`, with fallback to the next larger configured tier if one of those targets is absent.

## Assessment Of Unshipped Adaptive Branches

`hierarchical_v1` is not a strict upgrade over `tier_binary`. It trades broader,
bundled escalation for finer-grained family steps, but it still does not learn
`response_mode` and it adds a second learned decision. Treat it as an
experimental alternative, not a new default. `bwk_v1` is orthogonal because it
adds a run-level token budget rather than replacing the unconstrained budget
objective.

| Branch | Status | Recommendation |
|--------|--------|----------------|
| `hierarchical_v1` | Implemented, not previously templated | Expose now; use for architecture comparisons against `tier_binary`, not as a strict replacement |
| `thompson` | Implemented, not previously templated | Expose now; compare against LinUCB when exploration shape is the question |
| helper `v1_interactions` | Implemented, not previously templated | Expose now; meaningful only for the helper/evidence controller |
| `percentile_v1` scaling | Implemented, not previously templated | Expose now; most useful with enough online history or a warm-start state |
| `traceback_first` reveal | Implemented, not previously templated | Expose now; prompt-shape/context-efficiency ablation |
| heuristic snapshot pruning | Implemented, not previously templated | Expose now; prompt-efficiency ablation |
| gated `aggressive` / `conservative` | Implemented, not previously templated | Expose now; bounded control variants for the gated adaptive lane |
| `adaptive_tep_mode = "full"` | Implemented, not previously templated | Expose now; distinct prompt-state adaptation path |
| `adaptive_tep_mode = "legacy"` | Implemented, not templated | Document only; lower priority than `full` and overlaps the older retry-nudge path |
| policy `shadow` modes | Implemented, not templated | Document only; diagnostic/off-policy-evaluation path, not a primary benchmark preset |
| helper response override | Implemented, not templated | Document only; narrow transport/format knob rather than a controller comparison |
| feature scaling `none` | Implemented, not templated | Leave untemplated; sanity-check/diagnostic ablation rather than a research-ready preset |

The `*_fair_v2.template.toml` files are additive fairness-tuned lanes. They do
not replace the canonical baseline/native templates; they keep thinking enabled
and raise only the explicitly documented budget knobs.

Native baseline shorthand is also supported in config files:

```toml
[run_eval]
baseline = "cot_repair"
```

By default that resolves to the native `cot_repair` condition on the OpenAI
path. To keep the native adaptive controller path while preserving cot planner
semantics on attempt 1, pair it with exactly one adaptive condition:

```toml
[run_eval]
baseline = "cot_repair"
conditions = "layered_adaptive_llm"
```

`conditions = "layered_adaptive_gated"` is also supported for symmetry. Native
`baseline = "cot_repair"` does not use the external baseline adapter path, and
cannot be combined with `baseline_name`, `baseline_repo_dir`, or
`baseline_python`.

## Exact Study Presets

The `study_*.template.toml` files are the exact paper-facing matrix for the
shared-augmentation baseline study:

- `study_<baseline>_raw.template.toml`: raw baseline arm
- `study_<baseline>_hybrid_canonical.template.toml`: hybrid arm with
  `baseline_hybrid_guidance_mode = "fair_v1"`
- `study_<baseline>_hybrid_tuned.template.toml`: hybrid arm with
  `baseline_hybrid_guidance_mode = "tuned_v1"`

Separately, `study_intervenor_fixed_evidence_v1.template.toml` is an
INTERVENOR-first deterministic fixed-evidence arm. It is intentionally outside
the shared 5-baseline matrix.

Supported baselines:

- `ldb`
- `intervenor`
- `tracecoder`
- `lambda_rlm`
- `rlm`

These study presets intentionally pin:

- `k = 2`
- `jobs = 2`
- hybrid arms to `conditions = "layered_adaptive_gated"`
- `adaptive_controller_lambda_rlm = false`

The hybrid study presets use `layered_adaptive_gated` rather than
`layered_adaptive_llm` so the outer controller remains adaptive while still
supporting `jobs >= 2` without turning online policy updates into a
concurrent/batched-learning confound. Keep the learned online-policy condition
(`layered_adaptive_llm` with controller modes `on`/`shadow`) for separate
`jobs = 1` probes or final online-policy benchmarks.

For the exact command sequence, queueing pattern, and reporting caveats, see
[`docs/baseline_hybrid_study_runbook.md`](../../docs/baseline_hybrid_study_runbook.md).

## Config Structure

Every config lives under a single `[run_eval]` table with sub-tables.
The loader flattens TOML keys onto the real `argparse` destination names used by
`python -m evals.run_eval`, so CLI/config sync is automatic:

- nearly every non-secret CLI flag is config-addressable
- unknown config keys fail fast instead of being silently ignored
- explicit CLI flags still override config values

Common table prefixes:

| Section | Purpose |
|---------|---------|
| `[run_eval]` | Top-level: patcher backend, conditions, parallelism (`jobs`), attempts (`k`), case selection |
| `[run_eval.openai]` | Model endpoint: `base_url`, `model`, `http2`, response mode, timeouts, retry policy |
| `[run_eval.context]` | Prompt compaction: mode, scope, target size |
| `[run_eval.adaptive]` | Helper & budget online policies: algorithm, hyperparameters, state/event paths, decision-model role and sidecar (`decision_model_*`), policy-input switches |
| `[run_eval.patch]` | Patch sanitisation and verifier settings |
| `[run_eval.syntax]` | Syntax preflight checks before patch application |
| `[run_eval.container]` | Execution backend knobs such as `image`, `network`, and resource limits |
| `[run_eval.record]` | Run-metadata capture knobs such as `prompt`, `evidence`, `env`, and `deps` |

For example, `--container-image` maps to:

```toml
[run_eval.container]
image = "ghcr.io/example/eval:latest"
```

and `--record-prompt` maps to:

```toml
[run_eval.record]
prompt = "full"
```

## Secret Management

API keys are **never** stored in config files. Use environment-variable
indirection:

```toml
openai_api_key_env = "OPENAI_API_KEY"   # harness reads $OPENAI_API_KEY at runtime
```

For real remote providers, export the variable before running:

```bash
export OPENAI_API_KEY="sk-..."
```

For repo-local OpenAI-compatible servers, leaving it unset is fine: repo
workflows default `OPENAI_API_KEY` to `local`.

## Usage

```bash
python -m evals.run_eval --config evals/configs/openai_base_repair.latest.toml \
  --run-id my-run-001
```

CLI flags override config values — e.g. `--run-id`, `--k`, `--jobs`.

The shipped templates now standardize on:

- `k = 2`
- `jobs = 3`
- `[run_eval.openai].base_url = "http://127.0.0.1:12000/v1"`
- `[run_eval.openai].timeout_sec = 600.0`
- OpenAI patcher templates: `[run_eval.openai].max_propose_sec = 600.0`
- Baseline templates: `baseline_process_timeout_sec = 600.0`
- Baseline templates: `baseline_max_iterations = 2` is pinned explicitly to keep
  adapter-level retries conservative
- Baseline templates: `baseline_openai_compat_profile = "lmstudio_model_yaml"`
- Baseline templates: `baseline_openai_allow_reasoning_fallback = true`
- Baseline templates: `baseline_openai_enable_thinking = true`
- OpenAI patcher templates: `[run_eval.openai].compat_profile = "lmstudio_model_yaml"`
- OpenAI patcher templates: `[run_eval.openai].allow_reasoning_fallback = true`
- OpenAI patcher templates: `[run_eval.openai].enable_thinking = true`
- Canonical native templates pin `[run_eval.openai].max_tokens = 2048`
- Canonical `cot_repair` pins `[run_eval.openai].plan_patch_max_tokens = 768`
- Canonical TraceCoder template now pins its adapter-owned compaction, exact-test-context, and stage-token knobs explicitly
- Baseline templates: `conditions = "<baseline_name>"` is pinned explicitly even when
  `baseline = "<baseline_name>"` shorthand is used
- `baseline_lambda_rlm` and `baseline_rlm`: baseline-internal `timeout_sec` matches the
  shared `baseline_process_timeout_sec` so they do not carry a hidden stricter cap
- `baseline_lambda_rlm`: shipped templates now pin `prompt_contract = "canonical"`
  and `max_tokens = 8192`
- `baseline_rlm`: native inner `max_iterations = 24` while the outer adapter loop
  stays at `baseline_max_iterations = 2`
- `baseline_rlm`: shipped templates now pin `prompt_contract = "canonical"`,
  `max_tokens = 3072`, and `max_total_tokens = 200000`

Adaptive probe presets intentionally differ from the generic templates in one
important way:

- they pin `jobs = 1` so online-controller updates remain case-ordered and
  reproducible during sweeps and smoke comparisons

For fairness-oriented adaptive+baseline studies, keep the generic baseline
defaults instead:

- `k = 2`
- `jobs >= 2` (the shipped templates use `jobs = 3`)
- set `baseline_hybrid_guidance_mode = "fair_v1"` for the canonical shared
  augmentation lane
- set `baseline_hybrid_guidance_mode = "fixed_v1"` only for the INTERVENOR
  deterministic fixed-evidence lane with full-snapshot injection

The exact `study_*.template.toml` presets are stricter:

- they pin `jobs = 2` instead of the generic `jobs = 3`
- they use `layered_adaptive_gated` for the hybrid arms
- they keep controller-owned lambda-RLM decomposition off

For OpenAI-compatible external baselines, the shared compatibility knobs live on
the top-level `[run_eval]` table:

```toml
[run_eval]
baseline_openai_compat_profile = "lmstudio_model_yaml"
baseline_openai_allow_reasoning_fallback = true
baseline_openai_enable_thinking = true
baseline_hybrid_guidance_mode = "tuned_v1"
```

Profile guidance:

| Backend shape | Profile |
|---------------|---------|
| Qwen on vLLM / SGLang / raw OpenAI-compatible endpoints | `qwen_openai_raw` |
| Qwen on DashScope | `qwen_dashscope` |
| LM Studio / llama.cpp / model-template-driven local servers | `lmstudio_model_yaml` |
| Unknown endpoint | `auto` |

The shipped baseline templates assume the local llama.cpp / model-template
workflow in this repo, with `enable_thinking = true` and reasoning fallback
enabled by default. For `lmstudio_model_yaml`, actual thinking behavior still
depends on the server/model template, because that profile cannot express a
per-request thinking toggle; for Qwen profiles that can express the flag
directly, the request will keep thinking on unless you opt out. If your
baseline endpoint is Qwen on vLLM/SGLang or DashScope, override
`baseline_openai_compat_profile` to the matching Qwen profile. If you want
strict direct-answer behavior, set `baseline_openai_enable_thinking = false`
and, if desired, `baseline_openai_allow_reasoning_fallback = false`.

The native OpenAI patcher templates use the same local-server-safe defaults:

```toml
[run_eval.openai]
compat_profile = "lmstudio_model_yaml"
allow_reasoning_fallback = true
enable_thinking = true
```

If you disable thinking for either native or baseline OpenAI-compatible runs,
the harness still lets you leave reasoning fallback enabled. When thinking is
left on, the harness will automatically keep reasoning fallback enabled so
reasoning-only responses remain usable.

To inspect existing run artifacts for cap-risk rather than editing them, use:

```bash
python -m evals.budget_risk_report \
  evals/results/base-repair3-4B.jsonl \
  evals/results/cot-repair-4B.jsonl \
  evals/results/intervenor-4b.jsonl
```

For `lambda_rlm`, an opt-in misclassification workaround is also available on
the baseline-specific table:

```toml
[run_eval.baseline_lambda]
force_task_type = "qa"
```

Leave this unset by default; only enable it when the model keeps routing repair
prompts into the wrong upstream task type and returning prose instead of code.

Both recursive baselines also expose an explicit prompt-contract selector on
their baseline-specific tables:

```toml
[run_eval.baseline_lambda]
prompt_contract = "repair_fair_v1"

[run_eval.baseline_rlm]
prompt_contract = "repair_fair_v1"
```

The canonical templates now pin `prompt_contract = "canonical"` explicitly.
The shipped `*_repair_fair.template.toml` files remain secondary tuned-baseline
configs for practical comparison; they are not drop-in replacements for the
canonical paper-primary templates.

Benchmark caveat:

- `jobs = 3` is a throughput default, not a strict-online-learning default.
- When `layered_adaptive_llm` runs with either adaptive controller enabled
  (`online_policy_mode` or `budget_policy_mode` set to `shadow`/`on`), pass
  `--jobs 1` for final online-policy benchmarks so policy `decide()` /
  `record_outcome()` updates remain case-ordered and reproducible.
- Use `jobs > 1` only when you intentionally want concurrent/batched policy
  evaluation.

Docker wrapper example:

```bash
bash evals/docker/run_eval_docker.sh \
  --config evals/configs/openai_adaptive_gated.template.toml \
  --run-id my-docker-adaptive-001
```

## Creating New Configs

1. Copy the closest existing template:
   ```bash
   cp evals/configs/openai_adaptive.template.toml evals/configs/my_experiment.toml
   ```
   For the rule-based control arm, start from
   `evals/configs/openai_adaptive_gated.template.toml` instead.
2. Set `run_id` and adjust `conditions`, `model`, `k`, or `jobs` as needed.
   For strict online adaptive benchmarks, override `jobs` to `1`.
3. For adaptive configs, update `online_state_path` and `online_events_path` to
   avoid overwriting shared policy state.
4. For Docker baseline runs, prefer the shipped `baseline_*.template.toml` files;
   the wrapper will select the baseline-capable image automatically.
5. Commit the config alongside its results for reproducibility.

## Warm-Start Bootstrap Utility

If you only have historical policy event logs, bootstrap a compatible state JSON
first and then point the warm-start template at the generated artifact:

```bash
uv run python -m evals.bootstrap_policy_state \
  --events evals/results/<prior_run>.adaptive_policy.jsonl \
  --output-state evals/artifacts/policies/<prior_run>.bootstrapped-online.json
```

Budget-policy example:

```bash
uv run python -m evals.bootstrap_policy_state \
  --policy-kind budget \
  --events evals/results/<prior_run>.adaptive_budget_policy.jsonl \
  --output-state evals/artifacts/policies/<prior_run>.bootstrapped-budget.json
```

Bootstrap notes:

- replay order follows the expanded `--events` argument order and file line order
- duplicate decision keys resolve last-writer-wins
- percentile-style feature-scaling histories are not reconstructable from event
  logs, so the bootstrap utility preserves the latest cap metadata but starts
  history buffers empty
- `*_init_state_path` warm-starts learned controller state only; exact run
  continuation should keep using the run-scoped `*_state_path`
- for `bwk_v1`, warm-starting from `budget_init_state_path` resets the new run's
  remaining-budget / cumulative-charge / dual-variable telemetry from the
  current config instead of inheriting prior-run depletion
