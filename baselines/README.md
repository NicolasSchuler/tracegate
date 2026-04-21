# External Baselines for ISSRE Paper Experiments

This directory contains adapter infrastructure for running external
debugging/repair baselines under the **same protocol** as llmdebug's own
eval conditions (`base_repair`, `layered_adaptive_llm`, etc.). Baseline runs are
recorded under dedicated experiment arms (`ldb`,
`intervenor`, `tracecoder`, `lambda_rlm`, `rlm`) so they do not collide with
llmdebug-native conditions during resume or analysis.

For adapter architecture details, see [docs/baseline-adapters.md](../docs/baseline-adapters.md).

## Baselines

| Baseline | Paper | Role | Status |
|----------|-------|------|--------|
| **LDB** ([LLMDebugger](https://github.com/FloridSleeves/LLMDebugger)) | ICSE 2024 | Primary direct baseline (runtime tracing) | ✅ Integrated |
| **INTERVENOR** ([NEUIR/INTERVENOR](https://github.com/NEUIR/INTERVENOR)) | ACL 2024 | Secondary baseline (teacher-learner chain-of-repair) | ✅ Integrated |
| **TraceCoder** ([CSMA/TraceCoder](https://github.com/CSMA-Research-Group/TraceCoder)) | ICSE 2026 | Trace-driven multi-agent repair baseline | ✅ Integrated |
| **lambda-RLM** ([lambda-calculus-LLM/lambda-RLM](https://github.com/lambda-calculus-LLM/lambda-RLM)) | GitHub / MIT | Raw long-context recursive reasoning baseline adapted to repair | ✅ Integrated |
| **RLM** ([alexzhang13/rlm](https://github.com/alexzhang13/rlm)) | GitHub / MIT | Raw recursive REPL baseline adapted to repair | ✅ Integrated |

## Controller Backends

| Backend | Source | Role | Status |
|---------|--------|------|--------|
| **lambda-RLM** ([lambda-calculus-LLM/lambda-RLM](https://github.com/lambda-calculus-LLM/lambda-RLM)) | GitHub / MIT | Controller-owned decomposition and compaction backend for hybrid adaptive+baseline runs | ✅ Integrated |

## ⚠️ Licensing Notice

| Baseline | License |
|----------|---------|
| **LDB** | MIT License ✅ |
| **INTERVENOR** | **No license file** — all rights reserved by default |
| **TraceCoder** | **No license file** — all rights reserved by default |
| **lambda-RLM** | MIT License ✅ |
| **RLM** | MIT License ✅ |

Cloned repositories are **git-ignored** and must **never be committed** to
this repository.  They are fetched locally for experimental comparison only.
Do not redistribute code from unlicensed repositories.

## Fair Comparison Protocol

All baselines run under identical constraints:

- **Same cases**: Every case in `evals/cases/` (buggy.py + test_buggy.py)
- **Same LLM**: Local model via OpenAI-compatible API (same endpoint, same model)
- **Same budget**: `k=2` repair attempts per case, single-candidate generation
- **Same scoring**: pytest pass/fail on the patched code, run inside Docker
- **Same case/test timeout**: `BaselineCaseContext.timeout_sec` remains the
  per-case pytest budget used by the harness and baseline-owned test checks
- **Same API timeout default**: `--openai-timeout-sec` is the shared
  OpenAI-compatible request timeout
- **Same baseline process timeout default**: baseline-owned subprocess attempts
  default to the same timeout, with optional override via
  `--baseline-process-timeout-sec`

For the exact fairness-study matrix, use the `study_*.template.toml` files in
[`evals/configs`](../evals/configs). Those presets pin `k = 2`, `jobs = 2`, and
use `layered_adaptive_gated` for the hybrid arms so the llmdebug augmentation
interface is shared without introducing concurrent online-policy updates.
For the exact commands, see
[`docs/baseline_hybrid_study_runbook.md`](../docs/baseline_hybrid_study_runbook.md).

Each baseline receives what it would normally get: the buggy code, the test
file, and the traceback/error output. Baseline-only runs do **not** inject
`llmdebug_*` controller/evidence/history/verifier sidecars. Hybrid
adaptive/controller-backed runs may inject those additive sidecars into the
adapter input dict, but that is an explicit hybrid mode rather than the default
baseline path.

When a case cannot be represented faithfully for a baseline, the harness now
records an explicit compatibility failure instead of silently coercing the
task into a different shape.

Current repair scope is intentionally explicit:

- `ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, and `rlm` still run as
  **single-file repair baselines** targeting `buggy.py`
- the shared adapter contract is now **multifile-ready** underneath that
  policy: adapters can return a unified diff directly or structured multifile
  text edits, but the eval harness boundary still consumes one final unified
  diff

## Quick Start

### 1. Clone baseline repos

```bash
chmod +x baselines/setup_repos.sh
./baselines/setup_repos.sh
```

This clones the external baselines plus `lambda-RLM` and `rlm` into
`baselines/repos/` at pinned commits, applies any compatibility patches,
normalizes known git-apply formatting artifacts, and runs a smoke verification
pass over the patched clones. Broken syntax or missing llmdebug compatibility
markers now fail during setup instead of later inside an eval run or Docker
baseline image.

For the patched `ldb` and `tracecoder` checkouts specifically, those
compatibility patches also add timeout-safe telemetry sidecars. The adapters
prefer those sidecars when present, so future `evals/results/*.jsonl` rows
preserve recovered `llm_call_count`, timing, and token telemetry even when the
upstream process times out before writing its native final output artifact.

### 2. Create isolated runtime envs

```bash
bash baselines/setup_envs.sh
```

If the cloned repos live outside the default `baselines/repos/`, either pass
that repo root as the second argument or export `BASELINE_REPO_ROOT=/path/to/repos`
before running `setup_envs.sh`.

This always creates:

- `baselines/envs/ldb`
- `baselines/envs/intervenor`
- `baselines/envs/tracecoder`

If `baselines/repos/lambda-RLM` is present, it also creates:

- `baselines/envs/lambda_rlm`

If `baselines/repos/rlm` is present, it also creates:

- `baselines/envs/rlm`

`setup_envs.sh` now prefers `python3.12`, then `python3.11`, then `python3.10`
before falling back to generic `python3` / `python`. Both `lambda-RLM` and
`rlm` require Python `>=3.11`, so if your default `python3` is older, point the
script at a compatible interpreter explicitly:

```bash
BASELINE_ENV_PYTHON="$(command -v python3.11)" bash baselines/setup_envs.sh
```

If your system does not expose `python3.11` directly on `PATH` but `uv` has a
managed interpreter available, this also works:

```bash
BASELINE_ENV_PYTHON="$(uv python find 3.12 || uv python find 3.11)" \
  bash baselines/setup_envs.sh
```

The default `smoke` env mode is designed for adapter validation and mocked
integration tests. The LDB smoke env intentionally installs a tiny local
`vllm` stub directly into the env's `site-packages` so the upstream code can
import on GPT-style runs without pulling in the full vLLM stack or any build
backend tooling, and it also includes the upstream `graphviz` Python
package because the patched tracing modules import it even in mocked smoke
runs. It also pins `httpx<0.28` to stay compatible with the upstream
`openai==1.12.0` client used by the LDB baseline subprocess, and `astroid<4`
to preserve the upstream tracer's `AstroidBuilder()` constructor contract.
When the pinned local `lambda-RLM` or `rlm` checkout is available, the matching
env installs only that repo's declared runtime dependencies. The env clients
load the upstream code from the provided `repo_dir` at runtime, so the optional
envs do not need a packaged install of the cloned repo itself. That avoids
host-path editable links and fragile local-package builds on cluster Python
installs. If either repo is absent, `setup_envs.sh` skips the optional env and
still provisions the core baseline envs successfully.

Those smoke envs also back the post-provision `just test-baseline-runtime` lane
used for automated repo/env/adapter acceptance coverage.

External baseline eval runs now validate those prerequisites up front. The
resolved baseline repo dir must exist, and the baseline runtime must use either
an explicit `--baseline-python` override or the default
`baselines/envs/<baseline>/bin/python`. If that interpreter is missing or not
executable, `run_eval` aborts early with an actionable error instead of
falling back to the main eval interpreter.

`setup_envs.sh` now recreates those envs with copy-based interpreters and
installs `lambda-RLM` / `rlm` without editable path links so the resulting
`baselines/envs/*` runtimes remain usable when the repo is mounted at a
different root. If your envs were created with an older checkout, rerun
`bash baselines/setup_envs.sh` once before reusing them under a different repo
mount. The script now creates those venvs with `--without-pip` and lets
`uv pip install --python ...` seed the packages afterward; on Linux this also
avoids `ensurepip` failures with `uv`-managed copied interpreters by syncing
the source `libpython*.so` runtime libraries into each env when needed.

### 3. Run baselines via the eval harness

```bash
# LDB baseline
python -m evals.run_eval \
  --baseline ldb \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --k 2

# INTERVENOR baseline
python -m evals.run_eval \
  --baseline intervenor \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --k 2

# TraceCoder baseline
python -m evals.run_eval \
  --baseline tracecoder \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --k 2

# lambda-RLM raw baseline
python -m evals.run_eval \
  --baseline lambda_rlm \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --k 2

# RLM repair-adapted baseline
python -m evals.run_eval \
  --baseline rlm \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --k 2
```

`--baseline <name>` is the shorthand path for external baselines. It expands to
`--patcher baseline --baseline-name <name>`, selects the matching dedicated
condition label (`ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, or `rlm`),
defaults to `baselines/repos/<repo>` for the cloned repo, and auto-detects
`baselines/envs/<name>/bin/python` when that env exists.

The same pinned `lambda-RLM` checkout also remains available as a
controller-owned backend for hybrid adaptive runs. Enable that path explicitly:

```bash
python -m evals.run_eval \
  --baseline intervenor \
  --conditions layered_adaptive_llm \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --adaptive-controller-lambda-rlm
```

When enabled, the adaptive outer controller invokes the cloned
`baselines/repos/lambda-RLM` checkout through `baselines/envs/lambda_rlm/bin/python`
to generate a bounded decomposition block that is injected into the controller
packet seen by the selected baseline. Even without controller-owned
`lambda-RLM`, the hybrid controller now emits a deterministic typed
`strategy_summary` plus bounded history/evidence summaries for baseline prompts.

The same hybrid composition path also works with the standalone upstream `rlm`
baseline. The adaptive outer controller writes the same bounded controller
packet, and the `rlm` adapter folds that packet into the native RLM root prompt
while still preserving the upstream recursive REPL loop inside each harness
attempt:

```bash
python -m evals.run_eval \
  --baseline rlm \
  --conditions layered_adaptive_llm \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --k 2
```

Use `--baseline-repo-dir` or `--baseline-python` only when your repos or envs
live outside the standard layout.
Use `--baseline-process-timeout-sec` only when a baseline subprocess needs a
different wall-clock budget than the shared OpenAI request timeout.
Use `--baseline-openai-compat-profile` and
`--baseline-openai-allow-reasoning-fallback` to control OpenAI-compatible
reasoning/output adaptation across all external baselines.

### 3b. Run live baseline adapter E2E integration tests

The live adapter E2E suite (`tests/test_baseline_live_e2e.py`) now supports
runtime endpoint/model/profile overrides via pytest CLI flags. This suite is
manual-only; CI does not run it because it depends on a reachable local
OpenAI-compatible model server.

```bash
# Lambda-RLM only, thinking disabled
uv run pytest -m integration tests/test_baseline_live_e2e.py::TestLambdaRLMLiveE2E -v -k thinking_off \
  --live-openai-base-url http://127.0.0.1:1234/v1 \
  --live-openai-model mlx-community/Qwen3.5-4B-MLX-8bit \
  --live-openai-compat-profile auto
```

Available flags:

- `--live-openai-base-url` (default: `http://127.0.0.1:1234/v1`)
- `--live-openai-model` (default: `qwen_qwen3.5-4b`)
- `--live-openai-compat-profile` (`auto`, `qwen_openai_raw`, `qwen_dashscope`,
  `lmstudio_model_yaml`, `off`; default: `auto`)

## OpenAI-Compatible Reasoning Modes

External baselines now share one OpenAI-compatible request/response adapter.
Default behavior is strict: the baselines consume the final answer from
`message.content`; `message.reasoning` / `message.reasoning_content` are treated
as metadata only. If a server returns reasoning but no final content, the
baseline stops early with an explicit compatibility error instead of drifting to
the outer subprocess timeout.

Shared flags:

- `--baseline-openai-compat-profile {auto,qwen_openai_raw,qwen_dashscope,lmstudio_model_yaml,off}`
- `--baseline-openai-allow-reasoning-fallback`

Recommended profiles:

| Backend shape | Profile | Effect |
|---------------|---------|--------|
| Qwen via vLLM / SGLang / raw OpenAI-compatible servers | `qwen_openai_raw` | Sends `extra_body.chat_template_kwargs.enable_thinking=false` |
| Qwen via DashScope | `qwen_dashscope` | Sends top-level `enable_thinking=false` |
| LM Studio / llama.cpp / model-template-driven local servers | `lmstudio_model_yaml` | No request-side injection; disable thinking in the server or model template, or pair it with reasoning fallback if thinking stays enabled |
| Unknown / mixed OpenAI-compatible endpoint | `auto` | Conservative detection only; leaves unknown localhost servers unchanged |

Notes:

- Qwen3/Qwen3.5 backends are backend-specific. Do not infer behavior from the
  model name alone.
- If an OpenAI-compatible endpoint rejects requests with errors like
  `Assistant response prefill is incompatible with enable_thinking`, use
  `--baseline-openai-compat-profile qwen_openai_raw` for Qwen on vLLM/SGLang,
  or disable server-side thinking in local model-template servers.
- LM Studio and llama.cpp may expose reasoning-only output unless the model
  template disables thinking. For strict direct-answer baselines, keep
  `baseline_openai_allow_reasoning_fallback = false` and fix the server/model
  configuration instead of silently consuming hidden reasoning as the answer.
- The shipped eval baseline templates under `evals/configs/` assume the local
  llama.cpp thinking workflow used in this repo and therefore set
  `baseline_openai_allow_reasoning_fallback = true` by default. Override it
  back to `false` if you disable thinking server-side.

For Docker wrapper runs:
- host-side `baselines/setup_repos.sh` and `baselines/setup_envs.sh` are not
  required; the baseline-capable eval image bakes the pinned repos and Linux
  smoke envs into the container image
- do not pass custom `--baseline-repo-dir` or `--baseline-python` through
  `evals/docker/run_eval_docker.sh`

TraceCoder is integrated as a **repair-seeded** baseline on llmdebug: the
patched upstream CLI starts from the provided `buggy.py`, runs its own
trace/analyze/repair loop, and uses the case's native pytest command for
internal evaluation and instrumentation. Unsupported cases remain explicit
`case_incompatible` failures; there is no weaker fallback mode.

### 4. Compare results

```bash
python -m baselines.compare_results \
  --jsonl 'evals/results/<run_id>*.jsonl' \
  --conditions base_repair cot_repair layered_adaptive_llm ldb intervenor tracecoder lambda_rlm rlm \
  --reference base_repair \
  --primary layered_adaptive_llm \
  --bootstrap 10000 --seed 42 \
  --out reports/comparison.json \
  --latex reports/comparison_table.tex
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment protocol,
statistical analysis plan, and paper reporting guidelines.
For the harness-level callgraph and baseline routing diagram, see
[../docs/eval_baseline_callgraph.md](../docs/eval_baseline_callgraph.md).

## Architecture

```
baselines/
├── adapter_base.py          # BaselineAdapter ABC + BaselineConfig/BaselineResult
├── case_schemas.py          # Typed BaseCaseInput + per-adapter subclasses
├── extra_configs.py         # Typed extra-config schemas per adapter
├── compare_results.py       # Cross-baseline statistical comparison
├── normalize_repos.py       # Normalize cloned repos after setup
├── verify_repos.py          # Sanity-check the pinned repository checkouts
├── setup_repos.sh           # Clone + pin external repos
├── setup_envs.sh            # Create isolated runtime envs for baselines/controller backends
├── EXPERIMENTS.md           # Experiment protocol and reporting notes
├── envs/
│   └── runtime_manifest.json # Generated manifest for local runtime envs
├── ldb/
│   ├── adapter.py           # LDBAdapter(BaselineAdapter)
│   ├── case_converter.py    # llmdebug case → LDB format
│   ├── result_extractor.py  # LDB output → unified diff
│   ├── patches/             # Compatibility patches for the pinned upstream checkout
│   └── requirements-smoke.txt
├── intervenor/
│   ├── adapter.py           # IntervenorAdapter(BaselineAdapter)
│   ├── case_converter.py
│   ├── env_client.py        # Isolated env shim for standalone runs
│   ├── result_extractor.py
│   └── requirements-smoke.txt
├── lambda_rlm/
│   ├── adapter.py           # LambdaRLMAdapter(BaselineAdapter)
│   ├── case_converter.py    # llmdebug case → lambda-RLM repair prompt context
│   └── env_client.py        # Isolated env shim for standalone baseline + controller calls
├── rlm/
│   ├── adapter.py           # RLMAdapter(BaselineAdapter)
│   ├── case_converter.py    # llmdebug case → RLM repair prompt context
│   └── env_client.py        # Isolated env shim for standalone RLM runs
└── tracecoder/
    ├── adapter.py
    ├── case_converter.py
    ├── result_extractor.py
    ├── patches/             # Compatibility patches for the pinned upstream checkout
    └── requirements-smoke.txt

evals/
├── lambda_rlm_backend.py    # Controller-owned lambda-RLM decomposition bridge
└── patchers/
    └── baseline_patcher.py  # Wraps BaselineAdapter → Patcher protocol
```

### How It Works

1. `run_eval.py` instantiates a `BaselinePatcher` with the chosen adapter
2. For each case, the patcher calls:
   - `adapter.convert_case(case_dir)` → baseline-specific input dict
   - `adapter.run_repair(case_input, config)` → `BaselineResult`
   - `adapter.extract_diff(case_dir, result)` → unified diff string
3. The diff is applied and tested via pytest (same as all other conditions)
4. Results are recorded in the standard JSONL schema

### Harness Integration Contract

The eval harness and adapters communicate through three typed layers, each
validated at runtime.

#### 1. Typed case-input schemas (`case_schemas.py`)

Every adapter's `convert_case()` returns a **frozen dataclass** subclass of
`BaseCaseInput`.  The hierarchy is:

| Adapter | Schema class | Adapter-specific fields |
|---------|-------------|------------------------|
| LDB | `LDBCaseInput` | `prompt`, `seed`, `given_tests`, `dataset_type` |
| INTERVENOR | `IntervenorCaseInput` | `prompt`, `buggy_code` |
| TraceCoder | `TraceCoderCaseInput` | `complete_prompt`, `buggy_code`, `repair_seeded`, `execution_mode`, `category`, `difficulty`, `snapshot_dependency` |
| lambda-RLM | `LambdaRLMCaseInput` | `buggy_code`, `prompt_context`, `category`, `difficulty`, `snapshot_dependency` |
| RLM | `RLMCaseInput` (alias for `LambdaRLMCaseInput`) | same as lambda-RLM |

`BaseCaseInput.from_dict()` validates at construction time: missing required
fields or unrecognised keys (other than harness-injected `llmdebug_*` keys)
raise `ValueError`.

The harness lifecycle is:

1. `adapter.convert_case(case_dir)` → typed `BaseCaseInput` subclass
2. `BaselinePatcher` calls `.to_dict()` on the result
3. Patcher merges harness-injected `llmdebug_*` keys into the plain dict
4. `adapter.run_repair(case_input_dict, config)` receives the merged dict

This keeps adapters' `run_repair()` backward-compatible with plain dicts while
giving `convert_case()` compile-time and runtime guarantees.

#### 2. Typed extra-config validation (`extra_configs.py`)

Each adapter optionally overrides `validate_extra_config(config)`, which is
called by `build_baseline_patcher()` after `setup()`.  The validation:

- Constructs a typed `*ExtraConfig` dataclass from `config.extra`
- Raises on unknown keys in strict eval mode (the default)
- Logs a warning instead when strict mode is explicitly disabled

| Adapter | Config class | Recognised keys |
|---------|-------------|-----------------|
| LDB | `LDBExtraConfig` | *(none)* |
| INTERVENOR | `IntervenorExtraConfig` | *(none)* |
| TraceCoder | `TraceCoderExtraConfig` | `tracecoder_fake_responses_path` |
| lambda-RLM | `LambdaRLMExtraConfig` | `baseline_lambda_context_window_chars`, `baseline_lambda_accuracy_target`, `baseline_lambda_a_leaf`, `baseline_lambda_a_compose`, `baseline_lambda_verbose`, `baseline_lambda_temperature`, `baseline_lambda_top_p`, `baseline_lambda_max_tokens`, `baseline_lambda_stream`, `baseline_lambda_timeout_sec`, `baseline_lambda_force_task_type`, `baseline_lambda_prompt_contract` |
| RLM | `RLMExtraConfig` | `baseline_rlm_max_depth`, `baseline_rlm_max_iterations`, `baseline_rlm_max_errors`, `baseline_rlm_compaction`, `baseline_rlm_compaction_threshold_pct`, `baseline_rlm_verbose`, `baseline_rlm_max_total_tokens`, `baseline_rlm_temperature`, `baseline_rlm_top_p`, `baseline_rlm_max_tokens`, `baseline_rlm_timeout_sec`, `baseline_rlm_prompt_contract` |

#### 3. Sidecar schema versioning

Adaptive controller sidecars (written by `write_adaptive_controller_sidecars()`
in `evals/adaptive_controller_packet.py`) carry a `_sidecar_schema_version`
field at the top level.  When `parse_controller_packet()` loads a sidecar, it
compares the embedded version against the current `_SCHEMA_VERSION`.

- **Match or missing field**: parse normally (backward-compatible with
  pre-versioned sidecars)
- **Mismatch**: log a warning, continue best-effort parsing, and record
  `adaptive_sidecar_schema_version` in the patcher meta for post-run auditing

This is "warn + best-effort" by design: sidecars are transient within a run, so
hard failures would abort otherwise recoverable attempts.

#### 4. Model default delegation

Each adapter class declares a `default_model` attribute (e.g.
`default_model = "gpt-4"` for LDB, `"gpt-4o-mini"` for the rest).
`resolve_effective_baseline_model()` checks this before falling back to the
legacy `_BASELINE_DEFAULT_MODELS` lookup table.  New adapters should always set
`default_model` on the class.

#### 5. Harness-injected keys

After `convert_case()` returns, `BaselinePatcher.propose_patch()` may inject
up to four keys into the merged `case_input` dict:

| Key | Source | Present when |
|-----|--------|-------------|
| `llmdebug_controller_packet` | `llmdebug_controller_packet.json` sidecar | Adaptive/controller-backed runs |
| `llmdebug_evidence` | `llmdebug_evidence_bundle.json` sidecar | Evidence selected by the policy |
| `llmdebug_history` | `llmdebug_history_bundle.json` sidecar | Retry history available |
| `llmdebug_verifier_report` | `llmdebug_verifier_report.json` sidecar | Post-attempt verification ran |

These keys use the `llmdebug_` prefix, which is explicitly allowed through by
the case-schema validation layer (`_HARNESS_INJECTED_PREFIXES`).  Adapters may
read them but must treat missing keys as normal. Baseline-only runs never set
them; adaptive/controller-backed runs may set any subset.

#### 6. env_client monkey-patch contract

The lambda-RLM and RLM adapters each ship an `env_client.py` that runs inside
an isolated Python environment.  These shims monkey-patch upstream module
globals to inject tracking clients:

| Adapter | Patched targets |
|---------|----------------|
| lambda-RLM | `rlm.clients.get_client`, `rlm.lambda_rlm.get_client` + instance methods `_parse_task_type`, `_plan` |
| RLM | `rlm.clients.get_client`, `rlm.core.rlm.get_client` + instance method `_default_answer` |

The patches replace `get_client` with a `tracked_get_client` that wraps the
returned client in a `TrackingOpenAIClient` subclass, adding character-level
telemetry (`model_input_chars`, `model_output_chars`, `model_char_calls`) plus
per-call finish-reason / compat-recovery summaries to every completion call.
These patches are brittle by nature — they depend on the exact module layout of
the pinned upstream checkout — so the `_COMPATIBLE_WITH` value is now pinned to
a concrete commit from `baselines/pins.py`, and adapter `setup()` validates
both that declaration and the checked-out repo commit early.

### Compatibility Notes

- **LDB** uses a strict HumanEval-like bridge. Only synchronous single-function
  tasks that can be lowered into `check(candidate)` plus executable
  `given_tests` are supported. Unsupported cases are recorded as
  `case_incompatible`.
- **INTERVENOR** reuses the official teacher/learner prompt assets from the
  pinned repo, but the adapter owns the execution loop and runs each case's
  native test command inside a temporary copied workspace.
- **TraceCoder** runs the patched upstream CLI in a repair-seeded llmdebug
  mode. It keeps TraceCoder's own planning/repair flow, but the compatibility
  patch swaps its dataset/evaluation layer over to a copied llmdebug workspace
  and the case's native pytest command.
- **lambda-RLM** now has two roles:
  - a standalone raw baseline (`--baseline lambda_rlm`) that preserves
    lambda-RLM's native recursive execution inside each harness attempt, and
  - an optional controller backend used by `layered_adaptive_gated` and
    `layered_adaptive_llm` hybrid baseline runs to produce bounded
    decomposition/compaction hints.
- **RLM** is a separate repair-adapted baseline (`--baseline rlm`). The
  harness uses `k` as the outer retry budget, keeps the upstream recursive
  REPL loop inside each attempt, and adds a repair-specific final-turn fallback
  plus final-answer salvage when no workspace write survives.
- **LDB** and **TraceCoder** now checkpoint telemetry during execution into
  timeout-safe JSON sidecars, but only for the patched llmdebug-compatible
  upstream support shipped in this repository. The adapters recover those
  snapshots on timeout and serialize them into the standard eval JSONL, so
  missing native CSV/JSONL outputs no longer collapse completed-call metrics to
  zero.
- Baseline telemetry is now schema-validated. Older or partially patched LDB
  and TraceCoder repos/outputs are intentionally unsupported: missing required
  telemetry fields fail with `baseline_incompatible_output_schema` instead of
  being coerced to `False`/`0`.
- Run metadata captures the selected baseline name, dedicated baseline
  condition label, pinned repo commit, selected baseline python executable,
  and any compatibility patch files applied during `setup_repos.sh`.
