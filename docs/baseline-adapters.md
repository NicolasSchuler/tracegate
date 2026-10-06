# Baseline Adapter Architecture

llmdebug integrates five external debugging/repair systems as comparative
baselines: **LDB**, **INTERVENOR**, **TraceCoder**, **lambda-RLM**, and
**RLM**. Each runs under the same eval harness, the same LLM endpoint, and
the same scoring protocol as llmdebug's own conditions.

The **adapter pattern** makes this work. Every baseline implements a concrete
subclass of `BaselineAdapter`, translating between the harness's case format
and the baseline's native I/O. The `BaselinePatcher` orchestrator wraps any
adapter into the standard `Patcher` protocol so that `run_eval.py` treats
external baselines identically to llmdebug-native conditions.

**Why adapters?** Each upstream tool has its own input format, execution
model, and dependency tree. The adapter layer isolates those differences
behind a stable three-method contract while preserving each baseline's
original repair logic as faithfully as possible.

| Baseline | Adapter class | Name | Default model |
|----------|--------------|------|---------------|
| LDB | [`LDBAdapter`](../baselines/ldb/adapter.py) | `ldb` | `gpt-4` |
| INTERVENOR | [`IntervenorAdapter`](../baselines/intervenor/adapter.py) | `intervenor` | `gpt-4o-mini` |
| TraceCoder | [`TraceCoderAdapter`](../baselines/tracecoder/adapter.py) | `tracecoder` | `gpt-4o-mini` |
| lambda-RLM | [`LambdaRLMAdapter`](../baselines/lambda_rlm/adapter.py) | `lambda_rlm` | `gpt-4o-mini` |
| RLM | [`RLMAdapter`](../baselines/rlm/adapter.py) | `rlm` | `gpt-4o-mini` |

---

## Adapter Pipeline

Every baseline run follows a three-step pipeline, orchestrated by
[`BaselinePatcher`](../evals/patchers/baseline_patcher.py):

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  convert_case() │────▶│   run_repair()  │────▶│  extract_diff() │
│                 │     │                 │     │                 │
│ case_dir → typed│     │ case_input dict │     │ BaselineResult  │
│ BaseCaseInput   │     │ + config →      │     │ → unified diff  │
│ subclass        │     │ BaselineResult  │     │ string          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Step 1: `convert_case(case_dir, case_context=...)`

Reads the llmdebug case directory (`buggy.py`, `test_buggy.py`, `case.json`,
`oracle.patch`) and returns a typed `BaseCaseInput` subclass validated at
construction. If the case cannot be faithfully represented for a baseline,
the adapter raises `BaselineCaseIncompatibleError` with a structured reason.

The patcher then calls `.to_dict()` on the result and merges in any
harness-injected `llmdebug_*` sidecar keys (see
[Sidecar Injection](#sidecar-injection-hybrid-runs) below).

### Step 2: `run_repair(case_input, config)`

Executes the baseline's debugging/repair loop. This is where each adapter
diverges — LDB spawns a subprocess, INTERVENOR runs a teacher/learner LLM
chain, TraceCoder invokes its patched CLI, etc. Returns a `BaselineResult`
containing repaired code (or `None` on failure), iteration count, token
usage, and adapter-specific metadata.

### Step 3: `extract_diff(case_dir, result)`

Produces a unified diff from the repair result. The default implementation
in `BaselineAdapter` handles three shapes with this precedence:

1. **`result.patch_diff`** — authoritative diff supplied directly by the
   adapter
2. **`result.file_edits`** — synthesized multifile diff from structured
   edits
3. **`result.repaired_code`** — legacy single-file diff against the
   original `buggy.py`

Most adapters use the default; override only when the baseline produces a
non-standard diff format.

The harness then applies the diff and runs pytest — the same scoring path
used for all other conditions.

### Adaptive Helper Delegation

In adaptive baseline-backed runs, `BaselinePatcher` now splits the work into
two layers:

- **Stage A helper**: a cached shared
  [`OpenAICompatPatcher`](../evals/patchers/openai_compat_patcher.py)
  delegate owned by `BaselinePatcher`
- **Stage B repair**: the baseline adapter's normal `convert_case() ->
  run_repair() -> extract_diff()` pipeline

The Stage-A helper delegate is configured from the baseline run's resolved
model, API base URL/key, timeout, and the baseline OpenAI-compat extras
(`baseline_openai_compat_profile`,
`baseline_openai_allow_reasoning_fallback`,
`baseline_openai_enable_thinking`). This lets adaptive hybrids reuse the same
generic helper/retry machinery as native OpenAI-compatible patchers without
special-casing any one baseline.

Context compaction is still **not** delegated through baseline patchers in
this version. Baseline-backed adaptive runs gain request-context support and
strict-JSON Stage-A retries, while Stage-B patch generation remains fully
baseline-owned.

---

## BaselineContract

Each adapter returns a [`BaselineContract`](../baselines/contracts.py)
dataclass from its `contract()` method. This provides machine-readable
metadata about the adapter's integration fidelity and capabilities.

| Field | Type | Description | Example values |
|-------|------|-------------|----------------|
| `baseline_name` | `str` | Short unique identifier | `"ldb"`, `"tracecoder"` |
| `contract_version` | `str` | Schema version of the contract itself | `"v1"` |
| `upstream_commit` | `str` | Pinned git commit of the external repo | `"49ac191f..."` |
| `fidelity_level` | `str` | How faithfully the adapter reproduces upstream behavior | `"upstream_patched"`, `"prompt_asset_port"`, `"generic_api_adaptation"` |
| `repair_scope_policy` | `str` | What files the adapter may modify | `"single_file_only"`, `"adapter_defined"` |
| `prompt_contract_kind` | `str` | How the adapter constructs LLM prompts | `"upstream_native_prompt_contract"`, `"adapter_defined_code_repair_prompt"` |
| `output_contract` | `str` | Shape of the adapter's repair output | `"single_file_repaired_source"`, `"single_file_repaired_source_or_final_answer_salvage"` |
| `unsupported_case_policy` | `str` | Behavior on incompatible cases | `"explicit_failure"` |
| `accepted_case_shapes` | `tuple[str, ...]` | Case shapes the adapter can handle | `("single_file_buggy_py", "workspace_tests_available")` |
| `patch_scope_summary` | `tuple[str, ...]` | Categories of upstream patches applied | `("compat", "telemetry", "behavioral_adaptation")` |
| `notes` | `tuple[str, ...]` | Free-form adapter notes | — |

The contract is serialized into the eval JSONL metadata for every baseline
attempt, making it possible to audit exactly which adapter version and
configuration produced a given result.

---

## Per-Baseline Mechanism Summaries

### LDB

**Approach:** Runtime tracing + LLM diagnosis. LDB instruments the buggy
code with block-level runtime traces, feeds those traces to an LLM for
diagnosis, and generates a repair.

**Subprocess isolation:** LDB runs as a **subprocess** rather than
in-process. This is necessary because LDB pins different versions of
`openai` and other dependencies that conflict with llmdebug's own
environment. The adapter writes LDB-format JSONL files to a temp directory
and invokes `programming/main.py` via `run_monitored_subprocess()`.

**Case conversion:** The adapter lowers llmdebug cases into upstream
HumanEval-like rows: a callable prompt, seed code, and `check(candidate)`
test assertions. Only synchronous single-function tasks that fit this shape
are supported; others fail explicitly as `case_incompatible`.

**Fidelity level:** `upstream_patched` — runs the actual upstream LDB code
with compatibility patches for llmdebug dataset support and telemetry
sidecars.

**Key files:**

| File | Role |
|------|------|
| [`baselines/ldb/adapter.py`](../baselines/ldb/adapter.py) | `LDBAdapter` — subprocess orchestration, result parsing |
| [`baselines/ldb/case_converter.py`](../baselines/ldb/case_converter.py) | llmdebug case → LDB HumanEval-like row |
| [`baselines/ldb/patches/`](../baselines/ldb/patches/) | Compatibility patches applied to pinned upstream checkout |

---

### INTERVENOR

**Approach:** Teacher/learner chain-of-repair with iterative test feedback.
A "teacher" LLM call analyzes the buggy code and test failures, then a
"learner" LLM call generates a repair guided by the teacher's diagnosis.
The loop repeats with pytest feedback until tests pass or the budget is
exhausted.

**Orchestration:** The adapter reuses INTERVENOR's official prompt assets
(`prompt/code_teacher.txt`, `prompt/code_learner.txt`) from the pinned
repo but owns the execution loop itself. This allows it to run each case's
native pytest command inside a temporary workspace rather than depending on
INTERVENOR's own runner.

**Fidelity level:** `prompt_asset_port` — preserves the upstream prompt
templates and two-agent architecture, with adapter-owned orchestration.

**Key files:**

| File | Role |
|------|------|
| [`baselines/intervenor/adapter.py`](../baselines/intervenor/adapter.py) | `IntervenorAdapter` — teacher/learner loop, LLM calls, test execution |
| [`baselines/intervenor/case_converter.py`](../baselines/intervenor/case_converter.py) | llmdebug case → INTERVENOR input |
| [`baselines/intervenor/env_client.py`](../baselines/intervenor/env_client.py) | Isolated env shim for standalone runs |

---

### TraceCoder

**Approach:** Trace-driven repair via patched CLI. TraceCoder traces the
buggy program's execution, analyzes the traces to localize the fault, and
generates a repair. The adapter invokes the patched upstream
`trace_learn_coder.py` CLI as a subprocess, writes a single-row JSONL
dataset, and translates TraceCoder's CSV output back into the adapter
contract.

**Why patching was needed:** The upstream TraceCoder only supports its own
bundled datasets (HumanEval, MBPP). The compatibility patch
(`0001-add-llmdebug-repair-mode.patch`) adds an `llmdebug` dataset mode
that accepts external JSONL rows, routes them through the case's native
pytest command for evaluation and instrumentation, and emits
timeout-safe telemetry sidecars.

**Fidelity level:** `upstream_patched` — runs TraceCoder's actual
planning/repair flow with a swapped dataset/evaluation layer.

**Key files:**

| File | Role |
|------|------|
| [`baselines/tracecoder/adapter.py`](../baselines/tracecoder/adapter.py) | `TraceCoderAdapter` — subprocess orchestration, CSV result parsing |
| [`baselines/tracecoder/case_converter.py`](../baselines/tracecoder/case_converter.py) | llmdebug case → TraceCoder dataset row |
| [`baselines/tracecoder/patches/`](../baselines/tracecoder/patches/) | Compatibility patches for llmdebug repair-seeded mode |

**Failure interpretation:** When TraceCoder returns
`patch_meta.baseline_meta.candidate_status=unchanged_original_code`, the adapter is not
discarding a valid repair. It means TraceCoder's own `final_code` matched the original buggy
file, so the adapter clears the no-op candidate and records the status explicitly. In the
patched upstream loop, `best_code` starts as the original buggy program and is only replaced
when a debug candidate improves `passed_count` or passes all tests. As a result, runs that
stop with `No improvement streak reached` or `No repair plan produced` can legitimately end
with the original code as the reported `final_code`.

---

### lambda-RLM

**Approach:** Recursive REPL. lambda-RLM decomposes a problem recursively —
if a subproblem is too complex for a single LLM call, it spawns child
lambda-RLM calls to solve sub-parts, then composes the results. The adapter
wraps this around a code-repair prompt: the case's buggy code and test
context become the "question", and the LLM's response is extracted as
replacement source code.

**Key methods:**
- `_build_prompt()` — constructs the `Context: ... Question: ... Answer:`
  prompt from the case's `prompt_context`, failure feedback from previous
  attempts, and optional adaptive controller guidance.
- `_extract_candidate_code()` — extracts the repaired Python source from
  the LLM response. Prefers fenced code blocks; falls back to stripping
  a `buggy.py` header or using the raw response.

**Env client:** The adapter ships an `env_client.py` that runs inside an
isolated Python environment. It monkey-patches upstream `get_client` to
inject a `TrackingOpenAIClient` for character-level telemetry. The
`_COMPATIBLE_WITH` declaration is pinned to a concrete commit from
[`baselines/pins.py`](../baselines/pins.py).

**Fidelity level:** `generic_api_adaptation` — preserves lambda-RLM's
native recursive execution but wraps it in a code-repair prompt.

**Key files:**

| File | Role |
|------|------|
| [`baselines/lambda_rlm/adapter.py`](../baselines/lambda_rlm/adapter.py) | `LambdaRLMAdapter` — prompt construction, env client invocation, result parsing |
| [`baselines/lambda_rlm/case_converter.py`](../baselines/lambda_rlm/case_converter.py) | llmdebug case → lambda-RLM repair prompt context |
| [`baselines/lambda_rlm/env_client.py`](../baselines/lambda_rlm/env_client.py) | Isolated env shim with tracking client injection |

---

### RLM

**Approach:** Upstream generic REPL QA loop. RLM gives the LLM a set of
tools (`list_files`, `read_file`, `write_buggy_py`, `run_case_tests`,
`llm_query`, `rlm_query`) and lets it iteratively inspect the workspace,
modify code, and run tests. The adapter constructs a repair-specific root
prompt and lets the upstream recursive REPL engine run inside each harness
attempt.

**Termination:** The upstream RLM loop terminates via `FINAL()` or
`FINAL_VAR()` calls — the model's way of declaring "I'm done, here's my
answer." The adapter includes a repair-specific final-turn fallback: if no
workspace write survives (the model never called `write_buggy_py` or its
writes were rejected), it attempts to salvage a valid full-file Python
answer from the `FINAL(...)` payload.

**Env client:** Like lambda-RLM, the adapter ships an `env_client.py` with
monkey-patched tracking clients. The `_COMPATIBLE_WITH` pin is validated
during `setup()`.

**Fidelity level:** `generic_api_adaptation` — preserves the upstream
recursive REPL loop with a repair-adapted prompt and final-answer salvage.

**Key files:**

| File | Role |
|------|------|
| [`baselines/rlm/adapter.py`](../baselines/rlm/adapter.py) | `RLMAdapter` — root prompt construction, env client invocation, workspace diffing, final-answer salvage |
| [`baselines/rlm/case_converter.py`](../baselines/rlm/case_converter.py) | llmdebug case → RLM repair prompt context |
| [`baselines/rlm/env_client.py`](../baselines/rlm/env_client.py) | Isolated env shim with tracking client and tool definitions |

---

## Sidecar Injection (Hybrid Runs)

In **baseline-only** runs, adapters receive only the case data from
`convert_case()`. In **adaptive or controller-backed hybrid** runs,
`BaselinePatcher` injects up to four additional keys into the `case_input`
dict after `convert_case()` returns:

| Key | Source file | Content |
|-----|------------|---------|
| `llmdebug_controller_packet` | `llmdebug_controller_packet.json` | Full adaptive controller state: strategy summary, retry plan, evidence/history summaries |
| `llmdebug_evidence` | `llmdebug_evidence_bundle.json` | Evidence bundle for the current attempt |
| `llmdebug_history` | `llmdebug_history_bundle.json` | Retry history bundle from previous attempts |
| `llmdebug_verifier_report` | `llmdebug_verifier_report.json` | Patch verifier report from the previous attempt |

These keys are **additive**: missing keys are normal and must be treated as
absent by adapters. The `llmdebug_` prefix is explicitly allowed through by
the case-schema validation layer (`_HARNESS_INJECTED_PREFIXES` in
[`case_schemas.py`](../baselines/case_schemas.py)).

Adapters that consume sidecar data (all five currently do for controller
guidance) call `parse_controller_packet()` and the shared hybrid guidance
renderer from
[`baselines/controller_packet.py`](../baselines/controller_packet.py)
to fold the controller state into their prompts. The harness builds the packets in
[`evals/adaptive_controller_packet.py`](../evals/adaptive_controller_packet.py).

Three hybrid guidance modes are supported:

- `tuned_v1` (default): preserves each baseline's current prompt integration.
  `intervenor` keeps its richer teacher/learner-specific rendering, while the
  other four baselines keep the compact shared block.
- `fair_v1`: injects one shared canonical controller block across all five
  baselines. This equalizes the llmdebug augmentation interface, not the
  baselines' native solver loops or internal tools.
- `fixed_v1`: an INTERVENOR-first deterministic fixed-evidence block. It omits
  controller state, verifier, history, and retry-plan narration, injects the
  full snapshot plus compact failure-state evidence, and is intended for
  fixed-evidence ablations rather than the shared 5-baseline study matrix.

Separately, some shipped configs now expose `fair_v2` lanes as distinct
template files. Those are fairness-tuned budget presets, not replacements for
the canonical baseline templates.

---

## Common Failure Modes

### Adapter import errors

If a baseline's repository is not cloned or its isolated env is not set up,
the adapter class fails to import. The adapter registry in
`baseline_patcher.py` wraps each import in `try/except ImportError` and
silently omits unavailable adapters. Run `baselines/setup_repos.sh` and
`baselines/setup_envs.sh` to provision everything.

Separately, `run_eval` now performs a shared external-baseline runtime
preflight before case loading and before direct `make_patcher()` baseline
construction. External baseline runs fail immediately when the resolved
`baseline_repo_dir` is missing or when the resolved baseline interpreter is
missing or not executable. The harness no longer falls back to the main eval
interpreter when `baseline_python` is absent.

For mounted-repo workflows, older provisioned envs can still fail even when
`baselines/envs/<name>/bin/python` exists: a host-created venv may contain
absolute interpreter links or editable-install metadata that point back to the
original checkout instead of the mounted repo path. The current
`baselines/setup_envs.sh` recreates copy-based envs and removes that editable
path coupling; rerun it once after pulling the fix if your envs were created
earlier. The current script also avoids `ensurepip` during creation and repairs
copied `uv`-managed Linux interpreters by syncing `libpython*.so` runtime
libraries into the env when the copied binary cannot start on its own.

### Timeouts

Each baseline has two timeout layers:

- **`process_timeout_sec`** — wall-clock limit for the adapter's subprocess
  or outer repair loop (default: 300s, configurable via
  `--baseline-process-timeout-sec` or `baseline_process_timeout_sec` in
  TOML)
- **`api_timeout_sec`** — per-request timeout for OpenAI-compatible calls
  (defaults to `process_timeout_sec` if not set separately)

LDB and TraceCoder run as subprocesses and are killed on timeout. The
adapters recover whatever telemetry is available from timeout-safe sidecar
files. INTERVENOR, lambda-RLM, and RLM run in-process (or via env client
subprocess) and respect the same timeout budget.

### Compatibility failures

When `convert_case()` cannot faithfully represent a case for a baseline, it
raises `BaselineCaseIncompatibleError`. The patcher records the attempt as
`case_incompatible` with a structured reason rather than silently coercing
the case into a different shape. This is most common for LDB, which only
supports synchronous single-function tasks.

### `forced_noop_patch`

When a baseline produces `repaired_code=None` (repair failed) but the
harness is configured with `force_patch_attempt = true`, the eval runner
records the attempt with `note="forced_noop_patch"`. This means the
baseline was invoked but could not produce any candidate repair.

---

## Configuration

### Baseline CLI flags

Every `--baseline-*` flag of `run_eval.py`, also settable as a TOML key in a run
config. The table is generated from the argument parser, so the defaults are the
CLI's own:

<!-- BEGIN AUTO-GENERATED: BASELINE_CLI_DEFAULTS -->
<!-- Rendered from the code by `python -m evals.doc_sync --write`; do not edit. -->

| Flag | TOML key | CLI default | Description |
|------|----------|-------------|-------------|
| `--baseline` | `baseline` | unset | One-switch baseline selector. External baselines resolve to --patcher baseline --baseline-name NAME and auto-fill the standard baselines/repos/<repo> and baselines/envs/<name>/bin/python paths when present. Native shorthand 'cot_repair' resolves onto the native OpenAI path instead. |
| `--baseline-name` | `baseline_name` | unset | External baseline adapter name (for --patcher baseline). E.g. ldb, intervenor, tracecoder, lambda_rlm, rlm. |
| `--baseline-repo-dir` | `baseline_repo_dir` | unset | Path to the cloned external baseline repository (for --patcher baseline). |
| `--baseline-python` | `baseline_python` | unset | Optional Python executable for the selected baseline runtime env. If omitted, run_eval requires baselines/envs/<baseline>/bin/python. |
| `--baseline-process-timeout-sec` | `baseline_process_timeout_sec` | unset | Optional timeout for one baseline-owned subprocess attempt. Defaults to --openai-timeout-sec when unset. |
| `--baseline-max-iterations` | `baseline_max_iterations` | unset | Max internal iterations per baseline invocation. Controls the adapter's debug/repair loop count independently of --k (which sets the harness-level retry count). Defaults to 2 when omitted. |
| `--baseline-openai-compat-profile` | `baseline_openai_compat_profile` | `lmstudio_model_yaml` | Shared OpenAI-compatible request-shaping profile for external baselines. Use qwen_openai_raw for vLLM/SGLang-style Qwen endpoints, qwen_dashscope for DashScope-compatible Qwen endpoints, lmstudio_model_yaml for LM Studio / llama.cpp model-template setups, or off to disable request shaping. |
| `--baseline-openai-allow-reasoning-fallback` / `--no-baseline-openai-allow-reasoning-fallback` | `baseline_openai_allow_reasoning_fallback` | `true` | Allow external baselines to consume reasoning-only response fields when message.content is empty. |
| `--baseline-openai-enable-thinking` / `--no-baseline-openai-enable-thinking` | `baseline_openai_enable_thinking` | `true` | Enable thinking/reasoning mode for external baselines (e.g. Qwen enable_thinking=True). Also auto-enables --baseline-openai-allow-reasoning-fallback. |
| `--baseline-hybrid-guidance-mode` | `baseline_hybrid_guidance_mode` | `tuned_v1` | How llmdebug controller guidance is exposed to hybrid adaptive+baseline runs: 'tuned_v1' preserves baseline-specific prompt integration, 'fair_v1' uses one shared canonical guidance block across all baselines, and 'fixed_v1' uses a deterministic evidence block with full-snapshot injection intended for INTERVENOR fixed-evidence studies. |
| `--baseline-intervenor-teacher-max-tokens` | `baseline_intervenor_teacher_max_tokens` | `1024` | Teacher-stage max_tokens for the INTERVENOR adapter. |
| `--baseline-intervenor-learner-max-tokens` | `baseline_intervenor_learner_max_tokens` | `2048` | Learner-stage max_tokens for the INTERVENOR adapter. |
| `--baseline-intervenor-teacher-guidance-max-chars` | `baseline_intervenor_teacher_guidance_max_chars` | `4000` | Teacher-stage controller-guidance budget for INTERVENOR. |
| `--baseline-intervenor-learner-guidance-max-chars` | `baseline_intervenor_learner_guidance_max_chars` | `6000` | Learner-stage controller-guidance budget for INTERVENOR. |
| `--baseline-intervenor-append-output-contract` / `--no-baseline-intervenor-append-output-contract` | `baseline_intervenor_append_output_contract` | `true` | Append the explicit single-file output suffix to INTERVENOR learner prompts. |
| `--baseline-tracecoder-controller-guidance-max-chars` | `baseline_tracecoder_controller_guidance_max_chars` | `2000` | Controller-guidance budget appended to TraceCoder prompts. |
| `--baseline-tracecoder-max-no-improvement-streak` | `baseline_tracecoder_max_no_improvement_streak` | `2` | TraceCoder --max-no-improvement-streak value. |
| `--baseline-tracecoder-analysis-max-tokens` | `baseline_tracecoder_analysis_max_tokens` | `4096` | TraceCoder max_tokens for instrumentation/analysis-stage calls. |
| `--baseline-tracecoder-implementation-max-tokens` | `baseline_tracecoder_implementation_max_tokens` | `4096` | TraceCoder max_tokens for direct/code-implementation calls. |
| `--baseline-tracecoder-problem-summary-max-chars` | `baseline_tracecoder_problem_summary_max_chars` | `4000` | Max chars kept in the compact TraceCoder task/problem summary. |
| `--baseline-tracecoder-exact-test-context-max-chars` | `baseline_tracecoder_exact_test_context_max_chars` | `3000` | Max chars of verbatim failing-test context injected into TraceCoder analysis prompts. |
| `--baseline-tracecoder-failure-feedback-max-chars` | `baseline_tracecoder_failure_feedback_max_chars` | `6000` | Max chars retained in TraceCoder failure feedback summaries. |
| `--baseline-tracecoder-failure-feedback-tail-max-chars` | `baseline_tracecoder_failure_feedback_tail_max_chars` | `2000` | Max chars reserved for the bounded tail excerpt in TraceCoder failure feedback. |
| `--baseline-tracecoder-instrumentation-output-max-lines` | `baseline_tracecoder_instrumentation_output_max_lines` | `80` | Max lines retained from llmdebug instrumentation output before TraceCoder prompts. |
| `--baseline-tracecoder-instrumentation-output-max-chars` | `baseline_tracecoder_instrumentation_output_max_chars` | `12000` | Max chars retained from llmdebug instrumentation output before TraceCoder prompts. |
| `--baseline-lambda-context-window-chars` | `baseline_lambda_context_window_chars` | `100000` | Context window budget in chars passed to the lambda_rlm baseline. |
| `--baseline-lambda-accuracy-target` | `baseline_lambda_accuracy_target` | `0.8` | Target accuracy passed to the lambda_rlm planner. |
| `--baseline-lambda-a-leaf` | `baseline_lambda_a_leaf` | `0.95` | Leaf-call accuracy prior for the lambda_rlm planner. |
| `--baseline-lambda-a-compose` | `baseline_lambda_a_compose` | `0.9` | Composition accuracy prior for the lambda_rlm planner. |
| `--baseline-lambda-verbose` | `baseline_lambda_verbose` | `false` | Enable verbose native lambda_rlm plan logging inside the isolated env. |
| `--baseline-lambda-temperature` | `baseline_lambda_temperature` | unset | Optional temperature override passed to lambda_rlm's OpenAI-compatible backend. |
| `--baseline-lambda-top-p` | `baseline_lambda_top_p` | unset | Optional top-p override passed to lambda_rlm's OpenAI-compatible backend. |
| `--baseline-lambda-max-tokens` | `baseline_lambda_max_tokens` | unset | Optional max_tokens override passed to lambda_rlm's OpenAI-compatible backend. |
| `--baseline-lambda-stream` | `baseline_lambda_stream` | `false` | Enable streaming responses inside the lambda_rlm OpenAI-compatible backend. |
| `--baseline-lambda-timeout-sec` | `baseline_lambda_timeout_sec` | unset | Optional per-attempt timeout cap inside the lambda_rlm isolated env. Defaults to the effective baseline process timeout when unset. |
| `--baseline-lambda-force-task-type` | `baseline_lambda_force_task_type` | unset | Optional lambda-RLM task-type override forwarded into the isolated env (for example "qa") when small models misclassify repair prompts. |
| `--baseline-lambda-prompt-contract` | `baseline_lambda_prompt_contract` | unset | Optional lambda-RLM prompt contract override: 'canonical' or 'repair_fair_v1'. Defaults to canonical when unset. |
| `--baseline-lambda-controller-guidance-max-chars` | `baseline_lambda_controller_guidance_max_chars` | `2500` | Controller-guidance budget appended to lambda-RLM prompts. |
| `--baseline-lambda-failure-feedback-max-chars` | `baseline_lambda_failure_feedback_max_chars` | `3500` | Failure-feedback budget injected into lambda-RLM prompts. |
| `--baseline-lambda-failure-feedback-meta-max-chars` | `baseline_lambda_failure_feedback_meta_max_chars` | `2000` | Failure-feedback truncation budget stored in lambda-RLM attempt metadata. |
| `--baseline-rlm-max-depth` | `baseline_rlm_max_depth` | `2` | Maximum recursive depth for the standalone rlm baseline. |
| `--baseline-rlm-max-iterations` | `baseline_rlm_max_iterations` | `12` | Maximum native RLM iterations per outer attempt. |
| `--baseline-rlm-timeout-sec` | `baseline_rlm_timeout_sec` | unset | Optional per-attempt timeout cap inside the standalone rlm env. Defaults to the effective baseline process timeout when unset. |
| `--baseline-rlm-controller-guidance-max-chars` | `baseline_rlm_controller_guidance_max_chars` | `2500` | Controller-guidance budget appended to RLM root prompts. |
| `--baseline-rlm-prompt-contract` | `baseline_rlm_prompt_contract` | unset | Optional RLM root prompt contract override: 'canonical' or 'repair_fair_v1'. Defaults to canonical when unset. |
| `--baseline-rlm-failure-feedback-max-chars` | `baseline_rlm_failure_feedback_max_chars` | `3500` | Failure-feedback budget injected into RLM root prompts. |
| `--baseline-rlm-failure-feedback-meta-max-chars` | `baseline_rlm_failure_feedback_meta_max_chars` | `2000` | Failure-feedback truncation budget stored in RLM attempt metadata. |
| `--baseline-rlm-max-errors` | `baseline_rlm_max_errors` | `5` | Maximum consecutive REPL errors before aborting an rlm attempt. |
| `--baseline-rlm-max-tokens` | `baseline_rlm_max_tokens` | unset | Optional OpenAI-compatible max_tokens override passed to the standalone rlm backend. |
| `--baseline-rlm-max-total-tokens` | `baseline_rlm_max_total_tokens` | unset | Optional total token ceiling enforced by the standalone rlm loop. |
| `--baseline-rlm-compaction` | `baseline_rlm_compaction` | `false` | Enable native trajectory compaction inside the standalone rlm loop. |
| `--baseline-rlm-compaction-threshold-pct` | `baseline_rlm_compaction_threshold_pct` | `0.85` | Compaction trigger threshold for the standalone rlm loop. |
| `--baseline-rlm-temperature` | `baseline_rlm_temperature` | unset | Optional temperature override passed to the standalone rlm OpenAI-compatible backend. |
| `--baseline-rlm-top-p` | `baseline_rlm_top_p` | unset | Optional top-p override passed to the standalone rlm OpenAI-compatible backend. |
| `--baseline-rlm-verbose` | `baseline_rlm_verbose` | `false` | Enable verbose native rlm logging inside the isolated env. |

`unset` means the flag has no value of its own; its description says what applies instead. Shipped templates under `evals/configs/` may set other values.
<!-- END AUTO-GENERATED: BASELINE_CLI_DEFAULTS -->

### Per-baseline TOML config

Each baseline has a template under `evals/configs/`:

| Baseline | Template file |
|----------|--------------|
| LDB | [`evals/configs/baseline_ldb.template.toml`](../evals/configs/baseline_ldb.template.toml) |
| INTERVENOR | [`evals/configs/baseline_intervenor.template.toml`](../evals/configs/baseline_intervenor.template.toml) |
| INTERVENOR fair lane | [`evals/configs/baseline_intervenor_fair_v2.template.toml`](../evals/configs/baseline_intervenor_fair_v2.template.toml) |
| TraceCoder | [`evals/configs/baseline_tracecoder.template.toml`](../evals/configs/baseline_tracecoder.template.toml) |
| TraceCoder fair lane | [`evals/configs/baseline_tracecoder_fair_v2.template.toml`](../evals/configs/baseline_tracecoder_fair_v2.template.toml) |
| lambda-RLM | [`evals/configs/baseline_lambda_rlm.template.toml`](../evals/configs/baseline_lambda_rlm.template.toml) |
| RLM | [`evals/configs/baseline_rlm.template.toml`](../evals/configs/baseline_rlm.template.toml) |

The templates set per-baseline values for the keys in the table above; read the
template for the values a shipped lane uses.

Optional attempt metadata the adapters record when the provider or env shim
exposes it:

| Baseline | Attempt metadata | Description |
|----------|------------------|-------------|
| INTERVENOR | `teacher_finish_reason`, `learner_finish_reason` | Per-call finish reasons |
| lambda-RLM / RLM | `call_finish_reason_counts`, `call_length_finish_count`, `compat_recovery_count` | Per-attempt call-cap and compat telemetry |
| lambda-RLM | `planned_decomposition`, `executed_decomposition`, `task_detection_call_count`, `repl_llm_query_call_count` | Separates planned recursive structure from observed model-call execution |
| RLM | `outcome_classification`, `stop_due_to_guard`, `guard_name` | Distinguishes workspace-write wins, salvage wins, and search-exhaustion failures |

### Pinned repo commits

All baseline repos are pinned to specific commits in
[`baselines/pins.py`](../baselines/pins.py). The `setup()` method of each
adapter validates that the checked-out repo matches the expected pin. The
lambda-RLM and RLM adapters additionally validate env-client compatibility
declarations against the pinned commit.

---

## See Also

- [baselines/README.md](../baselines/README.md) — licensing, setup, fair
  comparison protocol, quick start
- [baselines/EXPERIMENTS.md](../baselines/EXPERIMENTS.md) — full experiment
  protocol and statistical analysis plan
- [docs/eval_baseline_callgraph.md](eval_baseline_callgraph.md) —
  harness-level callgraph and baseline routing diagram
- [docs/baseline_contracts.md](baseline_contracts.md) — contract metadata
  reference
- [evals/README.md](../evals/README.md) — eval framework overview (includes
  all eval condition definitions and baseline arms)
