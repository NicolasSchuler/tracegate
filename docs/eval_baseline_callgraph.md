# Eval Baseline Callgraph

This note maps the callgraph for the eval conditions used in the comparison runs:

- llmdebug-native: `base_repair`, `cot_repair`, `layered_adaptive_gated`, `layered_adaptive_llm`
- external baselines: `ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, `rlm`

The key structural split is:

- native conditions stay inside the generic eval harness and call a normal patcher directly
- external baselines are normalized onto `--patcher baseline`, then routed through `BaselinePatcher` and a baseline-specific adapter

## Mermaid Overview

```mermaid
flowchart TD
    A[CLI args] --> B[main<br/>evals/run_eval.py]
    B --> C[normalize baseline shorthand<br/>resolve conditions]
    C --> D[make_patcher]
    D --> E[_execute_eval_unit]
    E --> F[copy case workdir]
    F --> G[run_one_condition]

    G --> H{condition family}

    H --> I[base_repair / cot_repair]
    I --> I1[monolithic prompt assembly]
    I1 --> I2[base_repair: standard Stage-B prompt]
    I1 --> I3[cot_repair: CoT preamble only]
    I2 --> Z
    I3 --> Z

    H --> J[layered_adaptive_gated]
    J --> J1[adaptive bookkeeping]
    J1 --> J2[_adaptive_gate_should_invoke_helper]
    J2 --> J3[optional Stage-A helper / snapshot evidence]
    J3 --> J4[Stage-B prompt assembly]
    J4 --> Z

    H --> K[layered_adaptive_llm]
    K --> K1[adaptive bookkeeping]
    K1 --> K2[feature extraction + online policy]
    K2 --> K3[action: skip_helper / invoke_helper / snapshot_lite / compact_then_patch / retry_helper...]
    K3 --> K4[optional Stage-A helper / snapshot evidence]
    K4 --> K5[Stage-B prompt assembly]
    K5 --> Z

    H --> L[ldb / intervenor / tracecoder / lambda_rlm / rlm]
    L --> M[BaselinePatcher.propose_patch]
    M --> N[load baseline_case_context.json]
    N --> O[adapter.convert_case]
    O --> P{adapter}
    P --> P1[LDB adapter<br/>lower to HumanEval-like callable task<br/>subprocess into upstream LDB CLI]
    P --> P2[INTERVENOR adapter<br/>copy workspace, run tests,<br/>teacher -> learner repair loop]
    P --> P3[TraceCoder adapter<br/>write one-row dataset,<br/>subprocess into trace_learn_coder.py]
    P --> P4[lambda-RLM adapter<br/>recursive reasoning backend<br/>subprocess into lambda-RLM]
    P --> P5[RLM adapter<br/>recursive REPL backend<br/>subprocess into rlm]
    P1 --> Q[adapter.extract_diff]
    P2 --> Q
    P3 --> Q
    P4 --> Q
    P5 --> Q
    Q --> Z

    Z[_propose_patch_with_snapshot_context<br/>patcher.propose_patch(...)] --> R[apply_diff]
    R --> S[optional syntax preflight]
    S --> T[run_pytest]
    T --> U[record JSONL result + artifacts]
```

## Where The Branches Actually Happen

- `--baseline <name>` is normalized in `evals/run_eval.py` onto `--patcher baseline`, a matching baseline condition, and default repo/env paths.
- The generic per-case path is still shared until `run_one_condition(...)`.
- `base_repair` and `cot_repair` share the same harness path. The meaningful delta is prompt text in `evals/eval_prompts.py`.
- Only `layered_adaptive_gated` and `layered_adaptive_llm` are treated as snapshot conditions.
- All external baselines converge through `evals/patchers/baseline_patcher.py`, which reconstructs a `BaselineCaseContext` and then follows:
  `convert_case()` -> `run_repair()` -> `extract_diff()`.
- The adapter result contract is now additive: `extract_diff()` prefers
  adapter-supplied `patch_diff`, then structured `file_edits`, then legacy
  single-file `repaired_code`.
- The patcher's reporting metadata is intentionally separate: `repair_scope` /
  `repair_targets` prefer explicit adapter metadata first, then inferred
  `file_edits`, then `patch_diff`, then legacy `target_path`. This means the
  reported repair scope can preserve adapter intent even when diff extraction
  follows the result-shape precedence above.
- Current shipped baseline runtimes still operate as single-file `buggy.py`
  repair flows even though the shared contract is multifile-ready.
- `lambda_rlm` and `rlm` use the same baseline patcher path with their own
  adapters and default models.

## Condition-Specific Notes

### `base_repair`

- Straight Stage-B repair path.
- No Stage-A helper call.
- No adaptive policy.
- No snapshot plugin injection.

### `cot_repair`

- Same as `base_repair` structurally.
- Differs only by the chain-of-thought preamble in the repair prompt.

### `layered_adaptive_gated`

- Maintains adaptive state inside `run_one_condition(...)`.
- Uses a fixed helper gate to decide whether Stage-A should run.
- Can materialize structured snapshot context before Stage-B patch proposal.

### `layered_adaptive_llm`

- Maintains adaptive state inside `run_one_condition(...)`.
- Computes policy features and asks the online policy to choose the next action.
- The policy can skip helper use, invoke helper use, request lighter snapshot materialization, compact context, or select retry-helper variants.

### `ldb`

- The most aggressive task reshaping.
- Converts llmdebug cases into a strict HumanEval-like callable contract.
- Rejects incompatible case shapes explicitly instead of silently coercing them.
- Runs upstream LDB code in a subprocess, then parses logs back into a unified diff.

### `intervenor`

- Preserves a copied workspace and the case's native test command.
- The adapter owns the teacher/learner loop directly.
- Test execution remains close to the original case contract.

### `tracecoder`

- Writes a one-row llmdebug dataset for the patched upstream CLI.
- Runs `trace_learn_coder.py` in a subprocess.
- Parses CSV outputs and reconstructs a patch diff against `buggy.py`.

### `lambda_rlm`

- Runs the lambda-RLM adapter through the same `BaselinePatcher` contract.
- Preserves the recursive reasoning backend inside the isolated baseline env.
- Uses the same unified-diff handoff back into the eval harness.

### `rlm`

- Runs the standalone RLM adapter through the same `BaselinePatcher` contract.
- Preserves the upstream recursive REPL loop inside the isolated baseline env.
- Uses the same unified-diff handoff back into the eval harness.

## Main Convergence Points

- All conditions eventually feed a unified diff back into the eval harness.
- The harness applies the diff in the copied case workspace.
- The harness reruns pytest, records success or failure, and emits standard JSONL metrics.

## Validation Anchors

These focused tests are the quickest checks for the branch claims above:

```bash
uv run pytest tests/test_evals_execution_config.py -q -k "normalize_baseline_selection_tracecoder_sets_default_repo_dir or run_eval_baseline_shorthand_records_baseline_metadata"
uv run pytest tests/test_evals_cot_repair.py -q -k "request_context or cot_repair"
uv run pytest tests/test_evals_adaptive_conditions.py -q -k "layered_adaptive_gated_invokes_helper_after_repeated_signature"
```

## 100-Case Docker Smoke Commands

These commands use the Docker wrapper and the repo-preferred queueing pattern via `pueue`.

Assumptions:

- the Docker wrapper will auto-build `llmdebug-eval-baselines:latest` for these
  runs, including pinned baseline repos and Linux smoke envs
- your model server is exposed via an OpenAI-compatible URL; when you pass `127.0.0.1` or `localhost`, the Docker wrapper rewrites it as needed
- custom `--baseline-repo-dir` and `--baseline-python` are intentionally not
  supported through the Docker wrapper

### LDB

```bash
pueue add -- bash evals/docker/run_eval_docker.sh \
  --baseline ldb \
  --openai-base-url http://127.0.0.1:1234/v1 \
  --openai-model <MODEL> \
  --max-cases 100 \
  --k 1 \
  --run-id smoke-ldb-100
```

### INTERVENOR

```bash
pueue add -- bash evals/docker/run_eval_docker.sh \
  --baseline intervenor \
  --openai-base-url http://127.0.0.1:1234/v1 \
  --openai-model <MODEL> \
  --max-cases 100 \
  --k 1 \
  --run-id smoke-intervenor-100
```

### TraceCoder

```bash
pueue add -- bash evals/docker/run_eval_docker.sh \
  --baseline tracecoder \
  --openai-base-url http://127.0.0.1:1234/v1 \
  --openai-model <MODEL> \
  --max-cases 100 \
  --k 1 \
  --run-id smoke-tracecoder-100
```

### lambda-RLM

```bash
pueue add -- bash evals/docker/run_eval_docker.sh \
  --baseline lambda_rlm \
  --openai-base-url http://127.0.0.1:1234/v1 \
  --openai-model <MODEL> \
  --max-cases 100 \
  --k 1 \
  --run-id smoke-lambda-rlm-100
```

### RLM

```bash
pueue add -- bash evals/docker/run_eval_docker.sh \
  --baseline rlm \
  --openai-base-url http://127.0.0.1:1234/v1 \
  --openai-model <MODEL> \
  --max-cases 100 \
  --k 1 \
  --run-id smoke-rlm-100
```

Useful follow-up commands:

```bash
pueue status
pueue log <task_id>
pueue follow <task_id>
```
