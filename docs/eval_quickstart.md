# Eval Quickstart

Last verified: March 26, 2026

> **Looking for a quick command?** See the [Master Runbook](runbook.md) for all commands across setup, quality, eval ops, and analysis in one place.

> **Scope**: This doc covers infrastructure setup: building containers, starting
> model endpoints, and verifying connectivity. For condition-specific CLI details,
> see [Eval Framework](../evals/README.md) and [Eval Configs](../evals/configs/README.md).

This is the shortest maintained path for:

- building the eval containers
- starting a local model endpoint
- running TraceGate eval conditions
- running the external baseline conditions in Docker

For the full operations guide, see
[Local/Docker Eval Runbook](local_docker_eval_runbook.md).

## Prerequisites

- repo checked out locally
- `uv` installed
- Docker installed
- a local OpenAI-compatible endpoint, or the repo's optional `llama.cpp` Docker path

## 1. Start a local OpenAI-compatible endpoint

If you already have a local endpoint, skip to step 2.

Quick readiness check:

```bash
curl http://127.0.0.1:12000/v1/models
```

### Optional: use the repo's `llama.cpp` Docker image

Build the model-serving image:

```bash
docker build --no-cache -f Dockerfile.llama -t llama-server-local:latest .
```

For Blackwell hosts, build the dedicated variant instead:

```bash
docker build --no-cache -f Dockerfile.llama.blackwell -t llama-server-blackwell:latest .
```

Run it against a local GGUF model:

```bash
MODEL="/models/path/to/model.gguf"

docker run --gpus all \
  -v "$(pwd)":/models \
  --ulimit memlock=-1:-1 \
  -p 12000:8080 \
  --restart on-failure:10 \
  llama-server-local:latest \
  -m "$MODEL"
```

Swap the image tag to `llama-server-blackwell:latest` if you built the
Blackwell-specific Dockerfile.

The default `qwen35-coding-repro` preset uses 4 parallel slots and
`--ctx-size 1048576`, which gives each slot up to 256K context under
llama.cpp's split-KV server mode.
The shipped Docker image also defaults to `-ub 2048 --no-mmap` via
`LLAMA_UBATCH_SIZE=2048` and `LLAMA_NO_MMAP=1`.
The entrypoint also enables `--swa-full` by default; set `LLAMA_SWA_FULL=0`
to omit that flag while keeping the preset-managed defaults.
The container also auto-derives runtime CPU threads from visible cores:
`--threads ~= cores / parallel` and `--threads-batch = cores`. On a 32-core
host, that becomes `-t 8 --threads-batch 32` unless you set explicit
`LLAMA_THREADS` / `LLAMA_THREADS_BATCH` overrides.

The eval examples below assume:

- base URL: `http://127.0.0.1:12000/v1`
- model name: your server's exposed model alias

## 2. Build the eval containers

Build the standard eval image:

```bash
docker build -t llmdebug-eval:latest -f evals/docker/eval.Dockerfile .
```

Build the baseline-capable image:

```bash
docker build \
  --build-arg LLMDEBUG_EVAL_INCLUDE_BASELINES=1 \
  -t llmdebug-eval-baselines:latest \
  -f evals/docker/eval.Dockerfile .
```

Notes:

- `llmdebug-eval:latest` is used for the native `llmdebug` conditions.
- `llmdebug-eval-baselines:latest` is used automatically for `--baseline ...` runs,
  including the native shorthand `--baseline cot_repair`.
- The wrapper script auto-builds either image when it is missing or stale.
- To force a rebuild through the wrapper, set `LLMDEBUG_EVAL_BUILD=1`.
- To force a no-cache rebuild through the wrapper, set `LLMDEBUG_EVAL_BUILD_NO_CACHE=1`.

## 3. Smoke-test the Docker wrapper

No model endpoint required:

```bash
bash evals/docker/run_eval_docker.sh --patcher null --max-cases 5 --k 1
```

Model-backed smoke:

```bash
OPENAI_BASE_URL="http://127.0.0.1:12000/v1"
OPENAI_MODEL="<model-name>"

bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --openai-response-mode edits \
  --conditions base_repair,layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --jobs 2 \
  --k 1 \
  --max-cases 5
```

## 4. Run the native `llmdebug` eval conditions

Direct wrapper invocation:

```bash
RUN_ID="adaptive-$(date +%Y%m%dT%H%M%S)"
OPENAI_BASE_URL="http://127.0.0.1:12000/v1"
OPENAI_MODEL="<model-name>"

bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --openai-api-key local \
  --openai-timeout-sec 180 \
  --openai-max-propose-sec 300 \
  --openai-max-tokens 2048 \
  --openai-temperature 0.0 \
  --max-context-chars 200000 \
  --prompt-budget-total-chars 180000 \
  --conditions base_repair,layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --jobs 4 \
  --k 3 \
  --run-id "$RUN_ID"
```

Template-driven invocation:

```bash
cp evals/configs/openai_adaptive.template.toml evals/configs/my_adaptive.toml
```

Edit `run_id`, model, and policy-state paths in that copied file, then run:

```bash
bash evals/docker/run_eval_docker.sh \
  --config evals/configs/my_adaptive.toml
```

Native shorthand runs also work through the wrapper:

```bash
RUN_ID="cot-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --baseline cot_repair \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --k 1 \
  --run-id "$RUN_ID"
```

Adaptive native cot-baseline shorthand:

```bash
RUN_ID="adaptive-cot-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --baseline cot_repair \
  --conditions layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --adaptive-budget-topup-max-calls 1 \
  --adaptive-online-policy-mode on \
  --adaptive-budget-policy-mode on \
  --jobs 1 \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --run-id "$RUN_ID"
```

`--prompt-adaptive-stage-b-cot` remains a prompt-only ablation. It is not a
substitute for `--baseline cot_repair --conditions layered_adaptive_llm`,
because the shorthand also preserves the native first-attempt planner and plan
injection into the final Stage-B patch prompt.

Unsupported with native `--baseline cot_repair`:

- external-only flags such as `--baseline-name`, `--baseline-repo-dir`, and `--baseline-python`
- forcing `--patcher baseline`
- multiple conditions or conditions outside `cot_repair`, `layered_adaptive_llm`, and `layered_adaptive_gated`

## 5. Run the external baselines

The quickest route is the baseline shorthand. The wrapper will automatically
use `llmdebug-eval-baselines:latest`.

Shared variables:

```bash
OPENAI_BASE_URL="http://127.0.0.1:12000/v1"
OPENAI_MODEL="<model-name>"
```

### LDB

```bash
RUN_ID="ldb-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --baseline ldb \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --max-cases 100 \
  --k 1 \
  --run-id "$RUN_ID"
```

### INTERVENOR

```bash
RUN_ID="intervenor-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --baseline intervenor \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --max-cases 100 \
  --k 1 \
  --run-id "$RUN_ID"
```

### TraceCoder

```bash
RUN_ID="tracecoder-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --baseline tracecoder \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --max-cases 100 \
  --k 1 \
  --run-id "$RUN_ID"
```

### lambda-RLM

```bash
RUN_ID="lambda-rlm-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --baseline lambda_rlm \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --max-cases 100 \
  --k 1 \
  --run-id "$RUN_ID"
```

### RLM

```bash
RUN_ID="rlm-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --baseline rlm \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --max-cases 100 \
  --k 1 \
  --run-id "$RUN_ID"
```

Config-based baseline runs also work. Start from the shipped templates in
`evals/configs/`, for example:

```bash
cp evals/configs/baseline_ldb.template.toml evals/configs/my_ldb.toml
bash evals/docker/run_eval_docker.sh --config evals/configs/my_ldb.toml
```

## 6. Inspect the outputs

Results are written under:

- `evals/results/`
- `evals/artifacts/`

Quick summary:

```bash
just eval-summary <run_id>
```

Direct CLI equivalent:

```bash
uv run python -m evals.eval_summary <run_id>
```

For longer runs on a server, prefer the repo's queueing convention:

```bash
pueue add -- bash evals/docker/run_eval_docker.sh --baseline ldb ...
pueue status
pueue log <task_id>
```
