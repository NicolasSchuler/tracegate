# Local and Docker Eval Runbook

Last verified: March 26, 2026

> **Looking for a quick command?** See the [Master Runbook](runbook.md) for all commands across setup, quality, eval ops, and analysis in one place.

This is the maintained eval-ops guide for running `llmdebug` eval workflows
without any cluster-specific tooling. It covers:

- a local OpenAI-compatible model endpoint
- the Docker wrapper at `evals/docker/run_eval_docker.sh`
- optional local `llama.cpp` Docker usage
- Docker-based official SWE-Bench Lite scoring

## Scope and prerequisites

Required:

- the repo checked out locally
- `uv` and Docker available
- a local OpenAI-compatible endpoint for model-backed evals

Useful starting points:

- [Eval Framework](../evals/README.md)
- [Eval Configs](../evals/configs/README.md)
- [`scripts/README.md`](../scripts/README.md)

## 1. Start a local OpenAI-compatible endpoint

Bring your own local endpoint, or use one of the repo helpers.

### Option A: existing local endpoint

Any OpenAI-compatible server is fine as long as you know:

- base URL, for example `http://127.0.0.1:12000/v1`
- served model name
- whether the endpoint expects `edits`, `diff`, or `full_file` style responses

Quick health check:

```bash
curl http://127.0.0.1:12000/v1/models
```

### Option B: optional Blackwell-oriented vLLM helper

If you are on the validated Blackwell workstation path, the helper script
starts a known-good vLLM Docker setup:

```bash
evals/docker/run_vllm_blackwell.sh
```

Treat this as a hardware-specific convenience, not as the required eval path.

### Option C: local `llama.cpp` Docker image

Ada / RTX 4090 path:

Build the repo image:

```bash
docker build --no-cache -f Dockerfile.llama -t llama-server-local:latest .
```

Blackwell path:

```bash
docker build --no-cache -f Dockerfile.llama.blackwell -t llama-server-blackwell:latest .
```

Blackwell notes:

- `Dockerfile.llama.blackwell` defaults `GGML_CUDA_FA_ALL_QUANTS=ON`, matching
  the current upstream `llama.cpp` Blackwell guidance from issue `#19662`.
- If you want the older, faster compile path, override it explicitly with
  `--build-arg GGML_CUDA_FA_ALL_QUANTS=OFF`.

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

Notes:

- `-p 12000:8080` maps host to container in the expected order.
- Keep the model path quoted via `"$MODEL"`.
- Swap the image tag to `llama-server-blackwell:latest` if you built the
  Blackwell-specific Dockerfile.
- The default `qwen35-coding-repro` preset now starts `llama-server` with
  `--parallel 4` and `--ctx-size 1048576`, which yields 256K effective context
  per slot under `llama.cpp`'s split-KV server mode.
- The shipped Docker image also defaults `LLAMA_UBATCH_SIZE=2048` and
  `LLAMA_NO_MMAP=1`, so the container starts with `-ub 2048 --no-mmap` unless
  you override those env vars.
- The entrypoint enables `--swa-full` by default. Set `LLAMA_SWA_FULL=0` to
  omit that flag while keeping the rest of the preset-managed defaults.
- The entrypoint now also derives runtime CPU threads from the visible cores:
  `--threads ~= cores / parallel` and `--threads-batch = cores`. On a 32-core
  host with the default repro preset, that means `-t 8 --threads-batch 32`
  unless you override `LLAMA_THREADS` or `LLAMA_THREADS_BATCH`.
- `curl http://127.0.0.1:12000/v1/models` is the quickest readiness check.

## 2. Run a smoke test first

Fast smoke using the Docker wrapper:

```bash
bash evals/docker/run_eval_docker.sh --patcher null --max-cases 5 --k 1
```

Model-backed smoke against a local endpoint:

```bash
bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --openai-response-mode edits \
  --conditions base_repair,layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --jobs 2 \
  --k 1 \
  --max-cases 5
```

## 3. Run a larger local/Docker eval

Direct wrapper invocation:

```bash
RUN_ID="local-eval-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
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
bash evals/docker/run_eval_docker.sh \
  --config evals/configs/openai_adaptive.template.toml \
  --run-id my-docker-adaptive-001
```

If you want the direct Python path instead of the Docker wrapper:

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --k 1 \
  --max-cases 5
```

## 4. SWE-Bench Lite: local generation and Docker scoring

Generate paired predictions against a local endpoint:

```bash
python -m evals.swebench.cli generate-lite \
  --run-id <id> \
  --conditions baseline,llmdebug \
  --max-instances <n|0> \
  --instance-ids-file <optional> \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model_alias> \
  --openai-http2 \
  --openai-async \
  --jobs <workers> \
  --max-request-errors 25 \
  --max-request-error-rate 0.5
```

Validate the run directory before scoring:

```bash
python -m evals.swebench.cli prepare-handoff --run-id <id>
```

Score with the official harness on any Docker-capable machine:

```bash
python -m evals.swebench.cli score-official \
  --run-dir /path/to/evals/swebench_runs/<id> \
  --conditions baseline,llmdebug \
  --workers <n> \
  --cache-level <none|base|env|instance>
```

Convenience wrappers:

```bash
./scripts/score_swebench_lite_official.sh --run-dir /path/to/evals/swebench_runs/<id>
./scripts/smoke_swebench_lite_official.sh --run-dir /path/to/evals/swebench_runs/<id> --max-instances 2
```

## 5. Practical notes

- Keep the local endpoint and eval run in separate terminals or sessions.
- Prefer a smoke run before any long benchmark or ablation run.
- `--openai-response-mode full_file` is available for single-source-file cases; in v1 it
  falls back internally to `diff` when the copied case workdir contains more than one
  non-test `.py` file.
- For reproducibility-oriented comparisons, fix the seed and keep the same
  hardware, model artifact, server settings, and prompt path across runs.
