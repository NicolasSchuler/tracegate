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

### Optional: System-1 decision model (Cloudflare Clef)

Only needed for `--adaptive-decision-model-role` runs. The harness talks to any
SystemOne-compatible endpoint (`GET /health`, `POST /v1/systemone`); there are two ways to serve
`clef-flash`.

**llama-swap / llama.cpp (preferred on a shared GPU host).** llama.cpp serves Clef natively
since release `b11371` (PR #29831; the `/v1/systemone` endpoint came with PR #29818), from the
GGUFs in `ggml-org/Clef-Flash-GGUF` (Q4_K_M 6.5 GB, Q8_0 9.7 GB, BF16 18.2 GB). Merge
`evals/docker/decision_model/llama-swap.snippet.yaml` into the llama-swap config (its `groups:`
block belongs at the top level, next to `models:`) and make sure the image's llama.cpp is at
least `b11371`: pin `LLAMA_CPP_REF` to a tag or rebuild with `--no-cache`, because a plain
rebuild with `master` reuses Docker's cached clone layer. Then use the passthrough route with one
of the llama-swap API keys:

```bash
export TRACEGATE_DECISION_MODEL_API_KEY=<llama-swap key>
export TRACEGATE_DECISION_MODEL_URL=http://127.0.0.1:9292/upstream/clef-flash-q4
just decision-model-health   # the first call waits while llama-swap loads the model
just decision-model-smoke    # p50/p95 latency at ~1k/2k/4k tokens
```

- `run_eval_docker.sh` passes the key to the harness as `--adaptive-decision-model-api-key`
  (redacted in lockfiles), never as an environment variable inside the container. Config files
  may not contain the key.
- The preflight health check waits up to 300 s for a cold entry; afterwards each decision uses
  `--adaptive-decision-model-timeout-sec` (10 s).
- llama.cpp's Clef support is text-only and decodes one sequence per batch (`--parallel 1`).

**PyTorch sidecar (reference implementation).** `evals/docker/decision_model/` runs the model
card's own code, which makes it the reference for parity checks against llama.cpp. In bf16 it
needs roughly 20–24 GB free at 8k context (the weights alone are about 18 GB; Cloudflare quotes
41 GB at 64k context). Pin the model revision:

```bash
just decision-model-build
GPU=1 CLEF_REVISION=<commit-sha> just decision-model-up   # detached, port 8090
just decision-model-health                                # 503 while the model still loads
just decision-model-smoke
```

- `CLEF_QUANTIZATION=nf4` loads the backbone in 4-bit (bitsandbytes NF4; about 8.6 GB of VRAM on
  an RTX PRO 6000 with an 8k context) while the joint head and `lm_head` stay in bf16;
  `int8` and the default `none` (bf16, about 20 GB) are the alternatives.
- The container runs as the invoking user with their supplementary groups (`RUN_AS_USER`), so a
  group-writable shared HF cache (`HF_CACHE_DIR=/shared/cache/huggingface`) stays writable and
  nothing in it ends up root-owned.
- `run_decision_model.sh` options: `GPU`, `PORT` (8090), `BIND_ADDR` (`127.0.0.1`; the
  endpoint is unauthenticated, so use `0.0.0.0` only when the harness runs on another trusted
  host), `HF_CACHE_DIR`, `CLEF_MODEL_ID`, `CLEF_REVISION`, `CLEF_MAX_LENGTH` (8192),
  `DETACH=0` for the foreground.
- Linear-attention CUDA kernels are optional at build time
  (`--build-arg INSTALL_LINEAR_ATTENTION_KERNELS=1`); the server logs a warning when
  transformers would fall back to the slower path.

**Both ways.** The harness finds the endpoint via `--adaptive-decision-model-url`,
`[run_eval.adaptive] decision_model_url`, or `TRACEGATE_DECISION_MODEL_URL` (default
`http://127.0.0.1:8090`). `run_eval_docker.sh` forwards the variable, rewrites a localhost URL
for the container, and uses host networking on Linux when the URL is local. An enabled role
health-checks the endpoint before the first unit and stops the run with a clear error when it is
down or still loading.

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
