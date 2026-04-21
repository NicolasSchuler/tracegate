#!/usr/bin/env bash
# Launch a vLLM OpenAI server in Docker with a Blackwell-compatible runtime
# workaround for CUDA Error 803 (container compat-libs overriding host driver).
#
# Usage:
#   evals/docker/run_vllm_blackwell.sh
#
# Optional env overrides:
#   VLLM_IMAGE=vllm/vllm-openai:nightly-<tag>
#   MODEL_TAG=Qwen/Qwen3-Coder-30B-A3B-Instruct
#   SERVED_MODEL_NAME=qwen3coder30ba3b
#   VLLM_PORT=7777
#   HF_CACHE_DIR=/your/cache/huggingface
#   VLLM_DTYPE=bfloat16
#   VLLM_MAX_MODEL_LEN=131072
#   VLLM_GPU_MEMORY_UTILIZATION=0.96
#   VLLM_KV_CACHE_DTYPE=fp8
#   VLLM_CALCULATE_KV_SCALES=1
#   VLLM_MAX_NUM_SEQS=2
#   VLLM_MAX_NUM_BATCHED_TOKENS=8192
#   VLLM_EXTRA_ARGS="--trust-remote-code"
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,40p' "$0"
    exit 0
fi

resolve_default_hf_cache_dir() {
    if [[ -n "${HF_HOME:-}" ]]; then
        printf '%s\n' "$HF_HOME"
        return 0
    fi
    if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        printf '%s\n' "${XDG_CACHE_HOME}/huggingface"
        return 0
    fi
    if [[ -n "${HOME:-}" ]]; then
        printf '%s\n' "${HOME}/.cache/huggingface"
        return 0
    fi
    printf '%s\n' "$(pwd -P)/.cache/huggingface"
}

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-2cbf9656ce6013f7b531bc5a0909d03b88c14862}"
MODEL_TAG="${MODEL_TAG:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3coder30ba3b}"
VLLM_PORT="${VLLM_PORT:-7777}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$(resolve_default_hf_cache_dir)}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.96}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
VLLM_CALCULATE_KV_SCALES="${VLLM_CALCULATE_KV_SCALES:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

mkdir -p "$HF_CACHE_DIR"

DOCKER_ARGS=(
    --rm
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -p "${VLLM_PORT}:${VLLM_PORT}"
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
    -e "MODEL_TAG=${MODEL_TAG}"
    -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
    -e "VLLM_PORT=${VLLM_PORT}"
    -e "VLLM_DTYPE=${VLLM_DTYPE}"
    -e "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
    -e "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
    -e "VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE}"
    -e "VLLM_CALCULATE_KV_SCALES=${VLLM_CALCULATE_KV_SCALES}"
    -e "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
    -e "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}"
    -e "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}"
)

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    DOCKER_ARGS+=( -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" )
fi

if [[ -n "${DOCKER_EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_DOCKER_ARGS <<< "$DOCKER_EXTRA_ARGS"
    DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")
fi

CONTAINER_CMD='set -euo pipefail
set -x

for f in /etc/ld.so.conf.d/*.conf; do
    sed -i "/cuda-.*\\/compat/d" "$f"
done
rm -f /etc/ld.so.conf.d/cuda-compat.conf || true
ldconfig

python3 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no-gpu\")"

EXTRA_ARGS=()
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS <<< "$VLLM_EXTRA_ARGS"
fi
if [[ "${VLLM_CALCULATE_KV_SCALES}" == "1" ]]; then
    EXTRA_ARGS+=(--calculate-kv-scales)
fi

exec vllm serve "$MODEL_TAG" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    "${EXTRA_ARGS[@]}"'

exec docker run "${DOCKER_ARGS[@]}" --entrypoint bash "$VLLM_IMAGE" -lc "$CONTAINER_CMD"
