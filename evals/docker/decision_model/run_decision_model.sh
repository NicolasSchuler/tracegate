#!/usr/bin/env bash
# Launch the System-1 decision-model sidecar (Cloudflare Clef) on the eval server's GPU.
#
# Usage:
#   evals/docker/decision_model/run_decision_model.sh
#   just decision-model-up
#
# The container serves the SystemOne API that evals/decision_model/client.py speaks:
#   GET  /health
#   POST /v1/systemone
# It is unauthenticated, so keep the published port on a trusted network.
#
# Optional env overrides:
#   DECISION_MODEL_IMAGE=tracegate-decision-model:latest
#   CONTAINER_NAME=tracegate-decision-model
#   GPU=0                      Device index passed to --gpus device=<GPU>
#   PORT=8090                  Host port published for the container's 8090
#   BIND_ADDR=127.0.0.1        Host address the port is published on; use 0.0.0.0 only
#                              when the harness runs on another (trusted) host
#   DETACH=1                   0 runs in the foreground instead
#   HF_CACHE_DIR=/your/cache/huggingface
#   RUN_AS_USER=<uid>:<gid>    Container user (default: the invoking user, so files written to
#                              a shared HF cache are not root-owned)
#   CLEF_MODEL_ID=Cloudflare/clef-flash
#   CLEF_REVISION=<commit sha>  Pin this; the server warns loudly on a moving branch
#   CLEF_DEVICE=cuda
#   CLEF_DTYPE=bfloat16
#   CLEF_QUANTIZATION=none      none | nf4 (4-bit backbone, ~6-8 GB) | int8
#   CLEF_MAX_LENGTH=8192
#   CLEF_MAX_REQUEST_BYTES=524288
#   CLEF_LOG_LEVEL=INFO
#   DOCKER_EXTRA_ARGS="..."    Extra docker run flags
#
# UNVERIFIED: this script has not been run yet; the image can only be built on a CUDA host.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DECISION_MODEL_IMAGE="${DECISION_MODEL_IMAGE:-tracegate-decision-model:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-tracegate-decision-model}"
GPU="${GPU:-0}"
PORT="${PORT:-8090}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
RUN_AS_USER="${RUN_AS_USER:-$(id -u):$(id -g)}"
DETACH="${DETACH:-1}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$(resolve_default_hf_cache_dir)}"
CLEF_MODEL_ID="${CLEF_MODEL_ID:-Cloudflare/clef-flash}"
CLEF_REVISION="${CLEF_REVISION:-main}"
CLEF_DEVICE="${CLEF_DEVICE:-cuda}"
CLEF_DTYPE="${CLEF_DTYPE:-bfloat16}"
CLEF_QUANTIZATION="${CLEF_QUANTIZATION:-none}"
CLEF_MAX_LENGTH="${CLEF_MAX_LENGTH:-8192}"
CLEF_MAX_REQUEST_BYTES="${CLEF_MAX_REQUEST_BYTES:-524288}"
CLEF_LOG_LEVEL="${CLEF_LOG_LEVEL:-INFO}"

if ! docker image inspect "$DECISION_MODEL_IMAGE" >/dev/null 2>&1; then
    echo ">>> Building $DECISION_MODEL_IMAGE ..."
    docker build -t "$DECISION_MODEL_IMAGE" "$SCRIPT_DIR"
fi

if [[ "$CLEF_REVISION" == "main" || "$CLEF_REVISION" == "master" ]]; then
    echo ">>> WARNING: CLEF_REVISION=$CLEF_REVISION is a moving branch; runs will not be" >&2
    echo ">>>          reproducible. Pin a commit sha from huggingface.co/$CLEF_MODEL_ID." >&2
fi

mkdir -p "$HF_CACHE_DIR"

DOCKER_ARGS=(
    --name "$CONTAINER_NAME"
    --gpus "device=${GPU}"
    --ipc=host
    --restart "on-failure:10"
    -p "${BIND_ADDR}:${PORT}:8090"
    --user "${RUN_AS_USER}"
    -v "${HF_CACHE_DIR}:/hf"
    -e "HF_HOME=/hf"
    -e "HOME=/tmp"
    -e "CLEF_MODEL_ID=${CLEF_MODEL_ID}"
    -e "CLEF_REVISION=${CLEF_REVISION}"
    -e "CLEF_DEVICE=${CLEF_DEVICE}"
    -e "CLEF_DTYPE=${CLEF_DTYPE}"
    -e "CLEF_QUANTIZATION=${CLEF_QUANTIZATION}"
    -e "CLEF_MAX_LENGTH=${CLEF_MAX_LENGTH}"
    -e "CLEF_MAX_REQUEST_BYTES=${CLEF_MAX_REQUEST_BYTES}"
    -e "CLEF_LOG_LEVEL=${CLEF_LOG_LEVEL}"
)

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    DOCKER_ARGS+=( -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" )
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    DOCKER_ARGS+=( -e "HF_TOKEN=${HF_TOKEN}" )
fi

# --user drops supplementary groups; keep them when running as the invoking user so group-
# writable shared caches (for example an HF cache owned by a lab group) stay writable.
if [[ "$RUN_AS_USER" == "$(id -u):$(id -g)" ]]; then
    for gid in $(id -G); do
        DOCKER_ARGS+=( --group-add "$gid" )
    done
    # The image has no passwd entry for this uid; torch derives cache dirs from
    # getpass.getuser(), which reads USER/LOGNAME before the passwd database.
    DOCKER_ARGS+=(
        -e "USER=$(id -un)"
        -e "LOGNAME=$(id -un)"
        -e "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor"
        -e "TRITON_CACHE_DIR=/tmp/triton"
    )
fi

if [[ -n "${DOCKER_EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_DOCKER_ARGS <<< "$DOCKER_EXTRA_ARGS"
    DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")
fi

# --restart rules out --rm, so a previous container of the same name has to go first.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

if [[ "$DETACH" == "0" ]]; then
    exec docker run "${DOCKER_ARGS[@]}" "$DECISION_MODEL_IMAGE"
fi

docker run -d "${DOCKER_ARGS[@]}" "$DECISION_MODEL_IMAGE" >/dev/null
echo ">>> Started $CONTAINER_NAME on port ${PORT} (GPU ${GPU})."
echo ">>> The first start downloads ~18 GB of weights; /health reports \"loading\" until then."
echo ">>> Logs:   docker logs -f $CONTAINER_NAME"
echo ">>> Health: just decision-model-health"
echo ">>> Smoke:  just decision-model-smoke"
