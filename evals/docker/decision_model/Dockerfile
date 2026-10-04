# syntax=docker/dockerfile:1
#
# GPU sidecar that serves Cloudflare's Clef decision model behind the SystemOne HTTP API
# (`GET /health`, `POST /v1/systemone`) consumed by evals/decision_model/client.py.
#
# Build and run on the eval server (this image cannot be built or run on macOS):
#   docker build -t tracegate-decision-model:latest evals/docker/decision_model
#   evals/docker/decision_model/run_decision_model.sh
#
# UNVERIFIED: no build has run on the card yet. The base image tag, the wheel index and every
# pin below were chosen from upstream release notes; re-check them against the host driver
# (`nvidia-smi`) before the first build and commit whatever actually resolves.

ARG CUDA_TAG=12.8.1-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

# cu128 is the last CUDA 12.8 wheel series; torch >= 2.12 only publishes cu126/cu130.
ARG TORCH_VERSION=2.11.0
# The model card's load_release_model() builds Qwen's AutoProcessor, whose video processor
# hard-requires torchvision even for text-only requests; it must match TORCH_VERSION.
ARG TORCHVISION_VERSION=0.26.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TRANSFORMERS_VERSION=5.10.2

# Optional linear-attention CUDA kernels for the Qwen3.5 hybrid-attention backbone. They are
# off by default because the published wheels must match the exact torch/CUDA build and often
# have to be compiled on the card. When they are absent the sidecar logs a warning at startup
# ("linear-attention CUDA kernels missing ...") and transformers falls back to its slower
# reference path; decisions stay correct, only prefill latency suffers.
# UNVERIFIED: both pins and whether the Qwen3.5 path uses them at all.
ARG INSTALL_LINEAR_ATTENTION_KERNELS=0
ARG FLASH_LINEAR_ATTENTION_VERSION=0.4.0
ARG CAUSAL_CONV1D_VERSION=1.5.2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "${VIRTUAL_ENV}"

# Keep the torch install in its own layer: it is by far the largest download and it only has to
# be redone when TORCH_VERSION changes.
RUN pip install --upgrade pip \
    && pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt /opt/decision_model/requirements.txt
RUN pip install -r /opt/decision_model/requirements.txt "transformers==${TRANSFORMERS_VERSION}"

RUN if [ "${INSTALL_LINEAR_ATTENTION_KERNELS}" = "1" ]; then \
        pip install \
            "flash-linear-attention==${FLASH_LINEAR_ATTENTION_VERSION}" \
            "causal-conv1d==${CAUSAL_CONV1D_VERSION}"; \
    fi

COPY server.py /opt/decision_model/server.py

# Runtime configuration; see server.py for the full list. CLEF_REVISION must be pinned to a
# commit sha for reproducible runs - the sidecar warns loudly when it is left at "main".
ENV CLEF_MODEL_ID=Cloudflare/clef-flash \
    CLEF_REVISION=main \
    CLEF_DEVICE=cuda \
    CLEF_DTYPE=bfloat16 \
    CLEF_MAX_LENGTH=8192 \
    CLEF_HOST=0.0.0.0 \
    CLEF_PORT=8090 \
    CLEF_MAX_REQUEST_BYTES=524288

EXPOSE 8090

# The first start downloads ~18 GB of BF16 weights, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${CLEF_PORT}/health" || exit 1

ENTRYPOINT ["python", "/opt/decision_model/server.py"]
