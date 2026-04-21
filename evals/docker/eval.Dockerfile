# Containerized eval pipeline runner.
#
# Runs the ENTIRE eval harness (run_eval.py --executor local) inside a single
# container. No Docker-in-Docker, no testcontainers at runtime.
#
# Build:
#   docker build -t llmdebug-eval:latest -f evals/docker/eval.Dockerfile .
#
# Baseline-capable variant:
#   docker build \
#     --build-arg LLMDEBUG_EVAL_INCLUDE_BASELINES=1 \
#     -t llmdebug-eval-baselines:latest \
#     -f evals/docker/eval.Dockerfile .
#
# Run (prefer the wrapper script instead):
#   docker run --rm \
#     -v $(pwd)/evals/results:/repo/evals/results \
#     -v $(pwd)/evals/artifacts:/repo/evals/artifacts \
#     llmdebug-eval:latest --patcher null --max-cases 5 --k 1

ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:0ca07117081b2c6a8dd813d2badacf76dceecaf8b8a41d51b5d715024ffef7d8

FROM ${UV_IMAGE} AS uv

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    TMPDIR=/var/tmp/llmdebug-build-tmp

ARG LLMDEBUG_EVAL_INCLUDE_BASELINES=0

WORKDIR /repo

# Install uv from a pinned image so the builder stays reproducible.
COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git \
    && mkdir -p "$TMPDIR" \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install dependencies into an isolated virtual environment.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml uv.lock ./
# Export locked requirements, then install into /opt/venv.
# uv export reads uv.lock for reproducible, pinned installs.
# pytest is read from uv.lock to keep the eval runtime reproducible.
RUN PYTEST_VERSION="$(awk '/^\[\[package\]\]$/ { in_pkg=0 } /^name = "pytest"$/ { in_pkg=1; next } in_pkg && /^version = "/ { sub(/^version = "/, ""); sub(/"$/, ""); print; exit }' uv.lock)" \
    && if [ -z "$PYTEST_VERSION" ]; then \
        echo "pytest missing from uv.lock; refresh lockfile to include it" >&2; \
        exit 1; \
    fi \
    && uv export --frozen --no-dev --extra evals --extra cli --no-hashes --no-emit-project \
        > "$TMPDIR/requirements.txt" \
    && uv pip install --python /opt/venv/bin/python -r "$TMPDIR/requirements.txt" "pytest==$PYTEST_VERSION" \
    && rm -rf "$TMPDIR/requirements.txt" /root/.cache/uv

COPY baselines/ baselines/

RUN if [ "$LLMDEBUG_EVAL_INCLUDE_BASELINES" = "1" ]; then \
        bash baselines/setup_repos.sh /repo/baselines/repos \
        && bash baselines/setup_envs.sh /repo/baselines/envs; \
    fi

COPY README.md ./
COPY src/ src/
COPY evals/ evals/

# Keep runtime image lean even if server-side build context contains stale outputs.
RUN rm -rf evals/artifacts evals/results evals/swebench_runs \
    && mkdir -p evals/artifacts evals/results evals/swebench_runs

# Install the project itself (no-deps: deps already installed above).
RUN uv pip install --python /opt/venv/bin/python --no-deps -e . \
    && rm -rf /root/.cache/uv \
    && rm -rf "$TMPDIR" \
    && rm -f /usr/local/bin/uv

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/repo/src:/repo

ARG LLMDEBUG_EVAL_BUILD_FINGERPRINT=""
LABEL io.github.nicolasschuler.llmdebug.eval-build-fingerprint="${LLMDEBUG_EVAL_BUILD_FINGERPRINT}"

# OS deps: coreutils provides GNU timeout used by subprocess timeouts.
# patch is sufficient for eval patch application and is much smaller than git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends coreutils patch \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /repo

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /repo/README.md /repo/README.md
COPY --from=builder /repo/src /repo/src
COPY --from=builder /repo/evals /repo/evals
COPY --from=builder /repo/baselines /repo/baselines

# Sanity check: llmdebug is importable.
RUN python -c "import llmdebug; print(f'llmdebug {llmdebug.__version__} OK')"

ENTRYPOINT ["python", "-m", "evals.run_eval"]
CMD ["--help"]
