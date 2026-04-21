FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /tmp/runner-build
COPY uv.lock ./uv.lock

# Keep the runner aligned with the repository lockfile so testcontainers stay reproducible.
RUN PYTEST_VERSION="$(awk '/^\[\[package\]\]$/ { in_pkg=0 } /^name = "pytest"$/ { in_pkg=1; next } in_pkg && /^version = "/ { sub(/^version = "/, ""); sub(/"$/, ""); print; exit }' uv.lock)" \
    && FILELOCK_VERSION="$(awk '/^\[\[package\]\]$/ { in_pkg=0 } /^name = "filelock"$/ { in_pkg=1; next } in_pkg && /^version = "/ { sub(/^version = "/, ""); sub(/"$/, ""); print; exit }' uv.lock)" \
    && if [ -z "$PYTEST_VERSION" ] || [ -z "$FILELOCK_VERSION" ]; then \
        echo "pytest/filelock missing from uv.lock; refresh lockfile before rebuilding the runner image" >&2; \
        exit 1; \
    fi \
    && python -m pip install --no-cache-dir \
        "pytest==$PYTEST_VERSION" \
        "filelock==$FILELOCK_VERSION" \
    && rm -f uv.lock

WORKDIR /workspace
