#!/usr/bin/env bash
# setup_envs.sh — Create isolated runtime envs for external baselines and controller backends.
#
# Usage:
#   ./baselines/setup_envs.sh [env_root] [repo_root]
#
# repo_root defaults to:
#   - $BASELINE_REPO_ROOT when set
#   - otherwise baselines/repos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="${1:-${SCRIPT_DIR}/envs}"
REPO_ROOT="${BASELINE_REPO_ROOT:-${2:-${SCRIPT_DIR}/repos}}"
MODE="${BASELINE_ENV_MODE:-smoke}"

mkdir -p "$ENV_ROOT"

python_version_triplet() {
    local python_bin="$1"
    "$python_bin" -c 'import sys; print(".".join(str(x) for x in sys.version_info[:3]))' 2>/dev/null
}

python_satisfies_min_version() {
    local python_bin min_major min_minor version major minor patch
    python_bin="$1"
    min_major="$2"
    min_minor="$3"
    version="$(python_version_triplet "$python_bin" || true)"
    if [ -z "$version" ]; then
        return 1
    fi
    IFS=. read -r major minor patch <<EOF
$version
EOF
    [ "${major:-0}" -gt "$min_major" ] || {
        [ "${major:-0}" -eq "$min_major" ] && [ "${minor:-0}" -ge "$min_minor" ]
    }
}

resolve_venv_python() {
    local resolved=""
    local first_incompatible_path=""
    local first_incompatible_version=""

    maybe_select_candidate() {
        local candidate_path="$1"
        if [ -z "$candidate_path" ] || [ ! -x "$candidate_path" ]; then
            return 1
        fi
        if [ -n "${REQUIRED_ENV_PYTHON_MAJOR:-}" ] && [ -n "${REQUIRED_ENV_PYTHON_MINOR:-}" ]; then
            if python_satisfies_min_version \
                "$candidate_path" \
                "$REQUIRED_ENV_PYTHON_MAJOR" \
                "$REQUIRED_ENV_PYTHON_MINOR"
            then
                printf '%s\n' "$candidate_path"
                return 0
            fi
            if [ -z "$first_incompatible_path" ]; then
                first_incompatible_path="$candidate_path"
                first_incompatible_version="$(python_version_triplet "$candidate_path" || echo unknown)"
            fi
            return 1
        fi
        printf '%s\n' "$candidate_path"
        return 0
    }

    if [ -n "${BASELINE_ENV_PYTHON:-}" ]; then
        if [ ! -x "$BASELINE_ENV_PYTHON" ]; then
            echo "BASELINE_ENV_PYTHON is not executable: $BASELINE_ENV_PYTHON" >&2
            exit 1
        fi
        printf '%s\n' "$BASELINE_ENV_PYTHON"
        return 0
    fi

    for candidate in python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            resolved="$(command -v "$candidate")"
            if maybe_select_candidate "$resolved"; then
                return 0
            fi
        fi
    done

    if command -v uv >/dev/null 2>&1; then
        for request in 3.12 3.11 3.10; do
            resolved="$(uv python find "$request" 2>/dev/null || true)"
            if maybe_select_candidate "$resolved"; then
                return 0
            fi
        done
    fi

    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            resolved="$(command -v "$candidate")"
            if maybe_select_candidate "$resolved"; then
                return 0
            fi
        fi
    done
    echo "setup_envs.sh requires Python >=${REQUIRED_ENV_PYTHON_MAJOR}.${REQUIRED_ENV_PYTHON_MINOR} on PATH (or via uv) for ${REQUIRED_ENV_PYTHON_LABEL}." >&2
    if [ -n "$first_incompatible_path" ]; then
        echo "First incompatible candidate was $first_incompatible_path (Python $first_incompatible_version)." >&2
    fi
    echo "Set BASELINE_ENV_PYTHON to a compatible interpreter such as python3.10, python3.11, or python3.12." >&2
    exit 1
}

require_python_min_version() {
    local python_bin min_major min_minor requirement_label version
    python_bin="$1"
    min_major="$2"
    min_minor="$3"
    requirement_label="$4"
    if python_satisfies_min_version "$python_bin" "$min_major" "$min_minor"; then
        return 0
    fi
    version="$(python_version_triplet "$python_bin" || echo unknown)"
    echo "Selected interpreter $python_bin reports Python ${version}, but ${requirement_label} requires Python >=${min_major}.${min_minor}." >&2
    echo "Set BASELINE_ENV_PYTHON to a compatible interpreter such as python3.10, python3.11, or python3.12 and rerun baselines/setup_envs.sh." >&2
    exit 1
}

create_portable_env() {
    local env_path="$1"
    "$VENV_PYTHON" -m venv --clear --copies --without-pip "$env_path"
    ensure_portable_env_python_runtime "$env_path"
}

python_runtime_lib_dir() {
    "$VENV_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.executable).resolve().parent.parent / "lib")'
}

ensure_portable_env_python_runtime() {
    local env_path="$1"
    local env_python="$env_path/bin/python"
    local runtime_lib_dir=""
    local runtime_lib=""
    local copied_runtime_lib=0

    if "$env_python" -c 'import sys' >/dev/null 2>&1; then
        return 0
    fi

    # uv-managed Linux interpreters can produce copy-based venvs whose copied
    # python binary cannot resolve ../lib/libpython*.so during bootstrap.
    runtime_lib_dir="$(python_runtime_lib_dir)"
    if [ ! -d "$runtime_lib_dir" ]; then
        echo "Created env python at $env_python is not runnable, and the source runtime lib dir is missing: $runtime_lib_dir" >&2
        "$env_python" -c 'import sys' >/dev/null
        return 1
    fi

    mkdir -p "$env_path/lib"
    for runtime_lib in \
        "$runtime_lib_dir"/libpython*.so* \
        "$runtime_lib_dir"/libpython*.dylib \
        "$runtime_lib_dir"/python*.dll
    do
        if [ ! -e "$runtime_lib" ]; then
            continue
        fi
        cp -P "$runtime_lib" "$env_path/lib/"
        copied_runtime_lib=1
    done

    if [ "$copied_runtime_lib" -eq 0 ]; then
        echo "Created env python at $env_python is not runnable, but no Python shared runtime library was found under $runtime_lib_dir" >&2
        "$env_python" -c 'import sys' >/dev/null
        return 1
    fi

    if "$env_python" -c 'import sys' >/dev/null 2>&1; then
        return 0
    fi

    echo "Copied Python runtime shared libraries into $env_path/lib, but $env_python still failed to start." >&2
    "$env_python" -c 'import sys' >/dev/null
    return 1
}

venv_site_packages_dir() {
    local env_path="$1"
    local python_major_minor="$2"
    printf '%s/lib/python%s/site-packages\n' "$env_path" "$python_major_minor"
}

install_vllm_smoke_stub() {
    local python_major_minor="$1"
    local target_dir
    target_dir="$(venv_site_packages_dir "$LDB_ENV" "$python_major_minor")"
    mkdir -p "$target_dir"
    cp "$SCRIPT_DIR/stubs/vllm_stub/vllm.py" "$target_dir/vllm.py"
}

read_project_runtime_dependencies() {
    local repo_path="$1"
    "$VENV_PYTHON" - "$repo_path" <<'PY'
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

repo_path = Path(sys.argv[1])
pyproject_path = repo_path / "pyproject.toml"
payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
for dependency in payload.get("project", {}).get("dependencies", []):
    value = str(dependency).strip()
    if value:
        print(value)
PY
}

install_repo_runtime_dependencies() {
    local env_python="$1"
    local repo_path="$2"
    local runtime_dependencies=()
    local dependency=""
    while IFS= read -r dependency; do
        if [ -n "$dependency" ]; then
            runtime_dependencies+=("$dependency")
        fi
    done < <(read_project_runtime_dependencies "$repo_path")
    if [ "${#runtime_dependencies[@]}" -eq 0 ]; then
        return 0
    fi
    uv pip install --python "$env_python" "${runtime_dependencies[@]}"
}

LDB_ENV="$ENV_ROOT/ldb"
INTERVENOR_ENV="$ENV_ROOT/intervenor"
TRACECODER_ENV="$ENV_ROOT/tracecoder"
LAMBDA_RLM_ENV="$ENV_ROOT/lambda_rlm"
RLM_ENV="$ENV_ROOT/rlm"
LAMBDA_RLM_REPO="$REPO_ROOT/lambda-RLM"
RLM_REPO="$REPO_ROOT/rlm"
LAMBDA_RLM_MANIFEST=""
RLM_MANIFEST=""
REQUIRED_ENV_PYTHON_MAJOR=3
REQUIRED_ENV_PYTHON_MINOR=10
REQUIRED_ENV_PYTHON_LABEL="baseline smoke envs"

if [ -d "$LAMBDA_RLM_REPO" ] || [ -d "$RLM_REPO" ]; then
    REQUIRED_ENV_PYTHON_MINOR=11
    REQUIRED_ENV_PYTHON_LABEL="lambda_rlm/rlm baseline envs"
fi

VENV_PYTHON="$(resolve_venv_python)"
VENV_PYTHON_VERSION="$(python_version_triplet "$VENV_PYTHON" || echo unknown)"
VENV_PYTHON_MAJOR_MINOR="${VENV_PYTHON_VERSION%.*}"
require_python_min_version \
    "$VENV_PYTHON" \
    "$REQUIRED_ENV_PYTHON_MAJOR" \
    "$REQUIRED_ENV_PYTHON_MINOR" \
    "$REQUIRED_ENV_PYTHON_LABEL"

case "$VENV_PYTHON_MAJOR_MINOR" in
    *.*) ;;
    *)
        echo "Could not derive a major.minor Python version from: $VENV_PYTHON_VERSION" >&2
        exit 1
        ;;
esac

echo "=== Setting up external baseline runtime envs ==="
echo "Mode: $MODE"
echo "Env root: $ENV_ROOT"
echo "Repo root: $REPO_ROOT"
echo "Venv python: $VENV_PYTHON (Python $VENV_PYTHON_VERSION)"
echo

create_portable_env "$LDB_ENV"
create_portable_env "$INTERVENOR_ENV"
create_portable_env "$TRACECODER_ENV"

echo "[ldb] Installing smoke runtime requirements ..."
uv pip install --python "$LDB_ENV/bin/python" -r "$SCRIPT_DIR/ldb/requirements-smoke.txt"
install_vllm_smoke_stub "$VENV_PYTHON_MAJOR_MINOR"

echo "[intervenor] Installing smoke runtime requirements ..."
uv pip install --python "$INTERVENOR_ENV/bin/python" -r "$SCRIPT_DIR/intervenor/requirements-smoke.txt"

echo "[tracecoder] Installing smoke runtime requirements ..."
uv pip install --python "$TRACECODER_ENV/bin/python" -r "$SCRIPT_DIR/tracecoder/requirements-smoke.txt"

if [ -d "$LAMBDA_RLM_REPO" ]; then
    create_portable_env "$LAMBDA_RLM_ENV"
    echo "[lambda_rlm] Installing controller backend requirements ..."
    install_repo_runtime_dependencies "$LAMBDA_RLM_ENV/bin/python" "$LAMBDA_RLM_REPO"
    LAMBDA_RLM_MANIFEST=$(cat <<EOF
,
  "lambda_rlm": {
    "python_executable": "$LAMBDA_RLM_ENV/bin/python",
    "python_repo_relative": "baselines/envs/lambda_rlm/bin/python",
    "repo_dir": "$LAMBDA_RLM_REPO",
    "repo_dir_repo_relative": "baselines/repos/lambda-RLM"
  }
EOF
)
else
    echo "[lambda_rlm] Missing cloned repo at $LAMBDA_RLM_REPO. Skipping optional controller backend env."
fi

if [ -d "$RLM_REPO" ]; then
    create_portable_env "$RLM_ENV"
    echo "[rlm] Installing standalone RLM baseline requirements ..."
    install_repo_runtime_dependencies "$RLM_ENV/bin/python" "$RLM_REPO"
    RLM_MANIFEST=$(cat <<EOF
,
  "rlm": {
    "python_executable": "$RLM_ENV/bin/python",
    "python_repo_relative": "baselines/envs/rlm/bin/python",
    "repo_dir": "$RLM_REPO",
    "repo_dir_repo_relative": "baselines/repos/rlm"
  }
EOF
)
else
    echo "[rlm] Missing cloned repo at $RLM_REPO. Skipping optional standalone baseline env."
fi

cat > "$ENV_ROOT/runtime_manifest.json" <<EOF
{
  "mode": "$MODE",
  "creation_backend": "python_venv_copies",
  "ldb": {
    "python_executable": "$LDB_ENV/bin/python",
    "python_repo_relative": "baselines/envs/ldb/bin/python"
  },
  "intervenor": {
    "python_executable": "$INTERVENOR_ENV/bin/python",
    "python_repo_relative": "baselines/envs/intervenor/bin/python"
  },
  "tracecoder": {
    "python_executable": "$TRACECODER_ENV/bin/python",
    "python_repo_relative": "baselines/envs/tracecoder/bin/python"
  }
$LAMBDA_RLM_MANIFEST$RLM_MANIFEST
}
EOF

echo
echo "=== Baseline envs ready ==="
echo "[ldb]         $LDB_ENV/bin/python"
echo "[intervenor] $INTERVENOR_ENV/bin/python"
echo "[tracecoder] $TRACECODER_ENV/bin/python"
if [ -n "$LAMBDA_RLM_MANIFEST" ]; then
    echo "[lambda_rlm] $LAMBDA_RLM_ENV/bin/python"
else
    echo "[lambda_rlm] skipped (repo missing)"
fi
if [ -n "$RLM_MANIFEST" ]; then
    echo "[rlm]        $RLM_ENV/bin/python"
else
    echo "[rlm]        skipped (repo missing)"
fi
