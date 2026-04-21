#!/usr/bin/env bash
# setup_repos.sh — Clone and pin external baseline repositories.
#
# Usage:
#   ./baselines/setup_repos.sh [target_dir]
#
# Clones LDB, INTERVENOR, TraceCoder, lambda-RLM, and upstream RLM into <target_dir>/ at
# pinned commits for reproducibility. Applies compatibility patches stored in
# baselines/<name>/patches/.
#
# Default target_dir: baselines/repos/
#
# IMPORTANT — Licensing:
#   - LDB (LLMDebugger): MIT License
#   - INTERVENOR: NO LICENSE FILE in repository (all rights reserved by default)
#   - TraceCoder: NO LICENSE FILE in repository (all rights reserved by default)
#   - lambda-RLM: MIT License
#   - RLM: MIT License
#
#   Cloned repositories are git-ignored and MUST NOT be committed.
#   They are fetched locally for experimental comparison only.
#   Do not redistribute code from unlicensed repositories.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-${SCRIPT_DIR}/repos}"

mkdir -p "$TARGET_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "python3 or python is required to run baseline repo verification." >&2
        exit 1
    fi
fi

# -----------------------------------------------------------------------
# Pinned repos/commits — sourced from baselines/pins.py.
# -----------------------------------------------------------------------
read -r LDB_REPO LDB_COMMIT \
        INTERVENOR_REPO INTERVENOR_COMMIT \
        TRACECODER_REPO TRACECODER_COMMIT \
        LAMBDA_RLM_REPO LAMBDA_RLM_COMMIT \
        RLM_REPO RLM_COMMIT <<EOF
$(
    PYTHONPATH="${SCRIPT_DIR}/.." "$PYTHON_BIN" - <<'PY'
from baselines.pins import PINNED_REPOS

ordered_keys = ("ldb", "intervenor", "tracecoder", "lambda_rlm", "rlm")
values: list[str] = []
for key in ordered_keys:
    repo = PINNED_REPOS[key]
    values.extend((repo.repo_url, repo.commit))
print(" ".join(values))
PY
)
EOF

# -----------------------------------------------------------------------
# Clone helper
# -----------------------------------------------------------------------
clone_or_update() {
    local name="$1"
    local repo="$2"
    local commit="$3"
    local patch_name="$4"
    local dest="$TARGET_DIR/$name"
    local -a patch_apply_args=(
        --ignore-space-change
        --ignore-whitespace
        --inaccurate-eof
    )

    if [ -d "$dest/.git" ]; then
        echo "[$name] Already cloned at $dest — fetching latest …"
        git -C "$dest" fetch --quiet origin
    else
        echo "[$name] Cloning $repo → $dest …"
        git clone --quiet "$repo" "$dest"
    fi

    echo "[$name] Checking out pinned commit $commit …"
    git -C "$dest" checkout --quiet --detach "$commit"

    # Apply patches if any exist.
    local patch_dir="$SCRIPT_DIR/$patch_name/patches"
    if [ -d "$patch_dir" ] && compgen -G "$patch_dir/*.patch" > /dev/null; then
        echo "[$name] Applying compatibility patches from $patch_dir …"
        for p in "$patch_dir"/*.patch; do
            echo "  → $(basename "$p")"
            if git -C "$dest" apply "${patch_apply_args[@]}" --check "$p" 2>/dev/null; then
                git -C "$dest" apply "${patch_apply_args[@]}" "$p"
            elif git -C "$dest" apply --reverse "${patch_apply_args[@]}" --check "$p" 2>/dev/null; then
                echo "  ℹ Patch $(basename "$p") already applied — skipping."
            else
                echo "  ✗ Patch $(basename "$p") does not apply cleanly to $name at $commit." >&2
                return 1
            fi
        done
    fi

    echo "[$name] Ready at $(git -C "$dest" rev-parse --short HEAD)."
    echo
}

# -----------------------------------------------------------------------
# Clone all baselines
# -----------------------------------------------------------------------
echo "=== Setting up external baseline repositories ==="
echo "Target directory: $TARGET_DIR"
echo

clone_or_update "LLMDebugger" "$LDB_REPO" "$LDB_COMMIT" "ldb"
clone_or_update "INTERVENOR"  "$INTERVENOR_REPO" "$INTERVENOR_COMMIT" "intervenor"
clone_or_update "TraceCoder"  "$TRACECODER_REPO" "$TRACECODER_COMMIT" "tracecoder"
clone_or_update "lambda-RLM"  "$LAMBDA_RLM_REPO" "$LAMBDA_RLM_COMMIT" "lambda_rlm"
clone_or_update "rlm"         "$RLM_REPO" "$RLM_COMMIT" "rlm"

echo "=== Running baseline repo smoke verification ==="
"$PYTHON_BIN" "$SCRIPT_DIR/normalize_repos.py" "$TARGET_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/verify_repos.py" "$TARGET_DIR"
echo

echo "=== All baselines ready ==="
echo
echo "Next steps:"
echo "  1. Create runtime envs:      bash baselines/setup_envs.sh"
echo "  2. Run a baseline:           python -m evals.run_eval --baseline ldb --baseline-repo-dir $TARGET_DIR/LLMDebugger"
echo "  3. Run hybrid+lambda-RLM:    python -m evals.run_eval --baseline intervenor --conditions layered_adaptive_llm --adaptive-controller-lambda-rlm"
echo "  4. Run raw RLM baseline:     python -m evals.run_eval --baseline rlm --baseline-repo-dir $TARGET_DIR/rlm"
