#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"

usage() {
    cat <<USAGE
Usage: $0 --run-dir <PATH> [options]

Options:
  --run-dir <PATH>            Source run directory with full predictions (required)
  --max-instances <INT>       Subset size for smoke scoring (default: 2)
  --subset-run-dir <PATH>     Optional output run dir for subset (default: derived sibling)
  --conditions <CSV>          Must include baseline,llmdebug (default: baseline,llmdebug)
  --workers <INT>             Official evaluator workers (default: 2)
  --cache-level <LEVEL>       none|base|env|instance (default: base)
  --python <EXE>              Python executable (default: python)
  --module <MODULE>           Evaluator module (default: swebench.harness.run_evaluation)
  --clean                     Pass --clean to official evaluator
  --help                      Show this message
USAGE
}

RUN_DIR=""
MAX_INSTANCES="2"
SUBSET_RUN_DIR=""
CONDITIONS="baseline,llmdebug"
WORKERS="2"
CACHE_LEVEL="base"
PYTHON_EXE="python"
MODULE_NAME="swebench.harness.run_evaluation"
CLEAN_FLAG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --subset-run-dir)
            SUBSET_RUN_DIR="$2"
            shift 2
            ;;
        --conditions)
            CONDITIONS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --cache-level)
            CACHE_LEVEL="$2"
            shift 2
            ;;
        --python)
            PYTHON_EXE="$2"
            shift 2
            ;;
        --module)
            MODULE_NAME="$2"
            shift 2
            ;;
        --clean)
            CLEAN_FLAG=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: --run-dir is required"
    usage
    exit 2
fi

RUN_DIR="$(cd "$RUN_DIR" && pwd -P)"
if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run directory not found: $RUN_DIR"
    exit 2
fi

cd "$REPO_DIR"

PREPARE_CMD=(
    "$PYTHON_EXE" -m evals.swebench.cli prepare-scoring-smoke
    --run-dir "$RUN_DIR"
    --max-instances "$MAX_INSTANCES"
    --conditions "$CONDITIONS"
)
if [[ -n "$SUBSET_RUN_DIR" ]]; then
    PREPARE_CMD+=(--output-run-dir "$SUBSET_RUN_DIR")
fi

echo "Preparing scoring smoke subset..."
printf 'Command: '
printf '%q ' "${PREPARE_CMD[@]}"
echo ""
"${PREPARE_CMD[@]}"

if [[ -n "$SUBSET_RUN_DIR" ]]; then
    TARGET_RUN_DIR="$(cd "$SUBSET_RUN_DIR" && pwd -P)"
else
    TARGET_RUN_DIR="$(cd "$(dirname "$RUN_DIR")" && pwd -P)/$(basename "$RUN_DIR")-score-smoke-${MAX_INSTANCES}"
fi

SCORE_CMD=(
    "$SCRIPT_DIR/score_swebench_lite_official.sh"
    --run-dir "$TARGET_RUN_DIR"
    --conditions "$CONDITIONS"
    --workers "$WORKERS"
    --cache-level "$CACHE_LEVEL"
    --python "$PYTHON_EXE"
    --module "$MODULE_NAME"
)
if (( CLEAN_FLAG == 1 )); then
    SCORE_CMD+=(--clean)
fi

echo ""
echo "Running official scoring smoke..."
printf 'Command: '
printf '%q ' "${SCORE_CMD[@]}"
echo ""
"${SCORE_CMD[@]}"

echo ""
echo "Smoke scoring completed successfully."
echo "Subset run dir: $TARGET_RUN_DIR"
