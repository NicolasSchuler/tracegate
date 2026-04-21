#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"

usage() {
    cat <<USAGE
Usage: $0 --run-dir <PATH> [options]

Options:
  --run-dir <PATH>          Path to copied swebench run directory (required)
  --conditions <CSV>        baseline,llmdebug by default
  --workers <INT>           Official evaluator workers (default: 6)
  --cache-level <LEVEL>     none|base|env|instance (default: env)
  --dataset-name <NAME>     Dataset name (default: princeton-nlp/SWE-bench_Lite)
  --split <NAME>            Split name (default: test)
  --python <EXE>            Python executable (default: python)
  --module <MODULE>         Evaluator module (default: swebench.harness.run_evaluation)
  --clean                   Pass --clean to official evaluator
  --help                    Show this message
USAGE
}

RUN_DIR=""
CONDITIONS="baseline,llmdebug"
WORKERS="6"
CACHE_LEVEL="env"
DATASET_NAME="princeton-nlp/SWE-bench_Lite"
SPLIT="test"
PYTHON_EXE="python"
MODULE_NAME="swebench.harness.run_evaluation"
CLEAN_FLAG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"
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
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
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

CMD=(
    "$PYTHON_EXE" -m evals.swebench.cli score-official
    --run-dir "$RUN_DIR"
    --conditions "$CONDITIONS"
    --workers "$WORKERS"
    --cache-level "$CACHE_LEVEL"
    --dataset-name "$DATASET_NAME"
    --split "$SPLIT"
    --python-executable "$PYTHON_EXE"
    --module "$MODULE_NAME"
)

if (( CLEAN_FLAG == 1 )); then
    CMD+=(--clean)
fi

echo "Scoring SWE-Bench Lite predictions"
echo "Run dir:     $RUN_DIR"
echo "Conditions:  $CONDITIONS"
echo "Workers:     $WORKERS"
echo "Cache level: $CACHE_LEVEL"
echo "Dataset:     $DATASET_NAME"
echo "Split:       $SPLIT"
echo ""
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"

echo ""
echo "Done. Raw official outputs are under: ${RUN_DIR}/official_eval"
