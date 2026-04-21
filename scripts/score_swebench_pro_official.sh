#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"

usage() {
    cat <<USAGE
Usage: $0 --run-dir <PATH> [options]

Options:
  --run-dir <PATH>          Path to copied swebench_pro run directory (required)
  --conditions <CSV>        baseline,llmdebug by default
  --workers <INT>           Docker scorer workers (default: 4)
  --timeout <SEC>           Per-instance timeout in seconds (default: 3600)
  --dataset-name <NAME>     Dataset name (default: ScaleAI/SWE-bench_Pro)
  --split <NAME>            Split name (default: test)
  --dockerhub-username <U>  DockerHub org/user hosting images (default: jefzda)
  --run-scripts-dir <PATH>  Optional path to cached run_scripts
  --dockerfiles-dir <PATH>  Optional path to cached dockerfiles
  --docker-platform <PLAT>  Optional Docker platform override
  --python <EXE>            Python executable (default: python)
  --block-network           Run containers with network_mode=none
  --redo                    Re-run scoring even if cached outputs exist
  --help                    Show this message
USAGE
}

RUN_DIR=""
CONDITIONS="baseline,llmdebug"
WORKERS="4"
TIMEOUT="3600"
DATASET_NAME="ScaleAI/SWE-bench_Pro"
SPLIT="test"
DOCKERHUB_USERNAME="jefzda"
RUN_SCRIPTS_DIR=""
DOCKERFILES_DIR=""
DOCKER_PLATFORM=""
PYTHON_EXE="python"
BLOCK_NETWORK=0
REDO=0

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
        --timeout)
            TIMEOUT="$2"
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
        --dockerhub-username)
            DOCKERHUB_USERNAME="$2"
            shift 2
            ;;
        --run-scripts-dir)
            RUN_SCRIPTS_DIR="$2"
            shift 2
            ;;
        --dockerfiles-dir)
            DOCKERFILES_DIR="$2"
            shift 2
            ;;
        --docker-platform)
            DOCKER_PLATFORM="$2"
            shift 2
            ;;
        --python)
            PYTHON_EXE="$2"
            shift 2
            ;;
        --block-network)
            BLOCK_NETWORK=1
            shift
            ;;
        --redo)
            REDO=1
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
    "$PYTHON_EXE" -m evals.swebench_pro.cli score-docker
    --run-dir "$RUN_DIR"
    --conditions "$CONDITIONS"
    --dataset-name "$DATASET_NAME"
    --split "$SPLIT"
    --dockerhub-username "$DOCKERHUB_USERNAME"
    --workers "$WORKERS"
    --timeout "$TIMEOUT"
)

if [[ -n "$RUN_SCRIPTS_DIR" ]]; then
    CMD+=(--run-scripts-dir "$RUN_SCRIPTS_DIR")
fi
if [[ -n "$DOCKERFILES_DIR" ]]; then
    CMD+=(--dockerfiles-dir "$DOCKERFILES_DIR")
fi
if [[ -n "$DOCKER_PLATFORM" ]]; then
    CMD+=(--docker-platform "$DOCKER_PLATFORM")
fi
if (( BLOCK_NETWORK == 1 )); then
    CMD+=(--block-network)
fi
if (( REDO == 1 )); then
    CMD+=(--redo)
fi

echo "Scoring SWE-Bench Pro predictions"
echo "Run dir:            $RUN_DIR"
echo "Conditions:         $CONDITIONS"
echo "Workers:            $WORKERS"
echo "Timeout:            $TIMEOUT"
echo "Dataset:            $DATASET_NAME"
echo "Split:              $SPLIT"
echo "DockerHub username: $DOCKERHUB_USERNAME"
echo ""
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"

echo ""
echo "Done. Raw Docker scoring outputs are under: ${RUN_DIR}/docker_eval"
