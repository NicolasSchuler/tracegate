#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-ase2026-intervenor9b-demo-$(date +%Y%m%dT%H%M%S)}"
RUNNER="${RUNNER:-docker}"
BASELINE="${BASELINE:-intervenor}"
CONFIG="${CONFIG:-evals/configs/baseline_intervenor.template.toml}"
CASE_LIST="${CASE_LIST:-demo/ase2026/cases.txt}"

OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:12000/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-Qwen3.5-9B-UD-Q6_K_XL}"
BASELINE_OPENAI_COMPAT_PROFILE="${BASELINE_OPENAI_COMPAT_PROFILE:-lmstudio_model_yaml}"

SOURCE_RUN_PREFIX="${SOURCE_RUN_PREFIX:-analysis/data/final-intervenor-adaptive-repair-9b}"
ONLINE_EVENTS="${ONLINE_EVENTS:-${SOURCE_RUN_PREFIX}.adaptive_policy.jsonl}"
BUDGET_EVENTS="${BUDGET_EVENTS:-${SOURCE_RUN_PREFIX}.adaptive_budget_policy.jsonl}"
POLICY_DIR="${POLICY_DIR:-evals/artifacts/policies}"
ONLINE_INIT_STATE="${ONLINE_INIT_STATE:-${POLICY_DIR}/ase2026-intervenor9b-prewarm.online.json}"
BUDGET_INIT_STATE="${BUDGET_INIT_STATE:-${POLICY_DIR}/ase2026-intervenor9b-prewarm.budget.json}"
REBUILD_WARMUP="${REBUILD_WARMUP:-0}"

K="${K:-2}"
JOBS="${JOBS:-1}"
HELPER_EPSILON="${HELPER_EPSILON:-0.0}"
BUDGET_EPSILON="${BUDGET_EPSILON:-0.0}"
RECORD_PROMPT="${RECORD_PROMPT:-full}"
RECORD_EVIDENCE="${RECORD_EVIDENCE:-full}"
KEEP_WORKDIRS="${KEEP_WORKDIRS:-1}"
ENABLE_TUI="${ENABLE_TUI:-1}"

if [[ "$ENABLE_TUI" == "1" ]]; then
  # The eval runner uses the Rich live dashboard automatically on TTY stdout
  # when color has not been disabled.
  unset NO_COLOR
  export TERM="${TERM:-xterm-256color}"
fi

if [[ ! -f "$ONLINE_EVENTS" || ! -f "$BUDGET_EVENTS" ]]; then
  cat >&2 <<EOF
Missing policy event logs.

Expected:
  $ONLINE_EVENTS
  $BUDGET_EVENTS

Download/extract the ASE demo data into analysis/data/ first. The expected
source is the Zenodo record for the TraceGate evaluation artifacts.
EOF
  exit 2
fi

if [[ ! -f "$CASE_LIST" ]]; then
  echo "Missing case list: $CASE_LIST" >&2
  exit 2
fi

mkdir -p "$POLICY_DIR"

if [[ "$REBUILD_WARMUP" == "1" || ! -f "$ONLINE_INIT_STATE" ]]; then
  uv run python -m evals.bootstrap_policy_state \
    --policy-kind online \
    --events "$ONLINE_EVENTS" \
    --output-state "$ONLINE_INIT_STATE"
fi

if [[ "$REBUILD_WARMUP" == "1" || ! -f "$BUDGET_INIT_STATE" ]]; then
  uv run python -m evals.bootstrap_policy_state \
    --policy-kind budget \
    --events "$BUDGET_EVENTS" \
    --output-state "$BUDGET_INIT_STATE"
fi

keep_args=()
if [[ "$KEEP_WORKDIRS" == "1" ]]; then
  keep_args+=(--keep-workdirs)
fi

echo "ASE 2026 TraceGate demo run"
echo "  run_id: $RUN_ID"
echo "  cases:  $CASE_LIST"
echo "  model:  $OPENAI_MODEL"
echo "  url:    $OPENAI_BASE_URL"
echo "  policy: $ONLINE_INIT_STATE"
echo "  budget: $BUDGET_INIT_STATE"
echo "  runner: $RUNNER"
echo "  tui:    $ENABLE_TUI"

eval_args=(
  --config "$CONFIG" \
  --baseline "$BASELINE" \
  --conditions layered_adaptive_llm \
  --case-list-file "$CASE_LIST" \
  --run-id "$RUN_ID" \
  --jobs "$JOBS" \
  --k "$K" \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL" \
  --baseline-openai-compat-profile "$BASELINE_OPENAI_COMPAT_PROFILE" \
  --adaptive-online-policy-mode on \
  --adaptive-budget-policy-mode on \
  --adaptive-online-init-state-path "$ONLINE_INIT_STATE" \
  --adaptive-budget-init-state-path "$BUDGET_INIT_STATE" \
  --adaptive-online-state-path "evals/artifacts/policies/${RUN_ID}.online.json" \
  --adaptive-budget-online-state-path "evals/artifacts/policies/${RUN_ID}.budget.json" \
  --adaptive-online-events-path "evals/results/${RUN_ID}.adaptive_policy.jsonl" \
  --adaptive-budget-online-events-path "evals/results/${RUN_ID}.adaptive_budget_policy.jsonl" \
  --adaptive-online-warmup-decisions 0 \
  --adaptive-budget-online-warmup-decisions 0 \
  --adaptive-online-epsilon "$HELPER_EPSILON" \
  --adaptive-budget-online-epsilon "$BUDGET_EPSILON" \
  --adaptive-max-helper-calls 1 \
  --adaptive-budget-topup-max-calls 1 \
  --record-prompt "$RECORD_PROMPT" \
  --record-evidence "$RECORD_EVIDENCE" \
  --record-deps minimal \
  --progress-verbosity detailed \
  --progress-heartbeat-sec 15 \
  "${keep_args[@]}"
)

case "$RUNNER" in
  docker)
    # The Docker wrapper auto-selects the baseline-capable image for
    # --baseline runs, rebuilds when the image is missing/stale, and translates
    # localhost model endpoints for container networking.
    LLMDEBUG_EVAL_DOCKER_TTY="${LLMDEBUG_EVAL_DOCKER_TTY:-$ENABLE_TUI}" \
      bash evals/docker/run_eval_docker.sh "${eval_args[@]}"
    ;;
  local)
    uv run python -m evals.run_eval "${eval_args[@]}"
    ;;
  *)
    echo "Unsupported RUNNER=$RUNNER; expected docker or local." >&2
    exit 2
    ;;
esac

uv run python demo/ase2026/summarize_demo.py --run-id "$RUN_ID"
