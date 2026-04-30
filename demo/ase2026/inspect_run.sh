#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  cat >&2 <<'EOF'
Usage: demo/ase2026/inspect_run.sh <run-id> [case-id] [attempt]

Examples:
  demo/ase2026/inspect_run.sh ase2026-intervenor9b-demo
  demo/ase2026/inspect_run.sh ase2026-intervenor9b-demo cd_abc353_a_55035442_easy 2
EOF
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

args=(--run-id "$1")
if [[ $# -ge 2 && -n "$2" ]]; then
  args+=(--case "$2")
fi
if [[ $# -ge 3 && -n "$3" ]]; then
  args+=(--attempt "$3")
fi

uv run python demo/ase2026/inspect_run.py "${args[@]}"
