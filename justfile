set shell := ["bash", "-uc"]

src := "src/llmdebug"

# List available recipes
[private]
default:
    @just --list

# Run ruff linter and format check
lint:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} ruff check src tests
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} ruff format --check src tests

# Run ruff on eval harness code
lint-evals:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} ruff check evals
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} ruff format --check evals

# Auto-fix lint and formatting issues
format:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} ruff check --fix src tests
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} ruff format src tests

# Run pyright type checker
typecheck:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pyright

# Run tests without coverage
test-no-cov:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest -m "not integration"

# Run tests with coverage report
test:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest -m "not integration" --cov={{ src }} --cov-branch --cov-report=term-missing --cov-report=xml --cov-report=json --cov-fail-under=95

# Run tests without coverage (faster, stop on first failure)
test-quick:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest -x -q -m "not integration"

# Run only integration tests
test-integration:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest -m "integration and not baseline_runtime" -v

# Run provisioned baseline runtime integration tests
test-baseline-runtime:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest -m "integration and baseline_runtime" -v

# Run tests with evals/ coverage and write coverage-evals.json
test-evals-cov:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest -m "not integration" --cov=evals --cov-branch --cov-report=term-missing --cov-report=json:coverage-evals.json --ignore=evals/cases --ignore=evals/artifacts --ignore=evals/results

# Compare an existing eval coverage report with the committed informational ratchet
eval-coverage-no-regression:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python scripts/quality/check_coverage_no_regression.py --coverage-json coverage-evals.json --baseline-json quality/coverage_evals_baseline.json

# Run tests with combined src+evals coverage
test-combined-cov:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest -m "not integration" --cov={{ src }} --cov=evals --cov-branch --cov-report=term-missing --cov-report=json:coverage-combined.json --ignore=evals/cases --ignore=evals/artifacts --ignore=evals/results

# Run capture-overhead benchmarks and persist JSON results (track-only)
bench-capture:
    mkdir -p .benchmarks
    uv run pytest tests/test_benchmarks.py -k capture_overhead --benchmark-only --benchmark-group-by=group --benchmark-json=.benchmarks/capture-overhead-current.json -q

# Compare capture-overhead benchmark medians (baseline vs current)
bench-capture-compare:
    #!/usr/bin/env bash
    set -euo pipefail
    base=".benchmarks/capture-overhead-baseline.json"
    curr=".benchmarks/capture-overhead-current.json"
    if [[ ! -f "$base" ]]; then
        echo "Missing baseline benchmark file: $base"
        echo "Create one by copying a trusted run:"
        echo "  cp .benchmarks/capture-overhead-current.json .benchmarks/capture-overhead-baseline.json"
        exit 1
    fi
    if [[ ! -f "$curr" ]]; then
        echo "Missing current benchmark file: $curr"
        echo "Run: just bench-capture"
        exit 1
    fi
    uv run python scripts/bench_capture_compare.py "$base" "$curr"

# Show complex functions and maintainability index
complexity:
    @echo "=== Cyclomatic Complexity (C+ rated) ==="
    @uv run radon cc {{ src }} -a -nc -s
    @echo ""
    @echo "=== Maintainability Index ==="
    @uv run radon mi {{ src }} -s

# Report pyright type-completeness and compare to baseline (informational)
type-coverage:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python scripts/quality/check_type_coverage.py

# Fail if public type completeness regresses below the committed baseline
type-coverage-strict:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python scripts/quality/check_type_coverage.py --strict

# Fail when duplicated dev dependency lists in pyproject.toml drift apart
dependency-sync:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python scripts/quality/check_dev_dependency_sync.py

# Staged eval typing pass for a bounded subset of eval modules
typecheck-evals:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pyright -p pyrightconfig.evals.json

# Run xenon complexity gate
complexity-gate:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} xenon --max-absolute C --max-modules C --max-average B {{ src }}

# Run bandit security scanner
security:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} bandit -c pyproject.toml -r {{ src }}

# Run vulture dead code detection
dead-code:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} vulture {{ src }} vulture_whitelist.py --min-confidence 90

# Check dependency hygiene
deps:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} deptry {{ src }}

# Check dependency vulnerabilities
audit:
    # Pygments GHSA-5239-wwwm-4pmq / CVE-2026-4539 has no upstream fixed release as of 2026-03-26.
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pip-audit --ignore-vuln GHSA-5239-wwwm-4pmq --ignore-vuln CVE-2026-4539

# Synchronize auto-generated documentation blocks
docs-sync:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python -m evals.doc_sync --write

# Check documentation drift (auto-generated docs + changelog consistency)
docs-check:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python -m evals.doc_sync --check

# Base-repair eval run using the pinned template defaults (legacy recipe name kept for compatibility)
eval-traceback-only run_id:
    bash evals/docker/run_eval_docker.sh --patcher openai --openai-base-url "${OPENAI_BASE_URL:-http://127.0.0.1:12000/v1}" --config evals/configs/openai_base_repair.latest.toml --run-id {{run_id}}

# Extract failed base_repair cases into a reusable case list
eval-select-failed run_id condition="base_repair" max_attempt="2":
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python -m evals.select_cases --results-jsonl evals/results/{{run_id}}.jsonl --condition {{condition}} --mode failed --max-attempt {{max_attempt}} --out evals/artifacts/case_lists/{{run_id}}.failed.txt

# Adaptive rescue run on failed-only cases with online policy best-settings template (Docker wrapper by default)
eval-adaptive-online-failed source_run_id rescue_run_id:
    bash evals/docker/run_eval_docker.sh --patcher openai --openai-base-url "${OPENAI_BASE_URL:-http://127.0.0.1:12000/v1}" --config evals/configs/openai_adaptive.template.toml --run-id {{rescue_run_id}} --case-list-file evals/artifacts/case_lists/{{source_run_id}}.failed.txt --adaptive-online-state-path evals/artifacts/policies/{{rescue_run_id}}.json --adaptive-online-events-path evals/results/{{rescue_run_id}}.adaptive_policy.jsonl

# Build the System-1 decision-model sidecar image (CUDA host only)
decision-model-build:
    docker build -t "${DECISION_MODEL_IMAGE:-tracegate-decision-model:latest}" evals/docker/decision_model

# Start the decision-model sidecar container (CUDA host only)
decision-model-up:
    bash evals/docker/decision_model/run_decision_model.sh

# Query the decision-model sidecar health endpoint
decision-model-health:
    curl -fsS ${TRACEGATE_DECISION_MODEL_API_KEY:+-H "Authorization: Bearer $TRACEGATE_DECISION_MODEL_API_KEY"} "${TRACEGATE_DECISION_MODEL_URL:-http://127.0.0.1:8090}/health" && echo

# Latency smoke test (1k/2k/4k-token states) against a running decision-model sidecar
decision-model-smoke:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python scripts/decision_model_smoke.py

# Core summary for a single eval run (strict schema-v5)
eval-summary run_id:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python -m evals.eval_summary {{run_id}}

# Two-pass workflow summary (base_repair -> failed-only layered_adaptive_llm rescue)
eval-two-pass-summary baseline_run_id rescue_run_id baseline_condition="base_repair" rescue_condition="layered_adaptive_llm":
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python -m evals.eval_two_pass_summary {{baseline_run_id}} {{rescue_run_id}} --baseline-condition {{baseline_condition}} --rescue-condition {{rescue_condition}}

# Coverage on changed lines (PR/local branch check)
diff-coverage:
    git fetch origin main --depth=1
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} diff-cover coverage.xml --compare-branch=origin/main --fail-under=80

# Coverage no-regression gate (line and branch coverage)
coverage-no-regression:
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python scripts/quality/check_coverage_no_regression.py --coverage-json coverage.json --baseline-json quality/coverage_baseline.json

# Focused coverage gate for the baseline patch boundary seam
baseline-boundary-coverage:
    mkdir -p .pytest_cache
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} pytest tests/test_baseline_adapter_base.py tests/test_baseline_patcher.py tests/test_attempt_verification.py tests/test_evals_apply_diff_fallback.py tests/test_evals_patch_contract.py tests/test_baseline_boundary_runner.py tests/test_baseline_diff_roundtrip.py tests/test_baseline_intervenor.py tests/test_baseline_ldb.py tests/test_baseline_tracecoder.py --cov=baselines.adapter_base --cov=evals.patchers.baseline_patcher --cov=evals.eval_patches --cov=evals.attempt_verification --cov-branch --cov-report=term-missing --cov-report=json:.pytest_cache/baseline-boundary-coverage.json --cov-fail-under=0 -q
    uv run ${PYTHON_VERSION:+--python "$PYTHON_VERSION"} python scripts/quality/check_boundary_module_coverage.py --coverage-json .pytest_cache/baseline-boundary-coverage.json

# Run the pre-commit essentials (lint + types + coverage-gated tests)
check: lint typecheck test

# Run the local quality suite mirrored by GitHub CI
quality: check lint-evals coverage-no-regression baseline-boundary-coverage deps audit docs-check dead-code complexity complexity-gate security type-coverage-strict dependency-sync

# Run the single-version GitHub CI contract locally
ci: quality

# Generate a comprehensive quality report
report:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "=============================================================="
    echo "Code Quality Report - $(date +%Y-%m-%d)"
    echo "=============================================================="
    echo ""

    echo "-- Lines of Code ----------------------------------------------"
    find {{ src }} -name "*.py" -print0 | xargs -0 wc -l | sort -n
    echo ""

    echo "-- Tests ------------------------------------------------------"
    uv run pytest --co -q 2>/dev/null | tail -1
    echo ""

    echo "-- Coverage ---------------------------------------------------"
    COV_OUT=$(uv run pytest --cov={{ src }} -q 2>/dev/null)
    echo "$COV_OUT" | grep -E '(^src/|^TOTAL)' | \
        awk '{printf "  %-45s %5s %6s %5s\n", $1, $2, $3, $4}'
    echo "$COV_OUT" | grep -E '^Required' || true
    echo ""

    echo "-- Complexity Distribution ------------------------------------"
    uv run radon cc {{ src }} -a -j 2>/dev/null | uv run python -c "import json,sys; data=json.load(sys.stdin); total=sum(len(blocks) for blocks in data.values()); by_rank={}; [by_rank.__setitem__(b['rank'], by_rank.get(b['rank'], 0) + 1) for blocks in data.values() for b in blocks]; print(f'  Total blocks analyzed: {total}'); [print(f'  {rank}: {by_rank[rank]:>4} ({(by_rank[rank] * 100 // total) if total else 0}%)') for rank in sorted(by_rank)]"
    echo ""

    echo "-- High Complexity Functions (D+F rated) ----------------------"
    uv run radon cc {{ src }} -nd -s 2>/dev/null || echo "  (none)"
    echo ""

    echo "-- Maintainability Index --------------------------------------"
    uv run radon mi {{ src }} -j 2>/dev/null | uv run python -c "import json,sys; data=json.load(sys.stdin); [print(f\"  {info['rank']} ({info['mi']:5.1f})  {path.replace('src/llmdebug/', '')}\") for path, info in sorted(data.items())]"
    echo ""

    echo "-- Lint -------------------------------------------------------"
    uv run ruff check src tests --statistics 2>/dev/null && echo "  No issues" || true
    echo ""

    echo "-- Type Check -------------------------------------------------"
    uv run pyright 2>&1 | tail -1
    echo ""

    echo "-- Type Coverage (pyright --verifytypes) ----------------------"
    uv run python scripts/quality/check_type_coverage.py 2>&1 | sed 's/^/  /'
    echo ""

    echo "-- Security (bandit) ------------------------------------------"
    uv run bandit -c pyproject.toml -r {{ src }} 2>&1 | grep -E '(Total issues|Files skipped)' || true
    echo ""

    echo "-- Dead Code (vulture) ----------------------------------------"
    DEAD=$(uv run vulture {{ src }} vulture_whitelist.py --min-confidence 90 2>/dev/null | wc -l | tr -d ' ')
    echo "  Unused code items: $DEAD"
    echo ""

    echo "-- Dependencies (deptry) --------------------------------------"
    uv run deptry {{ src }} 2>&1 | tail -1
    echo ""
    echo "=============================================================="

# Remove build artifacts and caches
clean:
    rm -rf .pytest_cache .ruff_cache .pyright htmlcov .coverage coverage.xml
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Print path to the master command runbook
runbook:
    @echo "docs/runbook.md"
