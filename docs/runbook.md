# Master Runbook

Last verified: July 12, 2026

All shell commands, `just` recipes, and CLI invocations in one place.
For detailed prose, context, and config examples, follow the cross-references.

**Cross-references:**

| Topic | Detailed doc |
|-------|-------------|
| Package configuration | [configuration.md](configuration.md) |
| CLI flags reference | [cli-reference.md](cli-reference.md) |
| Quality gate map | [quality-map.md](quality-map.md) |
| Eval conditions & prompt structure | [eval_quickstart.md](eval_quickstart.md) |
| Full Docker/local eval ops | [local_docker_eval_runbook.md](local_docker_eval_runbook.md) |
| Baseline hybrid study matrix | [baseline_hybrid_study_runbook.md](baseline_hybrid_study_runbook.md) |
| Eval framework & case contract | [../evals/README.md](../evals/README.md) |
| Baseline adapters | [baseline-adapters.md](baseline-adapters.md) |
| Testing guide | [testing-guide.md](testing-guide.md) |
| Troubleshooting | [troubleshooting.md](troubleshooting.md) |

---

## 1. Quick Reference

Most-used commands at a glance.

```bash
# --- Setup ---
uv sync --group dev                                # Install dev deps
just --list                                        # List all just recipes

# --- Quality (pre-commit) ---
just check                                         # lint + typecheck + tests
just quality                                       # full CI contract
just test-quick                                    # fast fail-fast tests only

# --- Debug CLI ---
llmdebug quickstart                                # create + inspect an offline example
llmdebug                                           # latest snapshot (crash view)
llmdebug diff                                      # compare latest vs previous

# --- Eval smoke ---
bash evals/docker/run_eval_docker.sh --patcher null --max-cases 5 --k 1

# --- Eval summary ---
just eval-summary <run_id>

# --- Two-pass workflow (shorthand) ---
just eval-traceback-only <base_run_id>
just eval-select-failed <base_run_id>
just eval-adaptive-online-failed <base_run_id> <rescue_run_id>
just eval-two-pass-summary <base_run_id> <rescue_run_id>
```

---

## 2. Environment Setup

### Package installation

```bash
pip install 'llmdebug[cli]'          # pytest plugin + CLI (recommended)
pip install llmdebug                  # core + pytest plugin only
pip install 'llmdebug[mcp]'          # MCP server for IDE integration
pip install 'llmdebug[jupyter]'      # Jupyter/IPython plugin
pip install 'llmdebug[toon]'         # TOON output format
pip install 'llmdebug[evals]'        # eval dataset importers (HuggingFace)
```

### Development environment

```bash
uv sync --group dev                  # install all dev dependencies
just --list                          # list available just recipes
```

### External baseline repos and envs

```bash
chmod +x baselines/setup_repos.sh
./baselines/setup_repos.sh           # clone and patch all baseline repos

bash baselines/setup_envs.sh         # create isolated runtime envs
# optional: pass repo root if clones are non-default
BASELINE_REPO_ROOT=/path/to/repos bash baselines/setup_envs.sh
```

## 3. Development & Quality

### Quick pass (pre-commit essentials)

```bash
just check             # lint + typecheck + coverage-gated tests
just test-quick        # pytest fail-fast, no coverage
just lint              # ruff check + format check (src, tests)
just typecheck         # pyright strict mode (src/llmdebug)
```

### Full local quality suite (mirrors GitHub CI)

```bash
just quality           # complete quality contract
just ci                # alias for just quality
```

### Individual lint / format

```bash
uv run ruff check src tests               # lint
uv run ruff format --check src tests      # format check
uv run ruff check --fix src tests         # auto-fix lint
uv run ruff format src tests              # auto-format
just format                               # format src + tests via just
just lint-evals                           # ruff check + format check for evals/
uv run ruff check evals                   # eval harness lint (direct)
uv run ruff format --check evals          # eval harness format check (direct)
```

### Type checking

```bash
just typecheck                                            # strict mode, src/llmdebug
uv run pyright                                            # direct equivalent
just typecheck-evals                                      # staged basic mode for evals/
uv run pyright -p pyrightconfig.evals.json                # direct equivalent
just type-coverage                                        # informational verifytypes snapshot
just type-coverage-strict                                 # enforced no-regression gate
uv run python scripts/quality/check_type_coverage.py      # direct (informational)
uv run python scripts/quality/check_type_coverage.py --strict  # direct (enforced)
```

### Tests

```bash
just test-no-cov       # full suite, no coverage
just test              # full suite with coverage (--cov-fail-under=95)
just test-quick        # fail-fast, no coverage
just test-integration  # blocking integration lane, excludes baseline_runtime
just test-baseline-runtime  # provisioned baseline repo/env/runtime lane
just test-evals-cov    # generate evals/ coverage report
just eval-coverage-no-regression  # compare report with informational ratchet
just test-combined-cov # combined src + evals coverage report

# direct equivalents
uv run pytest -m "not integration"
uv run pytest -x -q -m "not integration"
uv run pytest -m "integration and not baseline_runtime" -v
uv run pytest -m "integration and baseline_runtime" -v
uv run pytest -m "not integration" \
  --cov=src/llmdebug --cov-branch --cov-report=term-missing \
  --cov-report=xml --cov-report=json --cov-fail-under=95
```

### Coverage gates

```bash
just diff-coverage            # changed-line coverage gate (>=80%) vs origin/main
just coverage-no-regression   # line + branch no-regression vs quality/coverage_baseline.json
just baseline-boundary-coverage   # focused seam gate for baseline patching

# direct equivalents
git fetch origin main --depth=1
uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
uv run python scripts/quality/check_coverage_no_regression.py \
  --coverage-json coverage.json --baseline-json quality/coverage_baseline.json
```

### Security, dead code, dependencies

```bash
just security     # bandit security scan
just dead-code    # vulture dead code detection (--min-confidence 90)
just deps         # deptry dependency hygiene
just audit        # pip-audit vulnerability scan

# direct equivalents
uv run bandit -c pyproject.toml -r src/llmdebug
uv run vulture src/llmdebug vulture_whitelist.py --min-confidence 90
uv run deptry src/llmdebug
uv run pip-audit --ignore-vuln GHSA-5239-wwwm-4pmq --ignore-vuln CVE-2026-4539
```

### Complexity

```bash
just complexity          # radon cyclomatic complexity report
just complexity-gate     # xenon gate (max-absolute C, max-modules C, max-average B)

# direct equivalents
uv run radon cc src/llmdebug -a -nc -s
uv run radon mi src/llmdebug -s
uv run xenon --max-absolute C --max-modules C --max-average B src/llmdebug
```

### Docs drift

```bash
just docs-check    # check for auto-generated block drift + changelog consistency
just docs-sync     # refresh auto-generated blocks (evals case inventory, etc.)
just dependency-sync   # verify dev dependency list sync in pyproject.toml

# direct equivalents
uv run python -m evals.doc_sync --check
uv run python -m evals.doc_sync --write
uv run python scripts/quality/check_dev_dependency_sync.py
```

### Benchmarks

```bash
just bench-capture          # run capture-overhead microbenchmarks
just bench-capture-compare  # compare medians: baseline vs current

# establish baseline, then compare
cp .benchmarks/capture-overhead-current.json .benchmarks/capture-overhead-baseline.json
# ... make changes ...
just bench-capture
just bench-capture-compare
```

### Comprehensive quality report

```bash
just report         # generate code quality summary to stdout
```

### Cleanup

```bash
just clean          # remove .pytest_cache, .ruff_cache, .pyright, htmlcov, .coverage, coverage.xml
```

---

## 4. Debug CLI (`llmdebug`)

Requires `pip install 'llmdebug[cli]'`.

### show

```bash
llmdebug quickstart                   # create and inspect an offline example
llmdebug                              # latest snapshot (crash-level detail)
llmdebug show                         # same as above
llmdebug show previous                # second most recent snapshot
llmdebug show --detail full           # all stack frames
llmdebug show --detail context        # everything (pytest, env, git, repro)
llmdebug show --json                  # crash-level JSON
llmdebug show --json --detail context # full JSON with context
llmdebug show --raw-session           # raw DebugSession envelope JSON
```

### list / frames / diff / git-context

```bash
llmdebug list                         # list all snapshots
llmdebug list --search "KeyError"     # search by exception/file/name

llmdebug frames -i 0                  # crash frame from latest snapshot
llmdebug frames previous -i 0        # crash frame from previous snapshot
llmdebug frames '#3' -i 2            # frame 2 from snapshot #3

llmdebug diff                         # compare latest vs previous
llmdebug diff #2 #1                   # compare by list index
llmdebug diff old.json new.json       # compare specific files
llmdebug diff --json                  # output diff as JSON

llmdebug git-context                  # latest snapshot, text view
llmdebug git-context --json           # JSON for tooling
llmdebug git-context '#2'            # specific snapshot reference
```

### export / clean

```bash
llmdebug export                               # HTML beside the snapshot file
llmdebug export --format json                 # machine-readable payload
llmdebug export previous --compare latest     # compare two snapshots
llmdebug export --output reports/failure.html
llmdebug export --mode redacted --redaction-profile prod
llmdebug export --bundle-dir reports/export-bundle

llmdebug clean -k 5                           # keep only 5 most recent snapshots
```

### Common debug workflow

```bash
pytest                                        # failure creates .llmdebug/latest.json
llmdebug                                      # read crash summary
llmdebug show --detail context                # full context when needed
pytest                                        # fix attempt → new snapshot
llmdebug diff                                 # see what changed
llmdebug show --detail full                   # see all frames
llmdebug frames -i 2                          # zoom into a specific frame
llmdebug git-context                          # check blame/diff for crash file
```

### CI configuration (environment variables)

```bash
export LLMDEBUG_REDACTION_PROFILE=ci
export LLMDEBUG_CAPTURE_LOGS=true
export LLMDEBUG_GIT_DIRTY_MODE=cached
pytest
```

---

## 5. Eval Infrastructure

See [local_docker_eval_runbook.md](local_docker_eval_runbook.md) for the full ops guide.

### Health check for local model endpoint

```bash
curl http://127.0.0.1:12000/v1/models
```

### Optional: llama.cpp Docker image

```bash
# Build
docker build --no-cache -f Dockerfile.llama -t llama-server-local:latest .

# Blackwell-specific build
docker build --no-cache -f Dockerfile.llama.blackwell -t llama-server-blackwell:latest .

# Run against a GGUF model
MODEL="/models/path/to/model.gguf"
docker run --gpus all \
  -v "$(pwd)":/models \
  --ulimit memlock=-1:-1 \
  -p 12000:8080 \
  --restart on-failure:10 \
  llama-server-local:latest \
  -m "$MODEL"
```

Use `llama-server-blackwell:latest` in the run command if you built the
Blackwell-specific Dockerfile.

The container derives runtime CPU threads from visible cores. With the default
`qwen35-coding-repro` preset on a 32-core host, it will start with
`-t 8 --threads-batch 32` unless you override `LLAMA_THREADS` or
`LLAMA_THREADS_BATCH`.
The shipped image also defaults to `-ub 2048 --no-mmap` via
`LLAMA_UBATCH_SIZE=2048` and `LLAMA_NO_MMAP=1`.

### Optional: System-1 decision-model sidecar

Needed only for `--adaptive-decision-model-role` runs (details:
`docs/local_docker_eval_runbook.md` and `docs/adaptive_online_policy_design.md`).

```bash
pueue add -- just decision-model-build
pueue add -- env GPU=1 CLEF_REVISION=<commit-sha> just decision-model-up
just decision-model-health      # 503 while loading
pueue add -- just decision-model-smoke
```

### Optional: Blackwell vLLM helper

```bash
evals/docker/run_vllm_blackwell.sh
```

### Build eval containers

```bash
# Standard image (llmdebug native conditions)
docker build -t llmdebug-eval:latest -f evals/docker/eval.Dockerfile .

# Baseline-capable image (includes external baseline envs)
docker build \
  --build-arg LLMDEBUG_EVAL_INCLUDE_BASELINES=1 \
  -t llmdebug-eval-baselines:latest \
  -f evals/docker/eval.Dockerfile .

# Force rebuild through the wrapper
LLMDEBUG_EVAL_BUILD=1 bash evals/docker/run_eval_docker.sh ...
LLMDEBUG_EVAL_BUILD_NO_CACHE=1 bash evals/docker/run_eval_docker.sh ...
```

---

## 6. Eval Runs — Docker Wrapper

See [local_docker_eval_runbook.md](local_docker_eval_runbook.md) for full options.

### Smoke tests

```bash
# No model required
bash evals/docker/run_eval_docker.sh --patcher null --max-cases 5 --k 1

# Model-backed smoke
bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --openai-response-mode edits \
  --conditions base_repair,layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --jobs 2 \
  --k 1 \
  --max-cases 5
```

### Full native run

```bash
RUN_ID="local-eval-$(date +%Y%m%dT%H%M%S)"

bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --openai-api-key local \
  --openai-timeout-sec 180 \
  --openai-max-propose-sec 300 \
  --openai-max-tokens 2048 \
  --openai-temperature 0.0 \
  --max-context-chars 200000 \
  --prompt-budget-total-chars 180000 \
  --conditions base_repair,layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --jobs 4 \
  --k 3 \
  --run-id "$RUN_ID"
```

### Template-driven run

```bash
# Copy a template and edit run_id, model, and policy paths
cp evals/configs/openai_adaptive.template.toml evals/configs/my_adaptive.toml

bash evals/docker/run_eval_docker.sh \
  --config evals/configs/my_adaptive.toml
```

### External baseline runs (Docker wrapper)

```bash
OPENAI_BASE_URL="http://127.0.0.1:12000/v1"
OPENAI_MODEL="<model-name>"

# LDB
bash evals/docker/run_eval_docker.sh \
  --baseline ldb \
  --openai-base-url "$OPENAI_BASE_URL" --openai-model "$OPENAI_MODEL" \
  --max-cases 100 --k 1 --run-id "ldb-$(date +%Y%m%dT%H%M%S)"

# INTERVENOR
bash evals/docker/run_eval_docker.sh \
  --baseline intervenor \
  --openai-base-url "$OPENAI_BASE_URL" --openai-model "$OPENAI_MODEL" \
  --max-cases 100 --k 1 --run-id "intervenor-$(date +%Y%m%dT%H%M%S)"

# TraceCoder
bash evals/docker/run_eval_docker.sh \
  --baseline tracecoder \
  --openai-base-url "$OPENAI_BASE_URL" --openai-model "$OPENAI_MODEL" \
  --max-cases 100 --k 1 --run-id "tracecoder-$(date +%Y%m%dT%H%M%S)"

# lambda-RLM
bash evals/docker/run_eval_docker.sh \
  --baseline lambda_rlm \
  --openai-base-url "$OPENAI_BASE_URL" --openai-model "$OPENAI_MODEL" \
  --max-cases 100 --k 1 --run-id "lambda-rlm-$(date +%Y%m%dT%H%M%S)"

# RLM
bash evals/docker/run_eval_docker.sh \
  --baseline rlm \
  --openai-base-url "$OPENAI_BASE_URL" --openai-model "$OPENAI_MODEL" \
  --max-cases 100 --k 1 --run-id "rlm-$(date +%Y%m%dT%H%M%S)"

# Config-based baseline run
cp evals/configs/baseline_ldb.template.toml evals/configs/my_ldb.toml
bash evals/docker/run_eval_docker.sh --config evals/configs/my_ldb.toml
```

---

## 7. Eval Runs — Python Direct

### base_repair

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --conditions base_repair \
  --k 2 --jobs 3 \
  --run-id my-base-repair-001

# via config
uv run python -m evals.run_eval --config evals/configs/openai_base_repair.latest.toml
```

### cot_repair

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --conditions cot_repair \
  --k 2 --jobs 3 \
  --run-id my-cot-repair-001
```

### layered_adaptive_gated

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --conditions layered_adaptive_gated \
  --adaptive-gate-policy balanced \
  --adaptive-max-helper-calls 1 \
  --k 2 --jobs 3 \
  --run-id my-gated-adaptive-001
```

### layered_adaptive_llm (LinUCB online policy)

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --conditions layered_adaptive_llm \
  --adaptive-online-policy-mode on \
  --adaptive-online-policy-algo linucb \
  --adaptive-max-helper-calls 1 \
  --k 2 --jobs 3 \
  --run-id my-llm-adaptive-001

# via config
uv run python -m evals.run_eval --config evals/configs/openai_adaptive.template.toml
```

With the System-1 decision model (sidecar running; start with `shadow`, then compare the
decision-model OPE targets before live roles):

```bash
pueue add -- uv run python -m evals.run_eval \
  --config evals/configs/openai_adaptive_probe_decision_model_shadow.template.toml \
  --run-id dm-shadow-001

# offline probe: score the decision model's answers against case labels, no repair budget
pueue add -- uv run python -m evals.decision_model.probe --executor local --split dev --jobs 4 --report
```

### External baselines (Python direct)

```bash
# LDB
uv run python -m evals.run_eval \
  --baseline ldb \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --k 2 --run-id my-ldb-001

# INTERVENOR
uv run python -m evals.run_eval \
  --baseline intervenor \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --k 2 --run-id my-intervenor-001

# TraceCoder
uv run python -m evals.run_eval \
  --baseline tracecoder \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --k 2 --run-id my-tracecoder-001

# lambda-RLM
uv run python -m evals.run_eval \
  --baseline lambda_rlm \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --k 2 --run-id my-lambda-rlm-001

# RLM
uv run python -m evals.run_eval \
  --baseline rlm \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --k 2 --run-id my-rlm-001
```

### Multi-condition comparison run

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-model <model-name> \
  --conditions base_repair cot_repair layered_adaptive_gated \
  --k 1 --jobs 3 --max-cases 10 \
  --run-id comparison-001
```

### Minimal smoke (no Docker)

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model-name> \
  --k 1 --max-cases 5
```

---

## 8. Two-Pass Workflow (Base → Rescue)

See [Eval Quickstart](eval_quickstart.md) for setup details and
[Eval Framework](../evals/README.md) for workflow details.

```bash
# Step 1: base_repair pass on full case set
just eval-traceback-only my-base-001
# equivalent:
bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url "${OPENAI_BASE_URL:-http://127.0.0.1:12000/v1}" \
  --config evals/configs/openai_base_repair.latest.toml \
  --run-id my-base-001

# Step 2: extract failed cases
just eval-select-failed my-base-001
# with explicit condition and max-attempt:
just eval-select-failed my-base-001 base_repair 2
# writes: evals/artifacts/case_lists/my-base-001.failed.txt

# Step 3: adaptive rescue on failed subset
just eval-adaptive-online-failed my-base-001 my-rescue-001

# Step 4: two-pass uplift report
just eval-two-pass-summary my-base-001 my-rescue-001
# with explicit conditions:
just eval-two-pass-summary my-base-001 my-rescue-001 base_repair layered_adaptive_llm
```

---

## 9. Hybrid Study Matrix

See [baseline_hybrid_study_runbook.md](baseline_hybrid_study_runbook.md) for guardrails and config details.

### Shared variables

```bash
export OPENAI_BASE_URL="http://127.0.0.1:12000/v1"
export OPENAI_MODEL="<model-name>"
export CASE_LIST="evals/artifacts/case_lists/<slice>.txt"
export RUN_TAG="<slice>"   # e.g. failed10 or matched157
```

### Single run

```bash
uv run python -m evals.run_eval \
  --config evals/configs/study_ldb_hybrid_canonical.template.toml \
  --run-id "${RUN_TAG}-ldb-hybrid-canonical" \
  --case-list-file "$CASE_LIST" \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$OPENAI_MODEL"
```

### Full 15-run matrix (pueue)

```bash
BASELINES=(ldb intervenor tracecoder lambda_rlm rlm)
ARMS=(raw hybrid_canonical hybrid_tuned)

for baseline in "${BASELINES[@]}"; do
  for arm in "${ARMS[@]}"; do
    cfg="evals/configs/study_${baseline}_${arm}.template.toml"
    run_id="${RUN_TAG}-${baseline}-${arm//_/-}"

    pueue add -- uv run python -m evals.run_eval \
      --config "$cfg" \
      --run-id "$run_id" \
      --case-list-file "$CASE_LIST" \
      --openai-base-url "$OPENAI_BASE_URL" \
      --openai-model "$OPENAI_MODEL"
  done
done
```

### Smoke matrix (one baseline, three arms)

```bash
for arm in raw hybrid_canonical hybrid_tuned; do
  pueue add -- uv run python -m evals.run_eval \
    --config "evals/configs/study_ldb_${arm}.template.toml" \
    --run-id "${RUN_TAG}-ldb-${arm//_/-}" \
    --case-list-file "$CASE_LIST" \
    --openai-base-url "$OPENAI_BASE_URL" \
    --openai-model "$OPENAI_MODEL"
done
```

### pueue monitoring

```bash
pueue status
pueue log <task_id>
pueue follow <task_id>
```

---

## 10. Results & Analysis

### Per-run summary

```bash
just eval-summary <run_id>
# direct equivalent:
uv run python -m evals.eval_summary <run_id>
```

### Two-pass cumulative summary

```bash
just eval-two-pass-summary <base_run_id> <rescue_run_id>
# direct equivalent:
uv run python -m evals.eval_two_pass_summary <base_run_id> <rescue_run_id> \
  --baseline-condition base_repair \
  --rescue-condition layered_adaptive_llm
```

### Aggregate analysis across multiple runs

```bash
uv run python -m evals.analyze_results \
  "evals/results/${RUN_TAG}-*.jsonl" \
  --out "evals/results/${RUN_TAG}.summary.json"
```

### Quick success-rate comparison from JSONL

```bash
python -c "
import json
results = {}
with open('evals/results/<run_id>.jsonl') as f:
    for line in f:
        r = json.loads(line)
        cond = r['condition']
        results.setdefault(cond, {'total': 0, 'success': 0})
        results[cond]['total'] += 1
        if r['success']:
            results[cond]['success'] += 1
for cond, s in sorted(results.items()):
    rate = s['success'] / s['total'] * 100 if s['total'] else 0
    print(f'{cond}: {s[\"success\"]}/{s[\"total\"]} = {rate:.1f}%')
"
```

### Result and artifact locations

```
evals/results/<run_id>.jsonl            # per-(case, condition, attempt) records
evals/artifacts/<run_id>/contexts/      # evidence bundles per case/condition
evals/artifacts/<run_id>/workdirs/      # working copies of test cases
evals/artifacts/policies/              # LinUCB policy state files
evals/artifacts/case_lists/            # case list files (from eval-select-failed)
```

---

## 11. SWE-Bench Lite

See [local_docker_eval_runbook.md](local_docker_eval_runbook.md#4-swe-bench-lite-local-generation-and-docker-scoring) for the full guide.

### Generate predictions

```bash
python -m evals.swebench.cli generate-lite \
  --run-id <id> \
  --conditions baseline,llmdebug \
  --snapshot-dir <dir_with_instance_snapshots> \
  --max-instances <n|0> \
  --instance-ids-file <optional> \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model <model_alias> \
  --openai-http2 \
  --openai-async \
  --jobs <workers> \
  --max-request-errors 25 \
  --max-request-error-rate 0.5
```

### Validate run directory

```bash
python -m evals.swebench.cli prepare-handoff --run-id <id>
```

### Official scoring

```bash
python -m evals.swebench.cli score-official \
  --run-dir /path/to/evals/swebench_runs/<id> \
  --conditions baseline,llmdebug \
  --workers <n> \
  --cache-level <none|base|env|instance>
```

### Convenience wrappers

```bash
./scripts/score_swebench_lite_official.sh \
  --run-dir /path/to/evals/swebench_runs/<id>

./scripts/smoke_swebench_lite_official.sh \
  --run-dir /path/to/evals/swebench_runs/<id> --max-instances 2
```

### Smoke scoring subset

```bash
python -m evals.swebench.cli prepare-scoring-smoke \
  --run-dir /path/to/evals/swebench_runs/<id> \
  --max-instances 2
```

---

## 12. Case Validation

```bash
# Schema-only validation (fast, no execution)
uv run python -m evals.validate_cases --schema-only

# Schema + execution with filters
uv run python -m evals.validate_cases \
  --category hidden_state \
  --difficulty easy \
  --snapshot-dependency required \
  --max-cases 5

# Full integrity validation (baseline fail + oracle pass)
uv run python -m evals.validate_cases

# Local executor fallback
uv run python -m evals.validate_cases --executor local
```

---

## 13. Maintenance & Cleanup

### Build artifact cleanup

```bash
just clean     # removes .pytest_cache, .ruff_cache, .pyright, htmlcov, coverage files

# manual
rm -rf .pytest_cache .ruff_cache .pyright htmlcov .coverage coverage.xml
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

### Snapshot cleanup

```bash
llmdebug clean -k 5     # keep only 5 most recent snapshots
```

### Troubleshooting diagnostics

```bash
# Check snapshot store
ls -la .llmdebug/
readlink .llmdebug/latest.json

# Remove stale snapshot lock
rm -f .llmdebug/.llmdebug.lock

# Force a snapshot via decorator (sanity check)
python -c "
from llmdebug import debug_snapshot
@debug_snapshot()
def f(): raise ValueError('test')
f()
"
ls .llmdebug/latest.json
```

### Ratchet coverage baseline

```bash
# Run coverage, then update quality/coverage_baseline.json
uv run pytest --cov=src/llmdebug --cov-branch --cov-report=json --cov-report=xml --cov-fail-under=95
# update line_percent, branch_percent, and updated_at_utc in the baseline
```

### Server command convention (long-running eval jobs)

```bash
pueue add -- <command>       # enqueue a command
pueue status                 # show task overview
pueue log <task_id>          # show logs for a completed task
pueue follow <task_id>       # follow live output
```
