# Scripts

Helper utilities for benchmarking, Docker-based SWE-Bench scoring, and CI
quality gates.

## Scripts

| Script | Purpose | Where to run |
|--------|---------|--------------|
| `bench_capture_compare.py` | Compare pytest-benchmark capture-overhead medians between two JSON reports | Local / CI |
| `score_swebench_lite_official.sh` | Run official SWE-Bench Lite scoring on generated predictions | Docker machine |
| `score_swebench_pro_official.sh` | Run Docker-based SWE-Bench Pro scoring on generated predictions | Docker machine |
| `smoke_swebench_lite_official.sh` | Quick smoke-test for official scoring on a small paired subset | Docker machine |
| `quality/check_coverage_no_regression.py` | Fail CI when line or branch coverage regresses below the configured baseline | Local / CI |
| `quality/check_type_coverage.py` | Report and optionally gate public type-completeness vs the committed baseline | Local / CI |
| `quality/check_boundary_module_coverage.py` | Enforce focused per-module coverage thresholds for the baseline patch seam | Local / CI |
| `quality/check_bench_regression.py` | Compare capture-overhead benchmark medians against a local baseline | Local |

## Prerequisites

| Script | Requirements |
|--------|--------------|
| `bench_capture_compare.py` | Python 3, `llmdebug` package installed |
| `score_swebench_lite_official.sh` | Python 3, `swebench` package, Docker |
| `score_swebench_pro_official.sh` | Python 3, Docker, `evals.swebench_pro` dependencies |
| `smoke_swebench_lite_official.sh` | Same as `score_swebench_lite_official.sh` |
| `quality/check_coverage_no_regression.py` | Python 3, `llmdebug` package, `coverage.json` from `pytest --cov-branch --cov-report=json` |
| `quality/check_type_coverage.py` | Python 3, `llmdebug` package, `pyright` available via `uv run` |
| `quality/check_boundary_module_coverage.py` | Python 3, `llmdebug` package, focused pytest-cov JSON report for the seam suite |
| `quality/check_bench_regression.py` | Python 3, `llmdebug` package, current and local baseline capture-overhead benchmark JSON reports |

## Usage

### bench_capture_compare.py

Compares median runtimes from two pytest-benchmark JSON reports and prints a
delta table.

```bash
python scripts/bench_capture_compare.py baseline.json current.json
```

### score_swebench_lite_official.sh

Runs the official SWE-Bench evaluation harness against predictions produced by
the local eval workflow. Intended for a Docker-capable machine.

```bash
./scripts/score_swebench_lite_official.sh \
  --run-dir /path/to/evals/swebench_runs/<RUN_ID> \
  --conditions baseline,llmdebug \
  --workers 6 \
  --cache-level env
```

Key options: `--clean` (re-run from scratch), `--cache-level none|base|env|instance`.

### score_swebench_pro_official.sh

Runs the llmdebug SWE-Bench Pro Docker scorer against generated predictions.
Intended for a Docker-capable scoring machine, not the generation host.

```bash
./scripts/score_swebench_pro_official.sh \
  --run-dir /path/to/evals/swebench_pro_runs/<RUN_ID> \
  --conditions baseline,llmdebug \
  --workers 6
```

### smoke_swebench_lite_official.sh

Prepares a small paired subset from a full run and scores it — a fast sanity
check before committing to the full evaluation.

```bash
./scripts/smoke_swebench_lite_official.sh \
  --run-dir /path/to/evals/swebench_runs/<RUN_ID> \
  --max-instances 2
```

### quality/check_coverage_no_regression.py

Compares current line and branch coverage against
`quality/coverage_baseline.json` and exits non-zero if either metric regressed
beyond a configurable tolerance.

```bash
python scripts/quality/check_coverage_no_regression.py \
  --coverage-json coverage.json \
  --baseline-json quality/coverage_baseline.json \
  --tolerance 0.05
```

### quality/check_type_coverage.py

Measures public type completeness with `pyright --verifytypes`, compares it to
`quality/type_coverage_baseline.json`, and exits non-zero only when `--strict`
is set.

```bash
python scripts/quality/check_type_coverage.py --strict
python scripts/quality/check_type_coverage.py --update-baseline
```

### quality/check_boundary_module_coverage.py

Checks focused per-module line and branch coverage for the baseline patch seam.

```bash
python scripts/quality/check_boundary_module_coverage.py \
  --coverage-json .pytest_cache/baseline-boundary-coverage.json
```

### quality/check_bench_regression.py

Compares capture-overhead benchmark medians against the compact regression
baseline `.benchmarks/capture-overhead-regression-baseline.json` by default and
fails when any median regresses beyond the configured threshold. This compact
baseline is separate from `.benchmarks/capture-overhead-baseline.json`, which is
the full pytest-benchmark JSON file used by `just bench-capture-compare`.

```bash
python scripts/quality/check_bench_regression.py
python scripts/quality/check_bench_regression.py --threshold 15
python scripts/quality/check_bench_regression.py --update-baseline
```
