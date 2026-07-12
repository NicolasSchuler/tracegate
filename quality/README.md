# Quality Baselines

Ratcheting quality baselines consumed by CI to prevent metric regressions.
Each baseline records a numeric floor; the check scripts fail when the current
measurement drops below the committed value (within a small tolerance). Baselines
only move **up** — improving quality is a one-way door.

## Baseline files

| File | Tracks | CI gate |
|------|--------|---------|
| `coverage_baseline.json` | Package line & branch coverage (%) | `check_coverage_no_regression.py` — fails if either metric drops |
| `coverage_evals_baseline.json` | Eval line & branch coverage (%) | `just eval-coverage-no-regression` — informational CI ratchet |
| `type_coverage_baseline.json` | Public type-completeness via `pyright --verifytypes` | `check_type_coverage.py --strict` — fails on regression |

## How baselines work

The committed baseline checks follow the same pattern:

1. CI produces a current measurement (coverage report or pyright output).
2. The corresponding check script loads the committed baseline and compares.
3. If the current value is **worse** than the baseline beyond a small tolerance,
   the checker exits non-zero. Package coverage and strict type coverage block
   CI; eval coverage is currently reported as an informational CI failure.
4. If the current value is **better**, CI passes — but the baseline is **not**
   auto-updated. A human must commit the improvement to ratchet the floor.

This "ratchet-up" policy ensures quality never silently degrades.

## Updating baselines

### Package coverage

```bash
# 1. Generate coverage.json and run the committed package ratchet
just test
just coverage-no-regression

# 2. Update quality/coverage_baseline.json with the new line_percent,
#    branch_percent, and current UTC timestamp in updated_at_utc
```

The update is manual — edit the JSON file and include the new values from
`coverage.json → totals`.

### Eval coverage

```bash
# Generate coverage-evals.json, then check the committed eval ratchet
just test-evals-cov
just eval-coverage-no-regression

# Then update both percentages and updated_at_utc in
# quality/coverage_evals_baseline.json.
```

The committed eval baseline is 98.2% line / 95.7% branch. The July 12, 2026
local measurement is 97.6278% line / 90.4516% branch, so the informational
ratchet currently fails and records existing debt; do not lower it merely to
make the lane green. The checker uses the same 0.05 percentage-point tolerance
as package coverage.

### Type coverage

```bash
uv run python scripts/quality/check_type_coverage.py --update-baseline
```

This overwrites `quality/type_coverage_baseline.json` with the current
`pyright --verifytypes` result.

## Thresholds

| Metric | Threshold | Tolerance | Notes |
|--------|-----------|-----------|-------|
| Line coverage floor | ≥ 95 % | — | Hard floor via `--cov-fail-under=95` |
| Package line & branch coverage | No regression | 0.05 pp | Compared against `coverage_baseline.json` |
| Eval line & branch coverage | No regression | 0.05 pp | Informational CI check against `coverage_evals_baseline.json` |
| Baseline patch seam coverage | Per-module thresholds | — | `just baseline-boundary-coverage` enforces focused eval seam coverage |
| Diff coverage (PRs) | ≥ 80 % | — | Changed lines only, via `diff-cover` |
| Type completeness | No regression | 0.05 pp | Compared against `type_coverage_baseline.json` |

## CI integration

The shared quality action runs `just quality`, which consumes the package
coverage and type-completeness baselines. On PRs it also runs
`just diff-coverage` for changed-line coverage gating. Eval coverage is kept
outside `just quality`: the action runs `just test-evals-cov` followed by
`just eval-coverage-no-regression` as a non-blocking informational step.

Locally, `just check` is the fast essentials lane (`lint`, `typecheck`, and
tests), while `just quality` runs the blocking contract including the package
coverage, type-completeness, and boundary baselines. `just ci` is exactly an
alias for `just quality`. The boundary gate targets:

- `baselines/adapter_base.py`
- `evals/patchers/baseline_patcher.py`
- `evals/eval_patches.py`
- `evals/attempt_verification.py`

## See also

- [CONTRIBUTING.md](../CONTRIBUTING.md) — full quality-gate list and development workflow
- [docs/quality-map.md](../docs/quality-map.md) — which checks are blocking vs staged
- [scripts/README.md](../scripts/README.md) — usage details for each check script
