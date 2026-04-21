# Quality Baselines

Ratcheting quality baselines consumed by CI to prevent metric regressions.
Each baseline records a numeric floor; the check scripts fail when the current
measurement drops below the committed value (within a small tolerance). Baselines
only move **up** — improving quality is a one-way door.

## Baseline files

| File | Tracks | CI gate |
|------|--------|---------|
| `coverage_baseline.json` | Line & branch coverage (%) | `check_coverage_no_regression.py` — fails if branch coverage drops |
| `type_coverage_baseline.json` | Public type-completeness via `pyright --verifytypes` | `check_type_coverage.py --strict` — fails on regression |

## How baselines work

Both committed baseline checks follow the same pattern:

1. CI produces a current measurement (coverage report or pyright output).
2. The corresponding check script loads the committed baseline and compares.
3. If the current value is **worse** than the baseline beyond a small tolerance, CI fails.
4. If the current value is **better**, CI passes — but the baseline is **not**
   auto-updated. A human must commit the improvement to ratchet the floor.

This "ratchet-up" policy ensures quality never silently degrades.

## Updating baselines

### Coverage

```bash
# 1. Run tests with branch coverage
uv run pytest -m "not integration" --cov=src/llmdebug --cov-branch \
  --cov-report=json --cov-report=xml --cov-fail-under=95

# 2. Update quality/coverage_baseline.json with the new line_percent,
#    branch_percent, and current UTC timestamp in updated_at_utc
```

The update is manual — edit the JSON file and include the new values from
`coverage.json → totals`.

### Type coverage

```bash
uv run python scripts/quality/check_type_coverage.py --update
```

This overwrites `quality/type_coverage_baseline.json` with the current
`pyright --verifytypes` result.

## Thresholds

| Metric | Threshold | Tolerance | Notes |
|--------|-----------|-----------|-------|
| Line coverage floor | ≥ 95 % | — | Hard floor via `--cov-fail-under=95` |
| Branch coverage | No regression | 0.05 pp | Compared against `coverage_baseline.json` |
| Baseline patch seam coverage | Per-module thresholds | — | `just baseline-boundary-coverage` enforces focused eval seam coverage |
| Diff coverage (PRs) | ≥ 80 % | — | Changed lines only, via `diff-cover` |
| Type completeness | No regression | 0.05 pp | Compared against `type_coverage_baseline.json` |

## CI integration

The baselines are consumed by the shared quality contract defined in
`.github/actions/run-quality-contract/action.yml`, which runs `just quality`.
On PRs the composite action also runs `just diff-coverage` for changed-line
coverage gating.

Locally, `just check` is the fast essentials lane (`lint`, `typecheck`, and
tests), while `just quality` runs the full contract including the additional
baseline gates. The quality contract also runs `just baseline-boundary-coverage`,
which targets the baseline patch seam:

- `baselines/adapter_base.py`
- `evals/patchers/baseline_patcher.py`
- `evals/eval_patches.py`
- `evals/attempt_verification.py`

## See also

- [CONTRIBUTING.md](../CONTRIBUTING.md) — full quality-gate list and development workflow
- [docs/quality-map.md](../docs/quality-map.md) — which checks are blocking vs staged
- [scripts/README.md](../scripts/README.md) — usage details for each check script
