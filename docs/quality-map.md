# Quality Map

This guide maps the repository's local and CI quality checks to the parts of
the codebase they protect. Use it alongside [CONTRIBUTING.md](../CONTRIBUTING.md)
when deciding which commands to run before a PR.

## Quick command map

| Command | Scope | Blocking? | Notes |
|---------|-------|-----------|-------|
| `just check` | `src/llmdebug` + default tests | Yes | Fastest full pre-PR pass |
| `just quality` | Repo-wide quality contract | Yes | Mirrors the shared GitHub quality gate, including eval lint |
| `just baseline-boundary-coverage` | Baseline patch seam modules | Yes | Focused per-module coverage gate for adapter-to-harness patch flow |
| `just test-integration` | Blocking `@pytest.mark.integration` tests excluding `baseline_runtime` | Yes | CI runs this lane after provisioning in the 3.13 quality job |
| `just test-baseline-runtime` | Provisioned baseline repo/env/runtime tests | No | Informational CI lane after `setup_repos.sh` and `setup_envs.sh` |
| `just type-coverage` | Public package type completeness | No | Informational snapshot of `pyright --verifytypes` |
| `just type-coverage-strict` | Public package type completeness | Yes | Fails on regression vs `quality/type_coverage_baseline.json` |
| `just lint-evals` | `evals/` Ruff lint + format check | Yes | Blocking eval-harness lint lane used by `just quality` |
| `just typecheck-evals` | `evals/` and `baselines/` minus holdouts | No | Staged `pyright` pass in `basic` mode |
| `just test-evals-cov` | `evals/` line + branch coverage | No | Writes `coverage-evals.json`; informational in CI |
| `just eval-coverage-no-regression` | Eval coverage ratchet | No | Compares the existing report with the committed baseline; informational in CI |

`just quality` directly composes `check`, `lint-evals`,
`coverage-no-regression`, `baseline-boundary-coverage`, `deps`, `audit`,
`docs-check`, `dead-code`, `complexity`, `complexity-gate`, `security`,
`type-coverage-strict`, and `dependency-sync`. `just ci` is exactly an alias
for `just quality`. Diff coverage, eval coverage, integration tests, the
baseline-runtime lane, and eval typing are orchestrated separately in GitHub
Actions as described below.

## Blocking contract for `src/llmdebug`

These checks are part of the normal shipping gate for the package:

- `ruff check` and `ruff format --check`
- strict `pyright` on `src/llmdebug`
- pytest with coverage over `src/llmdebug`
- line coverage floor `>= 95`
- line and branch coverage no-regression against `quality/coverage_baseline.json`
- public type-completeness no-regression against
  `quality/type_coverage_baseline.json`
- focused seam coverage via `just baseline-boundary-coverage`
- `bandit`, `deptry`, `pip-audit`, `vulture`, `radon`, `xenon`
- docs drift checks where applicable

The `pip-audit` lane currently carries one explicit temporary ignore:
`GHSA-5239-wwwm-4pmq` / `CVE-2026-4539` in `Pygments`. Keep that exception
narrow and remove it once PyPI/OSV publish a fixed upstream release.

## Staged contract for `evals/`

The eval harness is intentionally being brought under stronger static checks in
stages rather than all at once. Ruff lint/format is now blocking; staged typing
and staged coverage promotion remain separate.

- `just lint-evals` runs `ruff check evals` and `ruff format --check evals`
- `just typecheck-evals` runs `pyright -p pyrightconfig.evals.json`
- it covers all of `evals/` and `baselines/`, except generated trees, the GPU
  sidecar (`evals/docker/`), and holdouts that still have basic-mode errors:
  `evals/eval_attempt_flow.py`, the four `analysis_*` modules that read the
  untyped derived row, and the `rlm`/`lambda_rlm` adapters
- `tests/test_quality_gate_contract.py` lets the holdout list shrink but not
  grow, so new modules are type-checked by default
- this pass uses `basic` mode and is currently surfaced in CI as
  informational (`continue-on-error: true`)
- `just test-evals-cov` generates `coverage-evals.json`; the separate
  `just eval-coverage-no-regression` recipe checks line and branch coverage
  against `quality/coverage_evals_baseline.json`
- the current eval baseline is 98.2% line / 95.7% branch; this lane is also
  informational in CI (`continue-on-error: true`)
- the July 12, 2026 local measurement is 97.6278% line / 90.4516% branch, so
  the informational ratchet currently exposes existing debt rather than
  claiming a green baseline

## Test routing

- Default `just test-quick`, `just test`, and `just test-no-cov` runs exclude
  `@pytest.mark.integration`.
- Use `@pytest.mark.integration` for tests that need container wrappers, stub
  services, or other heavier infrastructure.
- Use `@pytest.mark.baseline_runtime` alongside `@pytest.mark.integration` when
  a test requires provisioned baseline repos or `baselines/envs/*` runtimes.
- Run the blocking integration subset with `just test-integration`.
- Run the provisioned post-setup baseline subset with `just test-baseline-runtime`.
- GitHub Actions provisions pinned baseline repos plus smoke envs first, then
  runs `just test-integration` as blocking and `just test-baseline-runtime` as
  a non-blocking post-provision lane.

## Updating baselines

Package coverage, eval coverage, and type-completeness baselines are committed
quality assets.

- Package coverage baseline: `quality/coverage_baseline.json`
- Eval coverage baseline: `quality/coverage_evals_baseline.json`
- Type-completeness baseline: `quality/type_coverage_baseline.json`

When intentionally ratcheting the package quality floor, regenerate the metric,
update the committed JSON file, and include the reason in the PR description.
Use `just test-evals-cov` followed by `just eval-coverage-no-regression` before
ratcheting the eval baseline, and use `check_type_coverage.py --update-baseline`
to ratchet type completeness.
