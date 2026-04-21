# Contributing to llmdebug

Code, tests, documentation, and benchmark improvements are all welcome.
This guide covers local setup, repository quality gates, and the contribution
expectations used during review. If you're new to the project, start with the
[README](README.md) for shipped behavior and interfaces, then use this document
for the development workflow.

## First-time setup

The repository uses `uv` for environment management and `just` for common task
wrappers.

1. Install [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/).
2. Clone the repository.
3. Install dependencies:

```bash
uv sync --group dev
```

4. Run the tests to verify your setup:

```bash
just test-quick
```

5. List available task recipes:

```bash
just --list
```

6. Find any command quickly: [`docs/runbook.md`](docs/runbook.md) is the canonical reference for all shell commands, `just` recipes, and CLI invocations across setup, quality, eval ops, and analysis.

> **When adding or removing `just` recipes**, update the `Last verified:` date in `docs/runbook.md` so the runbook stays trustworthy.

## Local quality checks

For most changes, run the quick pass before you open a PR. Use the full suite
when you touch tooling, generated docs, release paths, or operational scripts.

### Quick (pre-commit essentials)

```bash
just check    # lint + typecheck + tests with coverage
```

Default test recipes skip `@pytest.mark.integration` so the fast local loop
stays deterministic. Run heavier infrastructure-facing tests separately:

```bash
just test-integration
just test-baseline-runtime
```

Use `just test-integration` for the blocking integration lane and
`just test-baseline-runtime` for provisioned baseline repo/env/runtime checks
that only make sense after `baselines/setup_repos.sh` and
`baselines/setup_envs.sh` complete.

GitHub Actions runs both in the single-version 3.13 quality job: the blocking
`just test-integration` lane after provisioning, followed by a non-blocking
`just test-baseline-runtime` lane for provisioned baseline runtime coverage.

### Full suite

```bash
just quality  # mirrors GitHub CI: deps, audit, docs-check, complexity, security
just ci       # alias for the single-version local CI contract
just baseline-boundary-coverage  # focused 4-module seam coverage gate for baseline patching
just lint-evals  # blocking eval-harness lint/format check used by just quality
```

`just audit` runs `pip-audit` with a narrow temporary ignore for
`GHSA-5239-wwwm-4pmq` / `CVE-2026-4539` because PyPI/OSV do not yet publish a
fixed `Pygments` release. Remove that ignore as soon as upstream ships a fix.

### Individual commands

```bash
uv run ruff check src tests                  # Lint
uv run ruff format --check src tests         # Format check
uv run ruff check evals                      # Eval harness lint
uv run ruff format --check evals             # Eval harness format check
uv run pyright                               # Type checking (strict mode)
uv run pytest -x -q -m "not integration"     # Tests (fail-fast, default lane)
uv run pytest --cov=src/llmdebug --cov-report=term-missing --cov-fail-under=95  # Coverage
uv run pyright -p pyrightconfig.evals.json   # Staged eval typing pass
uv run python scripts/quality/check_type_coverage.py --strict  # Public type coverage gate
just baseline-boundary-coverage              # Baseline patch seam coverage gate
just type-coverage                           # Informational verifytypes snapshot
just type-coverage-strict                    # Enforced verifytypes no-regression gate
```

If you touch eval tooling or case metadata, also run:

```bash
uv run python -m evals.validate_cases --schema-only
just typecheck-evals
just test-evals-cov              # evals/ coverage with baseline regression check
just test-combined-cov           # src + evals combined coverage report
```

For a concise map of which checks are blocking for `src/llmdebug` versus staged
for `evals/`, see [docs/quality-map.md](docs/quality-map.md).

## Quality gates

`.github/actions/run-quality-contract/action.yml` is a reusable composite action
that encapsulates `just quality` for CI, so quality gate changes need only one edit.

These checks are part of the normal review contract, not optional extras:

- **Line coverage floor**: `--cov-fail-under=95` (src/llmdebug)
- **Branch coverage no-regression** against `quality/coverage_baseline.json`
- **Eval harness lint/format** via `just lint-evals`
- **Evals coverage no-regression** against `quality/coverage_evals_baseline.json` (96.9% line / 93.7% branch)
- **Public type-completeness no-regression** against `quality/type_coverage_baseline.json`
- **Baseline patch seam coverage** via `just baseline-boundary-coverage`
- **Diff coverage floor**: `diff-cover --fail-under=80` on changed lines
- **Complexity gate**: `xenon --max-absolute C --max-modules C --max-average B`
- **Type checking**: pyright strict mode, zero errors
- **Security**: bandit with zero unsuppressed findings
- **Dead code**: vulture at `--min-confidence 90`
- **Dependency hygiene**: deptry on `src/llmdebug`

The staged eval typing pass is tracked separately:

- `just typecheck-evals` targets a bounded module set in `evals/`
- it uses `pyrightconfig.evals.json`
- it is surfaced in CI as informational during the rollout

Evals coverage is also tracked separately:

- `just test-evals-cov` runs evals tests with coverage and baseline regression check
- `just test-combined-cov` reports combined src + evals coverage
- Baseline is `quality/coverage_evals_baseline.json` (96.9% line / 93.7% branch)
- The evals coverage gate is informational in CI (`continue-on-error: true`); promotion to blocking is planned once the eval harness type adoption reaches basic-mode clean (tracked via `just typecheck-evals`)

To intentionally ratchet the branch coverage baseline after improvements:

```bash
uv run pytest --cov=src/llmdebug --cov-branch --cov-report=json --cov-report=xml --cov-fail-under=95
```

Then update `quality/coverage_baseline.json` with the new `branch_percent` value and the current UTC timestamp in `updated_at_utc`:

```json
{
  "line_percent": 76.61,
  "branch_percent": 72.40,
  "source": "main",
  "updated_at_utc": "2026-03-09T12:00:00Z"
}
```

To ratchet the evals coverage baseline, run `just test-evals-cov` and update
`quality/coverage_evals_baseline.json` similarly.

## Benchmarking

Run capture-overhead microbenchmarks (local, non-gating):

```bash
just bench-capture
```

To compare against a local baseline:

```bash
cp .benchmarks/capture-overhead-current.json .benchmarks/capture-overhead-baseline.json
# ... make changes ...
just bench-capture
just bench-capture-compare
```

`just bench-capture-compare` prints per-benchmark median deltas for quick trend review.

## Coding guidelines

- Target Python 3.10+.
- Use 4-space indentation, max line length 100.
- Keep changes focused and minimal.
- Add regression tests for bug fixes and behavior changes.
- Prefer clear, typed APIs for new or changed code paths.
- Use `@pytest.mark.integration` for heavy infra/container/stub-server tests.
- Use `@pytest.mark.baseline_runtime` together with `@pytest.mark.integration`
  when a test requires provisioned baseline repos or `baselines/envs/*` runtimes.

## Commit and PR guidelines

- Prefer Conventional Commits:
  - `feat: ...`
  - `fix: ...`
  - `perf: ...`
  - `docs: ...`
  - `test: ...`
  - `chore: ...`
- Include a concise summary of problem and solution.
- Link related issues when applicable.
- Include evidence (test output or command output) for behavior changes.
- Update the relevant README/docs entry in the same PR when user-visible
  behavior or workflow guidance changes.

## Release notes

Keep changelog entries concise and user-facing. Automated package publishing is
not configured in this repository.
