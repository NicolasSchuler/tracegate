# Tests

Pytest test suite for **llmdebug**. The flat `tests/` directory currently contains
over 230 `test_*.py` files spanning the package, eval harness, baselines, and
quality contract.

## Running tests

```bash
# Fast feedback (fail on first error, quiet)
uv run pytest -x -q -m "not integration"

# With coverage (CI-equivalent)
uv run pytest -m "not integration" --cov=src/llmdebug --cov-branch --cov-report=term-missing --cov-fail-under=95

# Single file
uv run pytest tests/test_capture.py -v

# Single test
uv run pytest tests/test_capture.py::test_capture_basic_exception -v

# By keyword
uv run pytest -k "rca" -v

# Benchmarks only
uv run pytest tests/test_benchmarks.py --benchmark-only -v

# Skip benchmarks
uv run pytest tests/test_benchmarks.py --benchmark-skip

# All integration-marked tests
uv run pytest -m integration -v

# Blocking integration lane only
uv run pytest -m "integration and not baseline_runtime" -v

# Provisioned baseline runtime lane only
uv run pytest -m "integration and baseline_runtime" -v
```

Or use `just` recipes:

```bash
just test-quick     # fast loop, excludes integration
just test           # coverage run, excludes integration
just test-integration
just test-baseline-runtime
just typecheck-evals
just bench-capture  # capture-overhead benchmarks
just check          # lint + typecheck + tests (pre-commit essentials)
```

## Directory layout

All test files live in a flat `tests/` directory—no subdirectories. Files map to
source modules by naming convention. The tables below are representative rather
than exhaustive; newer refactored helper modules often share coverage with the
same subsystem-level tests.

### Core library → `src/llmdebug/`

| Test file | Source module | What it covers |
|---|---|---|
| `test_capture.py` | `capture.py` | Exception capture pipeline |
| `test_serialize.py`, `test_serialize_edge_cases.py` | `serialize.py` | Local variable serialization, array summarization, redaction |
| `test_output.py`, `test_output_edge_cases.py` | `output.py` | Snapshot file I/O |
| `test_compact_output.py` | `compact_keys.py` | Compact key expansion |
| `test_config_validation.py` | `config.py` | `SnapshotConfig` validation |
| `test_error_categories.py` | `error_categories.py` | Exception pattern matching |
| `test_git_context.py`, `test_git_context_helpers.py` | `git_context.py` | Git metadata capture |
| `test_async_context.py` | `async_context.py` | Async/coroutine context detection |
| `test_log_capture.py` | `log_capture.py` | Log interception |
| `test_hooks.py` | `hooks.py` | Production exception hooks, PII patterns |
| `test_gates.py` | `gates.py` | RCA verification gates |
| `test_hypotheses.py` | `hypotheses.py` | Hypothesis management |
| `test_detail_filter.py` | `detail_filter.py` | Frame/detail filtering |
| `test_redaction_policy.py` | `redaction_policy.py` | Redaction profiles (dev/ci/prod) |
| `test_snapshot_diff.py` | `snapshot_diff.py` | Snapshot diffing |
| `test_snapshot_loader.py` | `_snapshot_loader.py` | Snapshot file loading |
| `test_coverage_context.py` | `coverage_context.py` | Coverage context integration |
| `test_debug_session.py` | `_debug_session.py` | Debug session lifecycle |
| `test_middleware.py` | `middleware.py` | Middleware pipeline |
| `test_init_api.py` | `__init__.py` | Public API surface, re-exports |
| `test_module_reload_paths.py` | — | Module reload behavior |

### CLI

| Test file | Source module |
|---|---|
| `test_cli.py`, `test_cli_edge_cases.py` | `cli.py` |

### Pytest plugin

| Test file | Source module |
|---|---|
| `test_pytest_plugin.py` | `pytest_plugin.py` |

### Jupyter

| Test file | Source module |
|---|---|
| `test_jupyter_plugin.py` | `jupyter_plugin.py` |

### MCP (Model Context Protocol)

| Test file | Source module |
|---|---|
| `test_mcp_server.py` | `mcp_server.py` |
| `test_mcp_contract.py` | `mcp_contract.py` |
| `test_mcp_git_context.py` | `mcp_git_context.py` |
| `test_mcp_rca.py` | `mcp_rca.py` |

### RCA (Root Cause Analysis) engine

| Test file | Source module |
|---|---|
| `test_rca_engine.py` | `rca_engine.py` |
| `test_rca_state.py` | `rca_state.py` |
| `test_rca_backend.py` | `rca_backend.py` |
| `test_rca_identity.py` | `rca_identity.py` |
| `test_rca_store_sqlite.py` | `rca_store.py` |
| `test_rca_transitions.py` | `rca_transitions.py` |

### Eval framework → `evals/`

All `test_evals_*.py` files cover the evaluation/benchmarking framework in
`evals/`. The naming convention is `test_evals_<module>.py`. Examples:

- `test_evals_prompt_contract.py` — prompt rendering and contract assertions
- `test_evals_adaptive_conditions.py` — adaptive condition selection logic
- `test_evals_openai_patcher.py` — OpenAI API patcher
- `test_evals_swebench_*.py` — SWE-bench integration (CLI, generation, scoring)
- `test_evals_run_eval_*.py` — eval runner, resume, Docker wrapper

### Cross-cutting / meta

| Test file | Purpose |
|---|---|
| `test_quality_gate_contract.py` | Asserts CI config stays in sync (coverage thresholds, xenon, diff-cover) |
| `test_benchmarks.py` | Performance benchmarks (pytest-benchmark) |

## Naming conventions

- **Files**: `test_<module>.py` or `test_<module>_<variant>.py` (e.g., `test_serialize_edge_cases.py`).
  Eval tests use `test_evals_<module>.py`.
- **Functions**: `test_<behavior_under_test>` — descriptive, no class grouping needed for most files.
- **Classes**: Used sparingly for logical grouping (e.g., `TestRateLimiter`, `TestInstallHooks` in `test_hooks.py`).

## Markers

Defined in `pyproject.toml`:

```ini
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
    "baseline_runtime: marks provisioned baseline runtime integration tests",
]
```

Additional markers in use:
- `@pytest.mark.benchmark(group="...")` — benchmark grouping (pytest-benchmark)
- `@pytest.mark.skipif(...)` — conditional skips (e.g., optional `numpy` dependency)
- `@pytest.mark.asyncio` — async test functions

Use `@pytest.mark.integration` for tests that need heavier infrastructure such
as container wrappers or stubbed external services. Those tests are excluded
from the default `just test*` recipes and should run via `just test-integration`.
Use `@pytest.mark.baseline_runtime` with `@pytest.mark.integration` when the
test also depends on provisioned baseline repos or `baselines/envs/*` runtimes;
run that subset with `just test-baseline-runtime`.

Live repair E2E suites currently include:

- `test_baseline_live_e2e.py` — external baseline adapters against a live server
- `test_evals_native_live_e2e.py` — native `base_repair` / `cot_repair` against a live server

## Shared fixtures

### `conftest.py`

Intentionally minimal. One helper:

- **`read_latest_snapshot(out_dir)`** — Reads the most recent snapshot from a
  directory, expanding compact keys. Raises `AssertionError` if none exists.

### `_openai_stub_server.py`

Test-only HTTP stub for OpenAI API calls. Used by eval integration tests
(`test_evals_openai_integration.py`, `test_evals_openai_patcher.py`) and the
post-provision baseline runtime lane (`test_baseline_runtime_integration.py`).
Provides `OpenAIStubServer` with `enqueue_json()` for canned responses.

### `_live_e2e_support.py`

Shared live-test support for `test_baseline_live_e2e.py` and
`test_evals_native_live_e2e.py`. Holds the live endpoint/model fixtures,
reachability skip helper, model auto-discovery from `/v1/models`, shared
solvable/unsolvable cases, and local pytest rerun helpers.

### Common pytest fixtures

Most tests use standard pytest fixtures:
- **`tmp_path`** — isolated temp directory (used heavily by capture, output, CLI tests)
- **`monkeypatch`** — environment/attribute patching
- **`capsys`** — stdout/stderr capture

Some test files define local fixtures for config variants (e.g., `cfg_minimal`,
`cfg_standard`, `cfg_full` in `test_benchmarks.py`).

## Coverage expectations

From `CONTRIBUTING.md` and `pyproject.toml`:

| Gate | Threshold |
|---|---|
| Line coverage floor | `--cov-fail-under=95` |
| Branch coverage | No-regression vs `quality/coverage_baseline.json` |
| Baseline patch seam coverage | `just baseline-boundary-coverage` |
| Diff coverage (changed lines) | `diff-cover --fail-under=80` |

Coverage is measured over `src/llmdebug` only. Tests, `__pycache__`, and
`snapshot_types.py` are excluded. Lines guarded by `TYPE_CHECKING`, `@overload`,
or `pragma: no cover` are exempt.

Run the full coverage check:

```bash
uv run pytest -m "not integration" --cov=src/llmdebug --cov-branch --cov-report=term-missing --cov-report=json --cov-report=xml --cov-fail-under=95
just baseline-boundary-coverage
```

## Adding a new test

1. **Create the file** as `tests/test_<module>.py` (or `test_evals_<module>.py`
   for eval code). Match the source module name.

2. **Write test functions** named `test_<what_it_verifies>`. Keep each test
   focused on one behavior.

3. **Use `tmp_path`** for any file I/O — never write to the repo working tree.

4. **Add markers** if needed (`@pytest.mark.slow`, `@pytest.mark.integration`,
   `@pytest.mark.baseline_runtime`).

5. **Run locally** before pushing:
   ```bash
   uv run pytest tests/test_<module>.py -v        # your file
   uv run pytest -x -q -m "not integration"       # default suite, fail-fast
   just test-integration                           # if you touched integration tests
   just test-baseline-runtime                      # if you touched provisioned baseline runtime tests
   just check                                      # lint + typecheck + coverage
   ```

6. **Check coverage** on your diff:
   ```bash
   just test                # generates coverage.json + coverage.xml
   just diff-coverage       # checks changed-line coverage ≥ 80%
   ```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full quality gate contract.
