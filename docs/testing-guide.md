# Testing Guide

Comprehensive guide to running, writing, and debugging tests in llmdebug.
For first-time setup and quality-gate overview, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Quick Start

Install dev dependencies and run the test suite:

```bash
uv sync --group dev
uv run pytest -x -q -m "not integration"   # Fail-fast, quiet output
```

Common task recipes (via `just`):

| Command                | What it does                                    |
|------------------------|-------------------------------------------------|
| `just test-quick`      | Fast feedback loop, excludes `integration`      |
| `just test`            | Coverage run, excludes `integration`            |
| `just test-integration`| Blocking integration lane, excludes `baseline_runtime` |
| `just test-baseline-runtime` | Provisioned baseline repo/env/runtime lane |
| `just check`           | Lint + typecheck + coverage-gated tests         |
| `just quality`         | Full CI mirror: eval lint, deps, audit, docs    |
| `just lint-evals`      | Blocking Ruff lint/format check for `evals/`    |
| `just typecheck-evals` | Staged eval static check in `basic` mode        |
| `just test-evals-cov`  | Generate the eval coverage report               |
| `just eval-coverage-no-regression` | Compare eval report with informational ratchet |
| `just ci`              | Exact alias for `just quality`                  |

Run a single test file or function:

```bash
uv run pytest tests/test_capture.py -v
uv run pytest tests/test_mcp_server.py::test_show_latest -v
```

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                  # Shared fixtures and helpers
├── test_capture.py              # Exception capture logic
├── test_serialize.py            # Local-variable serialization
├── test_serialize_edge_cases.py # Serialization edge cases
├── test_output.py               # File I/O, latest symlink, locking
├── test_output_edge_cases.py    # Output path edge cases
├── test_rca_engine.py           # RCA state machine transitions
├── test_rca_backend.py          # RCA persistence backend
├── test_rca_store_sqlite.py     # SQLite RCA store
├── test_rca_transitions.py      # State transition guards
├── test_mcp_server.py           # MCP tool handlers
├── test_mcp_contract.py         # MCP JSON envelope contracts
├── test_mcp_rca.py              # MCP + RCA integration
├── test_cli.py                  # CLI commands (Click CliRunner)
├── test_cli_edge_cases.py       # CLI error paths
├── test_pytest_plugin.py        # Pytest plugin activation
├── test_hooks.py                # Production exception hooks
├── test_benchmarks.py           # Performance benchmarks
├── test_evals_*.py              # Eval harness tests
├── golden_eval/                 # Golden fixture runs of the eval harness
├── golden/eval_runs/            # Their committed outputs (gzipped JSON)
└── ...
```

### Naming Conventions

- **Files**: `test_<module>.py` — mirrors the source module under test.
- **Functions**: `test_<behavior_under_test>` — descriptive, no abbreviations.
- **Docstrings**: Every test function has a one-line docstring explaining
  *what* it verifies, not *how*.

```python
def test_capture_exception_collects_warning_paths(tmp_path, monkeypatch):
    """Capture should aggregate warning paths from optional enrichments."""
    ...
```

### Markers

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker         | Purpose                              | Usage                       |
|----------------|--------------------------------------|-----------------------------|
| `slow`         | Tests that take > 2 s                | `@pytest.mark.slow`         |
| `integration`  | Tests needing external resources     | `@pytest.mark.integration`  |
| `baseline_runtime` | Tests needing provisioned baseline repos/envs | `@pytest.mark.baseline_runtime` |

Skip slow tests during rapid iteration:

```bash
uv run pytest -x -q -m "not slow"
```

Default `just test-quick`, `just test`, and `just test-no-cov` commands also
exclude `integration` tests:

```bash
uv run pytest -x -q -m "not integration"
```

Run the blocking integration lane:

```bash
just test-integration
```

Run the provisioned baseline runtime lane after cloning repos and building
`baselines/envs/*` smoke runtimes:

```bash
just test-baseline-runtime
```

Run live baseline adapter E2E tests with dynamic OpenAI-compatible overrides:

```bash
uv run pytest -m integration tests/test_baseline_live_e2e.py::TestLambdaRLMLiveE2E -v -k thinking_off \
  --live-openai-base-url http://127.0.0.1:1234/v1 \
  --live-openai-model mlx-community/Qwen3.5-4B-MLX-8bit \
  --live-openai-compat-profile auto
```

Run live native eval-condition E2E tests:

```bash
uv run pytest -m integration tests/test_evals_native_live_e2e.py -v \
  --live-openai-base-url http://127.0.0.1:1234/v1 \
  --live-openai-model mlx-community/Qwen3.5-4B-MLX-8bit
```

Available live-test flags:

- `--live-openai-base-url` (default: `http://127.0.0.1:1234/v1`)
- `--live-openai-model` (default: first entry returned by `/v1/models`)
- `--live-openai-compat-profile` (`auto`, `qwen_openai_raw`, `qwen_dashscope`,
  `lmstudio_model_yaml`, `off`; default: `auto`, external baseline suite only)

## Fixture Patterns

### Shared Fixtures (`conftest.py`)

The top-level `conftest.py` provides a single utility:

```python
from tests.conftest import read_latest_snapshot

def test_something(tmp_path):
    # ... trigger a snapshot into tmp_path ...
    snapshot = read_latest_snapshot(tmp_path)
    assert snapshot["exception"]["type"] == "ValueError"
```

`read_latest_snapshot(out_dir)` reads and auto-expands compact keys, raising
`AssertionError` if no snapshot exists.

### Shared Live E2E Support (`_live_e2e_support.py`)

The live repair suites share a small helper module for:

- live endpoint/model fixtures
- server reachability skip behavior
- shared solvable/unsolvable cases
- local pytest rerun helpers

It is used by both `test_baseline_live_e2e.py` and
`test_evals_native_live_e2e.py`.

### `tmp_path` for Isolated Snapshots

Most tests use pytest's built-in `tmp_path` fixture as the snapshot output
directory to avoid cross-test interference:

```python
def test_capture_writes_snapshot(tmp_path):
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    capture_exception("test_name", exc, tb, cfg)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
```

### `monkeypatch` for Fault Injection

Capture tests frequently monkeypatch enrichment modules to verify graceful
degradation:

```python
def test_capture_survives_git_failure(tmp_path, monkeypatch):
    """Capture should emit a warning when git context fails."""
    monkeypatch.setattr(git_context_module, "get_git_context", _boom)
    exc, tb = _exception_with_traceback()
    cfg = SnapshotConfig(out_dir=str(tmp_path), include_git=True)
    capture_exception("test", exc, tb, cfg)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert any("git context failed" in w
               for w in snapshot.get("_capture_warnings", []))
```

### `sample_snapshot` for MCP / CLI Tests

The MCP and CLI test modules define a `sample_snapshot` fixture with a
representative expanded snapshot dict. This avoids re-creating realistic
payloads in every test:

```python
@pytest.fixture
def sample_snapshot() -> dict:
    """A representative expanded snapshot dict."""
    return {
        "schema_version": "1.0",
        "name": "test_example",
        "exception": {
            "type": "ValueError",
            "message": "invalid literal for int(): 'abc'",
        },
        "frames": [{ ... }],
        ...
    }
```

### Config Fixtures for Benchmarks

Benchmark tests use dedicated config fixtures to isolate variables:

```python
@pytest.fixture
def cfg_minimal():
    """Minimal config for baseline measurements."""
    return SnapshotConfig(
        locals_mode="none", source_mode="none",
        include_env=False, include_git=False,
    )
```

## Writing Tests for Each Layer

### Capture Tests (`test_capture.py`)

Test the exception → snapshot pipeline. Key patterns:

1. Create a real exception with traceback via a helper function.
2. Configure `SnapshotConfig` with `out_dir=str(tmp_path)`.
3. Call `capture_exception(...)`.
4. Assert on the snapshot payload.

```python
def _exception_with_traceback():
    try:
        marker = 1  # keep a local in the crash frame
        raise ValueError("boom")
    except Exception as exc:
        return exc, exc.__traceback__

def test_basic_capture(tmp_path):
    """Capture writes a valid snapshot with exception details."""
    exc, tb = _exception_with_traceback()
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    capture_exception("basic", exc, tb, cfg)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert snapshot["exception"]["type"] == "ValueError"
```

### Serialization Tests (`test_serialize.py`)

Test the `to_jsonlike` and `serialize_locals` functions in isolation:

```python
def test_primitives(cfg):
    """Test primitive type serialization."""
    assert to_jsonlike(None, cfg) is None
    assert to_jsonlike(42, cfg) == 42
    assert to_jsonlike("hello", cfg) == "hello"
```

### RCA Tests (`test_rca_engine.py`)

Test the state machine with explicit previous-state dicts:

```python
def test_initial_attempt():
    record = build_rca_attempt_record(
        previous=None,
        attempt_id="att-1",
        request=RCAAttemptRequest(
            session_id="sess-1",
            action="hypothesize",
            failure_signature="sig-1",
            evidence_refs=("exception:AssertionError",),
            hypothesis="off-by-one",
            experiment_plan="add boundary check test",
            gate_mode="soft",
            exploratory=False,
        ),
        now="2026-03-01T12:00:00Z",
    )
    assert record["state"] == "hypothesized"
    assert record["repeat_count"] == 1
```

### MCP Tests (`test_mcp_server.py`)

Test MCP tool handlers by writing snapshots to `tmp_path`, then calling
tool functions directly. Use the `_assert_json_error` helper for error
envelope validation:

```python
def test_show_returns_error_when_no_snapshot(tmp_path):
    result = llmdebug_show(out_dir=str(tmp_path))
    data = _assert_json_error(
        result, tool="show", code="NO_SNAPSHOT",
        message_contains="No snapshot",
    )
```

### CLI Tests (`test_cli.py`)

Use Click's `CliRunner` to invoke commands, writing fixtures to `tmp_path`:

```python
def test_show_command(tmp_path, sample_snapshot):
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    write_bundle(sample_snapshot, cfg)
    runner = CliRunner()
    result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
    assert result.exit_code == 0
```

### Eval Tests (`test_evals_*.py`)

Eval tests verify the evaluation harness — case loading, scoring, result
analysis. They often use fixtures or small inline data. These tests are
more integration-heavy and may carry the `slow` marker.

### Golden Fixture Runs (`golden_eval/`)

Golden runs prove that a harness refactor changes no behaviour. Each scenario in
`tests/golden_eval/scenarios.py` runs `run_eval.main` end to end (`--jobs 1`,
`--deterministic --seed 7`, local executor) on six synthetic cases from
`tests/golden_eval/cases.py`, against a stub model that scripts the n-th patch call of
each unit: wrong patches, syntax errors, sanitizer rejections, stale diffs, non-answers
and transport faults. The output keeps the result records, policy event logs and
states, the artifact tree, every model request and the `analyze_results`,
`report_run_metrics` and `eval_summary` output, with timings, ports, random IDs and
the run root replaced by placeholders. Together the scenarios cover about 80% of
`eval_attempt_flow.py`; baselines aren't covered because they need provisioned repos.

```bash
just golden-evals          # compare with tests/golden/eval_runs/ (about 15 s)
just golden-evals-update   # rewrite the golden files after an intended change
```

On a difference, the runner prints the first differing paths and writes the full
output to `.golden-diff/`. Prompts contain the checkout and interpreter paths, and
pytest's output depends on `uv.lock`, so golden files only compare in the environment
recorded in `tests/golden/eval_runs/environment.json`; elsewhere `just golden-evals`
exits 2 and the integration test skips. To use them on another machine, run
`just golden-evals-update` on a commit whose behaviour you trust first and commit the
regenerated `environment.json` with the fixtures.

Each scenario subprocess gets a pinned environment (`scenario_env()`): proxies off, so
the loopback stub cannot be routed to a real host, and `LLMDEBUG_*`, `PYTEST_ADDOPTS`,
`TZ` and the locale set by the harness rather than inherited, because they reach the
records without changing the environment fingerprint. `TIKTOKEN_CACHE_DIR` points at
`~/.cache/tracegate-golden-eval-tokenizer`, and `ensure_offline_tokenizer()` resolves
the tokenizer once up front, so the recorded token estimates come from local data
instead of a per-run download. Each scenario holds an exclusive lock on its run root
under `~/.cache/tracegate-golden-eval` for the whole run, so a second invocation of the
same scenario fails instead of deleting the first one's live work tree.

## Coverage Requirements

### Thresholds

| Gate                    | Threshold | Enforced by                       |
|-------------------------|-----------|-----------------------------------|
| Line coverage floor     | 95%       | `--cov-fail-under=95`             |
| Package line + branch coverage | no regress | `quality/coverage_baseline.json` |
| Eval line + branch coverage | 98.2% line / 95.7% branch | `quality/coverage_evals_baseline.json` (informational CI) |
| Baseline patch seam coverage | per-module | `just baseline-boundary-coverage` |
| Diff coverage (PRs)     | 80%       | `diff-cover --fail-under=80`      |

### Running Coverage Locally

```bash
# Full coverage with reports
uv run pytest \
    --cov=src/llmdebug \
    --cov-branch \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-report=json \
    --cov-fail-under=95

# Or via just
just test
just coverage-no-regression
just baseline-boundary-coverage
just test-evals-cov
just eval-coverage-no-regression
```

### Diff-Cover on Changed Lines

```bash
just diff-coverage
# Equivalent to:
# git fetch origin main --depth=1
# uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
```

### Updating the Coverage Baselines

After improving package coverage, generate the report and confirm the ratchet:

```bash
just test   # generates coverage.json
just coverage-no-regression
```

Then edit both percentages in `quality/coverage_baseline.json`:

```json
{
  "line_percent": 99.2,
  "branch_percent": 98.2,
  "source": "main",
  "updated_at_utc": "2026-03-26T12:50:58Z"
}
```

For eval coverage, `just test-evals-cov` generates `coverage-evals.json`; then
`just eval-coverage-no-regression` checks both metrics against
`quality/coverage_evals_baseline.json`. The current eval baseline is 98.2% line
/ 95.7% branch. Update both values and `updated_at_utc` manually only after an
intentional, evidence-backed ratchet decision.

### Coverage Exclusions

Configured in `pyproject.toml` under `[tool.coverage.report]`:

```
pragma: no cover      # Explicit opt-out
if __name__           # Script guards
if TYPE_CHECKING      # Type-only imports
raise NotImplementedError
@overload             # Typing overloads
...                   # Protocol stubs
```

## Debugging Test Failures

### Use llmdebug Itself

The pytest plugin automatically captures snapshots on test failure into
`.llmdebug/`. After a failure:

```bash
# View the crash snapshot
llmdebug show

# Full detail with all frames
llmdebug show --detail full

# JSON for programmatic inspection
llmdebug show --json | python -m json.tool

# Compare against the previous failure
llmdebug diff
```

### Inspect Raw Snapshot Files

```bash
# Latest snapshot is always symlinked
cat .llmdebug/latest.json | python -m json.tool

# List all snapshots
ls -lt .llmdebug/*.json

# Or via the CLI
llmdebug list
```

### Verbose Pytest Output

```bash
uv run pytest tests/test_capture.py -v --tb=long -s
```

### Debug Mode

Set `LLMDEBUG_DEBUG=1` to enable verbose capture diagnostics on stderr:

```bash
LLMDEBUG_DEBUG=1 uv run pytest tests/test_capture.py -x -v
```

### Common Failure Patterns

| Symptom                          | Likely cause                          |
|----------------------------------|---------------------------------------|
| `AssertionError: No snapshot`    | `out_dir` mismatch or capture skipped |
| `FileNotFoundError: latest.json` | Test didn't trigger an exception      |
| `JSONDecodeError`                | Concurrent write or partial flush     |
| `OperationalError: locked`       | SQLite contention in parallel tests   |

## Benchmark Tests

Benchmarks live in `tests/test_benchmarks.py` and use `pytest-benchmark`.
They are **track-only** (non-gating) — they measure overhead but don't
fail CI.

### Running Benchmarks

```bash
just bench-capture
# Equivalent to:
# uv run pytest tests/test_benchmarks.py \
#     -k capture_overhead \
#     --benchmark-only \
#     --benchmark-group-by=group \
#     --benchmark-json=.benchmarks/capture-overhead-current.json -q
```

### Comparing Against a Baseline

```bash
# Save the current run as baseline
cp .benchmarks/capture-overhead-current.json \
   .benchmarks/capture-overhead-baseline.json

# Make changes, then re-run
just bench-capture
just bench-capture-compare
```

`bench-capture-compare` prints per-benchmark median deltas using
`scripts/bench_capture_compare.py`.

### Skipping Benchmarks

Benchmarks run with `--benchmark-only` by default. When running the full
test suite, they are included but the benchmark harness is skipped:

```bash
uv run pytest --benchmark-skip     # Explicit skip
uv run pytest -x -q                # Benchmarks auto-skip without --benchmark-only
```

### Benchmark Config Fixtures

Tests use named config fixtures to isolate what's being measured:

| Fixture          | What it measures                                |
|------------------|-------------------------------------------------|
| `cfg_minimal`    | Bare capture overhead (no locals, no git)       |
| `cfg_standard`   | Default config (representative real-world cost) |

## CI Integration

### What CI Runs

The shared GitHub action runs `just quality`; `just ci` is exactly a local
alias for that recipe. `just quality` has these direct recipe dependencies:

```text
just check
just lint-evals
just coverage-no-regression
just baseline-boundary-coverage
just deps
just audit
just docs-check
just dead-code
just complexity
just complexity-gate
just security
just type-coverage-strict
just dependency-sync
```

`just check` expands to package/test lint and format checks, strict package
typing, and the non-integration package test run with the 95% coverage floor.
On pull requests, the shared action additionally runs blocking diff coverage.
It also runs `just test-evals-cov` followed by
`just eval-coverage-no-regression` as an informational eval line/branch
coverage check; both recipes are intentionally outside `just quality`.

GitHub Actions also runs `just test-integration` as a separate 3.13 lane and
surfaces `just typecheck-evals` as an informational staged eval-typing pass.
Before those post-quality lanes, CI provisions the pinned baseline repos plus
`baselines/envs/*` smoke runtimes. It then runs:

- `just test-integration` as the blocking integration lane
- `just test-baseline-runtime` as a non-blocking provisioned baseline runtime lane
- `just typecheck-evals` as a non-blocking staged eval-typing lane

The live baseline adapter E2E suite remains manual-only because it depends on a
reachable local OpenAI-compatible model server.

### Reproducing CI Locally

```bash
# Blocking local quality contract
just ci

# Additional action/workflow lanes
just diff-coverage        # Blocking; only meaningful on PR branches vs main
just test-evals-cov       # Generate the informational eval coverage report
just eval-coverage-no-regression  # Informational; currently exposes existing debt
just typecheck-evals      # Informational in CI
just test-integration     # Blocking after baseline provisioning
just test-baseline-runtime  # Informational after baseline provisioning
```

### Multi-Python Testing

CI runs on Python 3.10–3.13. To test locally with a specific version:

```bash
PYTHON_VERSION=3.12 just test-quick
```

### Eval Tooling Validation

When changing eval-related code, also run:

```bash
uv run python -m evals.validate_cases --schema-only
```

### Quality Gate Summary

| Gate               | Tool            | Threshold / mode           |
|--------------------|-----------------|----------------------------|
| Lint               | ruff            | Zero warnings              |
| Format             | ruff format     | Check mode                 |
| Types              | pyright         | Strict, zero errors        |
| Line coverage      | pytest-cov      | ≥ 95%                      |
| Package line/branch baseline | custom script | No regression vs baseline |
| Eval line/branch baseline | custom script | No regression; informational CI |
| Diff coverage      | diff-cover      | ≥ 80% on changed lines     |
| Complexity         | xenon           | max-absolute C, avg B      |
| Security           | bandit          | Zero unsuppressed findings |
| Dead code          | vulture         | min-confidence 90          |
| Dependencies       | deptry          | Zero violations            |
| Vulnerabilities    | pip-audit       | Zero findings after documented ignores |
