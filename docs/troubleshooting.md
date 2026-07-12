# Troubleshooting

Solutions for common issues when using llmdebug in development, testing,
and CI/CD. For configuration details see [configuration.md](configuration.md);
for the full test workflow see [testing-guide.md](testing-guide.md).

## Common Issues

### "No snapshot found"

**Symptoms**: `llmdebug show` prints nothing, `get_latest_snapshot()` returns
`None`, or tests fail with `AssertionError: No snapshot in <dir>`.

**Causes and fixes**:

| Cause                               | Fix                                        |
|--------------------------------------|--------------------------------------------|
| No exception was raised              | Verify code actually throws                |
| Wrong `out_dir`                      | Check `SnapshotConfig(out_dir=...)` path   |
| `latest.json` symlink is stale       | Delete `.llmdebug/latest.*` and re-run     |
| Pytest plugin disabled               | See [Plugin not activating](#pytest-plugin-not-activating) |
| `@pytest.mark.no_snapshot` applied   | Remove the marker if capture is expected   |

For a dependency-light check of capture and display, create an offline example:

```bash
llmdebug quickstart
```

This writes a synthetic failure snapshot without pytest, Docker, a model, or
evaluation data. Use `llmdebug quickstart --dir <path>` when troubleshooting a
custom snapshot directory.

Quick diagnostic:

```bash
# Check if any snapshot files exist
ls -la .llmdebug/

# Check if latest symlink points to a real file
readlink .llmdebug/latest.json

# Force a snapshot via the decorator
python -c "
from llmdebug import debug_snapshot
@debug_snapshot()
def f(): raise ValueError('test')
f()
"
ls .llmdebug/latest.json
```

### Lock Contention (`.llmdebug.lock` Timeouts)

**Symptoms**: Warning `llmdebug: lock timeout — latest symlink may be stale`
on stderr, or stale `latest.json` after concurrent test runs.

The output module uses `filelock` to serialize symlink updates and old-snapshot
cleanup. The default lock timeout is 5 seconds (1 second for production hooks).

**Fixes**:

```python
# Increase lock timeout for slow CI filesystems
cfg = SnapshotConfig(lock_timeout=10.0)
```

```bash
# Remove stale lock files
rm -f .llmdebug/.llmdebug.lock

# In CI, avoid sharing the same checkout across parallel jobs.
# The pytest plugin writes to .llmdebug/ in the current working tree.
```

When running parallel pytest workers (`pytest-xdist`), avoid sharing the same
working tree for snapshot writes. For non-pytest capture entry points
(`debug_snapshot`, `snapshot_section`, hooks, middleware), use distinct
`SnapshotConfig(out_dir=...)` values when you need separate snapshot stores.

### SQLite "database is locked" Errors

**Symptoms**: `sqlite3.OperationalError: database is locked` from
`state_db.py` during RCA operations or snapshot index updates.

The state database (`llmdebug_state.sqlite3`) uses WAL mode with a 30-second
busy timeout by default. Contention can occur when:

- Multiple processes write to the same `.llmdebug/` directory simultaneously.
- A network filesystem (NFS/CIFS) doesn't support SQLite locking properly.
- A previous process crashed and left a `-wal` or `-shm` file behind.

**Fixes**:

```bash
# Increase the busy timeout (milliseconds)
export LLMDEBUG_DB_BUSY_TIMEOUT_MS=60000

# Remove stale WAL files (safe when no other process is writing)
rm -f .llmdebug/llmdebug_state.sqlite3-wal
rm -f .llmdebug/llmdebug_state.sqlite3-shm

# On network filesystems, prefer a local checkout or local working directory.
# For non-pytest capture entry points, set SnapshotConfig(out_dir=...) explicitly.
```

The WAL mode initialization includes retry logic (8 attempts with exponential
back-off) to tolerate brief startup contention. If you still hit issues, check
whether multiple CI jobs share the same workspace directory.

### Snapshot Corruption / Malformed JSON

**Symptoms**: `JSONDecodeError` when reading snapshots, or `compact key
expansion failed` warnings.

**Common causes**:

- **Partial write**: Process killed during snapshot write. The output module
  uses atomic write (write to `.tmp` then rename), but a crash between the
  two steps can leave a partial file.
- **Concurrent format switch**: Switching between `json`, `json_compact`,
  and `toon` formats mid-session can leave a stale `latest.*` from a
  different format.
- **Disk full**: No space for the atomic rename.

**Fixes**:

```bash
# Remove corrupted snapshots and stale latest pointers
rm -f .llmdebug/latest.json .llmdebug/latest.toon
rm -f .llmdebug/*.tmp

# Re-run the failing test to regenerate
uv run pytest tests/test_example.py -x

# Validate a specific snapshot file
python -c "
from llmdebug._snapshot_loader import load_snapshot_from_file
snap = load_snapshot_from_file('path/to/snapshot.json')
print(snap['exception']['type'] if snap else 'INVALID')
"
```

### Pytest Plugin Not Activating

**Symptoms**: Tests fail but no `.llmdebug/` directory is created.

The plugin is registered as a `pytest11` entry point. Verify it's installed:

```bash
# Check entry point registration
uv run pytest --co -q 2>&1 | head -5

# Verify the plugin is loaded
uv run pytest --trace-config 2>&1 | grep llmdebug

# Check for import errors
uv run python -c "import llmdebug.pytest_plugin; print('OK')"
```

**Common causes**:

| Cause                                  | Fix                                     |
|----------------------------------------|-----------------------------------------|
| llmdebug not installed in the env      | `uv sync --group dev`                   |
| `-p no:llmdebug` in pytest args        | Remove the plugin disable flag          |
| `conftest.py` overrides `pytest_runtest_makereport` | Check for hook conflicts  |
| `LLMDEBUG_DEBUG=1` reveals import err  | Fix the import error shown on stderr    |

### MCP Server Connection Issues

**Symptoms**: IDE (Claude Code, Cursor, etc.) can't connect to the MCP
server, or tools return empty responses.

```bash
# Verify the MCP extra is installed
uv run python -c "import mcp; print(mcp.__version__)"

# Start the server manually to check for errors
uv run llmdebug-mcp

# Check if snapshots exist in the expected directory
ls -la .llmdebug/
```

**Common causes**:

| Cause                               | Fix                                       |
|--------------------------------------|-------------------------------------------|
| `mcp` package not installed          | `pip install 'llmdebug[mcp]'`             |
| No snapshots in `.llmdebug/`         | Run a failing test first                  |
| Wrong working directory              | Start MCP server from the project root    |
| Firewall / port conflict             | Check the MCP transport configuration     |

## CI/CD Issues

### Coverage Below Threshold

**Symptoms**: CI fails with `FAIL Required test coverage of 95% not reached`.

```bash
# Check current coverage locally
uv run pytest --cov=src/llmdebug --cov-report=term-missing --cov-fail-under=95

# Identify uncovered lines
uv run pytest --cov=src/llmdebug --cov-report=html
open htmlcov/index.html
```

**Strategies to improve coverage**:

- Add tests for the specific uncovered lines shown in `term-missing`.
- Use `# pragma: no cover` **only** for genuinely unreachable code
  (type guards, protocol stubs, `if __name__` blocks).
- New code must have ≥ 80% diff-coverage (enforced by `diff-cover`).

### Diff-Cover Failures on PRs

**Symptoms**: `diff-cover` reports `Coverage is below 80% on changed lines`.

```bash
# Run diff-cover locally against main
just diff-coverage

# See exactly which changed lines lack coverage
uv run diff-cover coverage.xml --compare-branch=origin/main --html-report diff.html
open diff.html
```

Make sure `coverage.xml` is up to date — run `just test` before
`just diff-coverage`.

### Type Checking Errors with Pyright Strict

**Symptoms**: `pyright` reports errors on new or changed code.

```bash
uv run pyright
```

Pyright runs in **strict mode** against `src/llmdebug` (tests and evals
are excluded). Common strict-mode issues:

| Error                               | Fix                                        |
|--------------------------------------|--------------------------------------------|
| `reportMissingParameterType`         | Add type annotations to all parameters     |
| `reportUnknownMemberType`            | Add explicit `cast()` or type narrowing    |
| `reportReturnType`                   | Annotate return types on all functions     |
| `reportGeneralTypeIssues`            | Use `Any` sparingly, prefer specific types |

## Performance Issues

### Slow Snapshot Capture

**Symptoms**: Test runs are noticeably slower with llmdebug installed,
or `just bench-capture` shows high overhead.

**Diagnose with benchmarks**:

```bash
just bench-capture
# Check median capture time in .benchmarks/capture-overhead-current.json
```

**Common causes and mitigations**:

| Cause                        | Mitigation                                       |
|------------------------------|--------------------------------------------------|
| Git dirty check (`git status`) | Set `LLMDEBUG_GIT_DIRTY_MODE=cached` or `off`  |
| Large local variables        | Use `locals_mode="meta"` to skip serialization   |
| Source context reading       | Use `source_mode="crash_only"` or `"none"`       |
| Many enrichment modules      | Disable unused: `include_git=False`, etc.        |
| Slow filesystem              | Write to a local tmpdir, not NFS                 |

```bash
# Fastest capture config for CI
export LLMDEBUG_GIT_DIRTY_MODE=off
export LLMDEBUG_OUTPUT_FORMAT=json_compact
```

The git metadata caching (introduced in v2.19.0) caches static git context
in-process. The dirty-check mode controls whether `git status` runs on
every capture (`always`), is cached after the first call (`cached`), or
is skipped entirely (`off`).

### High Memory Usage with Array-Like Objects

**Symptoms**: Memory spikes when capturing snapshots of frames with large
NumPy arrays, pandas DataFrames, or torch tensors.

The serializer computes summary statistics (shape, dtype, min/max/mean) for
array-like objects by default when `include_array_stats=True`.

**Mitigations**:

```python
# Disable array stats
cfg = SnapshotConfig(include_array_stats=False)

# Use metadata-only locals (no serialization, just types and sizes)
cfg = SnapshotConfig(locals_mode="meta")

# Limit the number of frames captured
cfg = SnapshotConfig(frames=5)

# Limit string/collection sizes
cfg = SnapshotConfig(max_str=200, max_collection=20)
```

For production hooks, array stats are disabled by default:

```python
import llmdebug
llmdebug.install_hooks(include_array_stats=False)  # default
```

## Environment Issues

### Missing Optional Dependencies

llmdebug core requires only `filelock` and `orjson`. Features behind
optional extras raise clear import errors when their dependencies are
missing.

| Feature        | Extra          | Package(s)             | Install command                   |
|----------------|----------------|------------------------|-----------------------------------|
| CLI            | `cli`          | `click`, `rich`        | `pip install 'llmdebug[cli]'`       |
| MCP server     | `mcp`          | `mcp`                  | `pip install 'llmdebug[mcp]'`       |
| TOON format    | `toon`         | `toons`                | `pip install 'llmdebug[toon]'`      |
| Jupyter        | `jupyter`      | `ipython`              | `pip install 'llmdebug[jupyter]'`   |
| Eval harness   | `evals`        | `datasets`, `httpx`, `polars`, `scikit-learn`, etc. | `pip install 'llmdebug[evals]'` |
| Dev / testing  | `dev`          | All of the above + test/lint tools | `uv sync --group dev`  |

**Diagnostic**:

```bash
# Check what's installed
uv run python -c "
extras = ['click', 'rich', 'mcp', 'toons', 'IPython']
for pkg in extras:
    try:
        __import__(pkg)
        print(f'  {pkg}: OK')
    except ImportError:
        print(f'  {pkg}: MISSING')
"
```

### Python Version Compatibility

llmdebug supports **Python 3.10+** (3.10, 3.11, 3.12, 3.13).

**Common version-specific issues**:

| Issue                                  | Versions affected | Notes                    |
|----------------------------------------|-------------------|--------------------------|
| `X | Y` union syntax in annotations   | < 3.10            | Use `from __future__ import annotations` |
| `match` statement in tests             | < 3.10            | Not used in llmdebug     |
| `tomllib` in stdlib                    | < 3.11            | Not used directly        |
| `ExceptionGroup`                       | < 3.11            | Graceful fallback        |

Check your Python version:

```bash
python --version
uv run python --version
```

To test with a specific version:

```bash
PYTHON_VERSION=3.10 just test-quick
```

## Quality Gate Failures

This section covers every gate in the blocking quality contract for
`src/llmdebug`. For the full gate list see [quality-map.md](quality-map.md).

### Coverage Below 95%

**Symptoms**: CI fails with `FAIL Required test coverage of 95% not reached`.

```bash
# Identify uncovered lines
uv run pytest --cov=src/llmdebug --cov-report=term-missing --cov-fail-under=95

# HTML report for detailed exploration
uv run pytest --cov=src/llmdebug --cov-report=html
open htmlcov/index.html
```

**Common causes**: new module added without corresponding tests, untested
error/exception paths, missing branch coverage on `if`/`else` blocks.

**Fix**: Add targeted tests for the uncovered paths. Use `# pragma: no cover`
only for genuinely unreachable code (type guards, protocol stubs).

### Line or Branch Coverage Regression

**Symptoms**: `check_coverage_no_regression.py` fails with a message that
line or branch coverage dropped below the baseline.

The script `scripts/quality/check_coverage_no_regression.py` compares the
current line and branch coverage against `quality/coverage_baseline.json`:

```json
{
  "line_percent": 99.2,
  "branch_percent": 98.2,
  "source": "main",
  "updated_at_utc": "2026-03-26T12:50:58Z"
}
```

**Fix**: Either improve coverage to meet the baseline, or intentionally
ratchet the baseline after an improvement:

```bash
# 1. Run coverage
uv run pytest --cov=src/llmdebug --cov-branch --cov-report=json --cov-report=xml --cov-fail-under=95

# 2. Update quality/coverage_baseline.json with the measured line_percent,
#    branch_percent, and current UTC timestamp in updated_at_utc

# 3. Include the reason for the ratchet in the PR description
```

See the [quality-gates section in CONTRIBUTING.md](../CONTRIBUTING.md#quality-gates)
for the full ratchet procedure.

### Type Coverage Regression

**Symptoms**: `check_type_coverage.py --strict` fails, reporting that public
type completeness dropped below the baseline.

```bash
# See all Pyright errors
uv run pyright

# Check type completeness score
just type-coverage-strict
```

**Fix**: Add type annotations to new or changed public APIs. Pyright strict
mode (`pyrightconfig.json`) requires annotations on all parameters, return
types, and class attributes in `src/llmdebug`.

### Xenon Complexity Violation

**Symptoms**: `xenon` fails with a message about functions exceeding the
complexity ceiling.

The gate runs: `xenon --max-absolute C --max-modules C --max-average B src/llmdebug`

| Grade | Cyclomatic complexity | Meaning          |
|-------|----------------------|------------------|
| A     | 1–5                  | Simple           |
| B     | 6–10                 | Moderate         |
| C     | 11–15                | Complex          |
| D     | 16–20                | Very complex     |
| E     | 21–25                | Highly complex   |
| F     | 26+                  | Unmaintainable   |

```bash
# Find complex functions in a specific file
uv run radon cc -s src/llmdebug/<file>.py

# Show all complex functions across the package
just complexity
```

**Fix**: Extract helper functions, simplify conditionals, reduce nesting
depth. The ceiling is **C** for any single function and **B** average across
the entire package.

### Bandit Security Finding

**Symptoms**: `bandit` reports a security finding (e.g., `B105`, `B608`).

```bash
# Run locally with the project config
uv run bandit -r src/llmdebug -c pyproject.toml
```

The project already skips `B101` (assert), `B311` (non-crypto random),
`B404`/`B603` (subprocess usage) in `pyproject.toml`.

**Fix**: If the finding is a **false positive** or **intentionally accepted
risk**, suppress with an inline comment and justification:

```python
password = get_from_vault(key)  # nosec B105 — not a hardcoded password
```

Only suppress when the finding is genuinely non-exploitable. For real issues,
fix the code.

### Vulture False Positive

**Symptoms**: `vulture` reports code as unused that is actually needed
(plugin entry points, protocol methods, `__all__` exports).

**Fix**: Add the symbol to `vulture_whitelist.py`:

```python
# vulture_whitelist.py
from llmdebug.pytest_plugin import pytest_configure  # entry point
```

If vulture found **genuinely dead code**, remove it instead of whitelisting.

### Deptry Dependency Issue

**Symptoms**: `deptry` reports a dependency rule violation.

| Code   | Meaning                   | Typical cause                          |
|--------|---------------------------|----------------------------------------|
| DEP001 | Missing dependency        | Optional extras (`torch`, `jax`, etc.) |
| DEP002 | Unused dependency         | Test-only or eval-only packages        |
| DEP004 | Misplaced dev dependency  | `pytest`, `numpy` used at runtime      |

**Fix**: Update `[tool.deptry.per_rule_ignores]` in `pyproject.toml` for
intentional optional/conditional imports:

```toml
[tool.deptry.per_rule_ignores]
DEP001 = ["torch", "jax", "pandas", "toons", "IPython"]
DEP002 = ["filelock", "datasets", "httpx", ...]
```

### pip-audit Finding

**Symptoms**: `pip-audit` reports a known vulnerability in a dependency.

**Fix**:

1. Check if an upstream fix exists — upgrade the dependency.
2. If no fix is available, add a temporary ignore with justification.

**Current known ignore**: `GHSA-5239-wwwm-4pmq` / `CVE-2026-4539` in
Pygments (no upstream fix yet). This exception is tracked in
[quality-map.md](quality-map.md) and should be removed once a fixed
Pygments release is published.

### Diff-Cover Below 80%

**Symptoms**: `diff-cover` reports `Coverage is below 80% on changed lines`.

```bash
# See exactly which changed lines lack coverage
uv run diff-cover coverage.xml --compare-branch=origin/main

# Or use the just recipe
just diff-coverage

# HTML report for detailed view
uv run diff-cover coverage.xml --compare-branch=origin/main --html-report diff.html
open diff.html
```

**Fix**: Write tests that exercise the specific lines you changed. Make sure
`coverage.xml` is up to date — run `just test` before `just diff-coverage`.

### Benchmark Regression

**Symptoms**: local benchmark comparison reports that capture overhead increased
more than expected.

```bash
just bench-capture
just bench-capture-compare
```

The local baseline lives in `.benchmarks/capture-overhead-baseline.json`.
Benchmark results are stored in `.benchmarks/capture-overhead-current.json`.

**Fix**:

- **Intentional** (e.g., added a serialization step): Refresh the local baseline
  from a trusted run and include the reason in your PR description.
- **Unintentional**: Profile the regression and optimize. Common causes:
  extra I/O per capture, un-cached git calls, or serializer changes.

## Getting Help

If none of the above resolves your issue:

1. **Check existing issues**: Search the repository issue tracker.
2. **Enable debug mode**: `LLMDEBUG_DEBUG=1` prints capture diagnostics
   to stderr.
3. **Collect diagnostic info**:
   ```bash
   uv run python -c "
   import llmdebug, sys, platform
   print(f'llmdebug: {llmdebug.__version__}')
   print(f'Python:   {sys.version}')
   print(f'Platform: {platform.platform()}')
   "
   ```
4. **Open an issue** with the diagnostic output and a minimal reproducer.
