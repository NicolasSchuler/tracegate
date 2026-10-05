# Configuration Reference

This document covers all capture entry points, parameters, output formats, and
API surface for `llmdebug`. For a quick overview, see the [README](../README.md).

## Installation Extras

```bash
pip install llmdebug              # Core library + pytest plugin
pip install 'llmdebug[cli]'       # + Click-based CLI (recommended)
pip install 'llmdebug[mcp]'       # + MCP server for IDE integration
pip install 'llmdebug[jupyter]'   # + Jupyter/IPython integration
pip install 'llmdebug[toon]'      # + TOON output format
pip install 'llmdebug[analysis]'  # + pandas/seaborn notebook analysis
pip install 'llmdebug[evals]'     # + Eval harness dependencies
```

## Capture Entry Points

### Pytest Plugin (automatic)

After installation, failing tests automatically create `.llmdebug/latest.json`.
No configuration required.

```bash
pytest  # Failures create .llmdebug/latest.json
```

Skip snapshot capture for specific tests:

```python
import pytest

@pytest.mark.no_snapshot
def test_expected_failure():
    ...
```

### Decorator

```python
from llmdebug import SnapshotConfig, debug_snapshot

cfg = SnapshotConfig(out_dir=".llmdebug", redaction_profile="ci")

@debug_snapshot(config=cfg)
def main():
    data = load_data()
    process(data)

if __name__ == "__main__":
    main()
```

### Context Manager

For targeted instrumentation when you need more detail:

```python
from llmdebug import SnapshotConfig, snapshot_section

cfg = SnapshotConfig(out_dir=".llmdebug", redaction_profile="prod")

with snapshot_section("data_processing", config=cfg):
    result = transform(data)
```

### Snapshot Privacy Defaults

`debug_snapshot()` and `snapshot_section()` are now config-first APIs.
If you do not pass any redaction settings, llmdebug emits a `UserWarning` so you can
opt into a safer profile explicitly. Legacy flat kwargs are still accepted
temporarily for backward compatibility, but they emit a `DeprecationWarning`.

Recommended:

```python
from llmdebug import SnapshotConfig, debug_snapshot, snapshot_section

ci_cfg = SnapshotConfig(redaction_profile="ci")      # safer dev/CI default
prod_cfg = SnapshotConfig(redaction_profile="prod")  # stricter production profile

@debug_snapshot(config=ci_cfg)
def run_job():
    ...

with snapshot_section("checkout", config=prod_cfg):
    ...
```

### Production Hooks

Capture unhandled exceptions automatically in production applications:

```python
import llmdebug

llmdebug.install_hooks(out_dir=".llmdebug")

# Any unhandled exception, thread crash, or unraisable exception
# will now produce a snapshot automatically.

# Optional: uninstall when done
llmdebug.uninstall_hooks()
```

Hooks install into `sys.excepthook`, `threading.excepthook`, and `sys.unraisablehook`.
They include rate limiting (default: 10 captures/min) and automatic PII redaction.

### Web Middleware

Zero-config crash capture for web frameworks:

```python
# Flask
app.wsgi_app = LLMDebugWSGIMiddleware(app.wsgi_app)

# FastAPI
from llmdebug import LLMDebugASGIMiddleware
app.add_middleware(LLMDebugASGIMiddleware)

# Django WSGI
from llmdebug import LLMDebugWSGIMiddleware
application = LLMDebugWSGIMiddleware(application)
```

Middleware captures request context (method, path, query string) alongside the crash
snapshot, with automatic PII redaction on query parameters.

### Jupyter / IPython

Automatic snapshot capture in notebooks with rich HTML display:

```python
# In a notebook cell:
%load_ext llmdebug

# Or programmatically:
import llmdebug
llmdebug.load_jupyter()
```

After any cell error, a compact banner shows the exception, crash location, and hints.
Use magic commands for deeper analysis:

```python
%llmdebug              # Show full snapshot with locals and context
%llmdebug list         # List recent snapshots
%llmdebug diff         # Compare latest vs previous snapshot
%llmdebug config       # Show active configuration
```

Requires the `jupyter` extra: `pip install 'llmdebug[jupyter]'`

## Parameter Reference

Prefer building a `SnapshotConfig` and passing it via `config=...`. The same
object can be reused across `@debug_snapshot(...)` and `snapshot_section(...)`.
Legacy flat kwargs still work temporarily, but they are deprecated.

```python
from llmdebug import SnapshotConfig, debug_snapshot

cfg = SnapshotConfig(
    # -- Capture scope --
    frames=5,                       # Stack frames to capture
    source_context=3,               # Lines of source before/after crash
    source_mode="all",              # "all" | "crash_only" | "none"
    locals_mode="safe",             # "safe" | "meta" | "none"
    include_args=True,              # Separate function arguments from locals
    include_modules=None,           # Filter frames by module prefix (None = all)
    max_exception_depth=5,          # Exception chain recursion limit

    # -- Truncation limits --
    max_str=500,                    # Truncate long strings
    max_items=50,                   # Truncate large collections

    # -- Redaction and privacy --
    redaction_profile="dev",        # Optional: "dev" | "ci" | "prod"
    # redact=(r"token=\\S+",),       # Optional explicit pattern override
    # redact_keys=False,             # Optional explicit profile override
    # redact_traceback=False,        # Optional explicit profile override
    # redact_exception_strings=True, # Optional explicit profile override

    # -- Context enrichment --
    include_env=True,               # Include Python/platform info
    include_git=True,               # Git commit/branch/dirty status
    git_dirty_mode="always",        # "always" | "cached" | "off"
    categorize_errors=True,         # Auto-classify errors with suggestions
    include_async_context=True,     # Asyncio task info
    include_array_stats=False,      # Compute min/max/mean/std for arrays
    capture_logs=False,             # Capture recent log records
    log_max_records=20,             # Max log records to capture
    include_coverage=True,          # Pytest-plugin coverage enrichment toggle

    # -- Output and storage --
    out_dir=".llmdebug",            # Output directory
    max_snapshots=50,               # Auto-cleanup old snapshots (0 = unlimited)
    output_format="json_compact",   # "json" | "json_compact" | "toon"
    lock_timeout=5.0,               # Seconds to wait for file lock
)

@debug_snapshot(config=cfg)
def main():
    ...
```

## Redaction Profiles

`redaction_profile` provides preset behavior:

- `dev`: minimal redaction defaults
- `ci`: stronger string redaction for non-local workflows
- `prod`: strictest defaults (includes traceback/exception-string redaction)

Profiles provide preset values; explicit `redact`/`redact_*` options always
take precedence. Omit an option to inherit its profile value, use `()` to
explicitly disable profile patterns, and use `True`/`False` to explicitly
override a profile boolean. The resulting `SnapshotConfig` attributes are
always concrete tuples and booleans.

`SnapshotConfig` is frozen. Use `config.with_updates(...)` when changing a
redaction profile or override so inherited-versus-explicit intent remains
unambiguous:

```python
prod_cfg = SnapshotConfig(redaction_profile="prod")
ci_but_strict_tracebacks = prod_cfg.with_updates(
    redaction_profile="ci",
    redact_traceback=True,
)
```

`dataclasses.replace()` remains safe for unrelated fields such as `out_dir`,
but intentionally rejects profile changes while any redaction setting is
inherited; it cannot distinguish an omitted setting from an explicitly
repeated resolved value.

Redaction defaults to leaf string values only. This avoids accidental key collisions
in nested dicts. Set `redact_keys=True` only if you explicitly need key-name redaction
and can accept possible key merging.

## Environment Variables

Selected configuration options can be set via environment variables for pytest:

```bash
LLMDEBUG_OUTPUT_FORMAT=json pytest               # Use pretty JSON
LLMDEBUG_INCLUDE_GIT=false pytest                # Disable git context
LLMDEBUG_GIT_DIRTY_MODE=cached pytest            # Dirty check strategy
LLMDEBUG_CAPTURE_LOGS=true pytest                # Enable log capture
LLMDEBUG_REDACTION_PROFILE=ci pytest             # Use CI redaction profile
LLMDEBUG_REDACT_TRACEBACK=true pytest            # Redact traceback text
LLMDEBUG_REDACT_EXCEPTION_STRINGS=true pytest    # Redact exception strings
LLMDEBUG_RCA_MAX_RECORDS=5000 pytest             # Cap persisted RCA history records (see MCP reference for RCA details)
```

This is a partial list. See [CLI Reference](cli-reference.md#environment-variables)
for the full `LLMDEBUG_*` table and additional capture, pytest-context, redaction,
MCP, and RCA knobs.

## Output Formats

llmdebug supports multiple output formats to optimize for different use cases:

| Format | Size | Best For |
|--------|------|----------|
| `json` | baseline | Human readability, external tools |
| `json_compact` (default) | ~40% smaller | LLM context efficiency |
| `toon` | ~50% smaller | Maximum token savings |

**Compact JSON** uses abbreviated keys (e.g., `_exc` instead of `exception`) to reduce
token usage. The `get_latest_snapshot()` function auto-expands keys and normalizes
DebugSession envelopes by default, so your code works identically regardless of format.

Each snapshot is written as `<UTC timestamp>_<name>_<nonce>.<ext>`, next to a
`.traceback.txt` sidecar. The `<name>` part keeps only `A-Z a-z 0-9 . _ -` (every other
character becomes `_`) and is cut to 120 characters plus a hash of the full name, so long
parametrized pytest node IDs stay within filesystem limits. The snapshot payload keeps the
original name.

## Snapshot Enrichment

Snapshots are automatically enriched with contextual data:

- **Schema metadata**: `schema_version`, `llmdebug_version`, `crash_frame_index`
- **Exception detail**: `qualified_type`, `args`, `notes`, `cause`, `context`, `exceptions` (ExceptionGroup), `error_category` with auto-classification and suggestions
- **Frame metadata**: `module`, `file_rel`, `locals_meta` (type/size hints), truncation markers
- **Git context**: commit hash, branch, dirty status
- **Pytest context**: `longrepr`, `capstdout`, `capstderr`, params, `repro` command
- **Coverage data**: executed/missing lines, branch stats (when pytest-cov is active)
- **Async context**: asyncio task name and state
- **Log records**: recent log entries (opt-in via `capture_logs=True`)
- **Capture config**: frames, locals_mode, truncation limits, redaction patterns

### Locals Format

Locals are serialized to JSON-safe values. A local and its direct elements are expanded
(containers cut at `max_items`, strings at `max_str`); anything nested deeper becomes its
`repr`. Dict keys that are strings
are emitted as-is (`{"k": "v"}`); other key types use their `repr` (`{1: "a"}` becomes
`{"1": "a"}`), and a key that collides with an earlier one gets a stable `#2`, `#3` suffix.
Snapshots before schema `2.1` wrote every key as its `repr` (`{"'k'": "v"}`). Integers
outside the 64-bit range are stored as their `repr` string, and a value that can't be
introspected becomes `"<serialization failed: <ErrorType>>"` without affecting the rest.

### Coverage Notes

`include_coverage` currently applies to pytest-plugin failure captures only. Coverage
data is attached when pytest-cov is active and `LLMDEBUG_INCLUDE_COVERAGE` is enabled.
Decorator/context-manager captures do not currently add coverage payloads.

### Git Cache Behavior

`include_git` uses a short in-process cache for static repo metadata (commit/branch).
`git_dirty_mode` controls dirty checks:

- `always`: refresh dirty status on every capture (default)
- `cached`: reuse dirty status while cache is valid
- `off`: skip dirty status detection

## API Reference

```python
from llmdebug import (
    # Capture
    debug_snapshot,          # Decorator for exception capture
    snapshot_section,        # Context manager for targeted capture
    get_latest_snapshot,     # Read the most recent snapshot (auto-expands keys)
    SnapshotConfig,          # Configuration dataclass
    RedactionProfile,        # Type alias: "dev" | "ci" | "prod"
    resolve_redaction_policy,# Resolve profile + explicit redaction settings

    # Analysis
    filter_snapshot,         # Layered disclosure: filter to crash/full/context detail
    DetailLevel,             # Type alias: "crash" | "full" | "context"

    # Production hooks
    install_hooks,           # Install sys.excepthook + thread + unraisable hooks
    uninstall_hooks,         # Restore original hooks
    PII_PATTERNS,            # Default PII redaction patterns (email, API keys, etc.)

    # Jupyter / IPython
    load_jupyter,            # Install into current IPython/Jupyter session

    # Web middleware
    LLMDebugWSGIMiddleware,  # WSGI middleware (Flask, Django)
    LLMDebugASGIMiddleware,  # ASGI middleware (FastAPI, Starlette)

    # Log capture
    enable_log_capture,      # Install log handler to capture recent records
)

# Hypothesis engine (separate import)
from llmdebug.hypotheses import generate_hypotheses, Hypothesis
```

### Common Patterns

```python
# Read the most recent snapshot
snapshot = get_latest_snapshot()  # Returns dict or None

# Filter to minimal detail for LLM context
filtered = filter_snapshot(snapshot, "crash")  # Exception + crash frame only

# Generate debugging hypotheses
hypotheses = generate_hypotheses(snapshot)
for h in hypotheses:
    print(f"[{h.confidence:.0%}] {h.description}")
    print(f"  Suggestion: {h.suggestion}")
```
