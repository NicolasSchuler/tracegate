# CLI Reference

Full command reference for the `llmdebug` CLI. For a quick overview, see the
[README](../README.md#cli).

Requires the `cli` extra: `pip install 'llmdebug[cli]'`

## Commands

### quickstart

Create one real synthetic failure snapshot and display it at the normal crash
detail level:

```bash
llmdebug quickstart
llmdebug quickstart --dir "snapshot examples"
```

This first-run path works offline and does not require pytest, Docker, a model,
or evaluation data. It disables environment and git enrichment for the example
and never removes existing snapshots. After displaying the failure, it prints
the snapshot path and copy-paste-safe `show` and `list` commands for the chosen
directory.

### show

Display a snapshot with configurable detail level.

```bash
llmdebug                                # Latest snapshot (crash-level detail)
llmdebug show                           # Same as above
llmdebug show previous                  # Second most recent snapshot
llmdebug show --detail full             # All stack frames
llmdebug show --detail context          # Everything (pytest/env/capture context)
llmdebug show --json                    # Crash-level JSON
llmdebug show --json --detail context   # Expanded JSON with full context
llmdebug show --raw-session             # Output raw DebugSession envelope JSON
```

JSON output (`--json`) returns the raw snapshot payload for programmatic
consumption or piping to other tools.

### list

List recent snapshots with metadata.

```bash
llmdebug list                           # List all snapshots
llmdebug list --search "KeyError"       # Search exception/file/name metadata
```

Search matches snapshot metadata only: name, exception type, exception message,
and crash-file location.

### frames

Inspect a specific stack frame from a snapshot.

```bash
llmdebug frames -i 0                    # Crash frame from latest snapshot
llmdebug frames previous -i 0           # Crash frame from previous snapshot
llmdebug frames '#3' -i 2               # Frame 2 from snapshot #3
```

### diff

Compare two snapshots to see what changed between runs.

```bash
llmdebug diff                           # Compare latest vs previous
llmdebug diff #2 #1                     # Compare numbered entries from `llmdebug list`
llmdebug diff old.json new.json         # Compare specific files
llmdebug diff --json                    # Output diff as JSON
```

### git-context

Get richer git-aware debugging metadata on demand (without inflating snapshot
capture payloads).

```bash
llmdebug git-context                    # Latest snapshot, text view
llmdebug git-context --json             # JSON output for tooling
llmdebug git-context '#2'               # Specific snapshot reference
```

Outputs metadata only:
- Crash-line blame metadata
- Recent commit metadata + shortstats
- Crash-file diffstat metadata

### clean

Remove old snapshots, keeping only the most recent.

```bash
llmdebug clean -k 5                     # Keep only 5 most recent snapshots
llmdebug clean --all                    # Delete every snapshot, sidecar and latest.*
llmdebug clean --all --purge-state      # ...and the state database and RCA history
```

`clean` removes the deleted snapshots' rows from the snapshot index. It never
deletes `.llmdebug.lock`, which another process may hold. The state database
(`llmdebug_state.sqlite3`, holding the RCA history) and `rca_attempts.jsonl` are
kept unless you pass `--purge-state`.

### export

Generate a self-contained artifact for sharing, CI logs, or review.

```bash
llmdebug export                         # Export latest snapshot as HTML beside the snapshot file
llmdebug export --format json           # Machine-readable export payload
llmdebug export previous --compare latest
llmdebug export --output reports/failure.html
llmdebug export --mode redacted --redaction-profile prod
llmdebug export --bundle-dir reports/export-bundle
```

Single-file exports default to `--mode full` for backward compatibility.
Bundle exports default to `--mode redacted --redaction-profile ci` and emit:

- `report.<fmt>` primary artifact
- `compare.<fmt>` when `--compare` is used
- `manifest.json`
- `index.html`

Use `--bundle-dir <path>` for collaboration-safe artifacts and `--output <path>`
for a single file. These flags are mutually exclusive.

`--mode redacted` needs a profile that redacts something: `--redaction-profile dev` is
rejected as a usage error (exit code 2), and `apply_export_redaction(...)` raises
`ValueError` for `dev` or no profile, instead of labelling an unredacted artifact
"redacted".

## Detail Levels

The `show` command defaults to **crash** level to keep evidence compact.
Expand to `full` or `context` only when the extra state is needed:

| Level | Content | Typical Size |
|-------|---------|--------------|
| `crash` (default) | Exception + crash frame only | ~2K tokens |
| `full` | All frames + traceback | ~5K tokens |
| `context` | Everything (repro, git, env, coverage) | ~10K tokens |

## Snapshot References

All inspection commands share one reference syntax:

| Reference | Meaning |
|-----------|---------|
| `latest` or no argument | Most recent snapshot |
| `previous` | Second most recent snapshot |
| `#N` | Numbered entry from `llmdebug list` (1-based) |
| file path | Load a specific snapshot file directly |

All commands accept `--dir <path>` to point at a custom snapshot directory.

## Environment Variables

All `LLMDEBUG_*` variables are read at plugin load time. Boolean variables
accept `1`, `true`, `yes`, `on` (case-insensitive); everything else is
treated as false.

### Snapshot Capture

| Variable | Default | Description |
|---|---|---|
| `LLMDEBUG_DEBUG` | `false` | Emit diagnostic warnings to stderr |
| `LLMDEBUG_OUTPUT_FORMAT` | `json_compact` | Snapshot file format: `json_compact`, `json`, or `toon` |
| `LLMDEBUG_INCLUDE_MODULES` | *(empty)* | Comma-separated module prefixes to include in stack frames |
| `LLMDEBUG_MAX_EXCEPTION_DEPTH` | `5` | Maximum chained-exception depth (minimum 1) |
| `LLMDEBUG_INCLUDE_GIT` | `true` | Include git metadata in snapshots |
| `LLMDEBUG_GIT_DIRTY_MODE` | `always` | Git dirty-file strategy: `always`, `cached`, or `off` |
| `LLMDEBUG_INCLUDE_ARGS` | `true` | Separate function arguments from other captured locals |
| `LLMDEBUG_INCLUDE_COVERAGE` | `true` | Include coverage context when available |
| `LLMDEBUG_ARRAY_STATS` | `false` | Include numpy/tensor array statistics in locals |
| `LLMDEBUG_ASYNC_CONTEXT` | `true` | Capture async task context |
| `LLMDEBUG_CAPTURE_LOGS` | `false` | Capture recent log records in snapshots |
| `LLMDEBUG_LOG_MAX_RECORDS` | `20` | Maximum log records to capture (minimum 0) |
| `LLMDEBUG_CATEGORIZE_ERRORS` | `true` | Add error category and suggestion to exceptions |

### Pytest Context

| Variable | Default | Description |
|---|---|---|
| `LLMDEBUG_INCLUDE_FIXTURES` | `true` | Include fixture names in pytest context |
| `LLMDEBUG_INCLUDE_MARKERS` | `true` | Include marker names in pytest context |
| `LLMDEBUG_INCLUDE_PYTEST_META` | `true` | Include pytest metadata (nodeid, outcome, etc.) |

### Redaction

| Variable | Default | Description |
|---|---|---|
| `LLMDEBUG_REDACTION_PROFILE` | `ci` | Redaction profile preset: `dev`, `ci`, or `prod`. |
| `LLMDEBUG_REDACT_KEYS` | *(profile)* | Override: redact sensitive-looking variable names |
| `LLMDEBUG_REDACT_TRACEBACK` | *(profile)* | Override: redact file paths in tracebacks |
| `LLMDEBUG_REDACT_EXCEPTION_STRINGS` | *(profile)* | Override: redact exception message content |

### MCP Server

| Variable | Default | Description |
|---|---|---|
| `LLMDEBUG_MCP_RATE_LIMIT` | `10` | Sustained MCP calls per second |
| `LLMDEBUG_MCP_RATE_BURST` | `20` | Maximum burst size for MCP rate limiter |
| `LLMDEBUG_RCA_GATE_MODE` | `soft` | Default RCA gate mode: `off`, `soft`, or `strict` |
| `LLMDEBUG_RCA_BACKEND` | `sqlite` | RCA storage backend: `sqlite` or `jsonl` |
| `LLMDEBUG_RCA_MAX_RECORDS` | `5000` | Maximum retained RCA records (0 = unlimited) |
| `LLMDEBUG_DB_BUSY_TIMEOUT_MS` | `30000` | SQLite busy-wait timeout in milliseconds |

### Display

| Variable | Default | Description |
|---|---|---|
| `NO_COLOR` | *(unset)* | Disable Rich color output when set (any value) |

### Example: CI configuration

```bash
export LLMDEBUG_REDACTION_PROFILE=ci
export LLMDEBUG_CAPTURE_LOGS=true
export LLMDEBUG_GIT_DIRTY_MODE=cached
pytest
```

## Common Workflows

### First failure

```bash
pytest                                  # Failure creates snapshot
llmdebug                                # Read crash summary
llmdebug show --detail context          # Get full context if needed
```

### Compare before/after a fix attempt

```bash
pytest                                  # Second failure creates new snapshot
llmdebug diff                           # See what changed
```

### Deep frame inspection

```bash
llmdebug show --detail full             # See all frames
llmdebug frames -i 2                    # Zoom into a specific frame
llmdebug git-context                    # Check blame/diff for crash file
```
