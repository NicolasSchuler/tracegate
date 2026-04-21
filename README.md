<p align="center">
  <img src="logo/bird.png" alt="TraceGate logo" width="200">
</p>

<h1 align="center">TraceGate</h1>

<p align="center">Structured debug snapshots for LLM-assisted debugging.</p>

> **Important analysis data note**
>
> `analysis/analysis.ipynb` requires the large result files under `analysis/data/`.
> They are not stored in this GitHub repository. Download the compressed data
> archive from Zenodo DOI
> [`10.5281/zenodo.19678467`](https://doi.org/10.5281/zenodo.19678467), then
> unzip it from the `analysis/` directory so the archive's `data/` folder is
> restored at `analysis/data/`.
>
> The Zenodo dataset is not public yet, so fresh clones cannot reproduce the
> notebook until the record is published or the archive is shared separately.

---

TraceGate captures failure-time evidence — exception details, prioritized stack
frames, local variables, and execution context — as a machine-readable artifact
that works for both humans and coding agents. The goal is to make the **first
failing run** useful, rather than reconstructing state after the fact.

This artifact keeps the existing `llmdebug` Python package, CLI, import paths,
and environment-variable names for compatibility.

## The Debugging Loop

Without structured evidence, a typical loop looks like:

```
fail → infer missing state → guess patch → rerun → repeat
```

With TraceGate, the loop becomes:

```
fail → read snapshot → ranked hypotheses → minimal patch → verify
```

## Installation

```bash
pip install 'llmdebug[cli]'   # Recommended: pytest plugin + CLI
pip install llmdebug           # Core library + pytest plugin only
pip install 'llmdebug[mcp]'    # MCP server for IDE integration
```

Other extras: `jupyter`, `toon`, `evals` — see [Configuration Reference](docs/configuration.md#installation-extras).

## Start Here

- **Use the package**: start with the quick start below, then the [Configuration Reference](docs/configuration.md), [CLI Reference](docs/cli-reference.md), and [MCP Reference](docs/mcp-reference.md).
- **Contribute to the package**: read [Contributing](CONTRIBUTING.md), [Testing Guide](docs/testing-guide.md), and [Quality Map](docs/quality-map.md).
- **Run evals locally or in Docker**: start with the [Eval Quickstart](docs/eval_quickstart.md), then [Eval Framework](evals/README.md), [Eval Configs](evals/configs/README.md), and the [Local/Docker Eval Runbook](docs/local_docker_eval_runbook.md).
- **Find any command fast**: [Master Runbook](docs/runbook.md) — setup, quality, debug CLI, eval ops, analysis, and maintenance all in one place.
- **Browse canonical docs first**: [Docs Index](docs/README.md) separates active product docs from dated research notes and analyses.

## Reproduce

The eval workflow assumes Docker, `uv`, and a local OpenAI-compatible model
endpoint.

```bash
git clone <repository-url>
cd llmdebug
uv sync --group dev

# Clone and provision external baseline repos/environments.
bash baselines/setup_repos.sh
bash baselines/setup_envs.sh

# Build the eval Docker image with a one-case smoke run.
LLMDEBUG_EVAL_BUILD=1 bash evals/docker/run_eval_docker.sh \
  --patcher null \
  --max-cases 1 \
  --k 1
```

Start the LLM server separately with any OpenAI-compatible runtime (for example
`llama.cpp`, vLLM, or a hosted-compatible local endpoint), then verify:

```bash
curl http://127.0.0.1:12000/v1/models
```

Run a default external baseline config:

```bash
BASELINE=intervenor
CONFIG=evals/configs/baseline_intervenor.template.toml
RUN_ID=<baseline-run-id>
MODEL=<model-name>

bash evals/docker/run_eval_docker.sh \
  --config "$CONFIG" \
  --baseline "$BASELINE" \
  --run-id "$RUN_ID" \
  --jobs 4 \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model "$MODEL"
```

For the other default baseline configs, use the matching template under
`evals/configs/`: `baseline_ldb.template.toml`,
`baseline_tracecoder.template.toml`, `baseline_lambda_rlm.template.toml`, or
`baseline_rlm.template.toml`.

Create a failed-case list from a completed run:

```bash
uv run python -m evals.select_cases \
  --results-jsonl "evals/results/${RUN_ID}.jsonl" \
  --condition "$BASELINE" \
  --mode failed \
  --out "evals/results/${RUN_ID}.failed.txt"
```

Run the adaptive condition on those failed cases:

```bash
ADAPTIVE_RUN_ID=<adaptive-run-id>

bash evals/docker/run_eval_docker.sh \
  --config "$CONFIG" \
  --baseline "$BASELINE" \
  --run-id "$ADAPTIVE_RUN_ID" \
  --case-list-file "evals/results/${RUN_ID}.failed.txt" \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model "$MODEL" \
  --conditions layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --adaptive-budget-topup-max-calls 1 \
  --adaptive-online-policy-mode on \
  --adaptive-budget-policy-mode on \
  --jobs 1
```

Run the fixed-evidence package config:

```bash
bash evals/docker/run_eval_docker.sh \
  --config evals/configs/study_intervenor_fixed_evidence_v1.template.toml \
  --run-id <fixed-evidence-run-id> \
  --case-list-file "evals/results/${RUN_ID}.failed.txt" \
  --openai-base-url http://127.0.0.1:12000/v1 \
  --openai-model "$MODEL"
```

## Quick Start

### Pytest (automatic)

Failing tests automatically create `.llmdebug/latest.json`:

```bash
pytest                                  # Failures create .llmdebug/latest.json
llmdebug                                # View crash summary
llmdebug show --detail context          # Full context (git, env, repro command)
llmdebug diff                           # Compare latest vs previous
```

### Decorator

```python
from llmdebug import SnapshotConfig, debug_snapshot

cfg = SnapshotConfig(out_dir=".llmdebug", redaction_profile="ci")

@debug_snapshot(config=cfg)
def main():
    data = load_data()
    process(data)
```

### Context Manager

```python
from llmdebug import SnapshotConfig, snapshot_section

cfg = SnapshotConfig(out_dir=".llmdebug", redaction_profile="prod")

with snapshot_section("data_processing", config=cfg):
    result = transform(data)
```

More entry points (production hooks, web middleware, Jupyter) in the [Configuration Reference](docs/configuration.md#capture-entry-points).

## Features

- **Automatic capture** — pytest plugin, decorator, context manager, production hooks, web middleware
- **Rich snapshots** — exception chain, prioritized frames, typed locals (array shapes, dtypes), source context
- **Layered detail** — `crash` (~2K tokens) → `full` (~5K) → `context` (~10K) disclosure levels
- **CLI inspection** — `show`, `list`, `frames`, `diff`, `git-context`, `clean`
- **Shareable exports** — `llmdebug export` generates single-file artifacts or redacted bundle reports for CI/team sharing
- **MCP server** — 9 tools for Claude Code, Cursor, and other MCP-capable editors
- **Hypothesis engine** — auto-generated ranked debugging hypotheses from snapshot patterns
- **Privacy controls** — PII redaction profiles (`dev` / `ci` / `prod`), pattern-based redaction
- **Jupyter integration** — cell-error banners + `%llmdebug` magic commands
- **Compact formats** — `json_compact` (~40% smaller) and `toon` (~50% smaller) for LLM context

## CLI

```bash
llmdebug                                # Latest snapshot (crash detail)
llmdebug show --detail full             # All frames
llmdebug show --detail context          # Everything (git, env, repro)
llmdebug show --json --detail context   # JSON output
llmdebug list                           # List snapshots
llmdebug list --search "KeyError"       # Search indexed snapshot metadata
llmdebug diff                           # Compare latest vs previous
llmdebug export latest --format html    # Shareable HTML artifact
llmdebug export --bundle-dir reports    # Redacted CI-safe bundle with manifest + index
llmdebug clean -k 5                     # Keep 5 most recent
```

| Level | Content | ~Tokens |
|-------|---------|---------|
| `crash` (default) | Exception + crash frame | ~2K |
| `full` | All frames + traceback | ~5K |
| `context` | Everything (repro, git, env, coverage) | ~10K |

Full reference: [docs/cli-reference.md](docs/cli-reference.md)

## MCP Server

```bash
pip install 'llmdebug[mcp]'
llmdebug-mcp  # Start the MCP server (stdio transport)
```

| Tool | Description |
|------|-------------|
| `llmdebug_diagnose` | Concise crash summary optimized for LLM consumption |
| `llmdebug_show` | Full expanded JSON snapshot with detail level control |
| `llmdebug_list` | List available snapshots with metadata |
| `llmdebug_frame` | Detailed view of a specific stack frame |
| `llmdebug_git_context` | On-demand enhanced git metadata for crash triage |
| `llmdebug_diff` | Compare two snapshots to show what changed |
| `llmdebug_rca_status` | Show latest RCA state for a session |
| `llmdebug_rca_history` | Show RCA attempt history |
| `llmdebug_rca_advance` | Manually advance RCA state machine |

Claude Code configuration (`.mcp.json`):

```json
{
  "mcpServers": {
    "llmdebug": {
      "command": "llmdebug-mcp"
    }
  }
}
```

Full reference: [docs/mcp-reference.md](docs/mcp-reference.md)

## Output

On failure, `.llmdebug/latest.json` stores a versioned `DebugSession` envelope:

```json
{
  "schema_version": "2.0",
  "kind": "llmdebug.debug_session",
  "session": {
    "name": "test_training_step",
    "timestamp_utc": "2026-01-27T14:30:52Z"
  },
  "snapshot": {
    "exception": {
      "type": "ValueError",
      "message": "operands could not be broadcast together..."
    },
    "frames": [
      {
        "file": "training.py",
        "line": 42,
        "function": "train_step",
        "code": "output = model(x) + residual",
        "locals": {
          "x": {"shape": [32, 64], "dtype": "float32"},
          "residual": {"shape": [32, 128], "dtype": "float32"}
        }
      }
    ]
  }
}
```

## Configuration

Prefer `config=SnapshotConfig(...)` for decorator/context-manager capture
settings. Legacy flat kwargs still work temporarily but emit a
`DeprecationWarning`. Pytest plugin settings can still be driven by environment
variables:

```bash
LLMDEBUG_OUTPUT_FORMAT=json pytest          # Use pretty JSON
LLMDEBUG_REDACTION_PROFILE=ci pytest        # Use CI redaction profile
LLMDEBUG_INCLUDE_GIT=false pytest           # Disable git context
```

Full parameter reference, output formats, and redaction profiles:
[docs/configuration.md](docs/configuration.md)

## Documentation

| Document | Description |
|----------|-------------|
| [Docs Index](docs/README.md) | Canonical docs map, grouped by reference, guides, ops, and research notes |
| [Configuration Reference](docs/configuration.md) | Parameters, env vars, output formats, API surface |
| [CLI Reference](docs/cli-reference.md) | Full CLI command reference |
| [MCP Reference](docs/mcp-reference.md) | MCP server JSON schemas and parameters |
| [Architecture](docs/architecture.md) | System layers, capture pipeline, and design decisions |
| [Testing Guide](docs/testing-guide.md) | Test workflows, snapshot-driven debugging, and integration guidance |
| [Troubleshooting](docs/troubleshooting.md) | Common setup/runtime failures and recovery steps |
| [Eval Quickstart](docs/eval_quickstart.md) | Fastest copy-paste path for eval container builds, local model serving, and baseline/native runs |
| [Eval Framework](evals/README.md) | Canonical starting point for running and interpreting evals |
| [Eval Configs](evals/configs/README.md) | Ready-made eval templates and TOML config mapping |
| [Local/Docker Eval Runbook](docs/local_docker_eval_runbook.md) | Canonical eval-ops guide for local model endpoints, Docker wrapper runs, and SWE-Bench scoring |
| [Contributing](CONTRIBUTING.md) | Development setup and quality gates |
| [Quality Map](docs/quality-map.md) | Which checks block package changes vs staged eval work |

Dated research notes, experiment logs, and roadmap documents remain under
`docs/`, but they are supporting material rather than the primary starting
point for package usage or eval operations.

## License

MIT

## Authors

Anonymous for double-blind review.
