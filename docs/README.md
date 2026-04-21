# Docs Index

Start here when you want the maintained documentation surface rather than the
full `docs/` tree.

## Canonical Docs

### Package Usage

| Document | Purpose |
|----------|---------|
| [README](../README.md) | Product overview, installation, quick start, and main docs router |
| [Configuration Reference](configuration.md) | Capture entry points, environment variables, output formats, and API surface |
| [CLI Reference](cli-reference.md) | Command reference for `show`, `list`, `frames`, `diff`, `export`, and `git-context` |
| [MCP Reference](mcp-reference.md) | MCP tool contracts, payload schemas, and error handling |
| [Troubleshooting](troubleshooting.md) | Common runtime, plugin, locking, environment, release, and quality gate recovery steps |

### Package Contribution

| Document | Purpose |
|----------|---------|
| [Contributing](../CONTRIBUTING.md) | Local setup, quality gates, and PR expectations |
| [Testing Guide](testing-guide.md) | Test organization, workflows, fixtures, and CI guidance |
| [Quality Map](quality-map.md) | Which checks are blocking for package code vs staged for eval code |
| [Architecture](architecture.md) | Capture pipeline, module boundaries, and design decisions |
| [Baseline Adapters](baseline-adapters.md) | Adapter architecture, contract semantics, per-baseline mechanisms, and failure modes |

### Evals and Local/Docker Ops

| Document | Purpose |
|----------|---------|
| [Master Runbook](runbook.md) | **All commands in one place** — setup, quality, debug CLI, eval ops, SWE-Bench, analysis, maintenance |
| [Eval Quickstart](eval_quickstart.md) | Fastest copy-paste path for eval container builds, local model serving, and baseline/native runs |
| [Eval Framework](../evals/README.md) | Canonical starting point for running and interpreting eval workflows |
| [Eval Configs](../evals/configs/README.md) | Ready-made TOML templates and CLI/config mapping |
| [Baseline Hybrid Study Runbook](baseline_hybrid_study_runbook.md) | Exact 15-run matrix for raw, hybrid-canonical, and hybrid-tuned baseline study arms |
| [Local/Docker Eval Runbook](local_docker_eval_runbook.md) | Local model endpoints, Docker wrapper runs, and SWE-Bench official scoring |
| [Adaptive Policy Reference](ADAPTIVE_POLICY_QUICK_REFERENCE.md) | LinUCB bandit parameters, action space, reward formula, and practical tuning guide |

## Research Notes

The top-level `docs/` directory may also contain dated research notes,
experiment plans, findings writeups, and roadmap documents. Those files are
useful for historical context, but they are not the primary product or
operations docs.
