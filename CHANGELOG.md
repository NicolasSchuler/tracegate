# CHANGELOG

## Unreleased (3.2.0)

### Features

- Add an offline `llmdebug quickstart` command that creates and displays a real synthetic snapshot.
- Add a System-1 decision-model advisor for the adaptive helper policy (`evals/decision_model/`): a SystemOne-compatible endpoint (e.g. Cloudflare clef-flash served by llama.cpp through llama-swap, or the bundled PyTorch sidecar) can run in `shadow`, `policy`, `features` or `prior` role via `--adaptive-decision-model-role`, with versioned question profiles, a redacted bounded decision state, policy-event schema v5, decision-model off-policy targets, and an `--adaptive-decision-model-api-key` (default `$TRACEGATE_DECISION_MODEL_API_KEY`) that `run_eval_docker.sh` passes as a flag and lockfiles redact.

### Bug Fixes

- SWE-Bench Lite/Pro generation: the `llmdebug` condition now fails fast without `--snapshot-dir` (it previously equaled `baseline`), and `run_metadata.json` records `prompt_policy` and per-run `snapshot_coverage` (DAT-03). Prompts no longer include `hints_text` or the FAIL_TO_PASS/PASS_TO_PASS test IDs by default; `--include-hints` / `--include-test-ids` restore them (DAT-01, DAT-02, D2).
- Stop sending dataset case descriptions and labels (`description`, `category`, `difficulty`, `snapshot_dependency`) to the external baselines; only the native conditions were blind to them (ATT-02, D1). Every baseline now also sees the test source, and each record carries `case_info_policy` and `case_information` (BAS-07). Importers no longer embed bug-type or fault text in `description`.
- Apply `SnapshotConfig` redaction-profile defaults on the config-first capture path, preserve explicit overrides, and provide intent-aware frozen-config updates.
- Compare both line and branch coverage in no-regression quality checks.
- Stop feeding oracle case labels (`case_difficulty_ord`, `case_snapshot_dependency_ord`, `case_bug_source_imported`) to the helper policy by default; `--adaptive-online-include-case-labels` restores the old behavior.
- Under `--adaptive-reveal-mode full`, the `snapshot_lite`, `frame_locals_lite` and `snapshot_full_lite` actions now replace the full snapshot section instead of duplicating evidence; `--adaptive-lite-snapshot-mode supplement` restores the old behavior.
- Log behavior propensity 1.0 for shadow-mode helper and budget decisions (the executed baseline is deterministic), fixing inflated off-policy weights.
- Label Thompson-sampling greedy helper decisions as `thompson_greedy` instead of `linucb_greedy`.

### Refactoring

- Remove a Stage-B reverse dependency and consolidate legacy eval metadata plumbing behind typed request objects.
- Move notebook/eval analytics libraries out of the core install and behind explicit extras.
- Move the helper-policy decision/event records to `evals/adaptive_policy_records.py` (re-exported from `evals.adaptive_online_policy`).

### Documentation

- Align architecture, coverage, testing, and first-run guidance with the executable repository contracts.

## v3.1.0 (2026-04-07)

### Features

- Add hybrid baseline submission helper.

### Bug Fixes

- Harden hybrid baseline submissions.
- Lower hybrid llama batch sizes.
- Stabilize baseline runtime and quality gates.

## v3.0.0 (2026-04-06)

### Features

- Expand the eval harness with renamed repair conditions, richer adaptive-policy tooling, baseline telemetry, failed-case extraction, and hybrid/SWE-Bench Pro workflows for the evaluation stack.
- Add snapshot export artifacts plus improved CLI display and progress reporting across local debug and eval flows.

### Bug Fixes

- Restore Python 3.10 compatibility and harden baseline diff, patch-seam, Docker, and CI handling across eval and adapter workflows.

### Refactoring

- Extract CLI display helpers and additional eval runtime/support modules to preserve cleaner seams as the harness grows.

### Documentation

- Add the master command runbook and broaden the maintained docs for baseline studies, telemetry, presets, and new eval workflows.

## v2.27.0 (2026-03-16)

### Features

- Add `llmdebug export` for shareable HTML/JSON artifacts, including redacted bundle outputs with manifests and RCA context.
- Add snapshot metadata search to `llmdebug list` and enrich dataframe summaries with dtype and null-count metadata.
- Expand eval Stage A context requests with `local_var` probes and track prompt-char/token usage throughout eval progress reporting.

### Documentation

- Document snapshot export workflows, searchable snapshot listing, and the richer eval progress telemetry.

## v2.26.0 (2026-03-12)

### Features

- Expand MCP tooling and strengthen repository quality checks.

### Bug Fixes

- Refactor snapshot cleanup to satisfy the repository quality gate.

### Refactoring

- Decompose `evals/run_eval.py` into focused helper modules.

### Documentation

- Polish the documentation suite, add broader inline clarification, and backfill placeholder changelog entries for older releases.

## v2.25.0 (2026-03-09)

### Features

- Add adaptive verifier repair loop.
- Add adaptive reward v2 eval guardrails.
- Harden adaptive helper reliability telemetry.

### Bug Fixes

- Restore CI quality gates.
- Prefer latest snapshot pointer for latest refs.
- Smooth llmdebug workflow inspection UX.
- Honor eval config openai key fallbacks.

## v2.24.0 (2026-03-05)

### Features

- Expand the adaptive eval policy workflow with shared policy feature scaling, Stage A contracts, and broader analysis/export support across the adaptive eval toolchain.

### Bug Fixes

- Install `just` in GitHub Actions for both the main CI workflow and the shared quality-contract action so the repository quality gates run successfully on GitHub-hosted runners.

### Documentation

- Polish the core project documentation, including contributor guidance, the MCP reference, and the docs site content.

## v2.23.0 (2026-03-04)

### Features

- Add adaptive budget online policy for adaptive eval conditions, including tiered propose-budget multipliers, shadow/on modes, per-attempt reward logging, state/event persistence, and summary/export analytics.

### Bug Fixes

- Return structured JSON error envelopes for MCP evidence tools when `response_format="json"` (with stable `error.code` + `metadata.status="error"` fields).
- Preserve requested `metadata.with_rca` in JSON error envelopes and normalize error `metadata.evidence_schema` to valid values.
- Reduce snapshot-capture latency by caching static git metadata in-process and adding configurable git dirty-check modes (`always`, `cached`, `off`).

### Notes

- Breaking: `evals.eval_summary` and `evals.analyze_results` now require eval result `schema_version=5`; older `evals/results/*.jsonl` artifacts are intentionally unsupported and must be regenerated before analysis.
- Breaking: `evals.export_policy_dataset` now requires adaptive policy event `schema_version=3`; older `evals/results/*.adaptive_policy.jsonl` artifacts are intentionally unsupported and must be regenerated before export.

## v2.22.0 (2026-03-04)

### Features

- Warn on early-terminated eval cases after DONE line.

## v2.21.0 (2026-03-03)

### Features

- Filter selected failed cases by max attempt.
- Default just workflows to docker runner.

## v2.20.0 (2026-03-03)

### Features

- Default fail-fast on request budget exhaustion.

## v2.19.2 (2026-03-02)

### Bug Fixes

- Improve runtime capture resilience and malformed payload handling.
- Harden snapshot diff and serialization against malformed data.
- Harden RCA input validation and retention.

## v2.19.1 (2026-03-02)

### Bug Fixes

- Harden eval resume repair and lock fallback.

### Testing

- Raise coverage gate to 85% and add edge-case suites.

## v2.19.0 (2026-03-01)

### Features

- Add strict eval summaries and two-pass uplift report.

### Performance

- Cache git metadata and add configurable dirty checks.

### Refactoring

- Migrate repository JSON usage to orjson adapters.

## v2.18.0 (2026-03-01)

### Features

- Add eval config files and harden MCP/json quality contracts.

### Refactoring

- Extract MCP helpers and decouple RCA engine.

## v2.17.2 (2026-02-27)

### Performance

- Cache eval context indexing and reduce IO overhead.

### Refactoring

- Modularize RCA state and harden MCP error envelopes.

## v2.17.1 (2026-02-27)

### Bug Fixes

- Retry transient OpenAI transport failures in eval runner.

## v2.17.0 (2026-02-27)

### Features

- Add adaptive online policy tooling and eval hardening.

## v2.16.0 (2026-02-23)

### Features

- Prompt-frozen quick wins and capture profiling.

## v2.15.0 (2026-02-22)

### Features

- Harden eval patcher and type MCP/eval payloads.

## v2.14.7 (2026-02-22)

### Bug Fixes

- Guard eval prompts against cosmetic UI drift.

## v2.14.6 (2026-02-21)

### Bug Fixes

- Tighten minimalist snapshot UI output in CLI.

## v2.14.5 (2026-02-20)

### Bug Fixes

- Harden patch fallbacks and switch qwen defaults in eval runner.

## v2.14.4 (2026-02-20)

### Bug Fixes

- Quick-win quality and safety remediations.

## v2.14.3 (2026-02-23)

### Refactoring

- Move eval/report analytics dependencies (`polars`, `scipy`, `scikit-learn`) to the `evals` extra while keeping dev installs ergonomic.

### Bug Fixes

- Improve snapshot-index fallback observability with explicit warnings when sqlite index reads fail and filesystem scan fallback is used.

### Performance

- Add stable capture-overhead benchmark workflow (`just bench-capture`, `just bench-capture-compare`) and prompt-freeze guardrails for eval prompt drift detection.

## v2.14.2 (2026-02-19)

### Bug Fixes

- Improve eval throughput reporting (parallel ETA) and default A100 flash-attn behavior.

### Documentation

- Document the batch-export comma-value gotcha in runbooks.
- Polish docs content for the academic/reporting workflow.

## v2.14.1 (2026-02-19)

### Bug Fixes

- Enforce source-first patch prompts and budget guardrails in eval flows.
- Track test-touch patch metrics and remove hypothesis hint surfaces from eval prompting.

## v2.14.0 (2026-02-19)

### Features

- Add SWE-Bench Lite hybrid pipeline support and smoke gates.

### Refactoring

- Sanitize batch defaults via local env overrides.

## v2.13.1 (2026-02-19)

### Bug Fixes

- Harden OpenAI transport reuse and runtime-dir handling.

### Notes

- Additional maintenance script updates.

## v2.13.0 (2026-02-19)

### Features

- Add top-3 importer pipeline and related tests.
- Add run report metrics tooling and long-context preset support.

### Bug Fixes

- Harden OpenAI-compatible patch parsing and fallback behavior.

## v2.12.0 (2026-02-18)

### Features

- Generalize the GGUF eval workflow and harden remote deployment scripts.

### Documentation

- Add remote eval runbook and batch quickstart documentation.

### Notes

- Additional eval script hardening and changelog maintenance updates.

## v2.11.4 (2026-02-18)

### Bug Fixes

- Add flush=True to eval progress prints for SSH/redirect visibility

## v2.11.3 (2026-02-18)

### Performance

- Reuse Docker containers via pool and show LLM/test timing in progress

## v2.11.2 (2026-02-18)

### Bug Fixes

- Suppress reportUnknownVariableType in jupyter_plugin.py

## v2.11.1 (2026-02-18)

### Bug Fixes

- Stabilize pyright strict across IPython stubs versions
- Restore defense-in-depth isinstance guards after pyright strict migration

### Refactoring

- Raise xenon gate to C and pyright to strict mode

### Evals

- Add retry-history telemetry and z.ai key fallback

## v2.11.0 (2026-02-17)

### Features

- Add resumable eval progress and harden sqlite WAL setup

## v2.10.1 (2026-02-17)

### Bug Fixes

- Implement quick wins for capture, privacy, and coverage gating

## v2.10.0 (2026-02-16)

### Features

- Add delta-first retry packaging and pairwise eval stats

## v2.9.0 (2026-02-16)

### Features

- Implement evidence-first MCP outputs and structured eval retry protocol

## v2.8.5 (2026-02-16)

### Performance

- Add sqlite state store and snapshot indexing

## v2.8.4 (2026-02-16)

### Bug Fixes

- Code-quality quick wins and medium refactors.

## v2.8.3 (2026-02-15)

### Bug Fixes

- Harden redaction and capture quick wins.
- Add missing changelog sections for recent tags.

## v2.8.2 (2026-02-14)

### Bug Fixes

- Make doc-sync compatible with Python 3.10.

### Documentation

- Add doc-sync utility and drift guard.

### Notes

- Add comprehensive uv/just quality pipeline.

## v2.8.1 (2026-02-14)

### Bug Fixes

- Harden eval importer validation and maintenance safety.

## v2.8.0 (2026-02-13)

### Features

- Add PyResBugs category mappings and update eval cases.

## v2.7.0 (2026-02-13)

### Features

- Import 1,267 ConDefects cases into eval suite.

## v2.6.0 (2026-02-11)

### Features

- Improve eval prompt template with structured snapshots and context budget.

### Refactoring

- Extract shared formatting helpers and harden edge cases.

### Documentation

- Fix 13 documentation discrepancies in CLAUDE.md and AGENTS.md.

## v2.5.1 (2026-02-10)

### Bug Fixes

- Resolve 6 code review findings across rendering, hooks, and importers.

### Refactoring

- Reduce complexity hotspots and enforce 80% diff-coverage gate.
- Set up comprehensive code quality tooling stack (bandit, vulture, deptry, radon, xenon).

## v2.5.0 (2026-02-10)

### Features

- Add enhanced git context for CLI and MCP.

## v2.4.0 (2026-02-10)

### Features

- Add configurable redaction profiles.

### Documentation

- Refresh project documentation.

## v2.3.0 (2026-02-10)

### Features

- Add RCA state-machine workflows and deterministic eval metadata.

## v2.2.0 (2026-02-10)

### Features

- Add Jupyter/IPython integration plugin and expand eval suite.

## v2.1.0 (2026-02-08)

### Features

- **Layered snapshot disclosure**: Read-time filter with three detail levels (`crash`, `full`, `context`) to control token usage. Wired into CLI (`--detail` flag), MCP server (`detail` parameter), and Python API (`filter_snapshot()`)
- **Hypothesis generation engine**: 10 pattern detectors that auto-generate ranked debugging hypotheses from snapshot data. New CLI command (`hypothesize`) and MCP tool (`llmdebug_hypothesize`)
- **Coverage-guided snapshot enrichment**: Extracts pytest-cov execution data (executed/missing lines, branch stats) for stack trace files into snapshots under `pytest.coverage`. Gracefully degrades when pytest-cov is absent

### Bug Fixes

- Eliminate silent failures across capture pipeline: replace bare `except Exception: pass` blocks with error sentinels, warnings, and visible error reporting across 14 modules
- Bootstrap CI off-by-one fix in stats module
- ASGI server IndexError guard in middleware
- Negative index validation in snapshot_diff
- CLI clean command now reports failed deletions
- Stale latest.json warning and temp file cleanup in output module
- Path traversal guard in snapshot loader
- Query string PII redaction in WSGI/ASGI middleware

## v2.0.1 (2026-02-08)

### Bug Fixes

- Eliminate silent failures across capture pipeline.

## v2.0.0 (2026-01-31)

### BREAKING CHANGES

- Default `output_format` changed from `"json"` to `"json_compact"`. Snapshots now use abbreviated keys by default for ~40% smaller files. Use `output_format="json"` or `LLMDEBUG_OUTPUT_FORMAT=json` to restore previous behavior.

### Features

- **Compact output formats**: `json_compact` (default, ~40% smaller) and `toon` (optional, ~50% smaller)
- **Git context capture**: commit hash, branch name, dirty status
- **Error categorization**: auto-classify exceptions with actionable suggestions
- **Async context capture**: asyncio task name and coroutine info
- **Function arguments**: captured separately from locals
- **Array statistics**: optional min/max/mean/std computation
- **Log capture**: capture recent log records for pre-crash context
- CLI for viewing and managing snapshots (`show`, `list`, `frames`, `clean`)
- MCP server for IDE integration (`llmdebug-mcp`) with 4 tools: `diagnose`, `show`, `list`, `frame`
- Snapshot diffing via CLI (`diff` command) and MCP (`llmdebug_diff`)
- Production exception hooks (`install_hooks` / `uninstall_hooks`) with rate limiting and PII redaction
- WSGI and ASGI middleware for web framework integration

## v1.0.1 (2026-01-28)

### Bug Fixes

- Set `major_on_zero=false` for proper semver during 0.x development

## v1.0.0 (2026-01-28)

### Features

- `max_snapshots` config for automatic storage cleanup (0 = unlimited)
- CI/CD pipeline with GitHub Actions

## v0.1.4 (2026-01-27)

### Features

- ML tensor introspection: NaN/Inf detection, device tracking, requires_grad
- Pytest-specific context capture: test nodeid, outcome, stdout/stderr
- Project logo

### Bug Fixes

- Deprecated datetime.utcnow() usage
- Handle None line numbers in frame extraction

## v0.1.0 (2026-01-27)

### Features

- Initial release
- Exception capture with structured JSON snapshots
- Pytest plugin for automatic capture on test failures
- Smart array summarization for NumPy arrays
- Atomic file writes for snapshot output
