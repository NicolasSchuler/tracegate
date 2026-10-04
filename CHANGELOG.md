# CHANGELOG

## Unreleased (3.2.0)

### Features

- Add an offline `llmdebug quickstart` command that creates and displays a real synthetic snapshot.
- Add a System-1 decision-model advisor for the adaptive helper policy (`evals/decision_model/`): a SystemOne-compatible endpoint (e.g. Cloudflare clef-flash served by llama.cpp through llama-swap, or the bundled PyTorch sidecar) can run in `shadow`, `policy`, `features` or `prior` role via `--adaptive-decision-model-role`, with versioned question profiles, a redacted bounded decision state, policy-event schema v5, decision-model off-policy targets, and an `--adaptive-decision-model-api-key` (default `$TRACEGATE_DECISION_MODEL_API_KEY`) that `run_eval_docker.sh` passes as a flag and lockfiles redact.

### Bug Fixes

- The importer determinism filter is an AST check (`evals/importers/determinism.py`): wall-clock calls, `datetime.now()`, unseeded `random`/`numpy.random` use, `uuid`, `os.urandom`, `secrets`, network imports and iteration over sets now reject a case; six substrings used to let all of these through (DAT-14). `python -m evals.importers.determinism` scans an existing corpus and flags 14 of the 1,828 committed cases.
- SWE-Bench Lite and Pro generation record each proposal's `finish_reason`, truncation, patcher error and request error in `<condition>/generation_meta.jsonl` (the official predictions file keeps its three keys), and `--openai-max-tokens` defaults to 4096 instead of 2048, which cut multi-file patches off without a trace (DAT-18).
- The `heuristic` smoke-test patcher's diffs now apply: its private diff builder put a blank line after every body line, so the patch was always rejected as malformed while the patcher reported `reason: patched` (LLM-08). It now uses the shared diff builder.
- A run whose OpenAI context window (manual or probed) leaves fewer than 1,024 prompt tokens after `--openai-max-tokens` and the 1,024-token reserve now fails at startup with `OpenAIPromptBudgetError` instead of capping every prompt at one character, which made every case look like a model failure, for example a llama.cpp server started with `-c 8192` and `--openai-max-tokens 8192` (LLM-06).
- OpenAI edits and full-file answers are applied faithfully (LLM-05, LLM-09, LLM-15, LLM-19): an empty `find` is rejected (`empty_find`) instead of inserting the replacement between every character, `count` below 1 means one replacement instead of all (the schema now requires `minLength: 1` and `minimum: 1`); CRLF files are matched as LF and keep CRLF in the generated diff; several edits to one file are chained into a single diff; diffs are built and parsed by splitting on `\n` only, so form feeds, `\x1c`-`\x1e`, `\x85` and U+2028 inside a line survive, and a missing final newline gets the `\ No newline at end of file` marker (the patcher and runner now share one builder); every edits payload-shape failure (`bad_edit_entry`, `bad_path`, `bad_find_replace`, `empty_find`, `path_outside_case`, `file_not_found`, `path_not_file`) takes the diff fallback like `find_not_found` already did; and a non-JSON chat-completion body (`bad_json_response`) is retried like a transport error.
- A completion that stops at `max_tokens` (`finish_reason="length"`) is a terminal `truncated_response` on every OpenAI parse path (diff, edits, JSON generation, Stage-A requests, compaction) and gets its own attempt note, instead of `_fix_hunk_counts` turning a cut-off hunk into a valid patch that drops lines or the attempt being recorded as `no_json_payload`/`no_patch_proposed` (LLM-02). Specific patcher failure reasons such as `file_not_found` or `no_json_payload` are no longer overwritten by an earlier stage's transport error, which is kept as `transport_error` (LLM-14).
- Model diffs are extracted completely (LLM-01, LLM-03, LLM-04, LLM-07, LLM-09, LLM-10): every fenced block and every file section is kept, not just the first (a later diff for the same file replaces an earlier draft); reasoning is cut at the last `</think>` and at an unclosed `<think>`, so drafts no longer leak in; hunks are read to their declared line counts, so trailing markdown bullets aren't absorbed; blank lines inside a hunk count as context; `--- /dev/null` headers create files (`-0,0` starts are kept); and CRLF files are patched as LF and keep their line endings. `tests/test_evals_diff_corpus.py` runs realistic responses through extraction, hunk-count repair and `apply_diff`.
- Records describe the prompt that produced the patch: every sent propose prompt's hash is kept in `propose_prompt_xxh3_128s`, the last is `final_prompt_xxh3_128`, and `--record-prompt full` writes `prompt_final.txt` (ATT-11). A transport-recovery replay deletes only what the failed call created instead of the whole attempt directory, so `prompt.txt`, snapshots, plan files and earlier patch-cycle diffs that records point to survive (ATT-12).
- Every LLM call is counted (POL-02, ATT-09, ATT-13, ATT-15): TEP critic/update calls are added to the attempt's tokens and call count before the reward is computed, TEP-equilibrium re-plans accumulate usage across rounds, compaction tokens are no longer taken from the propose call's usage when compaction returned no meta, and a transport-recovery replay keeps the failed first call's usage (`transport_recovery_first_call_*`).
- The adaptive `retry_gate_no_new_evidence` gate blocks at most once in a row, and a block no longer consumes a scored attempt: it is recorded as a non-scored `infra_event`, the attempt is retried with `retry_gate_reason="block_cap_reached"`, and the next record carries `retry_gate_blocks_before` (ATT-05, D9). Previously an adaptive case could get one real repair attempt where `base_repair` got k.
- A unit that raises now writes a terminal `note="unit_runtime_error"` failure record (resume still retries it), so it stays in its condition's denominator (ATT-08); records held back for credit resolution are flushed when a unit raises (ORC-10); and usage a failed unit already streamed counts in the run totals (ORC-17).
- Baseline OpenAI-compatible plumbing (BAS-16, BAS-17, BAS-19): LDB selects an upstream client only for genuine GPT-4/GPT-3.5 names (others route through the GPT path with the real model id) and records `ldb_model_routing`; the OpenRouter `extra_body={"usage": {"include": true}}` is sent only to OpenRouter via one shared helper; and the baseline thinking/reasoning-fallback defaults come from one pair of constants that match the CLI.
- RLM no longer rolls back a candidate whose import smoke test fails or exceeds 10 s; the failure is reported to the model and counted as `import_smoke_failure_count` (BAS-13).
- Baseline setup fails when an upstream checkout is not at its pinned commit (LDB, INTERVENOR and TraceCoder used to report the pin without checking; BAS-05), and every record carries `baseline_repo_state` (`observed_commit`, `dirty`, `tree_patch_sha256` of the applied patches; BAS-14).
- Baseline failures keep their progress (BAS-10, BAS-15): a timeout or failed model call in INTERVENOR, lambda-RLM or RLM returns the last candidate and accumulated usage (`error` `timeout`, `env_client_failed` or `generation_failed`) instead of discarding both, and LDB reports metrics and usage for attempts that found no repair.
- Baselines get retry context (BAS-02, D4): on harness attempt 2+ the failing pytest output is appended to the adapter's task prompt (INTERVENOR, TraceCoder, lambda-RLM, RLM); records carry `baseline_retry_context_consumed` (false for LDB, whose prompt has no free-text field).
- Baselines share the native generation contract (D4): `BaselineConfig.generation` carries `--openai-temperature`, `--seed`, `--openai-max-tokens` and transport retries to every adapter; INTERVENOR, lambda-RLM and RLM enforce it, LDB and TraceCoder receive `TRACEGATE_BASELINE_*` variables and are recorded as not enforced (BAS-03, BAS-04, BAS-11). Records carry `baseline_generation`, the run_repair wall clock and deadline status; INTERVENOR now honours the deadline (BAS-08), and TraceCoder's missing-context test timeout falls back to the process timeout instead of 10 s (BAS-12). `build_baseline_patcher(timeout_sec=...)`, a deprecated alias, is replaced by `generation=`.
- `run_monitored_subprocess` (local pytest runs, LDB, TraceCoder) no longer hangs when a grandchild outlives a timeout: children run in their own session, the whole group is killed on timeout, pipes are drained for at most `DRAIN_GRACE_SEC` after exit, stdin is `/dev/null`, and `timeout_sec=0` means no deadline instead of a busy loop (ORC-03, ORC-18, ORC-19).
- Baselines are scored on the cases they can represent (BAS-01, D3): rows whose patch metadata says `baseline_supported=False` (LDB converts 11 of 1,828 cases) are excluded from that condition's success rate and from paired tests, the summary reports `baseline_unsupported_cases`, and `run_metadata.baseline_case_support` records the convertible count and incompatibility reasons.
- Adaptive policy independence and reproducibility (POL-05, POL-23): with both adaptive conditions in a run, each gets its own budget-policy controller (state and event files `<stem>.<condition><suffix>`, listed in `patcher_config.adaptive_budget_controllers`) instead of one shared model; every controller seed is derived from `--seed` and its role/condition; and decisions draw from a per-unit stream keyed by case, condition and attempt, so exploration no longer depends on thread interleaving. Deterministic runs with `--jobs > 1` warn that learning order still depends on scheduling.
- `evals.run_eval` rejects abbreviated flags, which config-file values used to override silently (ORC-06), duplicate or empty `--conditions` (ORC-15), and `--run-id` values outside `[A-Za-z0-9._-]{1,128}` or made only of dots (ORC-21).
- Resuming a run whose settings fingerprint changed now aborts unless `--allow-settings-drift` is given, in which case new records carry their own run metadata (ORC-04). `--seed` is always recorded and part of the fingerprint, since it seeds bandit exploration even without `--deterministic` (ORC-05).
- `--deterministic --seed N` now reaches the pytest subprocesses (`PYTHONHASHSEED`, `TZ=UTC`) and the container pool, when requested, is actually used: both were assigned through a forwarding shim and never reached `run_eval` (ORC-01, ORC-02). `run_eval.set_active_*` setters replace the shim writes, and an architecture test bans them. Pooling is now opt-in (`--container-pool`) because it shares the workdirs mount across units; the default stays one fresh container per pytest call, which is what every earlier run actually used.
- Adaptive evidence enforcement (ATT-03, ATT-04, ATT-06, ATT-20, ATT-21, POL-12, D5): restricted helper actions no longer receive the whole-file fallback or seed bundle, the full snapshot reaches Stage-B under `--adaptive-reveal-mode full` only when the executed action grants it, a Stage-A request budget of 0 executes no request, fallback paths are validated against the workdir, and no scaled feature vector is logged without a decision under percentile scaling. Records gain `seed_bundle_chars`, `evidence_chars_by_type`, `adaptive_snapshot_requested` and `stage_b_snapshot_in_prompt`; `adaptive_snapshot_included` now says whether the snapshot was in the prompt.
- SWE-Bench Lite/Pro generation no longer saves failed requests as empty patches; the instance stays missing (paired validation blocks scoring) and `--resume` retries it (DAT-08).
- SWE-Bench Pro scoring: empty `fail_to_pass` lists, unexplained container failures, missing `output.json` and wait timeouts are infrastructure errors instead of wrong patches, with per-instance reasons persisted in `eval_results_detail.json` (DAT-04, DAT-09); patches are applied with the official fallback sequence and a non-applying patch is a model failure (DAT-19); cached outputs are reused only for the same patch (DAT-13); dataset values are shell-quoted in the entry script (DAT-11).
- Record `patch_meta.patch_apply_mode` (`strict`, `relaxed_fallback`, `runner_marker`) and report `relaxed_apply_attempts` and `test_surface_modified_attempts` per case and condition (ATT-10). Failed or fuzzy `patch` runs no longer leave `.rej`/`.orig` files in the workdir (ATT-16).
- `--force-patch-attempt` now defaults to off. When enabled, a runner marker patch applied after the model's diff failed keeps the attempt `patch_apply_failed` (with `runner_forced_patch_recovery*` modifiers), the patch metadata describes the applied marker diff (`model_patch_*` keeps the model's), and the marker is reverted so it can't leak into later prompts (ATT-07, D8). Runner and patcher forced-patch markers share one constant, so patcher-generated marker-only patches are recognized as no-ops (LLM-13).
- `--patch-verifier-scope` now defaults to `all`, so the verifier (rejections, recovery calls, test-edit penalties) applies to every condition; `adaptive_only` remains available as an ablation (ATT-14, D7).
- Eval success can no longer be produced by editing the test oracle: the harness restores the pristine test surface before each verification run and requires a JUnit report with at least one executed test and no failures or errors. Records carry `test_surface_modified` and `pytest_verdict_*` (ATT-01, D10).
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
