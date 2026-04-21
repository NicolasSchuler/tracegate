# Repository Guidelines

## Agent Workflow
- Start with the shipped-behavior docs before editing: `README.md` for product surface, `CONTRIBUTING.md` for the local quality contract, and `docs/README.md` for the canonical docs map.
- Then read the area-specific maintained doc for the subsystem you are touching: `docs/configuration.md`, `docs/cli-reference.md`, `docs/mcp-reference.md`, `docs/testing-guide.md`, `docs/quality-map.md`, `docs/troubleshooting.md`, `docs/baseline-adapters.md`, `docs/eval_quickstart.md`, `evals/README.md`, `evals/configs/README.md`, or `docs/local_docker_eval_runbook.md`.
- Keep repository exploration targeted. `evals/cases/` is large, and `evals/artifacts/`, `evals/results/`, `evals/swebench_runs/`, `evals/swebench_pro_runs/`, `evals/swebench_pro_assets/`, `.llmdebug/`, and `.benchmarks/` are generated trees. Prefer focused `rg`/file reads over bulk scans.
- When behavior, CLI/API usage, runbooks, or generated documentation blocks change, update the corresponding docs in the same patch.

## Debug Snapshot Workflow
- On any failing test or command, inspect `.llmdebug/latest.json` first, or run `llmdebug show --json --detail context`.
- Read the crash frame, exception message, locals, array shapes, and captured git/env context before editing.
- Form 2-4 ranked hypotheses from the evidence, then apply the smallest fix that matches the best-supported hypothesis.
- Re-run after each fix; avoid speculative multi-change patches.
- If the snapshot shows the symptom but not the cause, add targeted instrumentation such as `snapshot_section("stage_x")`, regenerate the snapshot, and continue the hypothesis loop.
- Useful CLI helpers:
  - `llmdebug show --detail full`
  - `llmdebug show --detail context`
  - `llmdebug show previous`
  - `llmdebug diff`

## Project Structure & Architecture
- `src/llmdebug/`: package code.
  - Capture, schema, and presentation: `capture.py`, `config.py`, `serialize.py`, `output.py`, `compact_keys.py`, `detail_filter.py`, `error_categories.py`, `hypotheses.py`, `_debug_session.py`, `_snapshot_loader.py`, `_formatting.py`, `_presentation.py`, `_json.py`, `_coerce.py`, `_hashing.py`, `snapshot_types.py`, `snapshot_diff.py`, `export_artifact.py`, `toon_encoder.py`.
  - RCA and persistence: `rca_state.py`, `rca_transitions.py`, `rca_engine.py`, `rca_backend.py`, `rca_identity.py`, `rca_types.py`, `gates.py`, `state_db.py`, `rca_store.py`.
  - Interfaces and contracts: `cli.py`, `mcp_server.py`, `mcp_snapshot_tools.py`, `mcp_diff_tools.py`, `mcp_git_context.py`, `mcp_rca.py`, `mcp_contract.py`, `mcp_types.py`, `pytest_plugin.py`, `jupyter_plugin.py`, `middleware.py`.
  - Runtime integrations and support: `hooks.py`, `redaction_policy.py`, `git_context.py`, `coverage_context.py`, `async_context.py`, `log_capture.py`, `_rate_limiter.py`, `_warnings.py`.
- `evals/`: benchmark and analysis harness.
  - Runners and summaries: `run_eval.py`, `eval_main_flow.py`, `eval_main_setup.py`, `eval_main_resume.py`, `eval_attempt_flow.py`, `eval_attempt_support.py`, `analyze_results.py`, `eval_summary.py`, `eval_two_pass_summary.py`, `report_run_metrics.py`, `select_cases.py`, `validate_cases.py`.
  - Adaptive controller, policy, and contracts: `adaptive_online_policy.py`, `adaptive_budget_online_policy.py`, `adaptive_controller_packet.py`, `adaptive_controller_runtime.py`, `adaptive_history.py`, `bootstrap_policy_state.py`, `controller_summarizer.py`, `eval_policy_budget.py`, `eval_policy_features.py`, `eval_policy_online.py`, `policy_feature_scaling.py`, `stage_a_contract.py`.
  - Runtime/evidence/patching support: `eval_config.py`, `eval_evidence.py`, `eval_patches.py`, `eval_patch_constants.py`, `attempt_metadata.py`, `attempt_verification.py`, `run_condition_builder.py`, `runtime_resolution.py`, `lambda_rlm_backend.py`, `token_counting.py`, `native_openai_attempt_flow.py`, `native_stage_b_flow.py`.
  - Support packages: `patchers/`, `importers/`, `swebench/`, `swebench_pro/`, `configs/`, `docker/`, `cases/`.
- `tests/`: pytest suite for package behavior, RCA flows, MCP tools, eval harnesses, quality-contract checks, and SWE-Bench helpers.
- `docs/`: user docs, protocol notes, local/Docker runbooks, and research notes.
- `baselines/`: external baseline adapters, runner contracts, and environment/bootstrap helpers.
- `scripts/`: helper utilities such as benchmark comparison and official SWE-Bench scoring wrappers.
- `quality/`: coverage baselines and other quality-policy assets.
- `Dockerfile.llama` and `llama-entrypoint.sh` provide the GGUF-based llama.cpp eval backend container. See `docs/local_docker_eval_runbook.md` for the maintained eval-ops guide.
- Generated artifacts should not be committed: `.llmdebug/`, `.benchmarks/`, `evals/artifacts/`, `evals/results/*.jsonl`, `evals/swebench_runs/`, `evals/swebench_pro_runs/`, `evals/swebench_pro_assets/`, `reports/`, `dist/`, `coverage.xml`, `coverage.json`, `htmlcov/`.

### Architecture guardrails
- Keep entrypoint modules (`cli.py`, `mcp_server.py`, `pytest_plugin.py`, `jupyter_plugin.py`, `middleware.py`) thin.
- Keep reusable logic in composable core modules; avoid reverse imports from interface layers into core.
- Preserve the RCA split: pure/domain logic in `rca_engine.py` and `rca_transitions.py`, persistence in `state_db.py` and `rca_store.py`, adapters at the boundaries.
- Prefer clear seams around capture, serialization, snapshot presentation, RCA state transitions, and eval patcher/importer boundaries instead of cross-cutting edits.
- Keep non-generated source files under 1200 lines as a soft limit. When materially editing an over-limit file, prefer extracting a coherent module seam; if a patch must keep or introduce an over-limit file, call out the exception explicitly in the patch or PR summary.

## Build, Test, and Development Commands
Use `uv` for environments and direct tool execution. Use `just` for the repo’s standard wrappers. Run `just --list` to see all available recipes.

> **Canonical command reference**: [`docs/runbook.md`](docs/runbook.md) — all commands in one place (quality, debug CLI, eval ops, SWE-Bench, analysis).

### Quick checks
```bash
uv sync --group dev
just lint
just typecheck
just test-quick
```

If you touch the eval harness or baseline adapters, also run:

```bash
just lint-evals
just typecheck-evals
just test-evals-cov
```

If you touch integration-marked flows, containers, or baseline env wiring, also run:

```bash
just test-integration
```

Equivalent direct commands:
```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest -x -q -m "not integration"
```

`just typecheck-evals` runs pyright with `pyrightconfig.evals.json` (basic mode, 7 core eval modules). This is informational in CI (`continue-on-error: true`), not blocking. The two-config setup is a staged typing strategy: `pyrightconfig.json` enforces strict mode on `src/llmdebug/` (the shipped package), while `pyrightconfig.evals.json` uses basic mode for gradually adopting types in the eval harness. Since JSON files don't support comments, this rationale lives here rather than in the configs.

### Full local quality pass
```bash
just check
just quality
```

`just check` runs lint, typecheck, and coverage-gated tests. `just quality` is the shared quality contract used by CI/release automation. Additional coverage recipes:

```bash
just test-evals-cov       # evals/ coverage with baseline regression check
just test-combined-cov    # combined src + evals coverage report
```

Key direct commands:
```bash
uv run ruff check src tests evals
uv run ruff format src tests evals
uv run pytest -m "not integration" --cov=src/llmdebug --cov-branch --cov-report=term-missing --cov-report=xml --cov-report=json --cov-fail-under=95
uv run python scripts/quality/check_coverage_no_regression.py --coverage-json coverage.json --baseline-json quality/coverage_baseline.json
uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
just baseline-boundary-coverage
uv run bandit -c pyproject.toml -r src/llmdebug
uv run vulture src/llmdebug vulture_whitelist.py --min-confidence 90
uv run deptry src/llmdebug
uv run pip-audit --ignore-vuln GHSA-5239-wwwm-4pmq --ignore-vuln CVE-2026-4539
uv run radon cc -s -a src/llmdebug
uv run xenon --max-absolute C --max-modules C --max-average B src/llmdebug
uv run python -m evals.doc_sync --check
uv run python scripts/quality/check_type_coverage.py --strict
uv run python scripts/quality/check_dev_dependency_sync.py
uv run pyright -p pyrightconfig.evals.json
uv build --no-sources
```

### Formatting, docs, and benchmarking
```bash
just format
just docs-check
just docs-sync
just bench-capture
just bench-capture-compare
```

### Eval helpers
```bash
uv run python -m evals.validate_cases --schema-only
just eval-summary <run_id>
just eval-two-pass-summary <baseline_run_id> <rescue_run_id>
```

## CI Contract
- `.github/workflows/ci-cd.yml` runs `just lint`, `just typecheck`, and `just test-no-cov` across Python 3.10, 3.11, 3.12, and 3.13.
- The 3.13 `quality` job then runs the shared quality contract, provisions baseline repos and smoke envs, runs `just test-integration`, and runs `just typecheck-evals` as an informational lane.
- `.github/actions/run-quality-contract/action.yml` is a composite action used by CI. It encapsulates `just quality` on Python 3.13, adds conditional `just diff-coverage` on pull requests, and measures evals coverage informationally against `quality/coverage_evals_baseline.json`.
- Always run CI/CD locally before pushing to `main`.
- Use `just ci` to mirror the single-version GitHub contract locally; use `PYTHON_VERSION=<version> just <recipe>` when a change is version-sensitive.

## Server Command Convention (`pueue`)
- When suggesting commands for a server, prefer queueing them:
  - `pueue add -- <command>`
  - `pueue status`
  - `pueue log <task_id>`
  - `pueue follow <task_id>`
- If you intentionally suggest a foreground command instead, say that the direct run is deliberate.

## Coding Style & Naming
- Python 3.10+, 4-space indentation, max line length 100.
- Ruff config lives in `pyproject.toml` and enforces `E`, `W`, `F`, `I`, `UP`, `B`, `SIM`, `C4`, `RUF`, `PERF`, and `FURB`.
- Pyright runs in `strict` mode for `src/llmdebug`.
- Prefer precise type annotations on new or changed APIs.
- Naming conventions: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Treat function signatures with more than 6 explicit named parameters (excluding `self`/`cls` and variadics) as a readability smell. Prefer grouped `*Config`, `*Inputs`, `*Request`, or `*State` objects and add or update a signature-budget regression test when you refactor those seams.
- Keep patches focused and local; prefer adapters or helper seams over broad refactors.

## Testing & Quality Expectations
- Framework: `pytest` with `testpaths = ["tests"]`.
- Add a regression test for each bug fix or behavior change when feasible.
- Use async tests when touching coroutine paths.
- Coverage floor is `>=95%` for `src/llmdebug`; branch coverage must not regress versus `quality/coverage_baseline.json`; changed-line coverage is `>=80%` on PRs.
- Evals coverage is tracked via `quality/coverage_evals_baseline.json` (96.9% line / 93.7% branch); run `just test-evals-cov` when touching eval code. This gate is informational in CI (`continue-on-error: true`).
- Public type completeness must not regress versus `quality/type_coverage_baseline.json`; `just quality` enforces this via `just type-coverage-strict`.
- Baseline patch-boundary coverage is enforced via `just baseline-boundary-coverage` when touching the adapter-to-harness seam.
- `just quality` also enforces eval-harness lint (`just lint-evals`) and dev dependency-list sync (`just dependency-sync`).
- Integration-marked tests are heavier and usually run separately via `just test-integration`; GitHub Actions runs that lane in the 3.13 quality job.
- Treat `bandit`, `deptry`, `pip-audit`, `vulture`, `radon`, `xenon`, and docs drift as part of the normal review surface when relevant.
- `just audit` currently carries a narrow temporary ignore for `GHSA-5239-wwwm-4pmq` / `CVE-2026-4539` in `Pygments`; remove that ignore once upstream ships a fixed release.
- The `[tool.deptry.per_rule_ignores]` exceptions in `pyproject.toml` cover optional research dependencies (torch, jax, pandas, etc.) that are imported conditionally behind extras but not listed as hard requirements. DEP001 suppresses missing-package warnings for these; DEP002 suppresses false-positive unused-dependency warnings for eval/test-only packages.
- When touching generated documentation blocks or release/changelog surfaces, run `just docs-check`; run `just docs-sync` when the generated content needs to be refreshed.
- When touching capture/serialization hot paths, benchmark-sensitive code, or perf claims, run `just bench-capture`. Use `just bench-capture-compare` to compare against a local trusted baseline.

## Eval & Research Notes
- Read `docs/eval_quickstart.md`, `evals/README.md`, and `evals/configs/README.md` before changing the eval harness, configs, cases, importers, patchers, or reporting code.
- Use `evals/EXPERIMENTS.md` and the dated notes under `docs/` for exploratory context, but treat the README/runbook/config docs as the maintained operational source of truth.
- The main eval conditions are `base_repair`, `cot_repair`, `layered_adaptive_gated`, `layered_adaptive_llm`, `ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, and `rlm`.
- Keep eval orchestration, patcher backends, importer logic, adaptive-policy logic, and SWE-Bench tooling loosely coupled; these pieces are reused across multiple flows.
- When extracting logic out of `evals/run_eval.py`, preserve its compatibility seams unless you update every downstream consumer in the same patch. In particular, `validate_cases.py`, importer utilities, and several tests still import or monkeypatch `run_eval`-level symbols such as `apply_diff` and the lambda-RLM invocation path.
- `evals/README.md` contains an auto-generated case inventory block; refresh it with `python -m evals.doc_sync --write` when case metadata changes.

### Policy-controlled evidence selection (design rationale)
- In the adaptive conditions (`layered_adaptive_gated`, `layered_adaptive_llm`), a **policy layer — not the LLM — controls what evidence the patching model receives**. This is a deliberate design decision backed by empirical evidence: when given free choice, LLMs consistently over-request context (snapshots, files, frames), inflating token usage without proportional accuracy gains.
- The architecture enforces evidence selection through 3 layers:
  1. **Informativeness pregate**: scores traceback diagnostic value; skips the helper entirely if the traceback is already sufficient.
  2. **Policy action selection**: a rule-based gate or LinUCB contextual bandit picks one of 8 actions constraining which *types* of evidence the Stage-A LLM may request.
  3. **Request filtering**: any LLM requests outside the allowed types are silently discarded before execution.
- This is a distinct contribution vs the baseline conditions (`ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, `rlm`), where either the harness or the LLM has unconstrained control over evidence. The policy-controlled approach treats evidence selection as a *resource allocation* problem optimized via online learning.
- When editing adaptive policy code (`adaptive_online_policy.py`, `adaptive_budget_online_policy.py`, `policy_feature_scaling.py`, `eval_attempt_flow.py`, `eval_policy_features.py`, `eval_policy_online.py`, and `run_eval.py` compatibility seams), preserve this 3-layer enforcement architecture. Do not bypass the policy by letting the LLM directly control evidence selection.

- SWE-Bench Lite support lives under `evals/swebench/` and uses a local/Docker paired workflow for generation, validation, and official scoring. Follow `evals/README.md` and `docs/local_docker_eval_runbook.md`.
- SWE-Bench Pro support lives under `evals/swebench_pro/` — a contamination-free, multi-language benchmark (731 instances). Uses Docker-based scoring against pre-built DockerHub images. CLI: `python -m evals.swebench_pro.cli <subcommand>`. See `evals/README.md`.

## PR Guidance
- CI automation lives in `.github/workflows/ci-cd.yml` and `.github/actions/run-quality-contract/action.yml`.
- Prefer Conventional Commits: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`.
- Use `feat!:` or a `BREAKING CHANGE:` footer for incompatible changes.
- Before opening a PR, run `just check` at minimum and `just quality` for tooling, docs, or eval-harness changes.
