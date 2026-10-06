# llmdebug evals

This folder provides a local multi-condition repair harness to measure whether
llmdebug snapshots improve automated debugging outcomes across a deterministic
synthetic suite.

- **Native conditions**: `base_repair`, `cot_repair`, `layered_adaptive_gated`,
  `layered_adaptive_llm`
- **External baselines**: `ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, `rlm`
- **Adaptive gated**: starts base-repair-first and invokes the snapshot helper
  when gate policy detects stagnation
- **Adaptive LLM**: starts base-repair-first and lets the model decide whether
  to use one of several context actions (`skip_helper`, `invoke_helper`,
  `frame_only`, `file_only`, `search_only`, `snapshot_lite`,
  `compact_then_patch`, `retry_helper_strict_json`)

Adaptive tooling supports two online controllers:
- **Helper policy** (the full context-action set above) for `layered_adaptive_llm`.
- **Budget policy** (`keep_budget` vs `increase_budget`) for adaptive conditions only. It can
  raise attempt propose budget via multiplier tiers (default `1.0,1.5,2.0`) and is enabled by
  default (`--adaptive-budget-policy-mode on`).

Key budget-policy flags:
- `--adaptive-budget-policy-mode {off,shadow,on}`
- `--adaptive-budget-policy-architecture {tier_binary,hierarchical_v1,bwk_v1}`
- `--adaptive-budget-policy-tiers 1.0,1.5,2.0`
- `--adaptive-budget-policy-max-ups-per-case 1`
- `--adaptive-budget-bwk-run-token-budget <int>` for `bwk_v1` run-level token caps
- `--adaptive-budget-reward-rho-budget-exceeded 0.5`
- `--adaptive-toon-encoding` — use TOON tabular encoding for adaptive
  LLM-bound snapshot and retry payloads. Stage-B snapshot payloads stay within
  the legacy prompt snapshot surface and are only substituted when they are
  shorter than the standard snapshot section. Requires `toons` for full TOON
  and gracefully falls back to compact JSON if missing.

Patcher backends:
- `command`: local command that returns a unified diff
- `openai`: OpenAI-compatible `/v1/chat/completions`
- `heuristic`: demo snapshot-aware patcher
- `null`: no-op patcher (smoke testing)

Command patcher safety defaults:
- command execution uses argv mode by default (`shell=False`)
- explicit shell opt-in is available via `--patch-cmd-shell` or `LLMDEBUG_EVAL_PATCH_CMD_SHELL=1`
- timeout defaults to 30s and can be changed with
  `--patch-cmd-timeout-sec` or `LLMDEBUG_EVAL_PATCH_CMD_TIMEOUT_SEC`
- each command attempt writes the current prompt to
  `$LLMDEBUG_EVAL_PROMPT_FILE` (default: `<context_dir>/prompt.txt`)

Artifacts are written to `evals/artifacts/` and results to `evals/results/*.jsonl`.

## SWE-Bench Lite (Local generation + Docker scoring)

SWE-Bench Lite integration lives under `evals/swebench/` and follows a paired
local/Docker workflow:

1. Generation:
- produces paired predictions for `baseline` and `llmdebug`.
- writes run artifacts under `evals/swebench_runs/<run_id>/`.

2. Official scoring:
- runs the official SWE-Bench evaluator on those predictions.
- writes raw official outputs under `official_eval/`.

Primary CLI:

```bash
# Generation against a local OpenAI-compatible endpoint
python -m evals.swebench.cli generate-lite \
  --run-id <id> \
  --conditions baseline,llmdebug \
  --snapshot-dir <dir_with_instance_snapshots> \
  --max-instances <n|0> \
  --instance-ids-file <optional> \
  --openai-base-url http://localhost:<port> \
  --openai-model <model_alias> \
  --openai-http2 \
  --openai-async \
  --jobs <workers> \
  --max-request-errors 25 \
  --max-request-error-rate 0.5

# Run-directory validation (writes handoff metadata used by the scoring flow)
python -m evals.swebench.cli prepare-handoff --run-id <id>

# Official scoring on a Docker-capable machine
python -m evals.swebench.cli score-official \
  --run-dir /path/to/evals/swebench_runs/<id> \
  --conditions baseline,llmdebug \
  --workers <n> \
  --cache-level <none|base|env|instance>

# Build a tiny paired scoring subset for smoke checks
python -m evals.swebench.cli prepare-scoring-smoke \
  --run-dir /path/to/evals/swebench_runs/<id> \
  --max-instances 2
```

`--snapshot-dir` is required whenever the `llmdebug` condition is requested: it
holds one `<instance_id>.txt|.md|.json` llmdebug artifact per instance, and
generation fails fast without it (DAT-03). `run_metadata.json` records
`prompt_policy` and `snapshot_coverage` (hits, misses, missing instance IDs).
Prompts carry only the issue text by default; `--include-hints` (Lite) and
`--include-test-ids` add `hints_text` and the FAIL_TO_PASS/PASS_TO_PASS test IDs,
which are not part of the official task (DAT-01, DAT-02). A request that fails is not saved
as an empty patch: the instance stays missing, `prepare-handoff` refuses the unequal paired
sets, and `--resume` retries it (DAT-08).

Local Docker convenience wrapper:

```bash
./scripts/score_swebench_lite_official.sh --run-dir /path/to/evals/swebench_runs/<id>

# Smoke-check official scoring on tiny paired subset
./scripts/smoke_swebench_lite_official.sh --run-dir /path/to/evals/swebench_runs/<id> --max-instances 2
```

Run directory contract:

`evals/swebench_runs/<run_id>/`

- `baseline/predictions.jsonl`
- `llmdebug/predictions.jsonl`
- `<condition>/generation_meta.jsonl`: one line per proposal with `finish_reason`,
  `truncated`, `patcher_error` and `request_error`; on resume the last line for an instance
  describes its prediction (DAT-18)
- `instance_ids.txt`
- `run_metadata.json`
- `run_index.json` (written by `prepare-handoff`)
- `COPY_INSTRUCTIONS.txt` (written by `prepare-handoff`)
- `official_eval/` (written during scoring)

Policy notes:

- `baseline` + `llmdebug` are mandatory paired conditions for generation, handoff,
  and scoring (`--conditions` must include both).
- generation fails fast when patch-request errors indicate likely systemic failure
  (all requests failing, absolute error threshold, or high error-rate threshold).
- generation fails fast when the selected SWE-Bench split contains duplicate
  `instance_id` values.

Each prediction row in `predictions.jsonl`:

- `instance_id` (string)
- `model_patch` (string, unified diff)
- `model_name_or_path` (string)

Lite and Pro generation default to `--openai-max-tokens 4096` (it was 2048, which cut off
multi-file patches; DAT-18). A truncated answer yields an empty patch and
`truncated: true` in `generation_meta.jsonl`.

Detailed operational guide:
- `docs/local_docker_eval_runbook.md`

The suite is signal-focused:
- `snapshot_dependency=required`: bugs where runtime state should materially help
- `snapshot_dependency=helpful`: snapshots may help, but traceback may still suffice
- `snapshot_dependency=none`: traceback-control cases

**Known limits of the case suite (DAT-17, DAT-21; decision D12).** Two properties of the
current 1,828-case corpus limit what results can show. Changing them is a research decision
(D12), so this section only documents them:

- *ConDefects grading is weak and visible (DAT-17).* The 1,267 ConDefects cases (69%) are
  graded by at most five sample input/output pairs (`MAX_TEST_PAIRS = 5`, examples first;
  1,096 cases have five, 171 have fewer), and `test_buggy.py` is in the workspace the model
  sees. A patch that special-cases the visible inputs passes. Read ConDefects success as
  "passes the samples", not as a verified repair.
- *`snapshot_dependency` is derived, not labelled (DAT-21).* Importers set it from the
  heuristic bug category (`evals/category_mapping.py:infer_snapshot_dependency`); only the
  hand-crafted cases are labelled by hand, and all 46 `required` cases are hand-crafted. The
  distribution is helpful 1,650 (90.3%), none 132, required 46. Per-stratum results
  therefore restate per-category results and are not independent evidence that snapshots
  help where they "should".
- *Run-to-run variance.* `python -m evals.importers.determinism` flags cases whose code uses
  clocks, unseeded randomness, entropy sources, network modules or set iteration (DAT-14);
  it flags 14 committed cases, which are kept until D12 decides.

## SWE-Bench Pro (Contamination-free, multi-language, Docker-scored)

SWE-Bench Pro is a contamination-free, multi-language benchmark with 731
instances drawn from recent commits across Python, JavaScript/TypeScript, Java,
Go, Rust, and C++ repositories.

- Dataset: [`ScaleAI/SWE-bench_Pro`](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) on HuggingFace
- Module: `evals/swebench_pro/`
- Run artifacts: `evals/swebench_pro_runs/<run_id>/`
- Cached assets: `evals/swebench_pro_assets/`

Key differences from SWE-Bench Lite:

| Aspect | Lite | Pro |
|--------|------|-----|
| Language | Python only | Multi-language (Python, JS/TS, Java, Go, Rust, C++) |
| Scoring | `swebench.harness` | Docker-based (`docker run` against pre-built images) |
| Images | Built locally | Pre-built on DockerHub |
| Contamination | Possible (pre-2024 data) | Contamination-free (recent commits) |

Primary CLI:

```bash
# Generation against a local OpenAI-compatible endpoint
python -m evals.swebench_pro.cli generate-pro \
  --run-id <id> \
  --conditions baseline,llmdebug \
  --snapshot-dir <dir_with_instance_snapshots> \
  --max-instances <n|0> \
  --openai-base-url http://localhost:<port> \
  --openai-model <model_alias> \
  --jobs <workers>

# Handoff preparation (validates predictions + writes handoff metadata)
python -m evals.swebench_pro.cli prepare-handoff --run-id <id>

# Docker-based scoring on Docker host
./scripts/score_swebench_pro_official.sh \
  --run-dir /path/to/evals/swebench_pro_runs/<id> \
  --conditions baseline,llmdebug

# Equivalent Python CLI
python -m evals.swebench_pro.cli score-docker \
  --run-dir /path/to/evals/swebench_pro_runs/<id> \
  --conditions baseline,llmdebug \
  --workers <n>

# Build a tiny paired scoring subset for smoke checks
python -m evals.swebench_pro.cli prepare-scoring-smoke \
  --run-dir /path/to/evals/swebench_pro_runs/<id> \
  --max-instances 2

# Fetch run scripts for a given instance
python -m evals.swebench_pro.cli fetch-run-scripts --instance-id <id>
```

Available subcommands:

- `generate-pro` — Produce paired predictions for each condition
- `prepare-handoff` — Validate predictions and write handoff metadata
- `score-docker` — Score predictions using pre-built DockerHub images and fail
  the command when Docker/scorer infrastructure errors occur. A container that exits
  non-zero for any reason other than a patch that doesn't apply, a missing `output.json`,
  a wait timeout and an empty `fail_to_pass` list are infrastructure errors (DAT-04,
  DAT-09); a patch that doesn't apply under the official sequence (`git apply --verbose`,
  then `--reject`, then `patch --fuzz=5`, DAT-19) is a model failure.
  `docker_eval/<condition>/eval_results_detail.json` keeps `error` and
  `infrastructure_error` per instance and is written before the command fails. Cached
  outputs are reused only for the identical patch (`*_output.patch_sha256`, DAT-13), and
  dataset values are shell-quoted in the container entry script (DAT-11).
- `prepare-scoring-smoke` — Build a tiny paired subset for quick scoring checks
- `fetch-run-scripts` — Retrieve run/test scripts for a specific instance

Quick-start example:

```bash
# 5-instance smoke gate
python -m evals.swebench_pro.cli generate-pro \
  --run-id pro-smoke-$(date +%Y%m%dT%H%M%S) \
  --max-instances 5 \
  --openai-base-url http://localhost:8080 \
  --openai-model qwen3coder30ba3b

# Score results via Docker
python -m evals.swebench_pro.cli score-docker \
  --run-dir evals/swebench_pro_runs/<run_id> \
  --conditions baseline,llmdebug
```

> **Note:** llmdebug snapshot capture relies on the Python `pytest` plugin.
> For non-Python repositories in the Pro benchmark, snapshots are not available
> and the `llmdebug` condition falls back to traceback-only evidence.

## Case contract

Each `evals/cases/<case_id>/` must contain:
- `buggy.py` (or small module set),
- `test_buggy.py`,
- `case.json`,
- `oracle.patch` (required; used by validator).

`case.json` schema:

```json
{
  "description": "short summary",
  "category": "hidden_state|mutability_aliasing|async_temporal|cross_file_contract|data_shape_runtime|parse_value|path_import_env|traceback_controls|syntax_error|reference_error|logic_error|type_coercion",
  "difficulty": "easy|medium|hard",
  "snapshot_dependency": "required|helpful|none",
  "expected_exception": "IndexError",
  "tags": ["multifile", "stateful"],
  "python_args": ["-m", "pytest", "-q"],
  "timeout_sec": 30,
  "bug_source": "hand_crafted|debugbench|pyresbugs|humanevalpack|condefects|mutmut|bugsinpy|pybughive|quixbugs|trickybugs",
  "contamination_risk": "none|low|medium|high",
  "expected_fix_lines": 3,
  "source_id": "debugbench_42",
  "difficulty_rationale": "optional explanation"
}
```

The last five fields (`bug_source` through `difficulty_rationale`) are optional
and used by imported cases for traceability.
At execution time, eval-only artifacts such as `case.json`/`meta.json` metadata and
`oracle.patch` are stripped from the copied workdir before prompting, so patchers do
not receive benchmark metadata or oracle diffs through the filesystem or prompt context.

`description`, `category`, `difficulty` and `snapshot_dependency` are analysis-only: no
condition, native or baseline, sees them (`case_info_policy = neutral_v1`, see
`docs/baseline_contracts.md`).

**Oracle integrity (ATT-01).** Before every post-patch verification run the harness restores
the case's pristine test surface: `test_*.py`, `*_test.py`, pytest config and hook files
(`conftest.py`, `pytest.ini`, `.pytest.ini`, `pytest.toml`, `pyproject.toml`, `setup.cfg`,
`tox.ini`), modules that would shadow pytest or run at interpreter start (`pytest.py`,
`pytest/`, `_pytest/`, `py.py`, `sitecustomize.py`, `usercustomize.py`, `*.pth`) and any
symlink. Edited files are reverted, added ones deleted, links removed without being followed.
Success then requires exit code 0 **and** a fresh per-run JUnit report (stale or planted
reports are deleted first) with at least one executed test and no failures or errors. The
patched code still runs inside the pytest process, so a deliberately adversarial patch could
forge a verdict; the guard targets reward hacking, not a hostile model. Records carry `patch_meta.test_surface_modified`,
`test_surface_restored_paths`, `pytest_verdict_source` (`junit`) and
`pytest_verdict_counts`; `attempt_metadata.test_surface_modified` mirrors the flag.

### Categories

| Category | Description |
|----------|-------------|
| `hidden_state` | Bugs involving mutable state, caching, stale values |
| `mutability_aliasing` | Reference/aliasing issues, shared mutable objects |
| `async_temporal` | Async/await ordering, race conditions, cancellation |
| `cross_file_contract` | Multi-file contract violations, API misuse |
| `data_shape_runtime` | Array shape/dimension mismatches, wrong dtypes |
| `parse_value` | Parsing or value extraction bugs |
| `path_import_env` | Path/import/environment configuration issues |
| `traceback_controls` | Control cases where traceback alone suffices |
| `syntax_error` | Missing colons, wrong indentation, operator misuse |
| `reference_error` | Undefined variables, wrong scope, missing attributes |
| `logic_error` | Wrong comparisons, off-by-one, incorrect conditions |
| `type_coercion` | Type mismatch, wrong cast, implicit conversion bugs |

### Contamination risk

Imported cases carry a `contamination_risk` field indicating how likely the
bug-fix pair appears in LLM training data:

| Level | Description | Example sources |
|-------|-------------|-----------------|
| `none` | Freshly generated, impossible to memorize | hand-crafted (the `mut_*` cases are hand-crafted mutations) |
| `low` | Post-training-cutoff or GPT-4-generated bugs | DebugBench, ConDefects |
| `medium` | Pre-cutoff but not widely published | PyResBugs, BugsInPy, PyBugHive |
| `high` | Extremely well-known benchmark | HumanEvalPack, TrickyBugs |

## Validate cases (required before benchmarks)

```bash
uv run python -m evals.validate_cases --schema-only
```

`evals.validate_cases` also supports the same execution flags as `run_eval`
(`--executor`, `--container-image`, `--container-*`, `--container-env`, `--jobs`).
Example local fallback:

```bash
uv run python -m evals.validate_cases --executor local
```

Useful filters:

```bash
uv run python -m evals.validate_cases --category hidden_state --difficulty easy --snapshot-dependency required --max-cases 5
```

Full executable integrity validation (baseline fail + oracle pass) is available for
later runs:

```bash
uv run python -m evals.validate_cases
```

## Current case inventory

<!-- BEGIN AUTO-GENERATED: CASE_INVENTORY -->
_This section is auto-generated from `evals/cases` by `python -m evals.doc_sync --write`._

1828 validated cases across 12 categories and 4 sources:

| Source | Cases |
|--------|-------|
| ConDefects | 1267 |
| DebugBench | 328 |
| HumanEvalPack | 164 |
| Hand-crafted | 69 |
| **Total** | **1828** |

Category distribution:

| Category | Count | Difficulty breakdown |
|----------|-------|---------------------|
| logic_error | 1293 | 589 easy, 391 medium, 313 hard |
| type_coercion | 294 | 198 easy, 47 medium, 49 hard |
| reference_error | 102 | 30 easy, 59 medium, 13 hard |
| parse_value | 49 | 3 easy, 46 medium |
| async_temporal | 30 | 7 easy, 16 medium, 7 hard |
| syntax_error | 26 | 16 easy, 6 medium, 4 hard |
| cross_file_contract | 12 | 1 easy, 10 medium, 1 hard |
| data_shape_runtime | 6 | 3 easy, 2 medium, 1 hard |
| hidden_state | 4 | 2 easy, 2 medium |
| mutability_aliasing | 4 | 2 easy, 1 medium, 1 hard |
| path_import_env | 4 | 3 easy, 1 medium |
| traceback_controls | 4 | 4 easy |
<!-- END AUTO-GENERATED: CASE_INVENTORY -->

## Import external cases

Install the `evals` extra for HuggingFace dataset access:

```bash
pip install 'llmdebug[evals]'
# or: pip install datasets>=2.0
```

### DebugBench (contamination: low)

Imports Python cases from `Rtian/DebugBench` on HuggingFace. Filters by code length,
fix size, determinism, and stdlib-only imports. Generates tests from LeetCode examples.
~34% of imported cases survive full execution validation.

```bash
# Full import (unlimited cases, auto-validate)
uv run python -m evals.importers.cli debugbench --validate

# Preview first 5 cases without writing
uv run python -m evals.importers.cli debugbench --max-cases 5 --dry-run
```

Every importer checks each generated case's schema. `--validate` additionally runs the case:
the baseline must fail with the expected exception and `oracle.patch` must fix it. Before
DAT-05, seven importers only checked the schema under `--validate`, and the schema check
rejected every native-repro case (BugsInPy, PyBugHive, TrickyBugs), which keep the source at
its repository path instead of `buggy.py`; native cases now need the files the oracle patch
changes.
When using `--validate`, importers honor the same execution flags as `run_eval`
and `validate_cases` (`--executor`, container resource/network flags, and `--container-env`).

### HumanEvalPack (~20 cases, contamination: high)

```bash
uv run python -m evals.importers.cli humanevalpack --max-cases 20 --validate
```

### ConDefects (contamination: low)

Requires cloning the ConDefects repo and downloading test data separately.

```bash
git clone https://github.com/appmlk/ConDefects.git
# Download Test.zip from OneDrive and extract into ConDefects/
uv run python -m evals.importers.cli condefects --repo-path ./ConDefects --validate
```

### QuixBugs (~40 cases, contamination: high)

Classic algorithm benchmark with 40 single-line Python bugs. Requires cloning the
QuixBugs repo. All bugs are logic errors in well-known algorithms.

```bash
git clone https://github.com/jkoppel/QuixBugs.git
uv run python -m evals.importers.cli quixbugs --repo-path ./QuixBugs --validate
```

### Deferred-download top 3 (BugsInPy, PyBugHive, TrickyBugs)

Use a two-step workflow so local preparation does not pull large artifacts.

1. Prepare a lock manifest (safe for low-storage laptops):

```bash
uv run python -m evals.importers.cli prepare-top3 \
  --manifest evals/manifests/top3.lock.json
```

2. Materialize on a server with enough disk just before evaluation:

```bash
uv run python -m evals.importers.cli materialize-top3 \
  --manifest evals/manifests/top3.lock.json \
  --data-root /mnt/evals/top3 \
  --download-policy missing \
  --jobs 4 \
  --validate
```

`materialize-top3` writes a structured import report to
`evals/artifacts/import_reports/top3-<timestamp>.json` with attempted/imported/skipped
counts, deterministic skip reasons, and hard-failure flags per source.

Download policies:
- `never`: do not download; fail/skip if required artifacts are missing locally.
- `missing`: download only missing artifacts; existing files are integrity-verified and
  re-downloaded once if verification fails (recommended default).
- `refresh`: re-download all pinned artifacts.

Strict reproducibility semantics (top3 only):
- native dataset test commands only (`repro_mode=native_dataset_test`).
- no synthetic marker-test conversion fallback.
- imported metadata includes `source_test_command`, `source_test_selector`,
  `source_buggy_commit`, and `source_fixed_commit`.
- non-convertible entries are skipped with explicit, stable reasons.

PyBugHive policy:
- offline dumps are mandatory and authoritative.
- full offline archive list is pinned in `prepare-top3` manifest metadata.
- required artifacts must carry integrity pins (`size_bytes` + `checksum`); legacy
  manifests without required integrity pins hard-fail deterministically.
- if mandatory PyBugHive artifacts fail availability/integrity checks, materialization
  still writes a report but exits non-zero.

Manual pre-step before `run_eval` (phase 1):

```bash
uv run python -m evals.importers.cli materialize-top3 \
  --manifest evals/manifests/top3.lock.json \
  --data-root /mnt/evals/top3 \
  --download-policy missing \
  --validate
uv run python -m evals.run_eval --bug-source bugsinpy --bug-source pybughive --bug-source trickybugs
```

Storage guidance (compressed inputs, approximate):
- BugsInPy metadata archive: MB scale.
- PyBugHive offline dump: tens of GiB.
- TrickyBugs archive: ~1.5 GiB.

## Benchmark runs

### Containerized test execution (default)

`evals.run_eval`, `evals.validate_cases`, and `evals.importers.cli --validate`
use `--executor testcontainers` by default.

Build the default runner image once:

```bash
docker build -f evals/docker/runner.Dockerfile -t llmdebug-eval-runner:latest .
```

Important execution flags:
- `--executor testcontainers|local` (default: `testcontainers`)
- `--jobs <int>` (default: `1`)
- `--container-image <image>` (default: `llmdebug-eval-runner:latest`)
- `--container-network none|bridge` (default: `none`)
- `--container-cpus <float>`, `--container-memory <str>`, `--container-pids-limit <int>`
- `--container-user <uid:gid>` (default: host `uid:gid` when available; fallback `0:0`)
- `--container-tmpfs-size-mb <int>` (default: `128`)
- `--container-env KEY=VALUE` (repeatable, explicit allowlist)
- `--container-pool`: reuse one long-lived container per job instead of a fresh container per
  pytest call. Off by default: pooled containers share the read-write workdirs mount and
  outlive units. Before the 2026-10-04 fix (ORC-02) the pool was started but never used, so
  every earlier result ran in one-shot containers. Recorded as
  `run_metadata.patcher_config.execution.container_pool`.
- stale eval containers are pruned automatically at run start (best-effort, 5-minute grace); this includes running pool containers whose owning run process on this host is gone, for example after `kill -9` (they are labelled with the owner's PID, host and run id)

Use `--executor local` if you want the legacy host-side pytest execution.

### Single-container eval runner

For running the whole eval pipeline in one container (without Docker-in-Docker),
use:

```bash
evals/docker/run_eval_docker.sh --patcher null --max-cases 5 --k 1
```

Behavior notes:
- auto-builds `llmdebug-eval:latest` for regular runs and
  `llmdebug-eval-baselines:latest` for external baseline runs
  (or if `LLMDEBUG_EVAL_BUILD=1`)
- strips `--executor` and `--container-*` flags, then forces `--executor local`
- for OpenAI-backed runs (`--patcher openai` and external baselines), resolves
  API key precedence as `Z_API`, then
  `OPENAI_API_KEY`, then `LLMDEBUG_EVAL_OPENAI_API_KEY`, then falls back to
  `local`, and passes the resolved value via `--openai-api-key` (instead of
  forwarding raw key env vars)
- rewrites localhost OpenAI URLs on macOS to `host.docker.internal`; on Linux,
  it uses `--network host` when a localhost URL is detected
- rewrites `--config <path>` to a staged path mounted inside the container, so
  host-local config files work out of the box
- reaps the container explicitly on exit or interrupt, removes anonymous
  volumes, and removes any staged config mount directories created for
  `--config`
- external baseline runs use the baseline-capable image, which bakes the pinned
  baseline repos and Linux smoke envs into the container image; that image is
  for local research use and should not be redistributed
- Docker baseline runs do not support custom `--baseline-repo-dir` or
  `--baseline-python`; rely on the image-baked `/repo/baselines/...` layout

### TOML config files (`--config`)

Use `--config <file.toml>` to move long argument lists into a versioned config.

Resolution order is:
- defaults
- TOML config values
- explicit CLI flags (highest precedence)

Config format:
- keys can live at the top level or inside `[run_eval]`
- nested tables are supported (`[run_eval.openai]`, `[run_eval.adaptive]`, etc.)
- table keys flatten to CLI destinations (`openai.base_url` -> `openai_base_url`)
- unknown keys fail fast, so config/CLI drift is surfaced immediately

Secret policy:
- `openai_api_key` literals are rejected in config files
- use `openai_api_key_env = "OPENAI_API_KEY"` instead
  - this follows the standard OpenAI key fallback chain: `Z_API`, then `OPENAI_API_KEY`, then `LLMDEBUG_EVAL_OPENAI_API_KEY`
  - when that chain is empty in repo-local workflows, the harness defaults `OPENAI_API_KEY` to `local`

Each run writes a resolved lock file:
- `evals/artifacts/configs/<run_id>.resolved.json`
- contains the final resolved args and redacts secret values

Example:

```bash
RUN_ID="local-qwen-$(date +%Y%m%dT%H%M%S)"
bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --run-id "$RUN_ID" \
  --config evals/configs/openai_adaptive.template.toml
```

Shipped template defaults:
- `k = 2`
- `jobs = 3`
- OpenAI-compatible base URL: `http://127.0.0.1:12000/v1`
- Baseline templates pin `conditions = "<baseline_name>"` explicitly so the intended
  comparison target is visible in the config itself, not only in derived runtime state.
- `baseline_lambda_rlm.template.toml` and `baseline_rlm.template.toml` keep their
  baseline-internal `timeout_sec` aligned with `baseline_process_timeout_sec` to avoid a
  hidden tighter inner cap than the other shipped external baselines.

Concurrency caveat for adaptive benchmarks:
- The shipped configs default to `jobs = 3` for throughput.
- For final benchmark runs that enable either online controller
  (`--adaptive-online-policy-mode on|shadow` or
  `--adaptive-budget-policy-mode on|shadow`) under `layered_adaptive_llm`,
  override to `--jobs 1` if you want strictly online, case-ordered policy
  updates and reproducible event logs.
- Leaving `jobs > 1` changes those runs into concurrent/batched policy
  evaluation, which is useful for throughput experiments but not directly
  comparable to single-threaded online learning.

The hierarchical bandit architecture (Budget → Evidence → Strategy levels),
its relationship to BOAD, and extension points (Thompson Sampling, TEP) are
documented in `docs/boad_architecture.md`.

Important distinction:
- the online and budget policies optimize **which action or resource envelope to
  choose**;
- `--adaptive-tep-mode full` optimizes **native prompt surfaces** for Stage A,
  context compaction, retry planning, and patch proposal after that path has
  been chosen.

TEP therefore complements the adaptive controllers rather than replacing them.
Trace-style end-to-end trajectory optimization remains a future research
direction and is not a shipped eval mode.

See template:
- `evals/configs/openai_adaptive_gated.template.toml`
- `evals/configs/openai_adaptive.template.toml`
- Probe presets:
  - `evals/configs/openai_adaptive_probe_decay_098.template.toml`
  - `evals/configs/openai_adaptive_probe_decay_0995.template.toml`
  - `evals/configs/openai_adaptive_probe_prompt_order_crash_first.template.toml`
  - `evals/configs/openai_adaptive_probe_prompt_order_evidence_last.template.toml`
  - `evals/configs/openai_adaptive_probe_current_knobs_compact.template.toml`
  - `evals/configs/openai_adaptive_probe_warm_start.template.toml`
  - `evals/configs/openai_adaptive_probe_bwk_v1.template.toml`
- Baseline templates:
  - `evals/configs/baseline_ldb.template.toml`
  - `evals/configs/baseline_intervenor.template.toml`
  - `evals/configs/baseline_intervenor_fair_v2.template.toml`
  - `evals/configs/baseline_tracecoder.template.toml`
  - `evals/configs/baseline_tracecoder_fair_v2.template.toml`
  - `evals/configs/baseline_lambda_rlm.template.toml`
  - `evals/configs/baseline_rlm.template.toml`
- Fairness-tuned native templates:
  - `evals/configs/openai_base_repair_fair_v2.template.toml`
  - `evals/configs/openai_cot_repair_fair_v2.template.toml`
- Hybrid adaptive+baseline runs are now allowed with `--patcher baseline` plus
  `--conditions layered_adaptive_gated` or `--conditions layered_adaptive_llm`.
  In that mode, the adaptive outer controller still owns history, verifier, and
  evidence sidecars, while the selected baseline owns the inner repair loop.
- Hybrid baseline runs now support three llmdebug augmentation modes:
  - `baseline_hybrid_guidance_mode = "tuned_v1"` preserves the current
    baseline-specific guidance integration.
  - `baseline_hybrid_guidance_mode = "fair_v1"` injects one shared canonical
    guidance block across all five baselines.
  - `baseline_hybrid_guidance_mode = "fixed_v1"` is an INTERVENOR-first
    deterministic fixed-evidence block that injects the full snapshot plus
    compact failure-state evidence, intended for fixed-evidence ablations.
  - For paper-facing reporting, treat these as separate arms:
    `raw baseline`, `hybrid-canonical`, and `hybrid-tuned`.
- Non-adaptive runs are now kept methodologically clean by construction. If the
  selected condition set does not include an adaptive arm, the harness forces
  `context_protocol=monolithic`, disables `adaptive_controller_lambda_rlm`, and
  forces the adaptive online/budget policy modes to `off` even when those knobs
  were requested in config or on the CLI. The resolved lockfile records those
  rewrites as `forced_off:no_adaptive_condition`.
- For plain baseline conditions (`ldb`, `intervenor`, `tracecoder`,
  `lambda_rlm`, `rlm`), llmdebug does not write controller/evidence/history or
  verifier sidecars. Those additions appear only for hybrid adaptive baseline
  runs under `layered_adaptive_gated` or `layered_adaptive_llm`.
- For native non-adaptive conditions (`base_repair`, `cot_repair`), the prompt
  still includes ordinary failure evidence from the test run, especially pytest
  stdout/stderr/traceback and optional monolithic project-file context. It does
  not include llmdebug runtime snapshot sections, Stage-A helper routing,
  adaptive evidence indexes, or controller framing unless adaptation is active.
- `cot_repair` now differs from `base_repair` semantically, not just via prompt
  wording: it runs a first-attempt planning call on the native OpenAI path,
  records the returned repair plan, and injects that plan into the final patch
  prompt before structured patch generation.
- `--baseline cot_repair` is a native shorthand, not an external baseline
  adapter. With no override it resolves to the native `cot_repair` condition.
  With `--conditions layered_adaptive_llm` or `layered_adaptive_gated`, it
  keeps the native adaptive path but still preserves the `cot_repair`
  first-attempt planner and Stage-B repair-plan injection.
- `--baseline-process-timeout-sec` optionally splits one baseline subprocess
  timeout from `--openai-timeout-sec`. The baseline adapter contract now treats
  `--openai-timeout-sec` as the shared API timeout and the case metadata
  timeout as the per-case pytest budget.
- The controller packet now carries bounded AST/snapshot evidence summaries and
  compact history lessons. Hybrid runs now always compute a bounded
  deterministic `strategy_summary`, and can additionally emit a bounded
  lambda-style decomposition block so baseline solvers can reason over a small
  set of explicit subproblems instead of one flat evidence dump. `intervenor`
  consumes the richer teacher/learner guidance path, while `ldb`,
  `tracecoder`, and `rlm` receive the same distilled controller guidance
  through their baseline prompt surface.
- `fair_v1` is the fairness-oriented lane: it equalizes the llmdebug
  augmentation interface, not the baselines' native solver affordances. RLM
  still keeps its tool loop, and INTERVENOR still keeps its teacher/learner
  solver structure.
- `fixed_v1` is the deterministic static lane: it injects the full snapshot
  plus a fixed failure-state evidence block derived from the current failing
  state, and is currently shipped only as an INTERVENOR-facing study preset.
- `fair_v2` template files are separate budget lanes. They keep thinking
  enabled, raise completion caps moderately, and leave the canonical native and
  baseline templates unchanged.
- TraceCoder now keeps `complete_prompt` compact and restores exact failing
  `test_buggy.py` excerpts only in llmdebug-native analysis prompts, bounded by
  `baseline_tracecoder_exact_test_context_max_chars`.
- `lmstudio_model_yaml` remains the default compat profile for local
  llama.cpp/model-template servers, but it cannot express per-request thinking
  toggles. The harness documents that limitation rather than disabling thinking
  to normalize behavior.
- To assess cap risk on historical artifacts without mutating them, run
  `python -m evals.budget_risk_report <jsonl>...`.
- `--adaptive-controller-lambda-rlm` turns that decomposition block into a real
  controller-owned backend call. The runner invokes the pinned
  `baselines/repos/lambda-RLM` checkout through
  `baselines/envs/lambda_rlm/bin/python` and records the returned
  strategy/decomposition metadata in `patch_meta` and the controller packet.
- Hybrid adaptive runs now also persist bounded RCA-backed attempt history
  under `evals/artifacts/<run_id>/contexts/<case_id>/<condition>/`. The
  controller reloads those lessons on later retries so adaptation remains
  stateful without writing free-form conversation history into baseline prompts.
- Stage-A request prompting is now backed by a typed schema/validation seam in
  `evals/stage_a_contract.py` rather than a handwritten JSON-only contract.
- The OpenAI-native adaptive repair path remains available as a legacy/native
  ablation path. New adaptive-controller integration work should target the
  hybrid baseline composition path first.
- Prompt-feature ablation templates:
  - `evals/configs/openai_base_repair_retry_reflection.template.toml`
  - `evals/configs/openai_adaptive_stage_b_cot.template.toml`
  - `evals/configs/openai_base_repair_repair_method_scaffold.template.toml`
  - `evals/configs/openai_base_repair_retry_plan_then_patch.template.toml`

Local/Docker quickstart:

Canonical runbook:
`docs/local_docker_eval_runbook.md`

```bash
# Terminal 1: optional Blackwell-oriented vLLM helper
evals/docker/run_vllm_blackwell.sh
```

```bash
# Terminal 2: smoke run against local vLLM
bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url http://127.0.0.1:7777/v1 \
  --openai-model qwen3coder30ba3b \
  --openai-response-mode edits \
  --conditions base_repair,layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --jobs 2 \
  --k 1 \
  --max-cases 5
```

```bash
# Larger local/Docker run
RUN_ID="local-qwen3coder30ba3b-$(date +%Y%m%dT%H%M%S)"
bash evals/docker/run_eval_docker.sh \
  --patcher openai \
  --openai-base-url http://127.0.0.1:7777/v1 \
  --openai-model qwen3coder30ba3b \
  --openai-api-key local \
  --openai-timeout-sec 180 \
  --openai-max-propose-sec 300 \
  --openai-max-tokens 2048 \
  --openai-temperature 0.0 \
  --max-context-chars 200000 \
  --prompt-budget-total-chars 180000 \
  --conditions base_repair,layered_adaptive_llm \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --jobs 4 \
  --k 3 \
  --run-id "$RUN_ID"
```

If you want continuity with `evals.run_eval` CLI defaults instead, use:
`--conditions base_repair,layered_adaptive_llm`.

Quick sanity (small subset):

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model deepseek/deepseek-r1-0528-qwen3-8b \
  --openai-response-mode edits \
  --jobs 2 \
  --no-files-context \
  --k 1 \
  --max-cases 5
```

Full suite (all discovered cases in `evals/cases`):

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model deepseek/deepseek-r1-0528-qwen3-8b \
  --openai-response-mode edits \
  --no-files-context \
  --k 1
```

Filtered run (category/difficulty/snapshot dependency/bug source):

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model deepseek/deepseek-r1-0528-qwen3-8b \
  --category async_temporal \
  --difficulty hard \
  --snapshot-dependency required \
  --k 3

# Filter by bug source
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model deepseek/deepseek-r1-0528-qwen3-8b \
  --bug-source hand_crafted \
  --k 1
```

`--openai-response-mode edits` is recommended for local reasoning models.
`--openai-response-mode full_file` is available for single-source-file cases; in v1 it
falls back internally to `diff` when the copied case workdir contains more than one
non-test `.py` file.
`--no-files-context` is useful when prompts are too long.
Runner-level forced patch attempts are off by default (`--force-patch-attempt` enables them).
They never change an outcome: a marker-only patch is recorded as `forced_noop_patch`, and a
marker applied after the model's diff failed keeps the attempt `patch_apply_failed` with
`runner_forced_patch_recovery*` modifiers, then is reverted (ATT-07, D8).
`no_patch_proposed`, `truncated_response` and `forced_noop_patch` are runner-level notes, not
baseline-native root causes. `truncated_response` means the model's answer stopped at
`--openai-max-tokens` (`finish_reason="length"`); it is never parsed into a patch (LLM-02).
For external baselines, the underlying cause may instead live in
`patch_meta.baseline_meta.error` (for example `timeout`, `learner_empty_response`)
or `patch_meta.compatibility_reason` (for example `subprocess_task_unsupported`).
For TraceCoder specifically, `patch_meta.baseline_meta.candidate_status=unchanged_original_code`
means the baseline returned `final_code` identical to the original buggy file. Treat this
as a baseline-side outcome, not a harness patch-application bug. In the patched TraceCoder
loop, `best_code` starts as the original buggy program and is only replaced when a candidate
strictly improves `passed_count` or fully passes, so `No improvement streak reached` or
`No repair plan produced` can still end with the original file as `final_code`.
For OpenAI patcher responses specifically, `--openai-force-patch-attempt` can further
force a minimal patch diff when model output is unparsable.
Forced marker patches are automatically skipped on transport/context-limit request failures,
and those attempts are recorded with note `patcher_request_failed`.
OpenAI transport recovery also applies to OpenAI-backed baseline adapters. When a baseline
throws a retryable infrastructure exception (for example model-loading 503s or connection
resets), the harness records a non-scoring `infra_event`, waits through the shared recovery
window, and retries the same outer attempt from a clean attempt workspace. If recovery does
not succeed in time, the unit remains unfinished/resumable under the same `--run-id` and is
excluded from failed-case accounting.
When OpenAI-compatible patcher responses return a non-transport no-diff error,
the runner performs one compact retry with source-focused guidance before falling back
to forced-patch handling.

### Progress output and safe resume

`evals.run_eval` prints per-unit progress lines while running:
- `START <case>/<condition>` before each unit,
- `DONE <case>/<condition>` with `status`, unit time, elapsed time, ETA, and cumulative
  prompt-char/token usage for the unit,
- `SKIP (resumed) <case>/<condition>` for units already completed in a resumed run.

Resuming requires the same run settings: when the run settings fingerprint (which covers the
patcher config, the case and condition selection, the record policy and `--seed`) differs from
the existing results, the run aborts instead of writing records that would claim the old
configuration (ORC-04). `--allow-settings-drift` resumes anyway; new records then carry fresh
run metadata with `settings_drift_from_fingerprint`. `--seed` is always recorded, with
`deterministic` as a separate flag, because it also seeds bandit exploration (ORC-05).

For long-running units, enable richer in-flight visibility:

```bash
uv run python -m evals.run_eval \
  --run-id z-ai-glm45air-main \
  --patcher openai \
  --openai-base-url https://api.z.ai/api/anthropic \
  --openai-model glm-4.5-air \
  --progress-verbosity detailed \
  --progress-heartbeat-sec 20
```

Progress controls:
- `--progress-verbosity basic|detailed|trace` (default `basic`)
- `--progress-heartbeat-sec <float>` (default `30`; `0` disables heartbeat)
- `detailed` adds `[HEARTBEAT]` and per-attempt timing lines with prompt/token usage after
  each completed attempt. When the OpenAI patcher is in flight, heartbeat lines and the Rich
  "Active Units" panel expose normalized live fields:
  `owner=harness|openai|baseline|pytest`,
  `source=phase|heartbeat|sidecar|stdout|stderr|transport`,
  the wait target (`wait=openai_server|tracecoder_subprocess|ldb_subprocess`),
  and the operation (`op=request_context|compact_context|plan_patch|propose_patch|baseline_repair`).
  This lets you distinguish harness-owned local work from OpenAI server waits, baseline
  subprocess work, pytest execution, and transport retries/backoff. For sidecar-aware baselines
  (`ldb`, `tracecoder`), detailed mode also shows observed counters such as
  `iters=<attempts_completed>`, `llm_calls=<observed_llm_calls>`, `eval=<sec>`, plus a concise
  `activity=...` excerpt when stdout/stderr tailing is the only live signal.
- `trace` additionally prints phase transitions (attempt start, patch propose cycle, retry phase)
- In the Rich TTY dashboard, heartbeat updates also feed the "Active Units" view even when
  inline detailed heartbeat lines stay suppressed, so long-running units can still show fresh
  activity state without extra stdout noise. Heartbeats are liveness pings only: they do not
  reset the dashboard's quiet/stall timer unless real progress changed, so `quiet <N>s` now
  means "no meaningful delta observed for N seconds" rather than merely "no redraw happened".

Use `--run-id` to make runs resumable:

```bash
# Start a named run
uv run python -m evals.run_eval \
  --run-id z-ai-glm45air-main \
  --patcher openai \
  --openai-base-url https://api.z.ai/api/anthropic \
  --openai-model glm-4.5-air \
  --k 1

# Resume later with the exact same settings + filters
uv run python -m evals.run_eval \
  --run-id z-ai-glm45air-main \
  --patcher openai \
  --openai-base-url https://api.z.ai/api/anthropic \
  --openai-model glm-4.5-air \
  --k 1
```

Resume safety guardrails:
- `--run-id` must exactly match the existing `evals/results/<run_id>.jsonl` filename for resume
  to engage; if no exact match exists and there is a unique close existing run-id, the runner now
  warns so timestamp typos are easier to spot
- if an existing `run_id` is missing `run_settings_fingerprint`, resume still aborts
  (cannot validate compatibility safely)
- if an existing `run_id` has a different `run_settings_fingerprint`, the runner now prints
  a warning and continues, reusing completed-unit state from existing records
- with an enabled decision-model role, the preflight resolves what the sidecar actually serves
  (`patcher_config.adaptive_decision_model_serving`: model id, revision, device, dtype,
  quantization, context limit) and resume aborts when it differs from the existing records'
  identity unless `--allow-settings-drift` is passed. The settings fingerprint cannot cover
  this: it is computed from the flags before the sidecar is contacted, and the request alias
  (`--adaptive-decision-model-name clef-flash`) stays the same across revisions and
  quantizations. Endpoints that report no identity on `/health` add no such entry

Crash/recovery safeguards:
- result records are appended incrementally during each unit, so progress survives process crashes
- trailing malformed JSON in the last line of a results file is auto-repaired during resume (file is truncated to the last valid record and a `.corrupt-tail-*.log` sidecar backup is written)
- a per-run lock file (`<run_id>.jsonl.lock`) prevents two writers from appending to the same run simultaneously when OS file-lock support is available; unsupported platforms continue with a warning
- the lock file stays after a run (removing it would let two processes each hold an "exclusive" lock on different files); it only records the last holder, so it is safe to leave or delete between runs
- each result record is fsynced as it is written, so a power loss or kernel crash loses at most the record being written
- with `--jobs N`, the first Ctrl-C stops scheduling and reports how many units are still running, then waits for them; a second Ctrl-C exits at once (status 130), and `--resume` continues the run

To compare retry strategies cleanly, run separate `run_id`s with identical settings except
`--retry-context-mode`:

```bash
# Stateless retries (new context each retry)
uv run python -m evals.run_eval \
  --run-id z-ai-glm45air-fresh \
  --patcher openai \
  --openai-base-url https://api.z.ai/api/anthropic \
  --openai-model glm-4.5-air \
  --retry-context-mode fresh

# Conversational retries (carry summarized history)
uv run python -m evals.run_eval \
  --run-id z-ai-glm45air-history \
  --patcher openai \
  --openai-base-url https://api.z.ai/api/anthropic \
  --openai-model glm-4.5-air \
  --retry-context-mode history
```

### Two-pass targeted rescue workflow

Use this when full adaptive runs are expensive and you want to spend helper/context
budget only on hard cases.

```bash
# Pass 1: traceback-only baseline over a broad set
RUN_ID_BASE="tb-$(date +%Y%m%dT%H%M%S)"
uv run python -m evals.run_eval \
  --run-id "$RUN_ID_BASE" \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --conditions base_repair \
  --k 3
```

```bash
# Build rerun list from terminal base_repair rows:
# include failed cases only (recommended default protocol).
CASE_LIST="evals/artifacts/case_lists/${RUN_ID_BASE}.failed.txt"
uv run python -m evals.select_cases \
  --results-jsonl "evals/results/${RUN_ID_BASE}.jsonl" \
  --condition base_repair \
  --mode failed \
  --max-attempt 2 \
  --out "$CASE_LIST"

# Shortcut for a single-condition run: write <results>.failed.txt next to the JSONL.
uv run python -m evals.extract_failed_cases \
  "evals/results/${RUN_ID_BASE}.jsonl"
```

`--mode failed` includes terminal `patcher_request_failed` rows, including
`request_budget_exceeded`, so those cases can still be targeted by adaptive rescue runs.
`--max-attempt 2` drops terminal attempt-3 rows so older `k=3` baselines can be
reused for newer `k=2` rescue workflows.

```bash
# Pass 2: adaptive rescue over failed subset only
RUN_ID_RESCUE="rescue-$(date +%Y%m%dT%H%M%S)"
uv run python -m evals.run_eval \
  --run-id "$RUN_ID_RESCUE" \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --conditions layered_adaptive_llm \
  --case-list-file "$CASE_LIST" \
  --adaptive-max-helper-calls 1 \
  --adaptive-budget-topup-max-calls 1 \
  --adaptive-online-policy-mode on \
  --adaptive-online-policy-algo linucb \
  --adaptive-online-state-path "evals/artifacts/policies/${RUN_ID_RESCUE}.json" \
  --adaptive-online-events-path "evals/results/${RUN_ID_RESCUE}.adaptive_policy.jsonl" \
  --k 3
```

Equivalent `just` shortcuts (Docker-backed by default):

```bash
RUN_ID_BASE="tb-20260301T120000"
RUN_ID_RESCUE="rescue-20260301T130000"

# 1) Base-repair baseline (Docker wrapper + evals/configs/openai_base_repair.latest.toml)
just eval-traceback-only "$RUN_ID_BASE"

# 2) Extract failed base_repair cases (default max_attempt=2)
just eval-select-failed "$RUN_ID_BASE"
# writes: evals/artifacts/case_lists/${RUN_ID_BASE}.failed.txt

# 3) Adaptive online-policy rescue on failed-only subset from step (2) via Docker wrapper
just eval-adaptive-online-failed "$RUN_ID_BASE" "$RUN_ID_RESCUE"

# 4) Workflow-level uplift summary (baseline + rescue)
just eval-two-pass-summary "$RUN_ID_BASE" "$RUN_ID_RESCUE"
```

Argument mapping:
- `eval-traceback-only <run_id>`: runs `evals/docker/run_eval_docker.sh` with traceback config, writes `evals/results/<run_id>.jsonl`.
- `eval-select-failed <run_id> [condition] [max_attempt]`: reads `evals/results/<run_id>.jsonl`, writes `evals/artifacts/case_lists/<run_id>.failed.txt` (default `max_attempt=2`; pass `0` to disable filtering).
- `eval-adaptive-online-failed <source_run_id> <rescue_run_id>`: runs `evals/docker/run_eval_docker.sh` with adaptive config and failed-case list, writes `evals/results/<rescue_run_id>.jsonl` plus online-policy state/events artifacts.
- `eval-two-pass-summary <baseline_run_id> <rescue_run_id>`: reports rescue coverage on baseline failures, rescued success rate on failed cases, the cumulative success after both passes and each pass's own token and runtime cost. The cumulative gain is the effect of extra compute on failures only, not a treatment effect, and cannot show a regression (STA-22).
- both Docker-backed `just` eval recipes honor `OPENAI_BASE_URL` (default: `http://127.0.0.1:12000/v1`).

### Reproducibility controls

Use deterministic mode when you need reproducible patch-generation runs:

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model deepseek/deepseek-r1-0528-qwen3-8b \
  --deterministic \
  --seed 12345 \
  --record-prompt hash \
  --record-evidence hash \
  --record-deps minimal
```

Key flags:
- `--deterministic --seed <int>`: enforce deterministic mode; fails fast if OpenAI-compatible backend does not acknowledge seed support.
- `--record-prompt full|hash|none`: controls prompt persistence in `attempt_metadata`.
- `--record-evidence full|hash|none`: controls evidence persistence in `attempt_metadata`.
- `--record-env KEY` (repeatable): allowlist environment keys into `run_metadata`.
- `--record-deps none|minimal|runtime`: dependency version capture scope.
- `--jobs <int>`: number of units to execute concurrently (default `1`).
- `--executor testcontainers|local`: test executor backend (default: `testcontainers`).
- `--case-list-file <path>`: newline-delimited case IDs to include (repeatable).
  Blank lines and `# comments` are ignored. Useful for targeted reruns.
- `--pytest-output-tail-lines <int>`: persist only the latest N lines in `pytest_before_*` / `pytest_after_*` artifacts (default `4000`, `0` disables line truncation; a hard per-file cap of `250000` chars still applies to protect disk usage).
- `--snapshot-artifact-max-string-chars <int>`: per-string truncation cap for persisted `snapshot.json` artifacts (default `16000`, `0` disables); patcher proposal calls still receive full snapshot context for compatibility.
- `--container-image <image>`: image used by `testcontainers` executor.
- `--container-network none|bridge`: container network mode (`none` is default and recommended).
- `--container-mount-repo-src` / `--container-no-mount-repo-src`: mount local `src/` for plugin imports.
- `--container-env KEY=VALUE`: explicit env allowlist passed into container (repeatable).
- stale eval containers are pruned automatically at run start (best-effort, 5-minute grace).
- `--openai-api-key <str>`: API key for OpenAI-compatible patcher; defaults to `Z_API`, then `OPENAI_API_KEY`.
- `--transport-fail-fast-streak <int>`: abort run after N consecutive terminal transport/backend failures (transport request failures, retryable HTTP transport statuses, or crash-like runtime errors such as segfault signatures; default `5`, `0` disables).
- `--case-fail-fast-on-request-budget-exceeded`: stop retrying the current case/condition as soon as an attempt returns `request_budget_exceeded` (default: enabled). Useful for conserving GPU time when per-attempt budget overruns are unlikely to recover.
- `--no-case-fail-fast-on-request-budget-exceeded`: disable that per-case fail-fast behavior when you want retries to continue (for example, adaptive rescue runs).
- `--openai-max-propose-sec <float>`: per-attempt shared wall-clock budget for OpenAI Stage-B propose calls (initial + context/no-diff recovery), where each call still uses its own internal retries/fallbacks (default `300`; set `0` to disable cap).
- `--openai-transport-max-retries <int>`: max immediate retries inside one OpenAI request for
  `request_failed` and retryable HTTP statuses (`408/409/425/429/5xx`) (default `6`).
- `--openai-transport-retry-backoff-base-sec <float>` and
  `--openai-transport-retry-backoff-max-sec <float>`: exponential backoff profile for those
  transport retries (defaults `1.0` and `8.0` seconds).
- `--openai-transport-recovery` / `--no-openai-transport-recovery`: runner-managed outage
  recovery for OpenAI-compatible backends (default: enabled). When a high-level OpenAI
  operation still ends in a terminal transport/backend-unavailable failure after its internal
  retries, the runner emits a non-scoring `infra_event` row, waits for readiness, and replays
  the same operation once.
- `--openai-transport-recovery-max-wait-sec <float>` and
  `--openai-transport-recovery-poll-sec <float>`: outage wait budget and readiness probe
  polling interval for that recovery loop (defaults `60` and `2.0` seconds).
- readiness probe order is `GET <origin>/health`, then `GET <base_url>/models`; any `2xx`
  response counts as healthy.
- per-attempt metadata records retry telemetry (`transport_retry_count`,
  `transport_retry_attempts`, and effective retry/backoff settings).
- `--openai-context-window-tokens <int>`: optional OpenAI-compatible context window guard for Stage-B prompt preflight (default `0` disables guard). Useful for LMStudio-like endpoints that return context-length HTTP 400s.
- `--openai-auto-context-window`: opt-in model metadata probe for context window tokens before execution starts (falls back to disabled guard on probe failure).
- probe adapter order is LM Studio REST (`/api/v1/models`, then `/api/v0/models`) followed by generic OpenAI-compatible `/models`.
- manual `--openai-context-window-tokens` always takes precedence over `--openai-auto-context-window`.
- with a known window, the run refuses to start when `window - --openai-max-tokens - 1024` (the reserve) is below 1,024 prompt tokens (`OpenAIPromptBudgetError`, LLM-06); previously the prompt cap collapsed to one character and every case looked like a model failure.
- `--openai-response-mode edits|diff|full_file`: Stage-B patch contract for the OpenAI patcher. `full_file` asks for the complete repaired contents of one target source file and diffs locally; in v1 it only applies when the copied case workdir has exactly one non-test `.py` file, otherwise it falls back internally to `diff`.
- `--openai-compat-profile auto|qwen_openai_raw|qwen_dashscope|lmstudio_model_yaml|off`: request-shaping profile for the internal OpenAI patcher. Use `qwen_openai_raw` for raw Qwen endpoints that return reasoning without final `message.content`; it sends `chat_template_kwargs.enable_thinking=false` on each request.
- `--openai-response-mode-autoswitch off|retry_to_diff`: opt-in retry policy for OpenAI patcher internal syntax/sanitize cycles (`retry_to_diff` switches `edits` → `diff` after retry-triggering failures).
- `--context-protocol monolithic|staged`: prompt protocol. `staged` adds a Stage-A evidence request pass before patch generation.
- `--context-stage-a-max-requests <int>`: maximum Stage-A retrieval requests (default 6).
- `--context-stage-a-max-chars <int>`: maximum Stage-A returned context size in characters (default 120000).
- Stage-A request schema (STRICT JSON, produced by the patcher and executed by the runner):
```json
{
  "diagnosis": "one-sentence root-cause hypothesis",
  "missing_fact": "specific fact needed, or null",
  "need_more_context": true,
  "need_snapshot": false,
  "need_compact_context": false,
  "requests": [
    {"type": "file", "path": "relative/path.py"},
    {"type": "search", "path": "relative/path.py", "pattern": "literal_text"},
    {"type": "frame", "index": 1}
  ],
  "snapshot_justification": "one short sentence (only when need_snapshot=true)",
  "compact_context_reason": "one short sentence (only when need_compact_context=true)",
  "sufficiency_rationale": "one short sentence"
}
```
- malformed Stage-A booleans fail closed to `false`; only `true/false/1/0/yes/no/on/off` and numeric `0/1` are accepted.
- invalid Stage-A `frame` / `local_var` request entries are ignored rather than repaired to frame `0`.
- `--retry-prompt-mode full|delta|auto`: retry packaging mode (`auto` defaults to `delta` for `layered_adaptive_llm`).
- adaptive conditions keep full retry prompts by default; use `--retry-prompt-mode delta` only when explicitly evaluating delta-first adaptive retries.
- `--retry-context-mode fresh|history`: retry context policy (`history` includes prior attempt transcript; default `history`).
- `--retry-history-max-chars <int>`: max characters for retry-history transcript (default 48000).
- `--prompt-retry-reflection` / `--no-prompt-retry-reflection`: enable a retry-only reflection block that asks why the previous attempt failed and what changed for the new retry. This is a legacy native-OpenAI ablation knob and defaults to `off`.
- `--prompt-adaptive-stage-b-cot` / `--no-prompt-adaptive-stage-b-cot`: enable CoT-style Stage-B reasoning scaffolding for adaptive conditions. This is a legacy native-OpenAI ablation knob and defaults to `off`. It is not equivalent to `--baseline cot_repair --conditions layered_adaptive_llm`, because the shorthand additionally preserves the native first-attempt planner and repair-plan injection.
- `--prompt-repair-method-scaffold` / `--no-prompt-repair-method-scaffold`: add a teacher-style modification-method scaffold to the Stage-B patch prompt. This is a legacy native-OpenAI ablation knob and defaults to `off`.
- `--prompt-retry-plan-then-patch` / `--no-prompt-retry-plan-then-patch`: on retries, run an extra planning call before the final Stage-B patch call. This currently requires `--patcher openai`, belongs to the legacy native path, and is disabled by default. This knob is separate from the built-in first-attempt planner used by `cot_repair`.
- `--prompt-budget-total-chars <int>`: Stage-B prompt total character budget (0 disables cap).
- `--prompt-budget-retry-delta-chars <int>`: retry packet per-section cap.
- `--prompt-budget-stdout-chars <int>`: pytest output per-section cap.
- `--prompt-budget-snapshot-chars <int>`: snapshot section cap.
- `--prompt-budget-evidence-chars <int>`: Stage-A evidence section cap.
- `--prompt-budget-retry-history-chars <int>`: retry-history section cap (default 12000).
- `--context-compaction-mode off|auto_overflow|always`: delegated context compaction policy for Stage-B (default `auto_overflow`).
- `--context-compaction-scope snapshot,pytest,files`: raw sections eligible for delegated compaction.
- `--context-compaction-max-input-chars <int>`: max raw characters sent to compaction calls.
- `--context-compaction-target-chars <int>`: target compacted context size in characters.
- `--context-compaction-max-calls <int>`: max compaction calls per unit (default `1`, `0` disables).
- `--conditions base_repair,cot_repair,layered_adaptive_gated,layered_adaptive_llm`:
  run explicit A/B + adaptive comparisons (default remains `base_repair,layered_adaptive_llm` for continuity).
- `--adaptive-gate-policy aggressive|balanced|conservative`: helper trigger policy for `layered_adaptive_gated`.
- `--adaptive-max-helper-calls <int>`: helper-call cap per unit (default `1`, `0` means unlimited).
- `--adaptive-budget-topup-max-calls <int>`: one-time helper budget top-up cap (default `1`, `0` disables top-ups). Top-ups only trigger under repeated no-progress failures after prior `need_more_context=true`.
- `--adaptive-stagnation-window <int>`: repeated-signature window for gate activation.
- `--adaptive-saturation-threshold <float>`: disables helper when low-yield helper ratio is too high.
- `--adaptive-no-progress-token-cap <int>`: cumulative no-progress token budget for adaptive hard-stop (`150000` default, `0` disables).
- `--adaptive-helper-budget-profile default|compact`: prompt budget profile when helper is invoked.
- `--adaptive-openai-helper-response-mode inherit|edits|diff|full_file`: per-helper-attempt OpenAI response-mode override (default `inherit`). `full_file` uses the same single-source-file-only v1 semantics as the top-level `--openai-response-mode` flag.
- `--adaptive-informativeness-pregate`: enables rule-based traceback informativeness pre-gate for adaptive conditions.
- `--adaptive-informativeness-pregate-threshold <float>`: pre-gate threshold in `[0.0, 1.0]` (default `0.7`).
- `--adaptive-online-policy-mode off|shadow|on`: online learning mode for `layered_adaptive_llm`.
- `layered_adaptive_llm` now includes a generalized stagnation override: when helper usage is repeatedly skipped under no-progress signatures, helper can be forced on the next retry (with one optional budget top-up if configured).
- `--adaptive-online-policy-algo linucb|thompson`: online policy algorithm.
- `--adaptive-budget-policy-algo linucb|thompson`: budget policy algorithm.
- `--adaptive-tep`: enable TEP-inspired stagnation nudges in retry prompts.
- `--adaptive-tep-equilibrium`: enable TEP-inspired retry-plan stabilization; this reruns planning until the plan text stabilizes or the configured cap is reached, but it is not full equilibrium propagation.
- `--adaptive-tep-mode off|legacy|full`: `legacy` preserves the old nudge/stability loop; `full` enables the native-only prompt-state optimizer with persisted mutable prompt surfaces. This mode does not replace helper or budget policy learning.
- `--adaptive-tep-state-path <path>` / `--adaptive-tep-init-state-path <path>`: persist or warm-start the full-mode TEP prompt state (default `evals/artifacts/policies/<run_id>.adaptive_tep_state.json`).
- `--adaptive-tep-beta-init <float>` / `--adaptive-tep-beta-decay <float>`: initial nudge strength and accepted-update decay for full mode (defaults `1.0` / `0.9`).
- `--adaptive-tep-full-equilibrium-max-iters <int>` / `--adaptive-tep-full-equilibrium-window <int>` / `--adaptive-tep-full-equilibrium-variance-threshold <float>`: full-mode local equilibrium controls (defaults `20`, `3`, `0.5`).
- `--adaptive-tep-validation-buffer-size <int>`: rolling per-node validation buffer for full-mode prompt updates (default `4`).
- `--adaptive-online-alpha <float>`: LinUCB exploration coefficient.
- `--adaptive-online-epsilon <float>`: epsilon-greedy exploration probability.
- `--adaptive-online-l2 <float>`: LinUCB regularization strength.
- `--adaptive-online-warmup-decisions <int>`: uniform warmup decisions before policy exploitation.
- `--adaptive-online-save-every <int>`: persist policy state every N updates.
- `--adaptive-online-feature-scaling-profile none|v1|v1_interactions|percentile_v1`: static feature scaling profile for online LinUCB (default `v1`). `v1_interactions` adds three helper-policy interaction terms (`ix_*`, products of `v1`-scaled factors; POL-19); `percentile_v1` keeps the same feature set and fits each count-like feature's cap once from its first 30 observations, then freezes it (POL-09).
- `--adaptive-online-action-score-biases <pairs>`: additive helper-action score priors as `action=bias` pairs (default `frame_only=-0.05,search_only=-0.10`; an empty string disables them). Each bias must be within ±0.1 of the success reward (decision D6, POL-07); larger values used to act as a veto on the arm.
- `--adaptive-online-init-state-path <path>`: warm-start online controller weights from an existing compatible state JSON while still saving to the run-scoped `--adaptive-online-state-path`. This is for warm starts, not exact run continuation. A state trained on a feature the run no longer produces is discarded at the first decision (`FeatureSetMismatch` load error, `state_resets` in the policy events; POL-21); the budget controller applies the same rule to its gate and family states.
- `--adaptive-online-state-path <path>`: policy state JSON path (default `evals/artifacts/policies/<run_id>.json`).
- `--adaptive-online-events-path <path>`: policy event log JSONL path (default `evals/results/<run_id>.adaptive_policy.jsonl`).
- `python -m evals.bootstrap_policy_state --events <jsonl> --output-state <json>`: reconstruct a warm-start compatible controller state from historical event logs when a saved state JSON is unavailable. Case-label features are dropped from replayed logs unless `--keep-case-label-features` is given. Exclude the cases (or runs) the warm-started run will be evaluated on with `--exclude-case-ids` / `--exclude-run-ids` (comma-separated or `@file`); `--target-case-ids` warns when replayed cases overlap them (POL-15). `percentile_v1` state is refused because its scaling caps can't be rebuilt from events (POL-14).
- `--adaptive-online-include-case-labels` / `--no-adaptive-online-include-case-labels`: feed the `case.json` difficulty/snapshot-dependency/bug-source labels to the helper policy as features. Off by default since 2026-10: the labels are not observable at deployment time.
- `--adaptive-lite-snapshot-mode replace|supplement`: with `--adaptive-reveal-mode full`, `replace` (default) makes `snapshot_lite`, `frame_locals_lite` and `snapshot_full_lite` replace the full snapshot section in the patch prompt; `supplement` restores the old behavior of adding them on top. Rows record `adaptive_lite_replaced_snapshot`.
- `--adaptive-decision-model-role off|shadow|policy|features|prior`: consult a SystemOne-compatible System-1 decision model (for example Cloudflare clef-flash served by llama.cpp through llama-swap, or the `evals/docker/decision_model/` PyTorch sidecar) for `layered_adaptive_llm` helper decisions. `shadow` logs only; `policy` lets it select; `features` adds six diagnostic LinUCB features; `prior` guides warm-up and adds a fading score prior. `policy`/`features`/`prior` require `--adaptive-online-policy-mode on`. See `docs/adaptive_online_policy_design.md`.
- `--adaptive-decision-model-url <url>` (default `$TRACEGATE_DECISION_MODEL_URL` or `http://127.0.0.1:8090`), `--adaptive-decision-model-api-key <key>` (default `$TRACEGATE_DECISION_MODEL_API_KEY`; bearer token for gateways such as llama-swap, redacted in lockfiles and rejected in config files), `--adaptive-decision-model-name <name>` (`clef-flash`), `--adaptive-decision-model-profile evidence_v1|evidence_v1_noaux|evidence_v1_text|evidence_v1_nohints|evidence_atomic_v1`, `--adaptive-decision-model-timeout-sec <float>` (`10`), `--adaptive-decision-model-temperature <float>` (`0` = argmax), `--adaptive-decision-model-prior-beta` / `--adaptive-decision-model-prior-anneal-updates` / `--adaptive-decision-model-warmup-mix` (`0.5` / `50` / `0.3`), `--adaptive-decision-model-log-state full|hash` (`full` writes rendered states to `<artifacts_root>/decision_model_states.jsonl`). The sidecar is health-checked before the first unit runs, and what it reports about the served release is recorded as `patcher_config.adaptive_decision_model_serving` (see the resume guardrails above). Results carry `adaptive_online_dm_*` fields and policy events schema v5 carries a `decision_model` block.
- `--prompt-order-mode current|crash_first|evidence_last`: reorder the shared Stage-B prompt sections without changing evidence membership.
- `--adaptive-reveal-mode full|traceback_first`: either send the policy-granted adaptive evidence immediately or start from traceback/failure-signature/snapshot-index-only prompting and reveal more only on explicit triggers. Under `full`, the full snapshot is in the prompt only when the executed helper action grants it (ATT-06, D5): rows record `adaptive_snapshot_requested`, `stage_b_snapshot_in_prompt` and `evidence_chars_by_type`.
- `--snapshot-frame-pruning-mode off|heuristic`: heuristically trim prompt-facing snapshot frames while always keeping the crash frame and adjacent user-code frames.
- Adaptive helper attempts now persist normalized helper telemetry in eval rows, including `adaptive_helper_outcome`, `adaptive_helper_failure_kind`, `adaptive_helper_retry_count`, and `adaptive_helper_transport_retry_count`.
- Adaptive helper attempts also persist action telemetry such as `adaptive_helper_action_selected`, `adaptive_helper_action_executed`, `adaptive_helper_action_fallback_reason`, and `adaptive_snapshot_lite_included`.
- `layered_adaptive_llm` can now choose cheaper targeted actions before patching: restricted Stage-A requests (`frame_only`, `file_only`, `search_only`), snapshot-index-only context (`snapshot_lite`), a forced compaction pass (`compact_then_patch`), or an OpenAI-only strict-JSON helper retry (`retry_helper_strict_json`).
- In `layered_adaptive_llm`, helper transport/backend failures and invalid helper payloads still log policy events, but they use explicit `adaptive_online_update_reason` values and do not update the online model state.
- `--adaptive-budget-policy-architecture tier_binary|hierarchical_v1|bwk_v1`: choose between scalar tier escalation, family-wise escalation, or a token-budgeted BwK-style controller state.
- Shared adaptive defaults now use the more conservative ladder `1.0,1.1,1.2,1.35,1.5` with `--adaptive-budget-policy-max-ups-per-case 1`.
- Under `tier_binary`, the dense ladder uses coarse promotion targets rather than literal `+1` rung stepping: `1.0 -> 1.2 -> 1.5 -> max tier` (falling back to the next larger configured tier when a target rung is absent).
- `hierarchical_v1` is an experimental alternative, not a strict upgrade over `tier_binary`: it can make finer-grained budget-family moves, but it still does not learn `response_mode` and it adds a second learned decision.
- `--adaptive-budget-init-state-path <path>`: warm-start the budget controller from a compatible prior state artifact while still writing the current run to `--adaptive-budget-online-state-path`. This reuses learned controller state only; under `bwk_v1`, run-scoped remaining-budget and dual-variable telemetry reset from the current run config rather than carrying over prior-run exhaustion.
- `--adaptive-budget-bwk-run-token-budget <int>`: required when `--adaptive-budget-policy-architecture bwk_v1`; logs remaining token budget plus a dual-variable trace in budget-policy metadata/events. `bwk_v1` paces spending to `budget / (cases × k)` tokens per decision: after each outcome the dual variable moves by `0.05 × (tokens − target) / target` (never below 0), and each gate arm's score is reduced by `λ × ĉ_k / target`, where `ĉ_k` is that arm's mean tokens per attempt (the target until observed). The dual variable used to be logged but never read, and its update stayed 0 until the budget was nearly exhausted (POL-04). Events carry `bwk_target_tokens_per_decision` and `bwk_cost_estimates` (budget event schema 7).
- `--adaptive-budget-feature-scaling-profile none|v1|v1_interactions|percentile_v1`: static feature scaling profile for budget LinUCB (default `v1`). In practice, `percentile_v1` is the main non-default budget variant today; the budget feature builder does not currently add extra interaction terms.
- `--adaptive-reward-lambda-tokens <float>` / `--adaptive-reward-mu-latency <float>`: reward penalties for token and latency cost.
- `--adaptive-reward-token-scale <float>` / `--adaptive-reward-latency-scale <float>`: normalization scales for reward penalties.
- `--adaptive-reward-eta-progress <float>` / `--adaptive-reward-kappa-stagnation <float>`:
  progress bonus and no-progress streak penalty terms for adaptive reward shaping.
- Adaptive online policy feature vectors include trajectory, helper-budget/top-up, lagged helper/compaction outcomes, case priors, and pregate exception flags in addition to the base pre-gate signals.
- When loading an existing online policy state, newly observed feature keys are auto-expanded into LinUCB matrices (existing learned weights are preserved).
- `--syntax-preflight-enabled` / `--no-syntax-preflight`: enable/disable Python syntax preflight (`py_compile`) after patch apply and before test rerun (default enabled).
- `--syntax-preflight-max-cycles <int>`: max internal syntax-repair cycles per logical attempt (default `2`); these cycles do not consume extra retry attempts.
- `--syntax-preflight-fallthrough` / `--no-syntax-preflight-fallthrough`: when enabled (default), preflight exhaustion falls through to test execution instead of terminating the attempt. Normalizes comparison with baselines that self-validate internally.
- `--patch-sanitize-enabled` / `--no-patch-sanitize`: enable/disable malformed patch-text sanitation guard before patch apply/syntax preflight (default enabled).
- `--patch-verifier-scope disabled|adaptive_only|all`: patch-verifier execution scope (default `all`; `adaptive_only` is an ablation, ATT-14/D7).
- `--patch-verifier-min-score <int>`: minimum relevance score required to pass patch verifier (default `2`).
- `--patch-verifier-infer-test-source-scope` / `--no-patch-verifier-infer-test-source-scope`: enable/disable patch-verifier source-scope inference from failing/crash test imports (default enabled).

### Adaptive workflow (tool-like evaluation)

Compare gate-driven vs model-discretion helper usage with the same run settings:

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --conditions layered_adaptive_gated,layered_adaptive_llm \
  --k 3 \
  --adaptive-gate-policy balanced \
  --adaptive-max-helper-calls 1 \
  --retry-prompt-mode full \
  --adaptive-helper-budget-profile compact \
  --adaptive-openai-helper-response-mode inherit
```

### Adaptive Probe Presets

The shipped probe presets under `evals/configs/` are thin, versioned wrappers
around the adaptive template for the roadmap experiments:

- `openai_adaptive_probe_decay_098.template.toml` and
  `openai_adaptive_probe_decay_0995.template.toml`: compare against
  `openai_adaptive.template.toml` for decay sweeps.
- `openai_adaptive_probe_prompt_order_crash_first.template.toml` and
  `openai_adaptive_probe_prompt_order_evidence_last.template.toml`: prompt
  ordering ablations.
- `openai_adaptive_probe_current_knobs_compact.template.toml`: current-knobs
  evidence-tightening probe using only existing caps and compaction controls.
- `openai_adaptive_probe_warm_start.template.toml`: explicit init-state reuse.
- `openai_adaptive_probe_bwk_v1.template.toml`: explicit-token-budget
  `bwk_v1` smoke path.
- `openai_adaptive_probe_decision_model_{shadow,policy,policy_atomic,features,prior}.template.toml`:
  System-1 decision-model roles (`policy_atomic` uses the `evidence_atomic_v1` profile). They
  need the decision-model sidecar (`just decision-model-up`); run `shadow` first and compare
  the decision-model OPE targets before live roles.

These presets intentionally pin `jobs = 1` so online-controller event logs stay
case-ordered and directly comparable.

Warm-start bootstrap example:

```bash
uv run python -m evals.bootstrap_policy_state \
  --events evals/results/<prior_run>.adaptive_policy.jsonl \
  --exclude-case-ids @<target_case_list>.txt \
  --output-state evals/artifacts/policies/<prior_run>.bootstrapped-online.json

uv run python -m evals.bootstrap_policy_state \
  --policy-kind budget \
  --events evals/results/<prior_run>.adaptive_budget_policy.jsonl \
  --output-state evals/artifacts/policies/<prior_run>.bootstrapped-budget.json
```

Enable online policy learning (`shadow` logs only; `on` updates state):

```bash
uv run python -m evals.run_eval \
  --patcher openai \
  --openai-base-url http://127.0.0.1:1234 \
  --openai-model <model> \
  --conditions base_repair,layered_adaptive_llm \
  --jobs 1 \
  --adaptive-informativeness-pregate \
  --adaptive-online-policy-mode on \
  --adaptive-online-policy-algo linucb \
  --adaptive-online-alpha 0.8 \
  --adaptive-online-epsilon 0.05 \
  --adaptive-online-state-path evals/artifacts/policies/run_online.json \
  --adaptive-online-events-path evals/results/run_online.adaptive_policy.jsonl
```

For strictly online final benchmarks with the two adaptive controllers enabled,
also keep budget policy explicit:

```bash
uv run python -m evals.run_eval \
  --config evals/configs/openai_adaptive.template.toml \
  --run-id <run_id> \
  --jobs 1 \
  --adaptive-online-policy-mode on \
  --adaptive-budget-policy-mode on
```

## Analyze and report uplift

Quick single-run core summary:

```bash
just eval-summary <run_id>

# Equivalent direct CLI:
uv run python -m evals.eval_summary <run_id>
```

This summary command reads `evals/results/<run_id>.jsonl` and enforces strict
schema validation (`schema_version=7`; `schema_version=6` and legacy
`schema_version=5` records are still accepted).

```bash
uv run python -m evals.analyze_results evals/results/*.jsonl --out evals/results/summary.json

# Optional: tighten/relax DR support requirements while keeping IPS/SNIPS unchanged.
uv run python -m evals.analyze_results \
  evals/results/*.jsonl \
  --adaptive-ope-min-action-support 2 \
  --adaptive-ope-propensity-floor 0.005 \
  --out evals/results/summary.json
```

`evals.analyze_results` now enforces strict schema validation for current
`schema_version=7` records and still accepts `schema_version=6` and legacy
`schema_version=5` outputs (the pre-xxhash digest format). Older harness versions remain
unsupported and should be regenerated with the current harness before analysis.

#### Record schema versions

- **7** (2026-10-04 code review remediation; see
  `docs/2026-10-04-code-review-findings.md`). Records written after the review fixes. Changes
  in record meaning versus v6: baselines no longer receive case descriptions or labels
  (`patch_meta.case_info_policy = "neutral_v1"`, ATT-02/BAS-07); success is judged against
  the pristine test surface and pytest's JUnit counts, so patches that edit tests or add
  `conftest.py` hooks no longer count (ATT-01); the patch verifier runs for every condition by
  default (ATT-14); `--force-patch-attempt` defaults to off and a forced marker recovery stays
  `patch_apply_failed` (ATT-07); `patch_apply_mode` distinguishes strict from relaxed patch
  application (ATT-10); adaptive evidence follows the policy: no fallback or seed bundle for
  restricted actions, and the full snapshot only when the action grants it (ATT-03, ATT-04,
  ATT-06; new `seed_bundle_chars`, `evidence_chars_by_type`, `adaptive_snapshot_requested`,
  `stage_b_snapshot_in_prompt`); runs with both adaptive conditions train separate budget
  controllers with derived seeds (POL-05, POL-23); baseline rows with
  `baseline_supported=False` are excluded from that baseline's rates and paired tests (BAS-01).
  A unit that raises writes a terminal `unit_runtime_error` failure record (ATT-08); a retry-gate
  block is a non-scored `infra_event` and the attempt is retried once (ATT-05). A completion
  cut off at `max_tokens` is recorded as `note="truncated_response"` (patcher error
  `truncated_response`) instead of being repaired into a patch or recorded as
  `no_patch_proposed` (LLM-02), and model diffs are taken from every fenced block with
  reasoning stripped, the last diff per file winning (LLM-01, LLM-04). Run metadata records
  `llmdebug_snapshot_schema_version`, the debug-session schema the pytest plugin wrote for
  snapshot-based conditions, and the settings fingerprint includes it, so resuming a run
  across a snapshot-format change counts as settings drift.
  Later review fixes extend this list.
- **6**: xxh3_128 prompt/evidence hash fields.
- **5**: legacy digest fields; still readable.
Current runs may also contain non-scoring telemetry rows with
`record_type="infra_event"` and `note="infra_transport_failure"`. These stay in the main
results JSONL for outage diagnostics, but resume logic, selection tools, and analysis ignore
them for attempt counting, terminal-row choice, and success-rate computation. Baseline-backed
infra interruptions also carry `infra_resume_eligible=true` plus normalized
`infra_failure_kind` / `infra_source` metadata so the unit can be resumed fairly instead of
counting as an ordinary failed attempt.
Runtime accounting now records canonical per-attempt wall-clock time as
`attempt_wall_clock_sec` (non-negative number) in each JSONL row. The analyzer
prefers this field when present across all attempts for a case-condition unit and
falls back to legacy cumulative/before/after duration fields for backward compatibility.
When resumed runs append duplicate attempt rows for the same
`(run_id, case_id, condition, attempt)`, the analyzer resolves them with
**last-write-wins** and reports `duplicate_attempt_rows_seen` /
`duplicate_attempt_rows_resolved` under input diagnostics.
For adaptive OPE, DR reward models are fit separately per
`*_feature_scaling_profile`; mixed-profile buckets still report IPS/SNIPS on the pooled
samples and now expose the contributing `feature_scaling_profiles` in the summary.
Adaptive OPE now also reports the analysis-only propensity floor used for IPS/SNIPS/DR,
the raw/trimmed minimum propensities, trimmed-sample counts, and the maximum resulting
importance weight.
Adaptive reward telemetry now uses `helper_credit_v2` / `budget_credit_v2` scopes while
remaining backward-compatible with legacy `*_local_v1` runs during analysis. Attempt
records also expose `adaptive_progress_score`, per-component
`adaptive_progress_component_*` fields, parsed pytest failure totals, pipeline-stage
labels/ordinals, and helper delayed-credit resolution metadata such as
`adaptive_online_reward_delayed_rescue_bonus` and
`adaptive_online_reward_credit_horizon_attempts`.
Aggregate adaptive summaries now include reward-distribution diagnostics
(`*_reward_std`, `*_reward_min`, `*_reward_p50`, `*_reward_p90`, `*_reward_max`)
for both the online and budget policies.
For adaptive budget policy, `increase_budget` now means “move to a broader control
profile,” not only “raise propose time.” Applied tiers can also widen OpenAI
`max_tokens`, helper-call allowance, Stage-A request/char budgets, retry-history
budget, compaction allowance, and, on the highest OpenAI tier, `response_mode`.
Attempts record the selected/applied/default control dicts via
`adaptive_budget_*_control_values`, per-control change flags, partial-override
telemetry, and flat `adaptive_budget_effective_*` values.

The analyzer reports:
- overall metrics,
- per-category metrics,
- per-difficulty metrics,
- per-snapshot-dependency metrics,
- per-bug-source metrics,
- per-contamination-risk metrics,
- explicit `uplift_success_rate = layered_adaptive_llm - base_repair`,
- `uplift_structured_vs_snapshot`,
- `uplift_structured_vs_traceback`,
- pairwise condition uplift matrix (`pairwise_uplifts`) across all discovered conditions,
- `retry_gate_block_rate`,
- prompt-size metrics including `mean_prompt_chars_stage_b` and per-condition means,
- retry/evidence packaging telemetry (`retry_prompt_mode`, evidence-ID counts, prompt-budget fields),
- attempt-efficiency telemetry (`mean_attempts_budget`, `mean_attempt_ratio_when_successful`,
  `solve_on_attempt_distribution`, `mean_attempts_when_failed` where failed
  cases now use observed executed attempts rather than configured budget),
- token telemetry (`mean_tokens_prompt`, `mean_tokens_completion`, `mean_tokens_total`,
  `mean_tokens_reasoning`, `mean_tokens_per_attempt`, `mean_reasoning_tokens_per_attempt`,
  `mean_llm_calls_per_case`, plus `reasoning_token_covered_cases` /
  `reasoning_token_coverage_rate` for covered-only reasoning-token reporting),
- baseline-only telemetry for external baseline conditions (`ldb`,
  `intervenor`, `tracecoder`, `lambda_rlm`, `rlm`) including timing/flow/feedback
  summaries; sparse
  baseline rates are reported over covered cases with explicit coverage counts,
  and external baseline token/call means can understate true cost when provider
  usage is missing. This is especially relevant for timeout-backed TraceCoder rows:
  if the adapter times out before a parseable CSV usage report is written, the
  terminal row is recorded with missing usage and aggregate token means become a
  lower bound rather than a full accounting,
- adaptive helper telemetry (`helper_invocation_case_rate`, `helper_success_when_invoked`,
  `helper_rescue_rate`, `helper_need_more_true_rate`, budget/saturation guard rates),
- adaptive online-policy telemetry (`adaptive_online_decisions`,
  `adaptive_online_selected_invoke_rate`, `adaptive_online_applied_invoke_rate`,
  per-action selected/applied/behavior count dictionaries,
  `adaptive_online_update_rate`, `adaptive_online_mean_reward`,
  `adaptive_online_mean_propensity`, `adaptive_online_mean_progress_score`,
  delayed-rescue bonus counts/means, and dominant `adaptive_online_progress_reason_counts`),
- adaptive budget-policy telemetry (`adaptive_budget_mean_reward`,
  `adaptive_budget_mean_propensity`, `adaptive_budget_mean_progress_score`,
  request-budget-exceeded rates, per-control change rates, partial-override rates,
  and mean effective control values),
- adaptive off-policy estimators (`adaptive_online_ope`, `adaptive_budget_ope`) with
  IPS/SNIPS/DR targets and effective sample size diagnostics,
- analyzer reproducibility metadata (`analysis_config.adaptive_ope_min_action_support`),
- retry-history telemetry (`mean_retry_history_chars`, `mean_retry_history_turns`,
  `retry_history_budget_applied_rate`, `retry_history_truncation_rate`).

Export online-policy logs to a flat training dataset:

```bash
uv run python -m evals.export_policy_dataset \
  --events "evals/results/*.adaptive_policy.jsonl" \
  --output-csv evals/results/adaptive_policy_dataset.csv
```

`evals.export_policy_dataset` enforces strict adaptive policy event schema validation: helper
events must be `schema_version` 4 or 5 and budget events 4. Older artifacts are intentionally
unsupported and should be regenerated with the current harness before CSV export. Version 5
helper events add decision-model columns (`dm_role`, `dm_status`, `dm_top_action`,
`dm_top_prob`, `dm_margin`, `dm_entropy`, `dm_atomic_action`, `dm_traceback_sufficient`,
`dm_fix_confidence`, `dm_latency_ms`, `dm_prior_beta`, ... plus `dm_p_<action>` and
`dm_missing_<option>`); events without a decision-model block leave them empty.

When rows carry decision-model advice, `analyze_results` OPE also reports
`decision_model_greedy`, `decision_model_stochastic` and `decision_model_atomic` targets (the
atomic target re-applies the `evidence_atomic_v1` thresholds to logged aux answers when the run
used another profile), and `shadow_corrected_samples` counts old shadow-mode rows whose behavior
propensity was corrected to 1.0 at read time. Those per-sample targets additionally report the
behavior support of their probability mass (`behavior_supported_target_mass`,
`unsupported_target_mass`, `unsupported_mass_samples`, `deterministic_behavior_samples`,
`estimates_identifiable`) and set `support_warning="unsupported_target_mass"` when target mass
sits on actions the behavior policy never could have taken; see
`docs/adaptive_online_policy_design.md`. Bootstrap intervals are suppressed for estimators
without behavior support (`ope_ci.ips`/`ope_ci.snips` are `null` rather than `[0, 0]`).

### Offline decision-model probe

`python -m evals.decision_model.probe` measures how well the decision model judges our failures
without any repair-model budget. For each case it runs the failing test once with snapshot
capture, builds the attempt-1 decision state exactly like the live harness (redacted, no case
metadata), queries the sidecar, and scores the diagnostic answers against the `case.json`
labels, next to the heuristic pregate on the same cases:

```bash
uv run python -m evals.decision_model.probe --executor local --split dev --jobs 4 --report
uv run python -m evals.decision_model.probe --executor local --dry-run --limit 20 --report  # no sidecar
```

- Cases are split deterministically by case family (the id without `_easy|_medium|_hard`):
  about 1 in 5 families is `test`. Tune profile wording and thresholds on `dev` only.
- `--rename-option-keys` checks option-order bias by renaming the action option ids on the wire.
- The default executor follows `--executor` (default `testcontainers`), so pass
  `--executor local` for local runs, as with `validate_cases`.
- Output: one JSONL record per case under `evals/artifacts/decision_model_probe/` plus a JSON
  report (AUROC against `snapshot_dependency`, Spearman of `fix_confidence` vs difficulty,
  action mix per category, calibration bins, latency).

By default (`--patch-verifier-scope all`), every condition runs the same pre-apply patch
verifier, so it can't favour one condition family (ATT-14, D7); `adaptive_only` restricts it to
the adaptive conditions and is reported as an ablation. Low-relevance patches are rejected
before apply with note
`patch_verifier_rejected` and corresponding
`patch_verifier_*` telemetry in `attempt_metadata` / `patch_meta`. When the rejected patch is
still substantive, adaptive runs now make one cheap same-attempt verifier-repair proposal before
giving up. The recovery result is logged via additive `patch_verifier_recovery_*` fields without
changing the record schema.

### Statistical analysis

Add `--stats` for rigorous significance testing:

```bash
uv run python -m evals.analyze_results evals/results/*.jsonl --stats --out evals/results/summary.json

# Evaluate structured packaging directly against traceback-only:
uv run python -m evals.analyze_results \
  evals/results/*.jsonl \
  --stats \
  --stats-pair-a base_repair \
  --stats-pair-b layered_adaptive_llm
```

The `--stats` flag adds:
- **McNemar's test**: Exact paired binary significance test (binomial on the discordant pairs at any sample size).
- **Cohen's h**: Effect size for proportions (small >= 0.2, medium >= 0.5, large >= 0.8).
- **Bootstrap CI**: 95% percentile interval for uplift via paired bootstrap (10k resamples), seeded with `--bootstrap-seed` (default 42) and recorded in `analysis_config`.
- **Holm-Bonferroni**: Multiple comparison correction for per-category p-values.
- **Power report**: Post-hoc power assessment to gauge result trustworthiness.
- **Pairwise condition matrix**: always reports pairwise A/B summaries for all conditions in the input.

For multi-run (k>1) analysis, use `--aggregation`. The case is the unit: each case gives one
paired observation, and the default `per_run` refuses input with several runs of a case
instead of counting them as independent pairs (STA-02).

```bash
# Majority vote: case succeeds if >50% of runs succeed; an exact tie leaves the case out
uv run python -m evals.analyze_results evals/results/*.jsonl --stats --aggregation majority_vote

# Any success: case succeeds if any run succeeds
uv run python -m evals.analyze_results evals/results/*.jsonl --stats --aggregation any_success
```

### Re-analysis audit of existing results

Results written before the 2026-10-04 review fixes can be checked without re-running anything:

```bash
uv run python -m evals.reanalysis_audit evals/results/<run>*.jsonl \
  --a base_repair --b layered_adaptive_llm \
  [--swebench-run-dir evals/swebench_runs/<id>] [--json audit.json]
```

It prints the headline metrics computed with the corrected analysis next to the same metrics
under the pre-review semantics (unmatched uplift, missing tokens as 0, run-level pairing,
chi-square McNemar, the old power formula, uncorrected pairwise significance), and the data
audits from "Impact on existing results" in `docs/2026-10-04-code-review-findings.md`:
successes that touched tests or pytest config, whole-file fallbacks and seed bundles on
restricted actions, full-reveal runs, policy decay below 1.0, OPE propensity problems,
pre-fix deterministic runs, repeated runs, unmatched case sets, missing token usage, LDB case
counts, retry-gate blocks scored as attempts, forced patches recorded as test failures, record
schema versions, and per SWE-Bench run: `--snapshot-dir`, hints/test IDs in prompts and empty
patches. It only reads the files.

### Paper-ready plain-text report

Use `evals.report_run_metrics` for a narrative, paper-facing summary with
primary endpoint, secondary slices, efficiency, and failure diagnostics:

```bash
uv run python -m evals.report_run_metrics \
  --jsonl "evals/results/*.jsonl" \
  --eval-log results/<run_id>/eval-<run_id>.out
```

Useful report flags:
- `--fixed-budget-token-thresholds 10000,25000,50000`: report fixed-budget
  success rates on matched A/B pairs at explicit total-token thresholds.
- when `tiktoken` is installed via the `evals` extra, report output also
  includes prompt-token estimate coverage/correlation diagnostics and records
  exact token estimates for pre-call prompt size and TOON snapshot savings.
  Without `tiktoken`, these fields degrade to `*_available=false` plus an
  explanatory error string; runs do not fail.

Baseline diagnostics in both `eval_summary` and `report_run_metrics` are now
restricted to the external baseline conditions (`ldb`, `intervenor`,
`tracecoder`, `lambda_rlm`, `rlm`). Coverage-sensitive baseline rates are printed as
covered-case rates with explicit `coverage x/y` counts.

Failure diagnostics metric glossary:
- `http_error_attempts`: attempt-incident counter where
  `patch_meta.error == "http_error"`; retries are included, so this is not a
  terminal failure count by itself.
- `context_overflow_attempts`: subset of attempts where the HTTP response body
  indicates context-size overflow.
- `context_overflow_cases`: case-level count (grouped by `run_id`, `case_id`,
  `condition`) where at least one attempt overflowed.
- `context_overflow_case_successes` / `context_overflow_case_failures`: among
  overflowed cases, whether any attempt eventually succeeded.

### Sample size guidance

| Cases | Discordant pairs (est.) | Power (uplift=0.2) | Recommendation |
|-------|------------------------|---------------------|----------------|
| 30 | ~10 | ~0.30 | Directional signal only |
| 60 | ~20 | ~0.55 | Moderate evidence |
| 100 | ~35 | ~0.75 | Near-publication quality |
| 150+ | ~50+ | ~0.90+ | Publication-grade |

Interpretation guidance:
- positive uplift in `required` buckets suggests llmdebug snapshots provide useful signal,
- near-zero uplift in `none` control buckets is expected and acts as a sanity check,
- positive uplift in a category suggests snapshots help that bug class,
- near-zero uplift suggests no clear advantage yet,
- negative uplift suggests prompt or patch quality regressions for that bucket.
