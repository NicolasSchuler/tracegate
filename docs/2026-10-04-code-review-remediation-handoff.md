# Handoff: remediation plan for the 2026-10-04 code review

> **How to use this file:** point a fresh session at it: "Read `docs/2026-10-04-code-review-remediation-handoff.md` and start Phase 0." It's written as instructions to the engineer or coding agent doing the work.

## Your task

Fix the issues documented in [2026-10-04-code-review-findings.md](2026-10-04-code-review-findings.md), in the order of the phases below. Each finding has a stable ID (for example `ATT-03`) with location, problem and suggested fix. Use these IDs in commits, tests and PR descriptions, and tick them off in this file as you go.

The order puts anything that changes experimental results first. The harness produces results for a paper, so no more data should be collected on top of these defects. Core-library fixes (Phase 4) are independent, so a second agent can do them in parallel on a separate branch.

## Read first

1. `AGENTS.md`: the repo workflow, architecture guardrails and quality commands. If it conflicts with this file, `AGENTS.md` wins.
2. The findings document: at least the Summary, "Impact on existing results", and the sections for your current phase.
3. `CONTRIBUTING.md`, `docs/testing-guide.md`, `docs/quality-map.md`.
4. The area docs for your phase:
   - eval harness: `evals/README.md`, `evals/configs/README.md`, `docs/eval_quickstart.md`
   - baselines: `docs/baseline-adapters.md`, `docs/baseline_contracts.md`
   - adaptive policy: `docs/adaptive_online_policy_design.md`, `docs/ADAPTIVE_POLICY_QUICK_REFERENCE.md`
   - metrics: `docs/METRICS_QUICK_REFERENCE.md`
   - core library: `docs/configuration.md`, `docs/mcp-reference.md`, `docs/cli-reference.md`

## Ground rules

- **Re-verify before fixing.** Line numbers refer to commit `06a5eb5`. Re-locate the code and reproduce the finding, ideally by writing the failing regression test first. If a finding turns out to be wrong or already fixed, don't fix it. Mark it in the tracker with a one-line reason.
- **Small, traceable changes.**
  - One finding, or one tight cluster of findings, per commit.
  - One phase or sub-phase per branch, for example `fix/review-1a-information-parity`.
  - Use Conventional Commits that reference the IDs, for example `fix(evals): stop seed bundle for restricted helper actions (ATT-04)`.
  - Don't push or open PRs unless the user asks.
- **Add a regression test for every fix** (required by AGENTS.md). Prefer tests that go through the real seam over tests that monkeypatch the global the bug is about. ORC-01 and ORC-02 survived because each side of the seam was tested in isolation.
- **Some existing tests lock in current behaviour.** Update them deliberately, in the same commit as the fix, and say so in the commit message:

  | Test | Finding |
  |------|---------|
  | `tests/test_evals_adaptive_conditions.py:1872` | ATT-06 |
  | `tests/test_evals_force_patch_policy.py:832` | ATT-07 |
  | `tests/test_evals_swebench_pro_generate_prompt.py:88` | DAT-01 |
  | `tests/test_evals_container_cleanup.py:56` | ORC-08 |
  | `tests/test_evals_run_eval_helpers_coverage.py:746`, `:883` and `tests/test_evals_main_flow.py:659` | ORC-01, ORC-02 |
  | `tests/test_cli_edge_cases.py:406`, `tests/test_cli.py:1909` | RCA-13 |
  | `tests/test_baseline_{ldb,intervenor,tracecoder}.py` and `tests/test_baseline_diff_roundtrip.py`, the `result_extractor` tests | BAS-18 |
  | `tests/test_evals_swebench_pro_types.py` | DAT structural notes |

- **Make result-affecting changes visible in the data.** When a fix changes what a record means or adds record fields:
  - Bump `EVAL_RECORD_SCHEMA_VERSION`. It's currently 6 and defined twice, in `evals/eval_types.py:120` and `evals/analyze_results.py:29`; collapse those to one definition first.
  - Add new knobs and effective values to `run_metadata`, `patcher_config` and the settings fingerprint.
  - Keep the analysis able to read older records.
  - Never silently change a default that affects comparability with earlier runs.
- **Keep the architecture guardrails:**
  - The three-layer policy enforcement. The Phase 1C fixes tighten it; don't route around it.
  - The RCA split in `src/llmdebug` and thin entrypoints.
  - The `run_eval` compatibility seams listed in AGENTS.md, until Phase 5 deliberately replaces them.
- **Update docs in the same patch** when behaviour, CLI flags, defaults or record fields change. Run `just docs-check`, and `just docs-sync` if generated blocks change.
- **Research decisions belong to the user.** Items tagged D1–D12 need the user's answer, so ask before implementing them. If you must proceed, put the change behind a flag that defaults to the recommended option, and record the flag in run metadata.
- **Don't run long or paid jobs.** That means no eval campaigns, no SWE-Bench generation or scoring, and no Docker image pulls. Use `tests/_openai_stub_server.py` and small local fixture cases. Re-running experiments is the user's call.

## Quality gates

After each sub-phase:

```bash
just lint && just typecheck && just test-quick
# if eval or baseline code changed:
just lint-evals && just typecheck-evals && just test-evals-cov && just eval-coverage-no-regression
# if containers, baseline env wiring, or integration-marked flows changed:
just test-integration
# if capture/serialization hot paths changed (Phase 4):
just bench-capture
just docs-check
```

Before declaring a phase done, run `just check`, and `just quality` for any change to the eval harness, tooling or docs.

## Decisions needed from the user (ask in Phase 0)

| ID | Question | Recommended default | Blocks |
|----|----------|---------------------|--------|
| D1 | Case `description` and `category`: remove them from baseline inputs, or give them to all conditions? | Remove from every model-facing input; keep them in case metadata for analysis. | ATT-02 |
| D2 | SWE-Bench prompts: drop `hints_text` and the FAIL_TO_PASS/PASS_TO_PASS test IDs, or keep them behind a recorded flag? | Drop them by default. Offer opt-in `--include-hints` and `--include-test-ids`, recorded in run metadata. | DAT-01, DAT-02 |
| D3 | LDB arm: run only on its convertible subset (report N), exclude incompatible rows from paired statistics, or widen the converter? | Convertible subset with an explicit N, reported as a separate comparison. | BAS-01 |
| D4 | Baseline generation contract: temperature, completion cap, transport retries, wall-clock deadline, retry context. | Match the native conditions: temperature 0.0, one shared completion cap, the same retries, one deadline at `run_repair`, and neutral retry context (or k=1 for baselines). | BAS-02, BAS-03, BAS-04, BAS-08, BAS-11 |
| D5 | In `adaptive_reveal_mode="full"`, should the policy decide whether the snapshot goes into the Stage-B prompt? Should the `*_lite` actions replace or add to the full snapshot? | The policy decides, and lite replaces. | ATT-06 |
| D6 | LinUCB defaults: decay and score-bias scale. | Decay 1.0, with decay < 1 opt-in and applied per arm. Biases as small priors, at most 0.1 of the reward scale. | POL-01, POL-07 |
| D7 | Patch-verifier scope for headline comparisons. | `all` (or `disabled`); the adaptive-only variant becomes an ablation. | ATT-14 |
| D8 | Default for `--force-patch-attempt`, and how its outcome is recorded. | Keep `note="patch_apply_failed"` with a modifier flag. Consider defaulting to off. | ATT-07 |
| D9 | Policy for the `retry_gate_no_new_evidence` gate. | At most one consecutive block, and a block doesn't consume a scored attempt. | ATT-05 |
| D10 | Test-surface integrity: reject patches that touch tests, or restore pristine test files before verification and record a flag? | Restore pristine test and config files before every verification run, and record `test_surface_modified`. | ATT-01 |
| D11 | MCP `out_dir` confinement: what is the allowed root? | An `LLMDEBUG_MCP_ROOT` environment variable, defaulting to the server's working directory. Read-only tools never create files. | RCA-02 |
| D12 | Benchmark design: ConDefects grading tests the model can see; `snapshot_dependency` labels; a common k for pass@k. | Document the current state now. Holding grading tests out and labelling independently are separate research tasks. Report pass@k at a common k. | DAT-17, DAT-21, STA-13 |

Which existing results to invalidate or re-run is entirely the user's decision. Phase 2.4 produces the audit that informs it.

---

## Phase 0: setup and triage

- [x] Create the branch from `main`, run `uv sync --group dev` and `just check`, and note the baseline state (coverage, failures). *(Branched from the M1 commit `bde82e2` per user decision. Baseline: `just check` green, 7,990 passed, 2 skipped, `src/llmdebug` coverage 99.52%.)*
- [x] Send the user decisions D1–D12 in one message, with the recommended defaults. *(2026-10-04: user accepted all recommended defaults.)*
- [x] Re-run the quick repros below. Note in the tracker any that no longer reproduce. *(All reproduce: ORC-01, CAP-01, CAP-03, BAS-01 = 11/1,828, LLM-01, STA-04 = 0.24, ATT-01 end-to-end.)*
- [x] Turn each Critical finding (ATT-01, ATT-02, BAS-01, POL-01, STA-01, STA-02, DAT-03, CAP-01) into a failing regression test, marked `@pytest.mark.xfail(strict=True, reason="<ID>")` until fixed, so progress is visible.

### Quick repros

Run from the repo root with `PYTHONPATH=. uv run python -c '…'`, or put them in a scratch file.

```python
# ORC-01 / ORC-02: run settings written through the shim never reach run_eval
from evals import eval_main_support as legacy, run_eval
from evals.eval_types import _ActiveRunSettings
legacy._ACTIVE_RUN_SETTINGS = _ActiveRunSettings(deterministic=True, seed=1234)
env = run_eval._base_pytest_env(output_format="json", snapshot_condition=False)
assert "PYTHONHASHSEED" in env  # fails at 06a5eb5
```

```python
# CAP-01 and CAP-03: a big-int local aborts capture; prod redaction keeps secret-named locals
import tempfile
from llmdebug import SnapshotConfig
from llmdebug.capture import capture_exception

def boom():
    big = 2**70          # remove this line to see CAP-03
    password = "hunter2"
    raise ValueError("x")

try:
    boom()
except Exception as e:
    cfg = SnapshotConfig(out_dir=tempfile.mkdtemp(), redaction_profile="prod")
    path = capture_exception("t", e, e.__traceback__, cfg)
    # with `big`: TypeError: Integer exceeds 64-bit range
    # without `big`: "hunter2" in path.read_text() is True
```

```python
# BAS-01: how many cases the LDB converter accepts
import pathlib
from baselines.ldb.case_converter import convert_case
from baselines.adapter_base import BaselineCaseIncompatibleError
ok = 0
for d in sorted(p for p in pathlib.Path("evals/cases").iterdir() if p.is_dir()):
    try:
        convert_case(d, case_id=d.name)
        ok += 1
    except BaselineCaseIncompatibleError:
        pass
print(ok)  # 11 of 1,828 at 06a5eb5
```

```python
# LLM-01: the second file is dropped when prose or a second fence separates diffs
from evals.patchers.openai_compat_patcher import _extract_unified_diff
A = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
     "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n")
B = A.replace("a.py", "b.py")
out = _extract_unified_diff("```diff\n" + A + "```\nAlso:\n```diff\n" + B + "```") or ""
assert "b.py" in out  # fails at 06a5eb5
```

```python
# STA-04: n=200 cases, 70 discordant pairs (n01=50, n10=20), uplift 0.15; true power is about 0.94
from evals.stats import power_report
print(power_report(n_cases=200, n_discordant=70, uplift=0.15))  # reports about 0.24
```

ATT-01 needs a small fixture case instead. Apply, through the real `apply_diff`, a diff that creates a `conftest.py` containing:

```python
def pytest_sessionfinish(session, exitstatus):
    session.exitstatus = 0
```

Then run `run_pytest` with the local executor and confirm the return code is 0 while the test still fails.

---

## Phase 1: result integrity (finish before collecting more data)

**Goal:** every recorded outcome means what it says, and every arm gets comparable inputs and budgets. Sub-phases are ordered by priority; 1A–1D can be split across branches.

### 1A. Information parity and leakage (D1, D2)
- [x] **ATT-02:** apply D1 to `baseline_case_context.json` (`evals/eval_attempt_flow.py:2498`, `:5940`) and to the adapters' converters. Remove fix hints from `description` at import time (`quixbugs_importer.py:334`, `pyresbugs_importer.py:393-396`). Record a `case_info_policy` value in run metadata.
- [x] **BAS-07:** define one case-information contract, covering which files and which test source each arm sees. Implement it in all five adapters, and record each arm's prompt artifacts per record.
- [x] **DAT-01, DAT-02:** apply D2 in both `generate.py` files and update the prompt test.
- [x] **DAT-03:** fail fast when the `llmdebug` condition runs without `--snapshot-dir`. Record `snapshot_dir` and per-instance snapshot hits and misses in `run_metadata.json`. Fix the README examples.
- **Acceptance:**
  - A test renders every arm's model-facing input from one fixture case with a sentinel description. The sentinel is absent from all arms, or present in all of them, per D1.
  - A CLI test covers the missing `--snapshot-dir` failure.

### 1B. Oracle and scoring integrity (D7, D8, D10)
- [x] **ATT-01:** apply D10 before every "after" pytest run.
  - Restore or protect `test_*.py`, `conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg` and `tox.ini`.
  - Determine success from pytest's counts (for example `--junitxml` into the attempt directory): require at least one test collected and zero failed or errored.
  - Record `test_surface_modified`.
- [x] **ATT-14:** set the patch-verifier scope per D7.
- [x] **ATT-07:** apply D8, and re-sync the diff metadata to the diff actually applied.
- [x] **LLM-13:** use one shared constant for the forced-patch marker.
- [x] **ATT-10:** record `patch_apply_mode`.
- [x] **ATT-16:** remove `.rej` and `.orig` files.
- [x] **DAT-04, DAT-09, DAT-13, DAT-11, DAT-19:** SWE-Bench Pro scoring *(infrastructure errors still fail the command loudly after details are persisted; an opt-in exclusion flag was not added because `run_docker_scoring` is at its signature-arity budget, revisit with the Phase 5 config object)*:
  - An empty `fail_to_pass` list becomes an infrastructure error.
  - Timeouts and non-test exits become infrastructure errors, persisted per instance in `eval_results.json`.
  - The cached-result reuse checks the patch hash.
  - Use quoted argv in the entry script.
  - Mirror the official apply sequence.
- [x] **DAT-08:** don't persist predictions whose request failed; re-queue them on `--resume`.
- **Acceptance:**
  - The `conftest.py` exit-status hack is recorded as a failure (end-to-end through `apply_diff` and `run_pytest` with the local executor).
  - Unit tests cover Pro scoring with empty lists, timeouts and non-zero exits.
  - A resume test shows errored predictions are retried.

### 1C. Evidence enforcement (D5)
- [x] **ATT-03:** when a restricted action's filtered request list is empty, return an empty Stage-A context. Add an `allow_fallback` parameter to `execute_context_requests`, and correct `requests_count` and `helper_outcome`. *(Implemented as `deny_fallback_for_restricted_action` at the call sites instead of a parameter: `execute_context_requests` is at its arity budget.)*
- [x] **ATT-04:** gate or scope the seed bundle by action, and record `seed_bundle_chars`.
- [x] **ATT-06:** apply D5. Add `stage_b_snapshot_in_prompt`, and rename the misleading `adaptive_snapshot_included`. *("Lite replaces" landed with M1; `adaptive_snapshot_included` is kept for compatibility but now equals `stage_b_snapshot_in_prompt`, and `adaptive_snapshot_requested` carries the old meaning.)*
- [x] **ATT-20, ATT-21:** validate fallback candidate paths, and make a request budget of 0 mean zero requests.
- [x] **POL-12:** log one consistent scaled feature vector, or none when there was no decision.
- [x] Add per-attempt `evidence_chars_by_type` (snapshot, seed, stage_a, frame, search, file).
- **Acceptance:**
  - A table-driven test over all 10 helper actions and representative request payloads asserts which evidence types reach the Stage-B prompt.
  - No restricted action yields more evidence characters than `invoke_helper` on the same fixture.

### 1D. Run settings and reproducibility
- [x] **ORC-01, ORC-02:** replace the `legacy.<NAME> = ...` writes with setter functions in `run_eval` (or a run-context object; see Phase 5.1). Add an architecture test that bans module-attribute writes through the shim. *(The pool is now opt-in via `--container-pool`: every earlier result ran in one-shot containers, and pooling shares the workdirs mount across units.)*
- [x] **ORC-04:** abort on a settings-fingerprint mismatch unless `--allow-settings-drift` is passed. **ORC-05:** always record the raw seed and include it in the fingerprint.
- [x] **ORC-06:** set `allow_abbrev=False`. **ORC-15:** validate `--conditions`. **ORC-21:** validate `--run-id`.
- [x] **POL-05, POL-23:** create one policy controller per condition. Derive per-unit RNGs from `(seed, case_id, condition, attempt)` and give each controller its own sub-seed. In deterministic mode, either serialize policy updates or document that `--jobs 1` is required.
- **Acceptance:**
  - An integration test goes through `run_main_flow` setup and asserts `PYTHONHASHSEED` is in the pytest environment and `pool.acquire` is used under the testcontainers executor (the pool can be stubbed).
  - A resume test aborts on a fingerprint mismatch.
  - A test shows argparse abbreviations are rejected.

### 1E. Baseline contract (D3, D4)
- [x] **BAS-01:** apply D3, and record the convertible-case count in run metadata.
- [x] **BAS-03, BAS-04, BAS-08, BAS-11, BAS-12:** extend `BaselineConfig` with temperature, seed, a completion cap, retry settings and one deadline. Forward them from `run_eval`, honour them in all five adapters and both upstream patches, and record the effective values per record. *(Partial: INTERVENOR, lambda-RLM and RLM enforce the contract; LDB and TraceCoder get `TRACEGATE_BASELINE_*` env vars and records say `enforced: false`, because editing their upstream patches needs the provisioned repos to verify. The single deadline is enforced in INTERVENOR and measured for every adapter; lambda-RLM/RLM still bound each inner call.)*
- [x] **BAS-02:** retry context per D4.
- [x] **BAS-10, BAS-15:** keep usage and the last candidate on timeouts and errors, and compute metrics on every return path.
- [x] **BAS-05, BAS-14:** record the observed commit plus a post-patch tree hash, and fail setup on mismatch. *(The tree hash is recorded per record; asserting it against a manifest is deferred until the upstream repos are provisioned.)*
- [x] **BAS-13, BAS-16, BAS-17, BAS-19:** RLM smoke gate as telemetry only, LDB model routing, `extra_body` capability check, one OpenAI-compatible defaults factory.
- [x] **ORC-03, ORC-18, ORC-19:** subprocess monitor. Use `start_new_session=True` and `killpg`, a bounded drain after exit, `stdin=DEVNULL`, and no deadline when the timeout is 0. Fix this before any baseline re-run.
- **Acceptance:**
  - A test asserts every adapter receives the same temperature, cap, retries and deadline from one config.
  - A subprocess-monitor test whose child spawns a grandchild returns within timeout plus grace.
  - LDB run metadata includes the convertible count.

### 1F. Record completeness and cost accounting (D9)
- [x] **ATT-08, ORC-10, ORC-17:** write a terminal `unit_runtime_error` record when a unit raises. Flush the deferred record queue in a `finally`. Keep streamed usage in the run totals.
- [x] **ATT-05:** retry gate per D9.
- [x] **POL-02, ATT-09, ATT-13, ATT-15:** count every LLM call: TEP critic and update calls, equilibrium re-plans, and the first call before a transport replay. Stop double-counting compaction tokens.
- [x] **ATT-11, ATT-12:** record the final prompt with per-cycle hashes, and don't delete artifacts that records reference.
- **Acceptance:** using the stub server's request log, the summed usage of every LLM call in a fixture attempt equals the recorded attempt total.

### 1G. LLM output handling that changes outcomes
- [x] First, add an end-to-end diff corpus test. About 15 realistic model outputs go through `_extract_unified_diff`, then `_fix_hunk_counts`, then `apply_diff`, and the test asserts the final file contents. Cover:
  - one fence, two fences, prose between diffs, trailing markdown bullets
  - a CRLF file, blank context lines without the leading space
  - a response truncated with `finish_reason=length`
  - an orphan `</think>` and an unclosed `<think>`
  - a new file created via `---`/`+++` headers
- [x] Diff pipeline: **LLM-01, LLM-02, LLM-03, LLM-04, LLM-07, LLM-09, LLM-10, LLM-15.**
- [x] **LLM-05:** edits schema and application. **LLM-06:** a degenerate prompt cap raises a configuration error. **LLM-08:** heuristic patcher. **LLM-14:** error-reason ordering. **LLM-19:** consistent recovery paths.
- [x] **DAT-18:** record `finish_reason` per SWE-Bench prediction, and raise the default completion cap.
- **Acceptance:** the corpus test passes, and the failure taxonomy has a distinct `truncated_response` reason.

### 1H. Case-suite validity (D12)
- [x] **DAT-05:** make `--validate` real in all importers. **DAT-14:** AST-based determinism check. **DAT-10:** oracle-patch trailing newline. **DAT-15:** pinned, isolated ruff. **DAT-16:** stable IDs; don't rename existing cases without a mapping. **DAT-20:** `literal_eval` for dataset values. (DAT-16: IDs kept as they are; imports now dedupe by `source_id`.)
- [x] **DAT-17, DAT-21:** document the current state in `evals/README.md` now. Further research changes follow D12.
- [x] Run the new determinism check over all of `evals/cases/` and report the flagged case IDs to the user. Full execution validation of the corpus is long, so ask before running it.

**Phase 1 exit criteria:**
- Every Critical finding is fixed or explicitly deferred by the user.
- The new run-metadata fields are present.
- `just quality` passes.
- A short changelog note or `evals/README.md` section describes what changed in record semantics and which schema version introduced it.

---

## Phase 2: statistics and reporting

**Goal:** reported numbers and tests are correct and reproducible, and existing results can be re-analysed with the fixed code.

### 2.1. Single sources of truth
- [x] **STA-19, DAT-24:** one `case_outcome(records)` helper used by `analyze_results`, `report_run_metrics`, `select_cases` and `extract_failed_cases`.
- [x] One canonical condition-name normalizer and condition-family registry.
- [x] **STA-05, STA-07:** one seeded bootstrap with one interval method (choose percentile or BCa and document it). Add `--bootstrap-seed`, recorded in `analysis_config`.
- [x] **STA-23:** use `scipy.stats.binomtest` for an exact McNemar, with α as a parameter. **STA-24:** Holm with `<=`. **STA-15:** a p-value formatter.

### 2.2. Correctness
- [x] **STA-01, STA-09:** missing tokens become `None`; add `token_covered_attempts` and `token_covered_fraction`.
- [x] **STA-02:** use the case as the unit. Aggregate runs per case, or use a cluster bootstrap, and refuse `per_run` pooling across `run_id`s.
- [x] **STA-03:** compare matched case sets only. Emit `cases_expected` and `cases_missing`, and refuse uplifts when the case sets differ. *(Uplifts are always computed over the matched cases and flagged `case_sets_match: false` with `n_matched_cases`, rather than suppressed.)*
- [x] **STA-04:** fix the power formula.
- [x] **STA-06, STA-20:** apply Holm or BH across the pairwise family and the slice family, and label exploratory tests as such.
- [x] **STA-08:** ratio of sums. **STA-10:** NaN κ. **STA-13:** unbiased pass@k at a common k (D12). **STA-14:** majority vote.
- [x] **STA-16:** bucket dilution. **STA-17:** timing keys. **STA-18:** de-duplication in `budget_risk_report`. **STA-21:** OPE parameters. **STA-22:** two-pass labelling. **STA-25:** `--strict-input`.
- [x] **BAS-06, BAS-09:** `compare_results` run collapsing and the unreadable-file crash.

### 2.3. Off-policy evaluation
- [x] **STA-11:** cross-fitted doubly-robust estimate; record the selected alphas. **STA-12:** drop and count samples with a missing propensity.
- [x] Add bootstrap CIs over cases for the OPE estimates.
- [x] Flag shadow-mode logs written before the POL-03 fix (Phase 3) as having biased propensities. *(M1 corrects them to 1.0 at read time; `shadow_corrected_samples` is counted and printed with the estimates.)*

### 2.4. Re-analysis audit, for the user
- [x] Write a tool, for example `python -m evals.reanalysis_audit <jsonl…>`. For a set of result files it should:
  1. recompute the headline metrics with the corrected code and print the change from the previously reported values;
  2. run the data audits listed under "Impact on existing results" in the findings document:
     - successes whose patch touched tests or config
     - fallback and seed-bundle evidence usage
     - the decay setting
     - `deterministic` flags
     - LDB case counts
     - SWE-Bench `snapshot_dir`
     - missing-usage rows
     - unmatched case sets
- [ ] Hand the output to the user. Don't edit any published numbers. *(No result files exist in this checkout; the command was handed to the user to run where the results live.)*

**Acceptance:**
- Unit tests check against scipy references: power within ±0.02 of a simulation, exact McNemar, and Holm.
- Golden-file tests for `eval_summary` and `report_run_metrics` use a fixture JSONL with unequal case sets, missing usage and multiple runs.
- `docs/METRICS_QUICK_REFERENCE.md` is updated.

---

## Phase 3: adaptive policy learning (D6)

**Goal:** the bandit can learn, its logs support valid off-policy evaluation, and the two adaptive conditions are independent. Changing defaults affects comparability with earlier runs, so record a `policy_defaults_version` in run metadata.

### 3.1. One bandit core first
- [x] Extract a `LinUCBBandit` core covering state, feature scaling, decay, feature expansion, persistence and Sherman–Morrison updates. *(Done as `evals/linucb_core.py`: state, feature expansion, scores, decay, rank-1 and Sherman–Morrison updates, tie-breaking and count parsing; percentile scaling is shared through `policy_feature_scaling.percentile_scale`. The state loaders stay per controller because their payload layouts differ.)* Use it from the helper controller, the budget gate and family controllers, and the bootstrap replay (part of X-04).
  - Land the behaviour-preserving extraction as its own commit, with parity tests against the old implementations.
  - Then make the fixes below, so each fix lands once.

### 3.2. Learning dynamics
- [x] **POL-01:** decay per D6, applied only to the updated arm.
- [x] **POL-06:** drop one one-hot level. *(Also dropped `attempts_remaining_frac`, always `1 − attempt_frac`.)*
- [x] **POL-07:** one source for the score biases, scaled per D6. *(Default `search_only=-0.10`, `frame_only=-0.05`; the CLI rejects |bias| > 0.1.)*
- [x] **POL-24:** random tie-breaking; record selected and applied counts.
- [x] **POL-22:** scoring never mutates the model. **POL-16:** NaN counter updated under the lock. **POL-21:** fail on a feature-set mismatch.
- [x] **POL-09, POL-19:** freeze scaling caps after warm-up; compute interactions from scaled features and rename them. *(Caps freeze per feature after `min_samples` = 30 observations.)*
- [x] **POL-20:** cache inverse matrices and maintain caps incrementally.

### 3.3. Rewards and budget
- [x] **POL-08:** charge only the incremental cost of escalation. *(`1 − default/applied` multiplier share of the attempt's tokens and time.)*
- [x] **POL-11:** award the rescue bonus only when the evidence appears in the successful prompt; remove the dead parameters.
- [x] **POL-04:** wire the BwK dual variable into the scores and fix its update, or rename the architecture. *(Wired: pacing target `budget / (cases × k)`, dual step 0.05, score penalty `λ · ĉ_k / target`.)*
- [x] **POL-10:** the context family widens by at least 1.

### 3.4. Logging and off-policy data
- [x] **POL-03:** correct propensities for shadow mode and warm-up. *(The helper paths were fixed in M1; the budget controller's events still used the selection's propensity in shadow mode. One shared rule now.)*
- [x] **POL-13:** export accepts the supported schema versions and rejects non-finite values. *(v3 predates the published repo and is counted as `records_unsupported_schema`.)*
- [x] **POL-18:** one chronological event loader. *(`evals/policy_event_log.py`; ties within a second keep log order.)*
- [x] **POL-14, POL-15:** bootstrap refuses `percentile_v1` state, adds `--exclude-case-ids` and `--exclude-run-ids`, and warns on overlap with the target cases. *(Overlap needs `--target-case-ids`.)*

### 3.5. TEP
- [x] **POL-17:** critic-parse flag, averaged or paired acceptance, robust score parsing. *(Paired: the critic scores the current and the candidate surface.)*
- [x] **ATT-17:** surface save failures.

### 3.6. Docs
- [x] **POL-25, X-08:** generate the policy defaults tables via `evals.doc_sync`, and fix the probe-template comment. *(The probe comment was already fixed with POL-01.)*

**Acceptance:**
- [x] A seeded synthetic test (linear rewards, about 53 features, 10 arms) shows that the default config's best-arm selection rate clearly beats uniform. *(58–69% vs 10%; `tests/test_evals_policy_acceptance.py`.)*
- [x] Propensity tests cover shadow mode, on mode and warm-up. *(`tests/test_evals_policy_propensities.py`.)*
- [x] Identical seeds produce identical event streams with `--jobs 1`. *(Controller-level, apart from decision id and timestamp.)*
- [x] Event and state schema versions are bumped if their fields change. *(Helper events 7, budget events 7, both states 4.)*

---

## Phase 4: core library hardening (`src/llmdebug`)

This phase is independent of Phases 1–3 and can run in parallel. Run `just bench-capture` before and after the capture-path changes, and include the numbers in the commit message.

### 4.1. Capture never silently fails
- [x] **CAP-01:** clamp ints outside the 64-bit range. **CAP-02:** `_safe_str`. **CAP-04, CAP-22:** bounded, sanitized filenames. **CAP-18:** unraisable-hook filter. **CAP-19:** `split_extra_context`. *(CAP-01 also covers pandas `head` cells, which bypassed `to_jsonlike`. A caller-supplied `extra` holding a wide int still raises, loudly; every internal producer goes through `to_jsonlike` or `safe_repr`.)*
- [x] Add a hostile-object fixture suite run end-to-end through `capture_exception`: broken `__str__`, `__repr__`, `__len__` and `__getattr__`; `2**70`; 300-character names; recursive containers; lazy objects. *(`tests/test_capture_hostile.py`. It found one more abort: a `__getattribute__` raising a non-`AttributeError` escapes `hasattr`/`isinstance`, so `to_jsonlike` is now a per-value guard. Lazy-object `len()` calls are left for CAP-21 in 4.6.)*

### 4.2. Redaction
- [ ] **CAP-20:** emit plain `str` keys. This is a prerequisite for key-aware matching, and since it changes the payload format, document it in `docs/configuration.md` and the changelog.
- [ ] **CAP-03:** redact values of variables with sensitive names (local names and dict keys), plus connection strings.
- [ ] **CAP-06:** refuse or mark redacted exports that redact nothing. **CAP-07:** atomic redactor cache.

### 4.3. RCA state machine
- [ ] **RCA-01, RCA-04, RCA-06, RCA-12, RCA-05.**
- [ ] **RCA-22:** a single `RCAState` and an `RCAAttemptRecord` TypedDict. **RCA-23:** `TransitionInputs` and `GateRequest` objects.

### 4.4. MCP and persistence
- [ ] **RCA-02:** root confinement per D11, and read-only tools never create files.
- [ ] **RCA-09:** async tools using `anyio.to_thread`. **RCA-10, RCA-17:** error envelope and `limit` validation.
- [ ] **RCA-11:** `PRAGMA user_version` with migrations. **RCA-16, RCA-18.**
- [ ] **RCA-03:** index consistency check, and one `snapshot_index` module. **RCA-21:** `clean`.

### 4.5. Interfaces
- [ ] **RCA-07, RCA-08:** Jupyter. **RCA-13:** dead Rich branch. **RCA-14:** git output encoding. **RCA-15:** exit codes. **RCA-19, RCA-20.**

### 4.6. Hot-path performance
- [ ] **CAP-09, CAP-10, CAP-12, CAP-13, CAP-08** (use `linecache`), **CAP-15, CAP-21.**

### 4.7. Remaining
- [ ] **CAP-05, CAP-11, CAP-14, CAP-16, CAP-17, CAP-23.**

**Acceptance:**
- Coverage stays at or above 95% with no regression against `quality/coverage_baseline.json`.
- `just quality` passes.
- `docs/configuration.md`, `docs/mcp-reference.md` and `docs/cli-reference.md` are updated, plus a changelog entry.

---

## Phase 5: structure and remaining hygiene

These are behaviour-preserving refactors, each in its own commit or PR with no functional change. Prove each is behaviour-preserving with golden fixture runs: the stub server plus fixture cases should produce identical records, ignoring timestamps and durations.

### 5.1. Replace the forwarding shims (X-02)
- [ ] Introduce a `RunContext` holding the execution config, run settings, container pool and controllers, and pass it into the main and attempt flows. Retire the module globals.
- [ ] Move shared helpers out of `run_eval` into modules that own them, imported in one direction.
- [ ] Reduce `eval_attempt_support` and `eval_main_support` to thin, typed re-exports, or delete them. Keep the monkeypatch seams that tests and importers rely on until every consumer is updated (AGENTS.md).
- [ ] Give the private `analyze_results` helpers that other modules import public names in a small module.
- [ ] Break the `baselines/` ↔ `evals/` cycle: move the shared pieces (`adaptive_controller_packet` types, `subprocess_monitor`) to a neutral module.
- [ ] Add an `evals/` architecture-boundary test, alongside `tests/test_architecture_boundaries.py`.

### 5.2. Split the very large functions (X-01)
- [ ] Move `run_eval.main` into `evals/eval_cli_args.py::build_parser()`.
- [ ] Split `run_one_condition_impl` along the seams in the ATT structural notes. Start with the terminal-failure recorder, then `adaptive_action_selection.py` and `stage_a_execution.py`.
- [ ] Split `run_main_flow`: metadata, policies, scheduler, and outcome handling with `RunAccumulators` and `UnitProgressMetrics` dataclasses.
- [ ] Split `analyze_results.py` and `openai_compat_patcher.py` along the seams in the STA and LLM structural notes.

### 5.3. Remove duplication (X-04) and dead code
- [ ] Sync/async twins: derive them from one core, or add a parity test that checks signatures and behaviour.
- [ ] Create `evals/swebench_common/`. Delete the unused `ProPredictionRecord` and `ProInstanceRow`, or adopt them.
- [ ] Move the shared adapter runtime into `adapter_base` or `baselines/_runtime.py`, with a `BaselineMetrics` dataclass. **BAS-18:** delete the dead extractors.
- [ ] Use one set of analysis formatters, one definition of `EVAL_RECORD_SCHEMA_VERSION`, and one `_build_diff_from_changes`.
- [ ] Delete the other dead code: **ORC-22, STA-26, CAP-17** (if not already done), and **DAT-25** (or rewrite it).

### 5.4. Types (X-03, X-05)
- [ ] Add TypedDicts or dataclasses for the attempt record, a composed `PatcherMeta`, the derived analysis row, and snapshot frames used by the code that builds them. Start with a record-keys constants module (**ORC-23**).
- [ ] Widen `pyrightconfig.evals.json` module by module as the seams land. Fix the undefined annotation names first.

### 5.5. Guardrails (X-06, X-07, X-08)
- [ ] Add a signature-budget test: AST-based, with an allowlist that can only shrink.
- [ ] Replace the silent `except` blocks with recorded events or fields, and use one logging channel for capture diagnostics.
- [ ] Generate defaults tables via `evals.doc_sync` wherever docs restate code constants.

### 5.6. Remaining Medium and Low findings
- [ ] **Orchestration:** ORC-07, ORC-08, ORC-09, ORC-11, ORC-12, ORC-13, ORC-14, ORC-16, ORC-20, ORC-24.
- [ ] **Attempt flow:** ATT-18, ATT-19, ATT-22, ATT-23.
- [ ] **LLM I/O:** LLM-11, LLM-12, LLM-16, LLM-17, LLM-18, LLM-20, LLM-21, LLM-22, LLM-23, LLM-24, LLM-25, LLM-26.
- [ ] **Importers:** DAT-06, DAT-07, DAT-12, DAT-22, DAT-23.

**Acceptance:**
- The golden fixture runs show no behaviour change.
- The complexity gate (`just complexity-gate`) improves or holds.
- The pyright eval include list grows.
- The signature-budget and `evals/` boundary tests pass.

---

## Coverage check

Every finding ID in the findings document is assigned to exactly one phase above:

| Area | Where its findings are assigned |
|------|----------------------------------|
| ATT (23) | 1A, 1B, 1C, 1F, 3.5, 5.6 |
| BAS (19) | 1A, 1E, 2.2, 5.3 |
| DAT (25) | 1A, 1B, 1G, 1H, 2.1, 5.3, 5.6 |
| ORC (24) | 1D, 1E, 1F, 5.3, 5.4, 5.6 |
| STA (26) | 2.1–2.3, 5.3 |
| POL (25) | 1C, 1D, 1F, 3.2–3.6 |
| LLM (26) | 1B, 1G, 5.6 |
| CAP (23) | 4.1, 4.2, 4.6, 4.7, 5.3 |
| RCA (23) | 4.3, 4.4, 4.5 |
| X (8) | 5.1–5.5, 3.6 |

If you defer a finding, leave its checkbox unticked and add `(deferred: <reason>, agreed with user on <date>)`.

## When you finish a phase

Report to the user:
- The IDs fixed, the IDs found invalid (with the reason), and the IDs deferred.
- New record or run-metadata fields and any schema-version bumps.
- Quality-gate results.
- For Phase 2, the re-analysis audit output.
- Which existing results the changes affect, so the user can decide what to re-run.
