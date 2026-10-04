# Code review findings: core library and eval harness (2026-10-04)

- **Commit reviewed:** `06a5eb5` on `main`. Line numbers refer to this commit; re-locate code before editing.
- **Scope:** `src/llmdebug/` (about 15k lines), `evals/` (about 68k lines, excluding generated trees and the `evals/cases/` data), `baselines/` (about 11k lines).
- **Method:** nine area reviews run in parallel, plus repository-wide static analysis (ruff, pyright in basic mode over `evals/` and `baselines/`, radon, vulture). The lead reviewer independently re-checked the highest-impact claims. The nine areas were:
  - core capture and serialization
  - core RCA, persistence and interfaces
  - eval orchestration
  - the per-attempt flow
  - adaptive policy and ML
  - patchers, prompts and LLM I/O
  - analysis and statistics
  - importers and SWE-Bench tooling
  - baseline adapters

  The review was read-only: no source files were changed.
- **Remediation plan:** [2026-10-04-code-review-remediation-handoff.md](2026-10-04-code-review-remediation-handoff.md)

## How to read this document

**Severity**

| Level | Meaning |
|-------|---------|
| Critical | Invalidates or biases headline experimental results, or makes the core library silently fail to produce the snapshot it exists to produce. |
| High | Materially wrong results or records in common configurations, a security exposure, or a structural problem that has already caused bugs. |
| Medium | Wrong under specific conditions, misleading telemetry, a notable performance cost, or maintainability debt with a concrete cost. |
| Low | Edge cases, hygiene, dead code. |

**Verification**

| Tag | Meaning |
|-----|---------|
| Verified | The lead reviewer reproduced it by running code, or confirmed it in the code, independently of the area reviewer. |
| Reproduced | The area reviewer demonstrated it by running code (a script or a targeted test). |
| Confirmed | The area reviewer confirmed it by reading the code and its callers. |
| Likely | Strong evidence, but the impact depends on runtime conditions or data not exercised during the review. |

**IDs.** Each finding has an area prefix and a number:
- `ATT`: per-attempt flow
- `BAS`: baseline adapters
- `DAT`: importers and SWE-Bench
- `ORC`: orchestration
- `STA`: analysis and statistics
- `POL`: adaptive policy
- `LLM`: patchers, prompts and LLM I/O
- `CAP`: core capture and serialization
- `RCA`: core RCA, persistence and interfaces
- `X`: cross-cutting

When two reviewers found the same issue, the entries were merged and marked "Also found by". Where the lead reviewer changed an area reviewer's severity, the entry says so.

## Summary

| Area | Prefix | Critical | High | Medium | Low | Total |
|------|--------|---------:|-----:|-------:|----:|------:|
| Per-attempt flow | ATT | 2 | 6 | 9 | 6 | 23 |
| Baseline adapters | BAS | 1 | 7 | 8 | 3 | 19 |
| Importers and SWE-Bench | DAT | 1 | 8 | 12 | 4 | 25 |
| Orchestration | ORC | 0 | 5 | 12 | 7 | 24 |
| Analysis and statistics | STA | 2 | 7 | 13 | 4 | 26 |
| Adaptive policy | POL | 1 | 7 | 13 | 4 | 25 |
| Patchers, prompts, LLM I/O | LLM | 0 | 9 | 12 | 5 | 26 |
| Core capture and serialization | CAP | 1 | 3 | 10 | 9 | 23 |
| Core RCA, persistence, interfaces | RCA | 0 | 4 | 10 | 9 | 23 |
| Cross-cutting | X | 0 | 2 | 3 | 3 | 8 |
| **Total** | | **8** | **58** | **102** | **54** | **222** |

**Overall assessment.** The core library is structurally sound: the RCA domain logic is free of side effects and the entrypoints are mostly thin. Its main problems are inputs that make capture fail silently (CAP-01, CAP-02, CAP-04), a redaction model that misses secrets held in local variables (CAP-03), and an MCP server that can write anywhere (RCA-02).

The eval harness carries the serious risk. Several independent defects bias or invalidate the comparisons the research relies on:
- Baselines receive information the native conditions don't (ATT-02, BAS-07).
- The LDB arm can run 0.6% of cases (BAS-01).
- The pass/fail oracle can be edited by the model (ATT-01).
- The three-layer evidence policy leaks evidence it denied (ATT-03, ATT-04, ATT-06).
- With default settings the bandit barely learns (POL-01, POL-06, POL-07).
- Several statistical procedures overstate significance or misreport cost (STA-01 through STA-09).

### Most important findings

| ID | Severity | Finding |
|----|----------|---------|
| ATT-01 | Critical | Success is pytest's exit code, and the model can edit tests or add a `conftest.py`. |
| ATT-02 | Critical | Root-cause and bug-type descriptions reach the external baselines only. |
| BAS-01 | Critical | The LDB converter accepts 11 of 1,828 cases; the rest count as LDB failures. |
| POL-01 | Critical | The default decay of 0.95 is applied to all arms on every update, so the bandit can barely learn. |
| STA-01 | Critical | Missing token telemetry is recorded as 0 tokens. |
| STA-02 | Critical | Paired tests treat repeated runs of the same case as independent. |
| DAT-03 | Critical | The SWE-Bench `llmdebug` condition equals `baseline` when `--snapshot-dir` is omitted (the default). |
| CAP-01 | Critical | A local integer wider than 64 bits aborts snapshot capture. |
| ATT-03, ATT-04, ATT-06 | High | Evidence the adaptive policy denied comes back through the fallback bundle, the seed bundle, and the reveal mode. |
| ORC-01, ORC-02 | High | `--deterministic --seed` and the container pool never take effect (written through a forwarding shim). |
| BAS-02, BAS-03, BAS-04, BAS-08, BAS-11 | High | Baselines run with different temperature, completion cap, timeout semantics and retry count, and without retry context. |
| STA-03, STA-04, STA-06 | High | Uplifts use unmatched case sets, the power formula is wrong, and there is no multiple-comparison correction. |
| X-01, X-02 | High | 6,674-line function; extracted modules depend back on `run_eval` through untyped shims. |

## Impact on existing results

Before reusing any existing result, check which of these apply.

1. **Any native-vs-baseline comparison.** Baselines received `case.description` (ATT-02) and differ from native conditions and from each other in case information, temperature, completion cap, retries, timeout semantics and retry context (BAS-02, BAS-03, BAS-04, BAS-07, BAS-08, BAS-11). For LDB, check which case list was used. On the full suite, the LDB column mostly measures the converter (BAS-01).
2. **Any success count.** Model patches could edit tests or add collection hooks (ATT-01). Audit successful attempts whose patch metadata has `patch_touches_tests` set, or whose patched files include `conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg` or `test_*.py`.
3. **Any claim that the adaptive policy restricts evidence.** Count attempts with Stage-A fallback reason `no_valid_stage_a_requests` and with `adaptive_helper_seed_bundle` evidence (ATT-03, ATT-04). In `full` reveal mode, the snapshot was always in the prompt (ATT-06).
4. **Any claim that the adaptive policy learned something.** Check the online and budget decay values in `patcher_config`. The default and the main `openai_adaptive.template.toml` both use 0.95 (POL-01). Also note the collinear feature set (POL-06) and score biases (POL-07), and that both adaptive conditions trained one shared controller (POL-05).
5. **Any shadow-mode off-policy estimate.** Behavior propensities are wrong (POL-03), and missing propensities were imputed as 1.0 (STA-12).
6. **Runs marked `deterministic: true`.** Test subprocesses were not hash-seeded (ORC-01), and the container pool was idle (ORC-02).
7. **Reported p-values, CIs and power.** Pooled multi-run analyses are pseudo-replicated (STA-02). Uplift matrices compare unmatched case sets (STA-03). Power numbers come from a wrong formula (STA-04). Nothing is corrected for multiple comparisons (STA-06, STA-20). The `analyze_results --stats` bootstrap is unseeded (STA-07).
8. **Token and cost comparisons.** Missing usage counted as zero (STA-01). Partial coverage shrank totals (STA-09). Adaptive cost is undercounted: TEP critic calls, re-plans and replays are not added (POL-02, ATT-09, ATT-15). Compaction tokens may be double-counted (ATT-13). "% of runtime" is mis-normalized (STA-08).
9. **SWE-Bench and SWE-Bench Pro.**
   - Check whether `--snapshot-dir` was passed (DAT-03).
   - Prompts included hints and hidden test IDs (DAT-01, DAT-02).
   - Infrastructure failures were scored as wrong patches (DAT-09).
   - Request errors became permanent empty patches on resume (DAT-08).
   - Multi-file diffs may have been cut down to the first file (LLM-01, LLM-03).
10. **Denominators.** Units that raised an error wrote no records (ATT-08). The adaptive retry gate could consume attempts without proposing a patch (ATT-05). With force-patch on, `patch_apply_failed` was recorded as `tests_still_failing` (ATT-07).

## Repository health snapshot

- **ruff:** clean on `src`, `tests`, `evals`, `baselines`.
- **pyright, strict, on `src/llmdebug`:** clean.
- **pyright for evals:** `pyrightconfig.evals.json` covers 14 files and is clean. Basic mode over all 129 files in `evals/` and `baselines/` reports **379 errors**:

  | File | Errors |
  |------|-------:|
  | `openai_compat_patcher.py` | 308 |
  | `analyze_results.py` | 19 |
  | `run_eval.py` | 11 |
  | `eval_main_flow.py` | 8 |
  | `adaptive_history.py` | 5 |
  | `eval_attempt_flow.py` | 5 |

  By rule: argument-type 131, general 129, assignment 44, call 26, return 24. `run_one_condition_impl` is reported as "too complex to analyze". See X-05.
- **radon:** 137 functions have cyclomatic complexity ≥ 20: 111 in `evals/`, 19 in `baselines/`, 7 in `src/`. See X-01 for the largest.
- **vulture (≥ 80% confidence):** two unused imports, at `evals/eval_main_support.py:26` and `evals/run_eval.py:1063`. More are hidden by a file-wide `noqa` (ORC-22).
- **Module boundaries:** 393 private (`_`-prefixed) names are imported across module boundaries (X-02).

---

## ATT: per-attempt flow

Scope: `evals/eval_attempt_flow.py`, `evals/eval_evidence.py`, `evals/eval_patches.py`, `evals/eval_recording.py`, `evals/attempt_metadata.py`, `evals/attempt_verification.py`, `evals/native_openai_attempt_flow.py`, `evals/native_stage_b_flow.py`.

#### ATT-01: Critical, Reproduced (scope Verified). Success is pytest's exit code, and the model can edit tests or add `conftest.py`
- **Where:**
  - `evals/eval_attempt_flow.py:7644`: `success = after.returncode == 0`
  - `evals/eval_patches.py:536`: `_validate_diff_paths_within_cwd` checks path containment only
  - Test files appear in the file index shown to the model (`evals/eval_evidence.py:236-256`).
  - Patch verifier at `evals/run_eval.py:6520-6560`; its default scope is `adaptive_only` (`evals/eval_types.py:177`, `evals/run_eval.py:2616`).
- **Problem:** A diff can weaken `test_buggy.py` or add a `conftest.py` hook such as `pytest_sessionfinish(session, exitstatus): session.exitstatus = 0`. Applied through the real `apply_diff`, that hook made pytest exit 0 while the test still failed, so the attempt was recorded as a success. `patch_touches_tests` is recorded but never enforced. The only guard is the patch verifier. It runs for adaptive conditions only, and it merely subtracts score points for test edits.
- **Fix:**
  - Before each verification run, restore pristine copies of the test surface: `test_*.py`, `conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`. Alternatively, reject diffs that touch these files.
  - Determine success from pytest's collected, passed, failed and errored counts (for example via `--junitxml`), not from the exit code.
  - Record `test_surface_modified`.
- **Also found by:** the DAT reviewer, who noted that the oracle is a mutable file in the workspace.

#### ATT-02: Critical, Confirmed. Bug descriptions, often root-cause statements or bug-type labels, reach the external baselines only
- **Where:**
  - Written to `baseline_case_context.json` at `evals/eval_attempt_flow.py:2498` and `:5940`, together with `category`, `difficulty` and `snapshot_dependency`.
  - Read by `baselines/ldb/case_converter.py:64,123-161`, `baselines/ldb/adapter.py:489`, `baselines/intervenor/case_converter.py:39,45,99`, `baselines/tracecoder/case_converter.py:42,54` and `baselines/lambda_rlm/case_converter.py:32,49`.
  - Hints also originate at import time: `evals/importers/quixbugs_importer.py:334` ("1-line bug") and `evals/importers/pyresbugs_importer.py:393-396`, which appends PyResBugs' natural-language fault description.
- **Problem:** The native conditions never see `case.description`. Real values include:
  - "DebugBench #2841: syntax error/illegal indentation (easy)"
  - "HumanEvalPack Python/1: operator misuse"
  - "Condition.notify() called before any coroutine enters wait(), causing a lost wakeup and deadlock."

  Every native-vs-baseline comparison is confounded in the baselines' favour.
- **Fix:**
  - Stop sending dataset-derived `description` and `category` to any patcher, or send the same neutral text to all arms (decision D1).
  - Record the policy in run metadata.
  - Stop embedding bug-type or fix text in `description` at import time.
- **Also found by:** the DAT reviewer.

#### ATT-03: High, Reproduced and Verified in code. The request filter (layer 3) is undone when it empties the request list
- **Where:** `evals/eval_evidence.py:525-535`; call site `evals/eval_attempt_flow.py:3691`; filter at `evals/eval_types.py:383`.
- **Problem:** Take `action=search_only` with an LLM request for `{"type": "file"}`. The filter correctly reduces the request list to empty. But `execute_context_requests` then sees `accepted == 0` and falls back to `build_fallback_context_bundle(...)`, which returns whole files: the crash file, the failing test and an import neighbour (2,235 characters in the repro). The attempt is logged with `requests_count=3`, `adaptive_context_requested=True` and `helper_outcome=materialized`, so the bandit is rewarded for evidence the policy denied. The filter function itself is correct; the leak is downstream.
- **Fix:** When the executed action is restricted and the filtered request list is empty, return an empty Stage-A context with reason `restricted_action_no_valid_requests`. Give `execute_context_requests` an `allow_fallback` parameter.

#### ATT-04: High, Confirmed and Verified in code. The "seed bundle" makes every restricted action non-restrictive
- **Where:** `evals/eval_attempt_flow.py:3914-3923`.
- **Problem:** `structured_seed_active = is_adaptive_condition and helper_materialized_context` injects the whole-file fallback bundle, up to `stage_a_max_chars`, and merges it into the Stage-B evidence. `frame_only` and `search_only` therefore deliver everything `invoke_helper` does plus their own block. Restricted actions can only add tokens, which contradicts the resource-allocation design described in `AGENTS.md`.
- **Fix:** Gate the seed bundle on the executed action: skip it, or scope it to the evidence types the action allows. Record `seed_bundle_chars` separately so earlier runs can be re-analysed.

#### ATT-05: High, Confirmed. The adaptive "no new evidence" retry gate can use up every remaining attempt without proposing a patch
- **Where:** `evals/eval_attempt_flow.py:5619-5681`.
- **Problem:** When the failure signature repeats and the evidence fingerprint is unchanged, the attempt is recorded as `success=False, note="retry_gate_no_new_evidence"` and consumes a scored attempt. Nothing changes, so the next attempt recomputes the same fingerprint and is gated again. There is no cap on consecutive blocks. An adaptive case can get one real repair attempt where `base_repair` gets k.
- **Fix:** Cap consecutive gate blocks (for example at one), don't consume a scored attempt when gating, or force an evidence-escalating action instead (decision D9).

#### ATT-06: High, Reproduced. In the default `adaptive_reveal_mode="full"`, the snapshot is always in the prompt while metadata says it isn't
- **Where:**
  - `evals/eval_attempt_flow.py:678-689` (`_resolve_stage_b_reveal_policy`) and `:4090-4098`
  - Locked in by `tests/test_evals_adaptive_conditions.py:1872`.
  - `--adaptive-reveal-mode full` is set in nearly every `evals/configs/*.template.toml`.
- **Problem:** `include_snapshot = bool(snapshot_available)`, regardless of `skip_helper`, the informativeness pregate, or `adaptive_snapshot_attempt1_denied`. Meanwhile `adaptive_snapshot_included` is recorded as `False`. The `snapshot` and `snapshot_lite` sections can coexist, so the `*_lite` actions add evidence instead of replacing it.
- **Fix:** Decide whether the policy should control snapshot inclusion in `full` mode (decision D5). At minimum, rename the field to `adaptive_snapshot_requested` and add `stage_b_snapshot_in_prompt`. Make the lite actions replace the full snapshot rather than supplement it.

#### ATT-07: High, Confirmed. Default force-patch turns `patch_apply_failed` into `tests_still_failing` and leaves a marker in the source
- **Where:**
  - `evals/run_eval.py:1857-1863`: `--force-patch-attempt` defaults to on
  - `evals/attempt_verification.py:107-142`
  - Stale diff metadata at `evals/eval_attempt_flow.py:6762`
  - Locked in by `tests/test_evals_force_patch_policy.py:832`
- **Problem:** When a patch fails to apply, the harness writes and applies a `# llmdebug-force-attempt-N` comment, sets `apply_ok=True`, and runs the "after" pytest pass, which yields `tests_still_failing`. As a result:
  - `patch_apply_failed` effectively disappears from results, mixing malformed diffs with wrong fixes.
  - `patch_diff_xxh3_128`, `patch_file_paths` and `patch_touches_tests` describe the model's diff, which was never applied.
  - The marker persists into later prompts.
- **Fix:** Keep `note="patch_apply_failed"` and add a `runner_forced_patch_recovery=True` modifier. Re-sync the diff metadata to the diff actually applied (decision D8).

#### ATT-08: High, Confirmed. Units that raise write no attempt records, which shrinks that condition's denominator
- **Where:** `evals/eval_attempt_flow.py:1131-1134`; `evals/eval_main_flow.py:622-664`; `evals/analyze_results.py:1206`.
- **Problem:** `execute_eval_unit_impl` converts exceptions into `runtime_error`, and `_handle_outcome` logs and returns before appending records. Because `success_rate = successes / len(condition_rows)`, a unit that dies on attempt 1 disappears from that condition's rate. The likelihood of crashing correlates with the condition (prompt size, external subprocesses).
- **Fix:** Emit a terminal `note="unit_runtime_error", success=False` record. Have the analyzer report missing units per condition (see STA-03).
- **Related:** ORC-10, ORC-17.

#### ATT-09: Medium, Confirmed. TEP-equilibrium re-plans aren't counted in token usage
- **Where:** `evals/native_stage_b_flow.py:289-301`; aggregation in `evals/eval_attempt_flow.py:5311-5326`.
- **Problem:** Extra `run_plan_patch` rounds discard `retry_plan_meta`, so only the first call's usage is aggregated. The flag is adaptive-only, so this understates exactly the cost the work claims to improve.
- **Fix:** Accumulate usage across all equilibrium rounds.

#### ATT-10: Medium, Confirmed. Relaxed ("fuzzy") patch application isn't recorded
- **Where:** `evals/eval_patches.py:79-82`; `evals/eval_attempt_flow.py:7346`, `:7387`.
- **Problem:** A success through `_apply_relaxed_replacements`, which re-anchors hunks using two or fewer lines of context, looks the same as a strict `patch -p1` success.
- **Fix:** Set `patch_meta["patch_apply_mode"]` to `"strict"` or `"relaxed_fallback"` and show it in summaries.

#### ATT-11: Medium, Confirmed. The recorded prompt isn't the one that produced the patch
- **Where:** `evals/eval_attempt_flow.py:4616` (recorded) vs `:5921` (used); `evals/eval_recording.py:1091`, `:1309-1314`.
- **Problem:** `prompt.txt`, `prompt_xxh3_128` and `stage_b_prompt_chars` describe the prompt from before the retry cycle. The deciding call used `prompt + syntax_retry_feedback`, or a re-budgeted recovery prompt.
- **Fix:** After the propose loop, record the final prompt plus a per-cycle list of prompt hashes.

#### ATT-12: Medium, Confirmed. Transport replay deletes artifacts that the record still points to
- **Where:** `evals/eval_attempt_flow.py:5924-5953`.
- **Problem:** The code runs `shutil.rmtree(attempt_dir)` and then rewrites only `baseline_case_context.json` and `pytest_before*`. Lost files include `prompt.txt`, `snapshot.json`, the plan prompt and response files, and earlier `patch_cycle_*.diff` files. `attempt_metadata["prompt_path"]` and similar fields still reference them.
- **Fix:** Delete only the files the patcher owns, or regenerate the deleted artifacts after the rebuild.

#### ATT-13: Medium, Confirmed. `compaction_tokens_*` can double-count the patch call
- **Where:**
  - `evals/run_eval.py:4841-4849`
  - `evals/eval_attempt_flow.py:4991`, `:6112`
  - The values are published via `evals/report_run_metrics.py:1344` and `evals/analyze_results.py:5094`.
- **Problem:** When compaction ran but returned no dict meta, the code sets `cc_meta = pm`, the whole `patch_meta`. That has already been updated with the propose call's usage, so the patch call's tokens are reported as compaction tokens.
- **Fix:** Drop the fallback, and always write `context_compaction_meta` when compaction runs.

#### ATT-14: Medium, Confirmed. The patch verifier runs only for adaptive conditions by default
- **Where:** `evals/eval_types.py:177`, `:699`; `evals/run_eval.py:2616`, `:5529`; documented in `evals/README.md:1400`.
- **Problem:** Only the adaptive conditions can lose attempts to `patch_verifier_rejected`, spend LLM calls on verifier recovery (`evals/eval_attempt_flow.py:6888` onward), and be penalized for test edits. That is an intervention applied to one family of conditions, sitting directly on the primary outcome.
- **Fix:** Run headline comparisons with `--patch-verifier-scope all` (or `disabled`) and report the adaptive-only variant as an ablation (decision D7).

#### ATT-15: Medium, Confirmed. Transport-recovery replay keeps only the second call's usage
- **Where:** `evals/eval_attempt_flow.py:1525-1545`.
- **Fix:** Merge the first call's usage into the record under `transport_recovery_*` keys.

#### ATT-16: Medium, Reproduced. Failed strict patches leave `.rej` and `.orig` files in the workdir
- **Where:** `evals/eval_patches.py:62-82`.
- **Problem:** `buggy.py.orig`, a verbatim copy of the pre-patch source, and `buggy.py.rej` persist across attempts and clutter workdirs kept with `--keep-workdirs`.
- **Fix:** Run `patch -r <tmpfile> --no-backup-if-mismatch`, or delete those files after a failed strict apply.

#### ATT-17: Medium, Confirmed. TEP state-save failures are swallowed
- **Where:** `evals/eval_attempt_flow.py:7818-7822` (`except Exception: pass`).
- **Problem:** If a save fails, later cases silently start from stale state, invalidating learning curves.
- **Fix:** Record `adaptive_tep_save_error` or emit an infrastructure event.

#### ATT-18: Low, Confirmed. Latent infinite loop if the syntax-cycle loop exits without a terminal state
- **Where:** `evals/eval_attempt_flow.py:7614-7615`. This `continue` doesn't consume the scored attempt. It is currently unreachable.
- **Fix:** Record `patch_cycle_incomplete`, consume the attempt, and continue.

#### ATT-19: Low, Confirmed. The patch sanitizer's docstring promises checks it doesn't perform
- **Where:** `evals/eval_patches.py:255-274`.
- **Problem:** The docstring promises detection of raw diff headers and markdown fences; the implementation doesn't check for them.
- **Fix:** Align the docstring with the code, or add those checks.

#### ATT-20: Low, Likely. The fallback bundle reads candidate paths without validating them
- **Where:** `evals/eval_evidence.py:568-612`.
- **Problem:** An absolute or `..`-containing `file_rel` from a snapshot makes `root / rel` escape the workdir.
- **Fix:** Run every candidate through `_normalize_context_rel_path` and the allow-list.

#### ATT-21: Low, Confirmed. A Stage-A request budget of 0 still executes one request
- **Where:** `evals/eval_evidence.py:308`: `max(1, max_requests)`.
- **Fix:** Use `max(0, …)` and return an empty context when the budget is 0.

#### ATT-22: Low, Confirmed. Compatibility shims parse `TypeError` messages to decide which kwargs to drop
- **Where:** `evals/native_openai_attempt_flow.py:285-299`; `evals/eval_attempt_flow.py:551-566`.
- **Problem:** An unrelated `TypeError` that mentions an unexpected keyword argument silently disables TEP equilibrium or progress reporting. The callees already accept these kwargs.
- **Fix:** Delete the shims, or check with `inspect.signature`.

#### ATT-23: Low, Confirmed. `build_attempt_metadata_request` is a 23-parameter pass-through
- **Where:** `evals/attempt_metadata.py:174-231`; its only caller is `evals/eval_attempt_flow.py:4611`.
- **Fix:** Construct `AttemptMetadataRequest(...)` directly.

### ATT structural notes

- **`run_one_condition_impl`** (`evals/eval_attempt_flow.py:1150-7823`, 6,674 lines, about 25 nested closures, cyclomatic complexity 648) is a single function. Extraction seams, in dependency order:

  | Lines | Responsibility | Target module |
  |-------|----------------|---------------|
  | 1413–1558 | transport recovery | `openai_transport_recovery.py` |
  | 1560–1725 | adaptive-history persistence and the pending-credit ledger | `adaptive_credit_ledger.py` |
  | 1722–2038 | progress refresh | `adaptive_progress.py` |
  | 2039–2299 | patcher capability probing | `patcher_capabilities.py` |
  | 3190–3479 | helper-action selection (the policy layer) | `adaptive_action_selection.py` |
  | 3480–3800 | Stage-A execution (where ATT-03 and ATT-04 live) | `stage_a_execution.py` |
  | 3800–4000 | evidence bookkeeping | `attempt_evidence_bookkeeping.py` |
  | 4600–5460 | about 860 lines of metadata assignment | `attempt_metadata_sync.py` |
  | 5851–7600 | the propose, recover, verify, apply and preflight cycle | `attempt_patch_cycle.py` |

- **Five terminal-failure blocks** repeat the same 8-step sequence in about 100 lines each, about 500 lines in total: lines 6652–6762, 7098–7192, 7219–7319, 7348–7434 and 7495–7574. One `_record_terminal_attempt(note=...)` helper would remove them and the drift between them.
- **Record schemas are flat dict literals:** `_empty_attempt_metadata` (363 lines, `evals/eval_recording.py:704-1067`), `_materialize_attempt_metadata_request` (261 lines), `_build_patcher_config` (306 lines). The dataclasses in `attempt_metadata.py` are flattened immediately.
- **Hand-maintained reset list:** an 80-key `patch_meta.pop` loop (`evals/eval_attempt_flow.py:6005-6110`) resets metadata between syntax cycles.
- **Condition dispatch** is about 60 independent boolean checks. Some feature gates depend on `is_openai_patcher` rather than the condition (for example `:6432`).
- **Frozen closure captures:** `_build_stage_b_sections_for_attempt` (`:4139-4208`) captures about 20 values through default arguments, freezing them at `def` time.
- `native_openai_attempt_flow.py` and `native_stage_b_flow.py` are clean, typed, dataclass-driven modules and a good template.

---

## BAS: baseline adapters

Scope: `baselines/` (`adapter_base.py`, `contracts.py`, `case_schemas.py`, `openai_compat.py`, `compare_results.py`, `extra_configs.py`, `normalize_repos.py`, `patch_manifest.py`, `pins.py`, `verify_repos.py`, and the `ldb/`, `intervenor/`, `tracecoder/`, `lambda_rlm/` and `rlm/` packages) and the seam to `evals/patchers/baseline_patcher.py`.

#### BAS-01: Critical, Verified. The LDB adapter can convert 11 of 1,828 cases, and the rest are scored as LDB failures
- **Where:**
  - `baselines/ldb/case_converter.py:196` (`_lower_case`; also `:227`, `:262`, `:329`)
  - `evals/patchers/baseline_patcher.py:718` sets `error="case_incompatible"`
  - `evals/report_run_metrics.py:1280` treats that as a terminal failure reason
  - Neither `evals/analyze_results.py` nor `baselines/compare_results.py` excludes these cases.
  - `evals/configs/study_ldb_raw.template.toml` sets `max_cases = 0` with no case list.
- **Problem:** The lead reviewer re-ran the converter over the suite. Rejection reasons:

  | Reason | Cases |
  |--------|------:|
  | `subprocess_task_unsupported` | 1,267 |
  | `non_function_entry_point_unsupported` | 207 |
  | `entry_point_not_found` | 164 |
  | `buggy_syntax_error` | 124 |
  | `async_task_unsupported` | 30 |
  | `test_support_code_unsupported` | 8 |
  | `non_assert_test_body_unsupported` | 7 |
  | `multifile_case_unsupported` | 6 |

  Unless the LDB runs used a curated case list, the LDB arm scores near 0% by construction and measures the converter, not LDB.
- **Fix:** Either run LDB only on its convertible subset and report N, or exclude `baseline_supported=False` rows from paired comparisons (decision D3). Record the convertible count in run metadata.

#### BAS-02: High, Confirmed. The harness repair prompt and retry history are discarded for every baseline
- **Where:** `evals/patchers/baseline_patcher.py:662`. `propose_patch(*, prompt, context_dir, cwd)` never reads `prompt`.
- **Problem:** `retry_prompt_mode = "full"` in the baseline templates has no effect. Each of the k harness attempts restarts cold from re-materialized files, while native conditions see the attempt-1 failure context.
- **Fix:** Pass failure and retry context through a neutral key in `case_input`, or run baselines with k=1 and document it (decision D4).

#### BAS-03: High, Confirmed. Sampling temperature differs across arms
- **Where:**
  - `baselines/adapter_base.py:130-139`: `BaselineConfig` has no `temperature` field.
  - `evals/run_eval.py:3491-3506` never forwards `--openai-temperature` (see also `:1456`, `:1614`).
  - Intervenor hardcodes 0.0 (`baselines/intervenor/adapter.py:890`, `env_client.py:132`).
  - LDB's upstream patch uses 0.2 (`baselines/ldb/patches/0001-…patch:107`); TraceCoder's uses 0.2 (`:2053`).
  - λ-RLM and RLM omit `temperature` unless a flag is set (`baselines/rlm/adapter.py:923-928`, `baselines/lambda_rlm/adapter.py:747-755`), so the server default applies. That default is stochastic on llama.cpp and LM Studio.
- **Fix:** Add `temperature` and `seed` to `BaselineConfig`, forward them from `run_eval`, honour them in every adapter and upstream patch, and record the effective values per record (decision D4).

#### BAS-04: High, Confirmed. Completion caps differ by up to 8×, and LDB is capped at 1,024 tokens
- **Where:** `baselines/ldb/adapter.py:580-605` exposes no `max_tokens` setting. At `:786-790`, a truncated output fails `ast.parse`, which yields an empty repair.
- **Problem:** Completion caps per arm:

  | Arm | Completion cap (tokens) |
  |-----|------------------------:|
  | TraceGate | 2,048 (+768 for the plan) |
  | Intervenor | 3,072 teacher / 6,144 learner |
  | TraceCoder | 4,096 |
  | λ-RLM | 8,192 |
  | RLM | 3,072 |
  | LDB | 1,024 (upstream default) |

  Full-file repairs often exceed LDB's cap.
- **Fix:** Add one shared completion-cap setting, plumb it to all five adapters, and record the effective cap per arm.

#### BAS-05: High, Confirmed. Adapters report the pinned upstream commit, not the one checked out
- **Where:** `baselines/ldb/adapter.py:429`, `baselines/intervenor/adapter.py:340`, `baselines/tracecoder/adapter.py:440`, `baselines/lambda_rlm/adapter.py:114`, `baselines/rlm/adapter.py:140`. Only `lambda_rlm/adapter.py:160-166` and `rlm/adapter.py:184-190` compare the two.
- **Problem:** LDB, Intervenor and TraceCoder can run against any checkout while the results claim the pinned commit.
- **Fix:** Record the observed `repo_commit` and a dirty-tree flag, and fail setup on mismatch for all five adapters.

#### BAS-06: High, Confirmed. `compare_results` silently keeps only the last run per condition
- **Where:** `baselines/compare_results.py:86-97` (`by_condition[cond][case_id] = success`), `:146-172`.
- **Problem:** Rows are keyed per `(run_id, case_id, condition)`, but this dict is keyed by case only. With repeated runs, earlier runs are overwritten in file-glob order, and the paired N and per-category N disagree.
- **Fix:** Key on `(run_id, case_id)`, and either require one run per condition or aggregate explicitly with a reported run count.

#### BAS-07: High, Confirmed. Baselines receive different case information from each other and from the native conditions
- **Where:**
  - λ-RLM and RLM inline the full `test_buggy.py` (`baselines/lambda_rlm/case_converter.py:18`, `:95-112`; `baselines/rlm/case_converter.py:21`).
  - TraceCoder withholds the test source but inlines `ast.unparse` of every assert (`baselines/tracecoder/case_converter.py:109-111`, `:196-215`).
  - Intervenor gets no test source at all (`baselines/intervenor/case_converter.py:80-107`).
  - Native conditions get test source via `collect_text_context` (`evals/eval_evidence.py:870`).
- **Fix:** Define one canonical case-information contract, implement it in all adapters, and record which artifacts entered each arm's prompt.

#### BAS-08: High, Confirmed. `process_timeout_sec` means three different things, and Intervenor has no wall-clock bound
- **Where:**
  - LDB and TraceCoder: a bound on the whole subprocess (`baselines/ldb/adapter.py:623`, `baselines/tracecoder/adapter.py:578`).
  - λ-RLM and RLM: a bound on each inner call (`baselines/lambda_rlm/adapter.py:356`, `:873-878`; `baselines/rlm/adapter.py:395-403`), so two iterations get about 2 × 600 s.
  - Intervenor: no bound (`baselines/intervenor/adapter.py:584-815`).
  - This contradicts `docs/baseline-adapters.md:376-388`.
- **Fix:** Enforce one deadline at `run_repair` entry in every adapter, and record the observed wall-clock time.

#### BAS-09: Medium, Verified. `compare_results` crashes when an input file is unreadable
- **Where:** `baselines/compare_results.py:63-68`.
- **Problem:** The code calls `len(errors['files_unreadable'])`, but that value is an int counter (`evals/analyze_results.py:339`, `:353`), so it raises `TypeError` exactly when it should warn.
- **Fix:** Print the counter directly.

#### BAS-10: Medium, Confirmed. Timeouts and transport errors inside adapters discard accumulated usage and the last valid candidate
- **Where:** `baselines/lambda_rlm/adapter.py:760-772`, `baselines/rlm/adapter.py:935-947`, `baselines/intervenor/adapter.py:950-962`. The exceptions are caught generically at `evals/patchers/baseline_patcher.py:793`.
- **Problem:** A timeout on attempt 2 discards attempt 1's `last_candidate` and all usage. The attempt is recorded as a forced no-op with zero tokens.
- **Fix:** Catch errors per attempt and return a `BaselineResult` carrying `last_candidate`, accumulated usage and `error="timeout"`.

#### BAS-11: Medium, Confirmed. Baselines get no transport retries; native conditions get six with backoff
- **Where:** `evals/run_eval.py:3491-3506` (baselines) vs `:3523-3526` (native).
- **Fix:** Add retry settings to `BaselineConfig` and apply them in the shared OpenAI-compatible call paths (decision D4).

#### BAS-12: Medium, Confirmed. TraceCoder's test timeout drops to 10 s when the case-context file is missing or invalid
- **Where:** `baselines/tracecoder/adapter.py:550`. `_load_baseline_case_context` never raises (`evals/patchers/baseline_patcher.py:106-114`).
- **Problem:** Other adapters fall back to `process_timeout_sec` (600 s); TraceCoder falls back to 10 s.
- **Fix:** Use the same fallback, and fail loudly on invalid JSON.

#### BAS-13: Medium, Likely. RLM rolls back any candidate that fails a 10-second import smoke test
- **Where:** `baselines/rlm/env_client.py:167-197`, `:736-748`; `baselines/rlm/adapter.py:640-676`.
- **Problem:** 830 of the 1,828 `buggy.py` files make top-level calls, so the smoke test runs the program. A partial repair that still raises, or takes longer than 10 s, can't be submitted, and RLM is the only arm with this filter.
- **Fix:** Keep the smoke test as telemetry only, and retain the rejected text as `last_candidate`.

#### BAS-14: Medium, Confirmed. The patched upstream baseline repositories can't be verified
- **Where:**
  - `baselines/setup_repos.sh:72-101` applies patches with `--ignore-space-change --ignore-whitespace --inaccurate-eof`.
  - `baselines/normalize_repos.py:23-156` then applies 30 hard-coded string fixes.
  - `baselines/verify_repos.py:178-208` checks only marker substrings and `py_compile`.
- **Fix:** Record and assert a hash of the post-patch tree, and run `git reset --hard && git clean -fd` before applying patches.

#### BAS-15: Medium, Confirmed. LDB records token and character metrics only when a valid repair was found
- **Where:** `baselines/ldb/adapter.py:793-906`. The metrics are built inside `if repaired:`.
- **Problem:** Failed LDB attempts systematically report less usage, which biases cost-per-success.
- **Fix:** Compute the metrics before the branch.

#### BAS-16: Medium, Likely. LDB model routing rewrites the model id and forwards only the port
- **Where:** `baselines/ldb/adapter.py:70-81`, `:607-616`.
- **Problem:** Names that merely contain `gpt-4` go to upstream unmodified. The host and path of the base URL depend on patch behaviour that is never validated.
- **Fix:** Match the GPT model family exactly, and log the resolved endpoint into the metadata.

#### BAS-17: Low, Confirmed. Intervenor always sends the OpenRouter-specific `extra_body={"usage": {"include": True}}`
- **Where:** `baselines/intervenor/adapter.py:188-189`, `:892`; `baselines/intervenor/env_client.py:35-37`, `:134`. RLM sends it conditionally (`baselines/rlm/env_client.py:454-455`).
- **Fix:** Decide whether to send it in one shared helper in `baselines/openai_compat.py`, based on the endpoint's capabilities.

#### BAS-18: Low, Confirmed. Three `result_extractor.make_diff` copies are unused in production but covered by tests
- **Where:** `baselines/{ldb,intervenor,tracecoder}/result_extractor.py`; the adapters call `super().extract_diff`. Also unused: `support_code` at `baselines/ldb/case_converter.py:271`, `:279-281`.
- **Fix:** Delete them, and point the diff tests at `BaselineAdapter.extract_diff`.

#### BAS-19: Low, Confirmed. The same OpenAI-compatible settings have different defaults on different code paths
- **Where:** `baselines/openai_compat.py:112-124` (`enable_thinking=False`, `allow_reasoning_fallback=False`) vs `baselines/extra_configs.py:84-100` and `evals/patchers/baseline_patcher.py:606-611` (both `True`).
- **Fix:** Use one config factory with a single set of defaults.

### BAS structural notes
- **Import cycle between `baselines/` and `evals/`:**
  - All five adapters import `evals.adaptive_controller_packet`.
  - LDB and TraceCoder also import `evals.subprocess_monitor`.
  - `evals` imports `baselines.*`.
  - `baselines/compare_results.py:30` imports private `evals.analyze_results` helpers.
- **`adapter_base.py` abstracts little:** most of its 486 lines are diff plumbing. Every adapter reimplements the same loop: precheck, iterate, extract a candidate, validate syntax, run pytest locally, format feedback, compute metrics.
- **About 1.2–1.5k lines are copy-pasted across adapters:**
  - `_package_case_files` is byte-identical in three converters, and `_extract_entry_point` exists in four.
  - `_CommandOutcome`, `_run_case_tests`, `_format_failure_feedback`, `_materialize_workspace` and `_resolve_repo_commit` are duplicated two to five times.
  - `_parse_metrics_sidecar` exists twice (about 160 lines each) with different `usage_available` semantics.
- **Typed configs are bypassed:** about 40 raw `config.extra.get("…") or <literal>` reads ignore the typed configs in `extra_configs.py`. Defaults diverge: for RLM max iterations, the typed config says 0, the adapter 12, the CLI 12, and the docs and templates 24.
- **`BaselineConfig` lacks the fairness fields** (temperature, top_p, seed, a shared max_tokens, retries) but carries a deprecated `timeout_sec` alias.
- **Undeclared metrics schema:** `_normalize_baseline_metrics` (`evals/patchers/baseline_patcher.py:356-436`) is a hand-maintained whitelist of about 40 keys.
- **Size:** `baselines/rlm/adapter.py` (1,388 lines) and `baselines/intervenor/adapter.py` (1,374) exceed the soft limit. `_final_metrics` and `_compatibility_failure_result` take 11–14 parameters.
- **Done right:**
  - The harness re-verifies baseline self-reported success with its own pytest run.
  - `case.json`, `meta.json` and `oracle.patch` are stripped before attempts.
  - `baselines/openai_compat.py` is genuinely shared with `evals/patchers/openai_compat_patcher.py`.

---

## DAT: importers and SWE-Bench tooling

Scope: `evals/importers/`, `evals/swebench/`, `evals/swebench_pro/`, `evals/validate_cases.py`, `evals/select_cases.py`.

#### DAT-01: High, Verified. SWE-Bench prompts include the hidden FAIL_TO_PASS and PASS_TO_PASS test IDs
- **Where:** `evals/swebench/generate.py:622-632`; `evals/swebench_pro/generate.py:716-734`. Asserted by `tests/test_evals_swebench_pro_generate_prompt.py:88`.
- **Problem:** These are exactly the node IDs the scorer checks. Both conditions receive them, so the comparison between conditions stays valid. But resolve rates aren't comparable to published SWE-Bench or SWE-Bench Pro numbers.
- **Severity:** the area reviewer rated this Critical; the lead lowered it to High because the paired comparison is unaffected.
- **Fix:** Remove them from the prompts, or put them behind an explicit flag recorded in `run_metadata.json` (decision D2).

#### DAT-02: High, Verified. The SWE-Bench Lite prompt includes `hints_text`
- **Where:** `evals/swebench/generate.py:620`, `:626-627`.
- **Problem:** `hints_text` is the issue discussion from before the fix and often contains the diagnosis or the patch itself. Official evaluation provides only `problem_statement`. The same comparability caveat as DAT-01 applies.
- **Fix:** As DAT-01 (decision D2).

#### DAT-03: Critical, Confirmed. The SWE-Bench `llmdebug` condition is identical to `baseline` when `--snapshot-dir` is omitted
- **Where:** `evals/swebench/cli.py:118`, `:296`; `evals/swebench_pro/cli.py:56`, `:176`; `evals/swebench/generate.py:655-657`.
- **Problem:** The default `""` becomes `snapshot_dir=None`, and the snapshot section is just the sentence "No snapshot directory was provided for this run." The README examples omit the flag, `run_metadata.json` doesn't record `snapshot_dir`, and nothing warns. Any SWE-Bench result produced this way compares two near-identical prompts.
- **Fix:** Fail fast when the `llmdebug` condition is requested without a snapshot directory. Count per-instance snapshot hits and misses, and persist both in run metadata.

#### DAT-04: High, Reproduced. SWE-Bench Pro scoring passes instances whose test lists are empty
- **Where:** `evals/swebench_pro/score_docker.py:341-353`.
- **Problem:** `fail_to_pass <= passed_tests` is trivially true for an empty set. A missing or renamed dataset column would score every instance as resolved for both conditions.
- **Severity:** the area reviewer rated this Critical; the lead lowered it to High because it requires empty or missing lists.
- **Fix:** Treat an empty `fail_to_pass` as an infrastructure error and exclude the instance.

#### DAT-05: High, Confirmed. The importer `--validate` flag only checks the schema in 6 of 8 importers
- **Where:**
  - `schema_only=True` is hardcoded in `evals/importers/quixbugs_importer.py:375`, `condefects_importer.py:468`, `humanevalpack_importer.py:188`, `mutmut_generator.py:327`, `bugsinpy_importer.py:280`, `pybughive_importer.py:378` and `trickybugs_importer.py:306`.
  - Only `debugbench_importer.py:354` and `pyresbugs_importer.py:423` run full validation.
  - `evals/importers/cli.py:330-336` checks pytest availability as if execution validation would happen.
- **Note:** a 12-case ConDefects sample passed full validation, so the current corpus appears to have been pruned separately.
- **Fix:** Pass `schema_only=not validate` through every importer.

#### DAT-06: High, Reproduced. The native "strict reproducibility" importers can never pass validation
- **Where:** `evals/importers/base.py:84` requires `buggy.py`, but `evals/importers/native_repro_helpers.py:159-184` writes files under their source-relative paths.
- **Problem:** Every BugsInPy, PyBugHive and TrickyBugs case is deleted as `schema_validation_failed`. This also diverges from `evals/validate_cases.py:205-221`.
- **Fix:** Have `validate_generated_case` delegate to `validate_case_contract`.

#### DAT-07: High, Likely. Native-dataset cases can't reproduce their bugs as built
- **Where:**
  - `evals/importers/bugsinpy_importer.py:138`, `:206-248`
  - `evals/importers/pybughive_importer.py:150-166`, `:207`, `:300-340`
  - `evals/importers/trickybugs_importer.py:215-240`, `:254`
- **Problem:**
  - Each case gets one changed source file plus its tests, with no package files or dependencies.
  - `expected_exception` is hardcoded to `AssertionError`.
  - PyBugHive's archive lookup either resolves buggy and fixed to the same file or rejects the match as ambiguous.

  No `bip_`, `pbh_` or `tb_` cases exist, so current results are unaffected.
- **Fix:** Materialize the full repository tree, or run in the dataset's own container. Derive `expected_exception` from an actual baseline run.

#### DAT-08: High, Confirmed. SWE-Bench generation saves request failures as empty patches, so `--resume` never retries them
- **Where:** `evals/swebench/generate.py` (about lines 470–500); `evals/swebench_pro/generate.py:449-500`, `:565-575`.
- **Problem:** A transient timeout is scored as "model produced no patch". Error rates differ by condition and time of day, so this biases the pairing.
- **Fix:** Don't persist records whose `request_error` is set, and re-queue them on resume.

#### DAT-09: High, Confirmed. SWE-Bench Pro scoring counts infrastructure failures as wrong patches and discards the reason
- **Where:** `evals/swebench_pro/score_docker.py:316-340`, `:364-374`, `:555-560`.
- **Problem:** A non-zero container exit, a missing `output.json` and a `container.wait` timeout all yield `infrastructure_error=False`. `eval_results.json` stores only `{instance_id: passed}`, so the reason can't be recovered afterwards.
- **Fix:** Classify these as infrastructure errors, persist `error` and `infrastructure_error`, and exclude such instances from both conditions.

#### DAT-10: Medium, Reproduced. `generate_oracle_patch` emits a corrupt diff when the source lacks a trailing newline
- **Where:** `evals/importers/base.py:21-48`.
- **Problem:** The last removed and added lines merge into one line, and `count_patch_lines` undercounts. All 1,828 committed patches apply, because ruff adds the newline. The native and HumanEvalPack importers don't run ruff, though.
- **Fix:** Normalize the trailing newline before diffing, or emit a `\ No newline at end of file` marker.

#### DAT-11: Medium, Confirmed. Dataset strings are interpolated unquoted into the Pro container entry script
- **Where:** `evals/swebench_pro/score_docker.py:94-115`.
- **Problem:** `selected_test_files` word-splits on spaces, so the wrong tests run. `base_commit`, `before_repo_set_cmd` and similar values are interpreted by the shell.
- **Fix:** Pass the test list as an argv array (`"$@"`) and `shlex.quote` every interpolated value.

#### DAT-12: Medium, Confirmed. Clone URLs from the dataset reach `git clone` with no `--` separator and no scheme check
- **Where:** `evals/importers/native_repro_helpers.py:130-152`; the URLs come from `evals/importers/bugsinpy_importer.py:88-91`.
- **Problem:** A value starting with `-` is parsed as an option, and git's `ext::` transport can execute commands.
- **Fix:** Require `https://` or `git@`, reject a leading `-`, and add `--` before positional arguments.

#### DAT-13: Medium, Confirmed. Pro scoring reuses cached results without checking the patch
- **Where:** `evals/swebench_pro/score_docker.py:210-216`.
- **Problem:** Regenerated predictions silently reuse the previous run's outcomes.
- **Fix:** Store a hash of the patch next to `output.json` and invalidate the cache on mismatch.

#### DAT-14: Medium, Reproduced. The case determinism filter misses common sources of nondeterminism
- **Where:** `evals/importers/base.py:224-229`.
- **Problem:** `time.time()`, `datetime.now()`, `random.randint`, `choice`, `shuffle` and `sample`, `uuid`, `os.urandom` and set iteration all pass. Flaky cases add run-to-run variance that looks like a condition effect.
- **Fix:** Detect these with an AST check and reject unseeded randomness.

#### DAT-15: Medium, Confirmed. Imported case content depends on whether `ruff` is on PATH and on the working directory
- **Where:** `evals/importers/base.py:232-246`.
- **Problem:** It runs a bare `ruff`, silently returns unformatted code when ruff isn't found, and picks up `pyproject.toml` from the current directory.
- **Fix:** Use `sys.executable -m ruff` with `--isolated` or an explicit config, and raise instead of silently falling back.

#### DAT-16: Medium, Confirmed. Difficulty is part of the case ID, so the same bug can be imported twice
- **Where:** `evals/importers/condefects_importer.py:409-412`, `:437`; `evals/importers/pyresbugs_importer.py:371`.
- **Fix:** Derive the ID from source identity only, and keep difficulty in `case.json`. Don't rename existing cases without a mapping.

#### DAT-17: Medium, Confirmed. ConDefects cases are graded by at most five sample tests that the model can see, with the examples first
- **Where:** `evals/importers/condefects_importer.py:51-52`, `:130-139`, `:201-227`.
- **Problem:** These are 1,267 of the 1,828 cases (69%). A patch that special-cases the visible inputs and outputs would pass.
- **Fix:** Include non-example tests, and keep the grading tests out of the workspace (decision D12).

#### DAT-18: Medium, Likely. SWE-Bench `--openai-max-tokens` defaults to 2,048 and truncation isn't recorded
- **Where:** `evals/swebench/cli.py`; `evals/swebench_pro/cli.py:53`; `generate.py` (`openai_max_tokens: int = 2048`).
- **Problem:** Multi-file patches can exceed the cap, and a truncated diff is scored as a wrong patch.
- **Fix:** Raise the default, and record `finish_reason` per prediction.

#### DAT-19: Medium, Likely. Pro scoring applies patches with `git apply -v` only
- **Where:** `evals/swebench_pro/score_docker.py:105`.
- **Problem:** There's no fallback with fuzz or a different `-p` level, unlike `evals/eval_patches.py:43-102` and the official harnesses, which deflates Pro numbers. See also LLM-03.
- **Fix:** Mirror the official apply sequence.

#### DAT-20: Medium, Confirmed. Generated tests embed raw dataset text as Python source
- **Where:** `evals/importers/debugbench_importer.py:139-148`; `evals/importers/condefects_importer.py:218-220`.
- **Problem:** Third-party dataset text becomes executable code that runs locally under the default `--executor local`.
- **Fix:** Parse values with `ast.literal_eval` and emit them with `repr()`, rejecting anything that doesn't round-trip.

#### DAT-21: Medium, Confirmed. `snapshot_dependency` is derived deterministically from a heuristic category
- **Where:** `evals/category_mapping.py:174-182`; `evals/importers/condefects_importer.py:263-285`.
- **Problem:** The distribution is helpful 1,650 (90.3%), none 132, required 46, and all 46 required cases are hand-crafted. Per-stratum results therefore just restate per-category results. See also LLM-24.
- **Fix:** Document the derivation in `evals/README.md`, and stop presenting `snapshot_dependency` as independent evidence until it is labelled independently (decision D12).

#### DAT-22: Low, Confirmed. `--max-cases 0` means "none" in two importers and "unlimited" in the rest
- **Where:** `evals/importers/humanevalpack_importer.py:111`; `evals/importers/mutmut_generator.py:238`.

#### DAT-23: Low, Confirmed. `import_humanevalpack` overwrites existing cases
- **Where:** `evals/importers/humanevalpack_importer.py:181-186`.
- **Problem:** There is no existence check, so re-running it discards earlier `fixup_expected_exceptions` corrections.

#### DAT-24: Low, Confirmed. `select_cases` breaks ties by JSONL file order
- **Where:** `evals/select_cases.py:70-74`.
- **Problem:** Under `--jobs > 1`, append order is nondeterministic. The comparison `attempt > prev or attempt == prev` is just `>=`. See STA-19.

#### DAT-25: Low, Confirmed. `mutmut_generator.py` can't produce usable cases
- **Where:** `evals/importers/mutmut_generator.py:29`, `:72-88`, `:262`.
- **Problem:**
  - Its parser doesn't match current `mutmut results` output.
  - `MAX_FIX_LINES` is unused.
  - The copied test's imports don't resolve.
  - The existing `mut_*` cases are hand-crafted.
- **Fix:** Delete it, or rewrite it against `mutmut --json`.

### DAT structural notes
- **SWE-Bench Lite and Pro are largely copy-pasted:**
  - `evals/swebench/io.py` and `evals/swebench_pro/io.py` (626 lines each) differ by 11 lines, about 98% identical.
  - `smoke.py` differs by 21 diff lines.
  - `generate.py` is about 75% shared.

  Every fix above currently has to be made twice. A shared `evals/swebench_common/` package would fix that.
- **Four parallel case schemas and two divergent validators:**
  - Schemas: `evals/importers/types.py`, `evals/swebench/types.py`, `evals/swebench_pro/types.py`, `baselines/case_schemas.py`.
  - Validators: `evals/importers/base.py:72` vs `evals/validate_cases.py:205`; the divergence causes DAT-06.
- **Unused types:** `ProPredictionRecord` and `ProInstanceRow` (`evals/swebench_pro/types.py`) are used only by their own tests. The real code uses the Lite `PredictionRecord` and `dict[str, Any]` rows.
- **No importer protocol:** five importers return preview dicts and three return `ImportRunSummary`. `evals/importers/cli.py:66-78` assumes the former.
- **Dependency direction:** `evals/importers/base.py:91` imports case enums from `evals.run_eval`. They belong in a small `evals/case_contract.py`.
- **`evals/importers/top3_bootstrap.py`** (1,006 lines) mixes five concerns. Its `jobs` parameter is accepted and discarded, and `DEFAULT_IMPORT_REPORT_DIR` is relative to the current directory.
- **Done right:** archive extraction (`top3_bootstrap.py:600-660`) rejects absolute paths, `..`, symlinks, hardlinks and device members. One gap remains: zip symlink detection misses archives created on Windows.
- **Swallowed exceptions:** `score_docker.py:214-215`, `:378-398`; `importers/base.py:240-243`; `native_repro_helpers.py:139-140`; `pybughive_importer.py:134-136`; `trickybugs_importer.py:133-137`.

---

## ORC: eval orchestration

Scope: `evals/run_eval.py`, `evals/eval_main_flow.py`, `evals/eval_main_setup.py`, `evals/eval_main_resume.py`, `evals/eval_main_support.py`, `evals/eval_config.py`, `evals/eval_types.py`, `evals/run_condition_builder.py`, `evals/runtime_resolution.py`, `evals/eval_containers.py`, `evals/subprocess_monitor.py`, `evals/docker_image_fingerprint.py`, `evals/progress_display.py`, `evals/progress_output.py`.

#### ORC-01: High, Verified (reproduced by the lead). `--deterministic --seed N` never reaches the pytest subprocesses
- **Where:**
  - Written at `evals/eval_main_flow.py:407` as `legacy._ACTIVE_RUN_SETTINGS = ...`, where `legacy` is `evals.eval_main_support`.
  - Read at `evals/run_eval.py:5673` by `_base_pytest_env`, the only place that sets `PYTHONHASHSEED`.
- **Problem:**
  - `eval_main_support` forwards attribute *reads* to `run_eval` through a module-level `__getattr__`, but assignments bind on the shim module.
  - So `run_eval._ACTIVE_RUN_SETTINGS` keeps its default, and `PYTHONHASHSEED` and `TZ=UTC` are never set.
  - Meanwhile `run_metadata` and `patcher_config` record `deterministic: true`.
  - The tests hide it: `tests/test_evals_run_eval_helpers_coverage.py:746` sets `run_eval`'s global, and `tests/test_evals_main_flow.py:659` checks the shim's.
- **Repro:** assign `legacy._ACTIVE_RUN_SETTINGS = _ActiveRunSettings(deterministic=True, seed=1234)`; then `run_eval._base_pytest_env(...)` has no `PYTHONHASHSEED`.
- **Fix:**
  - Use setter functions in `run_eval` (following `eval_config.set_active_execution_config`) or an explicit run-context object.
  - Ban `legacy.<NAME> = ...` writes with an architecture test.
  - Add an integration test that checks the environment actually reaches pytest.
- **Also found by:** the ORC reviewer, who rated it Critical. See X-02 for the root cause.

#### ORC-02: High, Verified. The container pool is started and never used
- **Where:** written at `evals/eval_main_flow.py:845`; read at `evals/run_eval.py:5626`; hidden by `tests/test_evals_run_eval_helpers_coverage.py:883`.
- **Problem:** Same root cause as ORC-01. Under the default `testcontainers` executor, N pool containers start and the display reports "Container pool: N container(s) ready". They then sit idle while every pytest call creates a one-shot container.
- **Fix:** As ORC-01.

#### ORC-03: High, Reproduced by two reviewers. `run_monitored_subprocess` can hang forever after a timeout
- **Where:** `evals/subprocess_monitor.py:107-117`, `:170-175`.
- **Problem:** On timeout, only the direct child is killed: there's no `start_new_session` and no `killpg`. The loop then waits for EOF on both pipes, which a surviving grandchild keeps open.
  - Repro: a child spawns a 60-second grandchild and is monitored with `timeout_sec=2`; the call never returned.
  - This affects LDB, which uses a `multiprocessing.Pool`, and TraceCoder.
  - The Docker wrapper forces `--executor local` (`evals/docker/run_eval_docker.sh:824`), so this is the main path.
- **Fix:** Use `start_new_session=True`, call `os.killpg` on timeout, and replace the wait-for-EOF requirement with a bounded drain after the process exits.
- **Also found by:** the BAS reviewer.

#### ORC-04: High, Verified. Resuming with a different settings fingerprint only warns, and new records carry the old config
- **Where:** `evals/eval_main_resume.py:106-112`; `evals/eval_main_flow.py:210-213`.
- **Problem:** On a mismatch, `run_metadata = dict(existing_run_metadata)`, so rows produced under configuration B claim configuration A. Nothing downstream can detect this.
- **Fix:** Abort on a mismatch unless `--allow-settings-drift` is passed. In that case, stamp each record with its own fingerprint and a fresh `patcher_config`.

#### ORC-05: High, Likely. `--seed` seeds the bandits even without `--deterministic`, but is recorded as null and left out of the fingerprint
- **Where:** `evals/eval_main_flow.py:317`, `:372`; `evals/eval_recording.py:241`, `:278`.
- **Problem:** The bandit exploration RNG can't be recovered from the results. Resumes with different seeds produce identical fingerprints.
- **Fix:** Always record the raw seed, with a separate `deterministic_mode` field, and include it in the fingerprint.

#### ORC-06: Medium, Verified. Abbreviated CLI flags are overridden by config-file values
- **Where:** `evals/eval_config.py:265-282`. `_collect_cli_specified_dests` does an exact lookup in `parser._option_string_actions`, while the parser at `evals/run_eval.py:1091` uses the default `allow_abbrev=True`.
- **Repro:** `--job 3 --config c.toml` with `jobs = 8` in the config gives `jobs == 8`, recorded with source `config:...`.
- **Fix:** Set `allow_abbrev=False`, or detect explicitly set options by comparing against a parse of the defaults.

#### ORC-07: Medium, Likely. The transport-recovery leader can leave the outage gate closed forever
- **Where:** `evals/run_eval.py:2893-2963`.
- **Problem:** The leader sets `_outage_active = True` and has no `try/finally`. If the leader dies, every worker blocks in `wait_if_open()`.
- **Fix:** Add `try/finally`, and key `_last_outcome` by generation.

#### ORC-08: Medium, Confirmed. Pool containers are left running after abnormal termination
- **Where:** `evals/eval_containers.py:131`, `:250-252`; `evals/eval_types.py:273`. Cleanup skips running containers by design (`tests/test_evals_container_cleanup.py:56`).
- **Problem:** Pool containers run `sleep infinity`, so after `kill -9` or an OOM kill they are never reaped.
- **Fix:** Label containers with the run ID and owner PID, and reap running ones whose owner no longer exists.

#### ORC-09: Medium, Likely. Ctrl-C during a parallel run blocks until every in-flight unit finishes
- **Where:** `evals/eval_main_flow.py:923-967`. There is no signal handling anywhere in `evals/`.
- **Fix:** Install a SIGINT handler that sets an abort flag, call `executor.shutdown(wait=False, cancel_futures=True)`, and show how many units are still in flight.

#### ORC-10: Medium, Likely. Deferred records can be lost when a unit raises after streaming
- **Where:** `evals/eval_attempt_flow.py:1146`, `:1353-1381`; `evals/eval_main_flow.py:667-668`.
- **Problem:** Records queued with `defer_sink=True` are written only on the normal path and the credit-resolution paths.
- **Fix:** Flush the deferred queue in a `finally`, and append records that weren't streamed on the runtime-error branch too.
- **Related:** ATT-08.

#### ORC-11: Medium, Confirmed. `--jobs 1` shares one patcher across units, while `--jobs N` builds one per unit
- **Where:** `evals/eval_main_flow.py:879`, `:912-918`; `evals/eval_attempt_flow.py:1103`.
- **Problem:** Today this only affects connection pooling, but any per-instance state added later would make the job count a confounding variable.
- **Fix:** Pick one policy (one patcher per unit is safest) and add a test that asserts it.

#### ORC-12: Medium, Likely. `make_patcher` mutates the shared `argparse.Namespace` from worker threads and raises `SystemExit`
- **Where:** `evals/run_eval.py:3450-3452`, `:3479-3480`.
- **Problem:** `SystemExit` escapes `except Exception` and aborts the run with a raw traceback.
- **Fix:** Resolve the patcher factory once into an immutable config, and raise `ValueError` or `MainFlowCliError` instead.

#### ORC-13: Medium, Likely. Budget-clamped patcher clones lose their progress callback
- **Where:** `evals/run_eval.py:6083-6090`; `evals/patchers/base.py:125-136`.
- **Problem:** `dataclasses.replace` creates a new instance whose per-instance `ContextVar` is unset, so the heartbeat display goes blind during long LLM calls.
- **Fix:** Copy the progress callback onto the clone.

#### ORC-14: Medium, Likely. The results JSONL is never fsynced, and the run lock file is unlinked on release
- **Where:** `evals/eval_recording.py:178-195`, `:213-214`.
- **Problem:** Unlinking the lock file while holding its `flock` lets two processes each hold an "exclusive" lock, on different inodes.
- **Fix:** Never unlink the lock file, and add an optional fsync cadence.

#### ORC-15: Medium, Confirmed. Duplicate or empty `--conditions` values aren't rejected
- **Where:** `evals/run_eval.py:903-990`.
- **Problem:** A duplicate condition doubles the work, and one copy is silently discarded during deduplication. An empty list exits 0 with "Nothing to do".
- **Fix:** Deduplicate while preserving order, and fail on an empty list.

#### ORC-16: Medium, Likely. The Docker image fingerprint includes generated directories
- **Where:** `evals/docker_image_fingerprint.py:10-16`. It misses `evals/swebench_pro_runs`, `evals/swebench_pro_assets` and `.benchmarks`.
- **Problem:** SWE-Bench Pro runs change the fingerprint, which triggers an image rebuild mid-experiment.
- **Fix:** Add those prefixes, ideally from one shared list of generated directories.

#### ORC-17: Medium, Likely. Units that hit a runtime error drop their already-streamed usage from the run totals
- **Where:** `evals/eval_main_flow.py:622-665`.
- **Fix:** Compute the progress metrics before branching on `runtime_error`.

#### ORC-18: Low, Reproduced. `timeout_sec=0` makes the monitor loop busy-spin
- **Where:** `evals/subprocess_monitor.py:139-147`. Measured at about 84% of one core.
- **Fix:** When `timeout_sec <= 0`, use no deadline.

#### ORC-19: Low, Likely. Monitored subprocesses inherit stdin
- **Where:** `evals/subprocess_monitor.py:107-117`.
- **Fix:** Pass `stdin=subprocess.DEVNULL`.

#### ORC-20: Low, Confirmed. The lockfile and argv capture redact only the API key
- **Where:** `evals/eval_types.py:276-279`; `evals/eval_config.py:434-461`, `:490-493`.
- **Problem:** `--container-env` values are written verbatim, and the lockfile write isn't atomic.
- **Fix:** Redact `container_env` values, and write via a temporary file plus `os.replace`.

#### ORC-21: Low, Confirmed. `--run-id` isn't validated before being used in file paths
- **Where:** `evals/run_eval.py:1178-1181`; `evals/eval_main_setup.py:257-304`.
- **Fix:** Validate against `^[A-Za-z0-9._-]{1,128}$`.

#### ORC-22: Low, Confirmed. Dead module state and imports hidden by a file-wide `noqa`
- **Where:**
  - `evals/run_eval.py:10` has `# ruff: noqa: F401, F811`.
  - `_ACTIVE_EXECUTION_CONFIG` at `:988` is never read; it shadows the real one in `eval_config`.
  - The `_ACTIVE_*_POLICY_CONTROLLER` globals are never read.
  - The `global` statement at `:1088-1089` has no matching assignment.
  - Nine imports are unused.
  - Vulture also flags `evals/eval_main_support.py:26` and `evals/run_eval.py:1063`.
- **Fix:** Delete the dead code, and use per-line `noqa` comments only on names that are really re-exported.

#### ORC-23: Low, Confirmed. Statuses and record keys are bare strings re-spelled across modules
- **Where:** `evals/eval_main_flow.py:674`; `evals/progress_output.py:127-141`; `evals/run_eval.py:4488`. Keys such as `infra_resume_eligible`, `adaptive_progress_stop_note` and `prompt_chars_total` are each spelled out in three or four files.
- **Fix:** Use `Literal` aliases and a record-keys constants module (see X-03).

#### ORC-24: Low, Confirmed. The auto-generated `run_id` uses local time while metadata uses UTC
- **Where:** `evals/eval_main_setup.py:258-260`.
- **Fix:** Use `datetime.now(timezone.utc)`.

### ORC structural notes
- **`run_eval.main`** (`evals/run_eval.py:1079-2685`) is 1,607 lines of argparse setup plus a three-line dispatch. Moving it to `evals/eval_cli_args.py::build_parser()` is a mechanical cut that removes about 24% of the file.
- **`run_main_flow`** (`evals/eval_main_flow.py:64-1045`, 982 lines, 7 closures, about 14 `nonlocal` variables). Seams:

  | Lines | Responsibility | Target |
  |-------|----------------|--------|
  | 210–293, 341–347, 395–401 | metadata assembly | `eval_main_metadata.py` |
  | 295–403, 972–991 | policy-controller lifecycle | `eval_main_policies.py` |
  | 787–967 | scheduler | `eval_main_scheduler.py`, with a `RunAccumulators` dataclass |
  | 466–578, 601–785 | outcome handling | `eval_main_outcome.py`; `_extract_progress_metrics` returns an 11-tuple that should become a dataclass |

- **Import cycle:** `run_eval` → `eval_main_flow` → `eval_main_support` → `run_eval` survives only because of function-local imports (`evals/run_eval.py:2683`, `:3064`, `:5549`). `tests/test_architecture_boundaries.py` guards only `src/llmdebug`.
- **`RunConditionInputs` has three representations:**
  - grouped `*Config` objects
  - a flat dataclass of about 90 fields built by `from_components` (13 parameters, 149 lines)
  - `**run_inputs.__dict__` unpacked into legacy kwargs (`evals/eval_attempt_flow.py:1114`)
- **Done right:** resume mechanics are well engineered: byte-offset tail repair, `flock`, and de-duplication by `(run_id, case_id, condition, attempt)`. The weakness is the policy (ORC-04), not the mechanics.
- **JSON:** `_SynchronizedJsonlWriter` uses stdlib `json.dumps` (`evals/eval_recording.py:208`) rather than the harness's orjson adapter.

---

## STA: analysis and statistics

Scope: `evals/analyze_results.py`, `evals/report_run_metrics.py`, `evals/stats.py`, `evals/eval_summary.py`, `evals/eval_two_pass_summary.py`, `evals/budget_risk_report.py`, `evals/extract_failed_cases.py`, `evals/doc_sync.py`.

#### STA-01: Critical, Verified. Missing token telemetry is recorded as 0 tokens
- **Where:** `evals/analyze_results.py:4787-4805` (`_token_usage_breakdown`).
- **Problem:**
  - `token_usage_available is False` sets `found_attempt_meta = True` and skips the record.
  - If no attempt has usage, the function returns `(0.0, 0.0, 0.0)` instead of `None`.
  - A baseline that timed out with missing usage then looks cheapest.
  - Token coverage is inflated to 100%.
  - `fixed_budget_success` (`evals/report_run_metrics.py:614-623`) counts a zero-token success as solved within budget.
- **Fix:** Return `None` when no attempt contributed a count, and add an explicit `token_coverage` field per row.

#### STA-02: Critical, Verified in code. Paired tests treat repeated runs of the same case as independent
- **Where:** `evals/stats.py:449-476` (`_extract_pairs` keys on `(run_id, case_id)`); mirrored at `evals/report_run_metrics.py:371`, `:502`, `:709`.
- **Problem:** With k runs of the same case set, each case contributes k "independent" pairs to McNemar and the bootstrap. In the repro, tripling an identical 28-case result changed McNemar from p=0.289 to p=0.0227, and the bootstrap CI from [-0.036, 0.321] to [0.024, 0.262]. This matters for any analysis that pools more than one `run_id`.
- **Fix:**
  - Use the case as the unit: aggregate runs per case first, or use a case-level cluster bootstrap and a case-level McNemar.
  - Refuse `per_run` aggregation when more than one `run_id` is present.

#### STA-03: High, Reproduced. Success rates and uplifts are computed over unmatched case sets
- **Where:** `evals/analyze_results.py:1205-1207` (`cases = len(condition_rows)`), `:2887-2910` (`_pairwise_uplifts`), `:5212-5227`. Printed by `evals/eval_summary.py:108-117`.
- **Problem:** Suppose condition A ran 10 cases (3 successes) and B only the 5 easiest (3 successes). The report says A=30%, B=60%, uplift +30pp; the matched-pair path says +0.0pp.
- **Fix:** Compute uplifts only over the intersection of case sets, report `cases_expected` and `cases_missing` per condition, and refuse to print uplifts when the case sets differ.

#### STA-04: High, Verified. `power_report` uses a wrong formula and badly understates power
- **Where:** `evals/stats.py:286-289`.
- **Problem:** It computes `sqrt(n_discordant) * |uplift| / sqrt(1 - uplift**2)`, mixing the number of discordant pairs with the overall difference in success rates. Examples:

  | Setting | Reported power | True power |
  |---------|---------------:|-----------:|
  | n=200, π01=0.25, π10=0.10 | 0.245 | 0.949 (simulated) |
  | n=100, π01=0.20, π10=0.10 | 0.079 | 0.386 (simulated) |

  The lead reviewer also checked the n=200 case analytically: reported about 0.24, correct about 0.94.
- **Fix:** `z = |n01 − n10| / sqrt(n_discordant) − z_{α/2}`. This matched simulation to three decimals.

#### STA-05: High, Reproduced. The same paired token delta gets two contradictory 95% CIs in one report
- **Where:** `evals/report_run_metrics.py:605-610` uses `evals.stats.bootstrap_mean_ci` (percentile method); `:747-752` and `:828-865` use a local `scipy.stats.bootstrap(method="basic")`.
- **Problem:** On the same 40 deltas with the same seed, one CI is [274, 19296], which excludes 0, and the other is [-5375, 13818], which includes it.
- **Fix:** Use one bootstrap helper and one interval method everywhere.

#### STA-06: High, Confirmed. No multiple-comparison correction across the pairwise condition tests
- **Where:** `evals/analyze_results.py:5660-5703`, printed at `:5844-5863`; `evals/stats.py:87`, `:394-416`.
- **Problem:** Nine conditions give 36 McNemar tests at an uncorrected α=0.05, so about 1.8 spurious "significant" pairs are expected. Holm is applied only within the per-category family of one primary pair.
- **Fix:** Apply Holm or Benjamini–Hochberg across the pairwise family, and report adjusted p next to raw p.

#### STA-07: High, Reproduced. The `analyze_results --stats` bootstrap is unseeded
- **Where:** `evals/analyze_results.py:5776-5789` calls `analyze_paired_results(...)` with no seed; `evals/stats.py:359` defaults to `seed=None`.
- **Problem:** Two runs on identical data gave different CIs. `report_run_metrics` defaults to seed 42, so the two tools disagree.
- **Fix:** Add `--bootstrap-seed` (default 42), pass it through, and record it in `analysis_config`.

#### STA-08: High, Reproduced. "% of runtime" divides a subset mean by an all-rows mean
- **Where:** `evals/report_run_metrics.py:1381-1395`.
- **Problem:** With 10 cases of 100 s each, one of which has `llm_total_sec=80`, the true LLM share is 8% but 80% is reported. The same applies to the overhead percentages. Truthiness checks also drop genuine zeros.
- **Fix:** Compute a ratio of sums over the same rows, and report how many rows each percentage covers.

#### STA-09: High, Reproduced. Partial token coverage silently shrinks per-case totals and per-attempt means
- **Where:** `evals/analyze_results.py:4787-4790`, `:595-599`.
- **Problem:** Covered attempts are summed, but the divisor is all executed attempts.
- **Fix:** Divide by covered attempts, and expose `token_covered_attempts` and `token_covered_fraction`.

#### STA-10: Medium, Reproduced. An undefined (NaN) Cohen's κ is reported as "almost_perfect" agreement
- **Where:** `evals/report_run_metrics.py:383-401`.
- **Problem:** NaN fails every comparison and falls through to `"almost_perfect"`. `json.dumps` then writes a bare `NaN`, which isn't valid JSON.
- **Fix:** Guard with `math.isfinite` and report "undefined" or `None`.

#### STA-11: Medium, Confirmed. The doubly-robust off-policy estimate is fit in-sample, and the reported `ridge_alpha` isn't the one used
- **Where:** `evals/analyze_results.py:4322`, `:4381-4462`, `:4601-4646`.
- **Problem:** The reward model is fit on the same samples whose residuals form the correction term. `RidgeCV` chooses alphas per action and stratum, but the report hardcodes the configured value.
- **Fix:** Use K-fold cross-fitting, and record the selected alphas.

#### STA-12: Medium, Confirmed. A missing propensity silently becomes 1.0 in IPS, SNIPS and DR
- **Where:** `evals/analyze_results.py:4153-4162` (`_sanitize_propensity`).
- **Fix:** Drop those samples and report how many were dropped.

#### STA-13: Medium, Confirmed. No pass@k estimator, and multi-attempt success uses each condition's own budget
- **Where:** `evals/analyze_results.py:581` (`success = any(...)`), `:4735-4751`; `evals/stats.py:336-348`.
- **Problem:** Conditions with different `max_attempts` are compared head-to-head, and `any_success_aggregation` mixes pass@1, pass@2 and pass@3.
- **Fix:** Add the unbiased pass@k estimator (Chen et al.) with explicit n and k, and report success at a common k (decision D12).

#### STA-14: Medium, Confirmed. `majority_vote_aggregation` requires unanimity when k is even
- **Where:** `evals/stats.py:331-332`.
- **Problem:** The test is a strict `>` against k/2, so for k=2 one success out of two counts as a failure.
- **Fix:** Use `>= ceil((k+1)/2)`, or reject even k.

#### STA-15: Medium, Reproduced. p-values are rounded to two or four decimals
- **Where:** `evals/analyze_results.py:5555-5563`, `:5861`; `evals/report_run_metrics.py:2130-2133`.
- **Problem:** 0.0004 and 0.004 both print as `0.00`, and 0.049 and 0.051 both print as `0.05`.
- **Fix:** Use a dedicated p-value formatter.

#### STA-16: Medium, Confirmed. Bucket-level adaptive rates are diluted by non-adaptive conditions
- **Where:** `evals/analyze_results.py:1942`, `:2344-2432`; fallback source in `evals/eval_summary.py:205`.
- **Fix:** Restrict bucket-level rollups to conditions that have adaptive telemetry.

#### STA-17: Medium, Reproduced. Three documented baseline timing fields never appear in the overhead section
- **Where:** `evals/report_run_metrics.py:1320-1339` (`_TIMING_SUMMARY_KEYS`) vs `:1871-1893` (`_TIMING_DISPLAY_LABELS`).
- **Problem:** `baseline_internal_total_sec`, `baseline_eval_sec` and `baseline_orchestration_sec` have labels but are never accumulated.
- **Fix:** Derive both from one list of keys.

#### STA-18: Medium, Confirmed. `budget_risk_report` neither de-duplicates retried attempts nor keys by `run_id`
- **Where:** `evals/budget_risk_report.py:32-42`, `:283-289`.
- **Fix:** Reuse `_load_and_group_records` and include `run_id` in the grouping key.

#### STA-19: Medium, Confirmed. Three incompatible definitions of "case succeeded"
- **Where:**
  - `evals/analyze_results.py:581`: any attempt succeeded.
  - `evals/select_cases.py:44-73` and `evals/extract_failed_cases.py:87`: the highest attempt succeeded, excluding `infra_resume_eligible` rows.
  - `evals/report_run_metrics.py:902-994`: mixes both.
- **Problem:** The same case can be a success in the summary and a failure in `<run>.failed.txt`.
- **Fix:** One shared `case_outcome(records)` helper.

#### STA-20: Medium, Confirmed. Four overlapping slice tests are reported without correction
- **Where:** `evals/report_run_metrics.py:161-195`, `:1523-1532`, `:1947-1960`.
- **Problem:** Primary, All, Required-only and Controls are four tests on nested subsets of the same data.
- **Fix:** Pre-register the primary endpoint, label the others as exploratory, and name the correction family explicitly.

#### STA-21: Medium, Likely. The report's off-policy numbers can differ from the analyzer's
- **Where:** `evals/report_run_metrics.py:154-156`.
- **Problem:** The OPE minimum-support and propensity-floor parameters aren't passed through, so the report always uses the defaults.
- **Fix:** Expose the same flags, and echo the effective values in the report.

#### STA-22: Medium, Confirmed. The two-pass "absolute uplift vs baseline" isn't a treatment effect
- **Where:** `evals/eval_two_pass_summary.py:187-193`, `:222-227`.
- **Problem:** The rescue pass runs only on baseline failures, so the comparison is about 2× compute against a single pass and can never show a regression.
- **Fix:** Relabel it as cumulative compute, and report the rescue pass's own cost.

#### STA-23: Low, Reproduced. McNemar switches to a chi-square approximation above 25 discordant pairs
- **Where:** `evals/stats.py:68-87`.
- **Problem:** scipy is a hard dependency, so the approximation is unnecessary. α is also hardcoded inside the test.
- **Fix:** Use `scipy.stats.binomtest` unconditionally, and take α as a parameter.

#### STA-24: Low, Reproduced. Holm uses a strict `<`
- **Where:** `evals/stats.py:242`.
- **Problem:** p exactly at α/(m−k+1) isn't rejected.
- **Fix:** Use `<=`, and derive `rejected` from the adjusted p.

#### STA-25: Low, Confirmed. `--strict-input` doesn't control schema validation
- **Where:** `evals/analyze_results.py:163-167`, `:381`.
- **Problem:** One schema-invalid record aborts the analysis even without the flag.
- **Fix:** Apply the flag to schema validation too, or reword its help text.

#### STA-26: Low, Confirmed. Dead statistics code and a counter that carries no information
- **Where:**
  - Unreachable scipy fallbacks: `evals/stats.py:494-540`.
  - No callers: `analyze_condition_matrix` (`evals/stats.py:421-440`), `_tokens_used` (`evals/analyze_results.py:4933-4936`), `_failure_reason` (`evals/report_run_metrics.py:1194-1203`).
  - Unused computation: `behavior_hat` for non-target actions (`evals/analyze_results.py:4637-4643`).
  - `duplicate_attempt_rows_resolved = duplicate_attempt_rows_seen` (`evals/analyze_results.py:392`) always equals the other counter.
- **Fix:** Delete them, or make `analyze_condition_matrix` the single implementation.

### STA structural notes
- **`analyze_results.py`** (5,867 lines) has four very large functions:

  | Function | Lines | Location |
  |----------|------:|----------|
  | `_adaptive_rollup_metrics` | 939 | L1929–2867 |
  | `_summarize_bucket` | 608 | L1185–1792 |
  | `_derive_case_condition_rows` | 606 | L569–1172 |
  | `_adaptive_budget_policy_attempt_metrics` | 441 | L3582–4022 |

  Extraction seams:

  | Lines | Target module |
  |-------|---------------|
  | 400–567 | `eval_record_schema.py` |
  | 324–397 and 4722–4751 | `eval_record_loader.py` |
  | 4023–4646 | `adaptive_ope.py` (the only consumer of numpy and sklearn) |
  | 3125–4022 | `adaptive_attempt_metrics.py` |
  | 5230–5492 and 5555–5634 | `analyze_output.py` |
  | 5646–5863 | a statistics bridge module |

- **Private helpers used as an API:** `report_run_metrics`, `eval_summary`, `eval_two_pass_summary` and `baselines/compare_results` import `analyze_results`' private helpers (`_derive_case_condition_rows`, `_load_and_group_records`, `_summarize_bucket`, and others).
- **Duplicated helpers:**
  - Formatters are implemented four times with different precision.
  - `_format_ope_targets` exists twice, with different output.
  - The bootstrap and the exact binomial test each have two implementations.
  - `_ratio`, `_mean`, `_median` and `_expand_paths` exist in two or three copies.
  - `EVAL_RECORD_SCHEMA_VERSION` is defined in both `evals/eval_types.py:120` and `evals/analyze_results.py:29`.
- **Untyped rows:** the derived row has about 170 keys read as string literals across five modules. `float(x or 0.0)` appears about 60 times, which conflates "missing" with zero (see X-03).
- **Inconsistent condition-name normalization:**
  - `analyze_results` doesn't `.strip()`, while `budget_risk_report` and `extract_failed_cases` do.
  - `eval_summary` detects adaptive conditions by the substring "adaptive".
- **No uncertainty on OPE:** no off-policy estimate carries a confidence interval or standard error.

---

## POL: adaptive policy and online learning

Scope: `evals/adaptive_online_policy.py`, `evals/adaptive_budget_online_policy.py`, `evals/adaptive_controller_packet.py`, `evals/adaptive_controller_runtime.py`, `evals/adaptive_history.py`, `evals/adaptive_tep.py`, `evals/bootstrap_policy_state.py`, `evals/eval_policy_online.py`, `evals/eval_policy_features.py`, `evals/eval_policy_budget.py`, `evals/policy_feature_scaling.py`, `evals/stage_a_contract.py`, `evals/controller_summarizer.py`, `evals/_policy_utils.py`, `evals/export_policy_dataset.py`.

#### POL-01: Critical, Verified (decay default and code) and Reproduced (simulation). The default decay of 0.95 is applied to every arm on every update, so the bandit can barely learn
- **Where:**
  - `evals/adaptive_online_policy.py:940-953` (`_apply_recency_decay`, called from `:928` on every update); `evals/adaptive_budget_online_policy.py:1191`
  - Defaults: `DEFAULT_ADAPTIVE_ONLINE_DECAY = 0.95` and `DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY = 0.95` (`evals/eval_types.py:157`, `:162`)
  - `evals/configs/openai_adaptive.template.toml` doesn't override it, although `evals/configs/openai_adaptive_probe_decay_098.template.toml:3` says that template uses 1.0.
- **Problem:** Every update applies `A ← γA + (1−γ)λI` and `b ← γb` to all 10 arms. The effective window is about 20 observations in total, roughly two per arm, against 53 features (56 with interactions). In the reviewer's simulation:
  - After 2,000 updates, the correlation between the estimated and true weights was 0.39, against 0.99 at γ=1.0.
  - In a Bernoulli-reward simulation, the default put 18% of post-warm-up decisions on the best arm (uniform random gives 10%), against 51% at γ=1.0.

  A "learned policy" from a default run is close to uniform exploration plus the fixed score biases.
- **Fix:**
  - Default to 1.0 (decision D6).
  - When decay < 1, decay only the arm being updated, behind a documented flag for non-stationary settings.
  - Add telemetry for the effective evidence per arm (`trace(A_k) − dλ`).

#### POL-02: High, Confirmed. TEP critic and update LLM calls are missing from token, latency and reward accounting
- **Where:** `evals/adaptive_tep.py:754` (`run_post_attempt_learning`), called from `evals/eval_attempt_flow.py:7751`, after the last usage aggregation at `:7076`. `_aggregate_attempt_llm_usage` reads only `stage_a_meta` and `patch_meta` (`evals/run_eval.py:4166`).
- **Problem:** Up to three `generate_json` calls per node over up to four nodes, so up to 12 calls per attempt, are never counted.
- **Fix:** Merge each call's metadata into a TEP usage bucket and fold it into the attempt totals.

#### POL-03: High, Reproduced. In shadow mode, `behavior_propensity` is wrong, inflating importance weights up to 10×
- **Where:** `evals/eval_policy_online.py:92-95`; duplicated at `evals/adaptive_online_policy.py:675-680` and `evals/eval_policy_budget.py:47-49`. Consumed by `evals/analyze_results.py:4088`.
- **Problem:** In shadow mode the executed action is always the baseline action, so its probability is 1. The code instead returns `decision.propensity` whenever the selected action equals the behavior action.
  - With ε=0.10 and no warm-up, 368 of 400 agreeing events get 0.91 instead of 1.0.
  - During warm-up, agreeing events get 0.1, an importance weight of 10.

  Shadow mode is the documented recommended first rollout.
- **Fix:** Return 1.0 whenever the executed action wasn't sampled by the policy, that is, when `mode != "on"` or the applied action differs from the selected one.

#### POL-04: High, Confirmed. `bwk_v1` has no Lagrangian cost control: the dual variable is computed and logged but never used
- **Where:** `evals/adaptive_budget_online_policy.py:856`; the only enforcement is the hard mask at `:700`.
- **Problem:**
  - `_bwk_dual_variable` appears only in init, load, save and log code.
  - The decision and reward paths never read it.
  - Its update is wrong too: `overspend = max(0, charged − remaining_before)` is 0 until the budget is nearly exhausted.

  Calling this arm "Bandits with Knapsacks" isn't supported by the implementation.
- **Fix:** Subtract `λ · cost_estimate / budget` from each arm's UCB index and update `λ ← max(0, λ + η(spend_rate − target_rate))`. Otherwise, rename the architecture and drop the dual telemetry.

#### POL-05: High, Confirmed. One policy controller is shared across worker threads and across both adaptive conditions
- **Where:**
  - `evals/eval_main_flow.py:923`; `--jobs` defaults to 3 (`evals/run_eval.py:1126`).
  - `evals/eval_attempt_flow.py:2941`; `evals/adaptive_online_policy.py:379`, `:574`, `:577`.
- **Problem:**
  - The `RLock` makes each call atomic, but not their order. RNG draw order, `decision_index` and the warm-up boundary vary from run to run under `--seed`.
  - The budget policy is eligible for both `layered_adaptive_gated` and `layered_adaptive_llm`, so the two conditions being compared update the same model and aren't statistically independent.
- **Fix:**
  - Create one controller per condition.
  - Derive per-unit RNGs from `(seed, case_id, condition, attempt)`.
  - Either serialize updates in deterministic mode or document that `--jobs 1` is required for reproducible learning.
- **Also found by:** the ORC reviewer.

#### POL-06: High, Reproduced. The `bias` feature equals the sum of the `last_helper_action_*` one-hot features
- **Where:** `evals/eval_policy_features.py:146` (`"bias": 1.0`) and `:210-211` (one-hot over all 10 actions, initialized to `skip_helper` at `evals/eval_attempt_flow.py:2245`).
- **Problem:** Over 2,000 simulated rows, `max|Σ one-hots − bias| = 0.0`. That direction is permanently unidentified, and its exploration bonus never shrinks.
- **Fix:** Drop one one-hot level, or drop the explicit `bias`.

#### POL-07: High, Reproduced. The runtime score biases are as large as a full success reward, and three sources disagree on the default
- **Where:**
  - `evals/eval_types.py:158` (`search_only=-1.00,frame_only=-0.50`; the argparse default at `evals/run_eval.py:2216`)
  - vs `evals/adaptive_online_policy.py:76` (−0.50 and −0.25) and both policy docs (−0.50 and −0.25)
  - `default_online_action_score_biases()` is never used for CLI runs (`evals/eval_config.py:172`).
- **Problem:** The success reward is 1.0 (`evals/eval_types.py:1182`), so −1.0 acts as a hard veto. `search_only` and `frame_only` got 1.7% and 3.4% of post-warm-up decisions, against about 10% under uniform selection.
- **Fix:** Use one source of truth, and express biases as a small fraction of the reward scale or as a prior on `b_k` (decision D6).

#### POL-08: High, Likely. The budget reward charges `increase_budget` the full attempt cost and treats `keep_budget` as free
- **Where:** `evals/eval_policy_budget.py:378-382`.
- **Problem:** The Stage-B proposal spends tokens either way. At `lambda_tokens=0.08` and about 5k tokens, escalating costs a 0.40 penalty against a success reward of 1.0, so it must raise P(success) by 40 points just to break even.
- **Fix:** Charge only the incremental cost of escalation, or charge both arms.

#### POL-09: Medium, Reproduced. `percentile_v1` re-fits scaling caps on every decision, and the caps only affect count-like features
- **Where:** `evals/adaptive_online_policy.py:439-471`; `evals/policy_feature_scaling.py:245-256`.
- **Problem:**
  - The same raw value scales differently over a run: `stagnation_streak=3` became 1.26, then 0.71, then 0.60 as history grew from 40 to 200 samples. Those incompatible vectors are summed into one linear model.
  - Caps fitted for features that aren't count-like are logged but never applied.
  - An all-zero history produces cap=1.0, which amplifies values.
- **Fix:** Freeze caps after the first `min_samples` observations (or refit on a schedule and reset A and b), and stop reporting caps that aren't applied.

#### POL-10: Medium, Reproduced. The `hierarchical_v1` "context" family does nothing under the default config
- **Where:** `evals/eval_policy_features.py:270-285`, `:414-428`.
- **Problem:** With the default ladder and `max_ups_per_case=1`, the widening delta is `floor(2·1/4) = 0`, so the discrete controls don't change. `tier_binary` does change them on the same ladder, so the two architectures aren't comparable at default settings.
- **Fix:** Make the first step above base widen by at least 1, or document that `max_ups_per_case ≥ 2` is required.

#### POL-11: Medium, Confirmed. The delayed rescue bonus is awarded on timing alone, and two declared inputs are unused
- **Where:** `evals/eval_policy_online.py:59-75`. The `case_difficulty` and `case_snapshot_dependency` parameters are never read.
- **Problem:** Any helper invocation one or two attempts before a success earns up to 0.75. `skip_helper` can never earn it, so this is a built-in bias toward invoking the helper.
- **Fix:** Award the bonus only when the evidence actually appears in the successful attempt's prompt (evidence-ID overlap is already tracked), and remove the dead parameters.

#### POL-12: Medium, Confirmed. The logged feature vector is scaled differently depending on whether a decision was made
- **Where:** `evals/eval_attempt_flow.py:3313-3316` (helper) and `:2937-2940` (budget); consumed by the doubly-robust regression at `evals/analyze_results.py:4103`.
- **Problem:** Under `percentile_v1`, one run logs feature vectors in two different feature spaces.
- **Fix:** Add a `scale_only()` method to the controller, or don't log a scaled vector when no decision was made.

#### POL-13: Medium, Confirmed. `export_policy_dataset` accepts only schema version 4, and `inf` values pass through
- **Where:** `evals/export_policy_dataset.py:32-35`, `:374`.
- **Problem:** Version 3 logs are silently skipped, even though the design doc still describes version 3.
- **Fix:** Support an explicit tuple of versions with per-version field fallbacks, and reject non-finite values.

#### POL-14: Medium, Confirmed. Bootstrapped state never rebuilds `feature_scaling_history`
- **Where:** `evals/bootstrap_policy_state.py:824-832`, `:851-854`.
- **Problem:** A `percentile_v1` warm start resumes in a different feature space from the one its A and b matrices were built in.
- **Fix:** Refuse to bootstrap `percentile_v1` state, or persist the caps as frozen overrides.

#### POL-15: Medium, Likely. Bootstrap replay has no case exclusion and no propensity correction
- **Where:** `evals/bootstrap_policy_state.py:197-258`, `:807`.
- **Problem:** The documented warm-start workflow lets the policy see the exact cases it will be evaluated on. Replaying shadow-mode logs loads evidence into one arm only.
- **Fix:** Add `--exclude-case-ids` and `--exclude-run-ids`, and refuse (or warn loudly) when the replayed cases overlap the target case list.

#### POL-16: Medium, Confirmed. `_nan_counts` is read and written outside the lock in `decide()`
- **Where:** `evals/adaptive_online_policy.py:520-524`; `evals/adaptive_budget_online_policy.py:663-680`.
- **Problem:** Concurrent increments are lost, and this counter is the only signal of NaN features.
- **Fix:** Update it under the lock.

#### POL-17: Medium, Confirmed. TEP equilibrium can't tell a failed critic from convergence, and its accept test compares two draws from the same distribution
- **Where:** `evals/adaptive_tep.py:167-175`, `:443-478`, `:784`, `:801`.
- **Problem:**
  - A missing critic payload reads as `stable`.
  - The "free" and "nudged" critics differ only by a label, so prompt updates are accepted about 50% of the time on noise.
  - `float(data.get("overall_score_10") or 0.0)` raises on string scores.
- **Fix:** Add a `critic_parsed` flag, use averaged or paired comparisons before accepting an update, and parse scores defensively.

#### POL-18: Medium, Confirmed. The two event-dedup implementations disagree on replay order, and neither is chronological
- **Where:** `evals/bootstrap_policy_state.py:524-529` (pop and re-insert moves duplicates to the end); `evals/export_policy_dataset.py:162` (sorted by `case_id`).
- **Problem:** Because decay depends on order, bootstrapped state depends on glob order. Both implementations also fall back to second-resolution timestamps as a tiebreaker.
- **Fix:** One shared loader that sorts by `(created_at_utc, run_id, case_id, attempt, decision_index)` and keeps the first occurrence.

#### POL-19: Medium, Reproduced. `v1_interactions` builds interaction features from raw values, and one is log-scaled by an accidental name match
- **Where:** `evals/eval_policy_features.py:216-224`; `evals/policy_feature_scaling.py:67-71`.
- **Problem:** `attempt_frac_x_stagnation_streak` ends in `_streak`, so it is treated as a count and log-scaled. The interactions end up on three inconsistent scales.
- **Fix:** Compute interactions from scaled features, and rename them so they don't match the count heuristic.

#### POL-20: Medium, Confirmed. Each decision does avoidable work
- **Where:** `evals/adaptive_online_policy.py:453-459`, `:864-879`.
- **Problem:** Every decision deep-copies and re-sorts the whole scaling history, and runs 10 independent 53×53 `np.linalg.solve` calls.
- **Fix:** Cache each arm's inverse matrix with Sherman–Morrison rank-1 updates, invalidated on decay or feature expansion, and maintain the caps incrementally.

#### POL-21: Medium, Confirmed. `_ensure_feature_state` silently discards a learned arm on a shape mismatch and zero-fills removed features
- **Where:** `evals/adaptive_online_policy.py:800-802`, `:833-839`.
- **Problem:** The docs say feature dimensions must match, but only the scaling profile is checked.
- **Fix:** Fail (set `load_error`) when a feature disappears, and emit telemetry whenever an arm is reset.

#### POL-22: Low, Confirmed. A read-only scoring path mutates the model
- **Where:** `evals/adaptive_online_policy.py:870-877`; `evals/adaptive_budget_online_policy.py:1130-1137`.
- **Problem:** The re-regularized matrix from a `LinAlgError` is written back into `A`, adding λI each time with no telemetry.
- **Fix:** Use the regularized matrix for the current score only.

#### POL-23: Low, Confirmed. The helper and budget controllers get the same `random_seed`
- **Where:** `evals/eval_main_flow.py:317`, `:372`.
- **Problem:** Their exploration streams are correlated.
- **Fix:** Derive a distinct sub-seed per controller.

#### POL-24: Low, Confirmed. Cold-start tie-breaking is biased, and shadow-mode `action_counts` record the counterfactual action
- **Where:** `evals/adaptive_online_policy.py:570-572`, `:595`; `evals/adaptive_budget_online_policy.py:616-619`.
- **Problem:** Ties go deterministically to `invoke_helper`, or to `time` for the budget family. In shadow mode, `action_counts` count the selected action, which never ran.
- **Fix:** Break ties randomly with the policy's RNG, and record both selected and applied counts.

#### POL-25: Low, Confirmed. Documentation and runtime defaults have drifted apart
- **Where:** `docs/adaptive_online_policy_design.md`; `docs/ADAPTIVE_POLICY_QUICK_REFERENCE.md`.
- **Problem:**

  | Item | Docs | Code |
  |------|------|------|
  | Actions | 8 | 10 |
  | Reward hints (success / progress / materialization / stagnation) | 0.25 / 0.5 / 0.25 / 0.5 | 1.0 / 1.0 / 0.5 / 1.0 |
  | Helper event schema version | 3 | 4 |
  | `alpha` | 0.6 and 0.8 in different places | 0.6 CLI, 0.8 dataclass |
  | Warm-up | decisions 1–200 | 50 |
  | Algorithms | "linucb only" | `linucb` and `thompson` |

  `bwk_v1` and `v1_interactions` aren't documented at all.
- **Fix:** Generate the defaults tables from the constants via `evals.doc_sync`.

### POL structural notes
- **Three copies of the LinUCB logic:** `adaptive_online_policy.py` and `adaptive_budget_online_policy.py` duplicate about 400 lines (feature state, scoring, Thompson sampling, update, decay, scaling, persistence, event append). `bootstrap_policy_state.py:730-821` is a third copy. Extract one `LinUCBBandit` core.
- **Size:** `adaptive_budget_online_policy.py` (1,450 lines) and `adaptive_controller_packet.py` (1,745) exceed the soft limit.
- **Event schemas written by hand:** `BudgetPolicyEvent` has 70+ fields plus a hand-written `to_json`, and `_sync_adaptive_budget_policy_metadata` restates them again as a 120-key dict. Derive the dict from `dataclasses.fields()`.
- **Fragile interfaces:**
  - Reward helpers return 10- and 12-element positional tuples (`eval_policy_features.py:712`, `:1051`).
  - `_finalize_adaptive_online_attempt` takes 44 parameters and `_finalize_adaptive_budget_policy_attempt` 48, some of them dead.
  - The `**legacy_kwargs` escape hatches and `getattr(..., default)` lookups let typos degrade to default values.
- **What the filter does and doesn't cover:** the layer-3 filter function (`_filter_context_request_payload_for_action`, `eval_types.py:383`) is correct and called on every Stage-A payload. The leaks are downstream; see ATT-03, ATT-04 and ATT-06.
- **What the training data represents:** the pregate, budget exhaustion, saturation guards and single-feasible-action shortcuts all skip `decide()` without logging an event. The learned weights and the OPE estimates therefore describe only the non-pregated subpopulation with more than one feasible action. Say so wherever full-set pass rates are reported next to them.
- **Done right:** `stage_a_contract.py`, `adaptive_history.py`, `controller_summarizer.py`, `adaptive_controller_runtime.py` and `_policy_utils.py` are clean. The single-solve trick in `linucb_exploit_explore` is correct.

---

## LLM: patchers, prompts and LLM I/O

Scope: `evals/patchers/` (`openai_compat_patcher.py`, `base.py`, `baseline_patcher.py`, `command_patcher.py`, `heuristic_patcher.py`, `null_patcher.py`, `payload_types.py`), `evals/eval_prompts.py`, `evals/lambda_rlm_backend.py`, `evals/token_counting.py`, `evals/toon_llm_encoder.py`, `evals/category_mapping.py`.

The default `response_mode` is `edits` (JSON find/replace) in the CLI and in all 27 shipped configs that set it. The diff-parsing findings below affect diff mode, the fallback from edits to diff, and the SWE-Bench paths.

#### LLM-01: High, Verified. Multi-file diffs are reduced to the first file
- **Where:** `evals/patchers/openai_compat_patcher.py:830` (`_extract_git_diff_blocks`; `i = starts[0]` at `:836`, `break` at `:877`).
- **Problem:** Only the first `diff --git` block is consumed, and scanning stops at the first line that doesn't look like a diff. The lead's repro:

  | Layout | Second file kept? |
  |--------|-------------------|
  | Adjacent diffs in one fence | yes |
  | Prose between diffs in one fence | no |
  | Two separate fences | no |
  | Unfenced with prose between | no |

  The result is a structurally valid single-file patch, so a half-repair is recorded as a genuine attempt.
- **Severity:** the area reviewer rated this Critical; the lead lowered it to High because edits mode is the default.
- **Fix:** Loop over all start positions, capture each block independently, and concatenate them.

#### LLM-02: High, Reproduced. `finish_reason == "length"` isn't checked outside full-file mode, and `_fix_hunk_counts` turns truncated diffs into valid ones
- **Where:** `evals/patchers/openai_compat_patcher.py:6027`, `:3383`; the check exists only at `:5892`.
- **Problem:** A hunk cut off at `max_tokens` gets its counts recomputed from the truncated body, so `@@ -6,4 +6,4 @@` becomes `@@ -6,2 +6,1 @@`: a valid patch that deletes a line without adding its replacement. In edits mode, the recorded failure cause is wrong (`no_json_payload`). Truncation is more likely in conditions whose prompts ask for prose first.
- **Fix:** Record `finish_reason` in the `_chat_completion` metadata, and treat `"length"` as a distinct terminal error on every parse path.

#### LLM-03: High, Reproduced. Blank context lines without the leading space are neither counted nor normalized
- **Where:** `evals/patchers/openai_compat_patcher.py:6065` (`if not b: continue`); `evals/eval_patches.py:619`.
- **Problem:** Locally, the whole patch is rejected as malformed. On SWE-Bench there's no validator, and `git apply` silently drops every change after the undercount (`evals/swebench_pro/score_docker.py:105`).
- **Fix:** Rewrite empty body lines to `" "` and count them as context.

#### LLM-04: High, Reproduced. Reasoning text leaks into the content, so the model's abandoned draft diff wins over its final answer
- **Where:**
  - `evals/patchers/openai_compat_patcher.py:734` (`_strip_think`); `baselines/openai_compat.py:35`, `:188`
  - `allow_reasoning_fallback` and `enable_thinking` default to on (`evals/run_eval.py:1702`, `:1718`).
  - Reasoning is copied into the content at `:4684`.
- **Problem:** The regex needs a matched `<think>…</think>` pair. An orphan closing tag or an unclosed opening tag leaves the whole chain of thought in place, and the first diff block, which is the rejected draft, is extracted.
- **Fix:** Drop everything up to the last `</think>`, strip an unclosed `<think>` to the end, and prefer the last diff block.

#### LLM-05: High, Reproduced. Schema-legal `find=""` and `count=0` corrupt the target file
- **Where:** `evals/patchers/openai_compat_patcher.py:5679` (`replace_count = count or -1`), `:5681`, `:5698`; the schema at `:404-420` allows both values.
- **Problem:** `count or -1` turns 0 into "replace all", and `"" in text` is always true. `find="", count=0` on `"abc\n"` produced `'# a# b# c# \n# '`.
- **Fix:** Require `minLength: 1` on `find` and `minimum: 1` on `count`, reject an empty `find`, and treat `count <= 0` as 1.

#### LLM-06: High, Reproduced. A misconfigured `max_tokens` silently collapses the prompt to one character
- **Where:** `evals/native_stage_b_flow.py:101-113` (returns `1`); used at `evals/eval_attempt_flow.py:4081`.
- **Problem:** When `max_tokens + 1024 >= context_window`, the prompt cap becomes 1, and the model receives the literal prompt `"Y"`. For example, window 8,192 with max_tokens 8,192. The window is probed from the server, so a llama.cpp server started with `-c 8192` and a config using 8,192 max tokens turns every case into an apparent total model failure.
- **Fix:** Raise a configuration error (or clamp `max_tokens`) when the usable budget is ≤ 0, and enforce a sensible minimum prompt size.

#### LLM-07: High, Reproduced. Markdown bullets after a diff are absorbed as deletion lines
- **Where:** `evals/patchers/openai_compat_patcher.py:861-871`.
- **Problem:** A trailing `- This changes b to 20…` line becomes a `-` line. The patch validates but fails to apply, so a correct one-line fix is recorded as an apply failure.
- **Fix:** Require body lines to look like code, and stop once each hunk's declared counts are satisfied.

#### LLM-08: High, Reproduced. `HeuristicSnapshotPatcher` can never produce a diff that applies
- **Where:** `evals/patchers/heuristic_patcher.py:130-142`.
- **Problem:** `splitlines(True)` combined with `lineterm=""` and `"\n".join(...)` inserts a blank line after every body line. The diff is rejected as malformed, while the patcher reports `reason: patched`. The documented smoke-test purpose isn't met.
- **Fix:** Use `lineterm="\n"` with `"".join(...)`, and add a test that the diff applies.

#### LLM-09: High, Reproduced. A single edit to a CRLF file rewrites every line ending in the file
- **Where:** `evals/patchers/openai_compat_patcher.py:5692-5699`; `evals/eval_patches.py:727`.
- **Problem:** The code returns LF-normalized text while the original keeps CRLF. Strict `patch` fails, and the relaxed fallback writes the whole file as LF. On SWE-Bench, which has no fallback, the patch simply fails.
- **Fix:** Re-apply the original line-ending style, and normalize newlines only for matching.

#### LLM-10: Medium, Reproduced. Creating a new file through a plain unified diff is always rejected
- **Where:** `evals/patchers/openai_compat_patcher.py:762-767`.
- **Problem:** The emitted headers are `diff --git a//dev/null b/new.py` and `--- a//dev/null`.
- **Fix:** Emit `--- /dev/null` and `diff --git a/<new> b/<new>`.

#### LLM-11: Medium, Confirmed. `generate_text` and `generate_json` ignore the `max_propose_sec` deadline
- **Where:** `evals/patchers/openai_compat_patcher.py:1276`, `:1335`.
- **Problem:** Each call can take up to 7 × timeout plus backoff, about 14 minutes, against a 300-second budget. TEP makes three such calls per attempt, and Stage A two.
- **Fix:** Pass a deadline through both methods, as `propose_patch` does.

#### LLM-12: Medium, Reproduced. Fuzzy block recovery burns about 24 s of CPU per failed edit, outside any deadline
- **Where:** `evals/patchers/openai_compat_patcher.py:5748` (`_apply_fuzzy_block_replacement`) and `:5728`.
- **Problem:** Measured at 2.8 s for a 2,000-line file and 23.7 s at the 5,960-line cap. The whitespace-normalizing regex at `:5728` also backtracks polynomially, 1.65 s in one measurement.
- **Fix:** Pre-filter candidate windows cheaply, cap the total wall time, and merge the adjacent whitespace character classes.

#### LLM-13: Medium, Reproduced. The forced no-op safety check never fires for patcher-generated forced patches
- **Where:** `evals/patchers/openai_compat_patcher.py:5987` uses the marker `"# llmdebug-force-attempt"`; `evals/eval_patches.py:827` checks for `"# llmdebug-force-attempt-"`, with a trailing dash. Consumer at `evals/eval_attempt_flow.py:6645`.
- **Problem:** The marker-only patch is scored as a genuine repair attempt.
- **Fix:** Share one constant.

#### LLM-14: Medium, Confirmed. The specific failure reason is overwritten by a stale transport error
- **Where:** `evals/patchers/openai_compat_patcher.py:3554`, `:3592`, `:3872`, `:4179`, `:4216`, `:4491`.
- **Problem:** `{"error": reason, **meta}` lets `meta["error"]` replace the specific reason, such as `no_json_payload` or `find_not_found`, so failure statistics misattribute parse failures to transport errors.
- **Fix:** Build `{**meta, "error": reason}` and keep the transport error under a separate key.

#### LLM-15: Medium, Reproduced. `str.splitlines()` is used to generate diffs
- **Where:** `evals/patchers/openai_compat_patcher.py:5934`; `evals/eval_patches.py:891`, a byte-identical duplicate.
- **Problem:** It splits on `\f`, `\x0b`, `\x1c`, `\x85` and ` `, and strips `\r`. There's also no `\ No newline at end of file` marker.
- **Fix:** Normalize newlines explicitly, split on `"\n"`, and emit the marker.

#### LLM-16: Medium, Likely. Estimating tokens as chars/4 under-counts code
- **Where:** `evals/eval_types.py:141`; `evals/native_stage_b_flow.py:114`.
- **Problem:** Code tokenizes at roughly 3.3–3.7 characters per token, so "fitting" prompts still overflow, and the server's 400 response wastes a call.
- **Fix:** Check the cap with the existing tiktoken-based `estimate_text_token_count`.

#### LLM-17: Medium, Likely. Retries change the system message, which defeats prefix caching
- **Where:** `evals/patchers/openai_compat_patcher.py:3416-3430`, `:3526`.
- **Fix:** Keep the system message byte-identical, and append the stronger formatting instruction as a trailing turn.

#### LLM-18: Medium, Likely. Structured-output support is re-probed on every case
- **Where:** `evals/patchers/openai_compat_patcher.py:3526-3540`.
- **Problem:** The `json_schema`, then `json_object`, then plain escalation isn't remembered. A server that rejects `json_schema` costs one wasted call per case, and a server that ignores `response_format` produces duplicate calls.
- **Fix:** Cache the working tier per endpoint and model for the run.

#### LLM-19: Medium, Confirmed. Recovery is inconsistent: some parse failures fall back to diff mode, structurally identical ones abort
- **Where:** `evals/patchers/openai_compat_patcher.py:3721-3771`, `:4962`, `:5657`.
- **Problem:** `find_not_found` falls back to diff mode, but `bad_edit_entry`, `bad_path` and `file_not_found` abort. `bad_json_response` is never retried.
- **Fix:** Route all payload-shape failures through the diff fallback, and treat `bad_json_response` as retryable.

#### LLM-20: Medium, Confirmed. The Stage-B section `priority` field is never read, despite the docstring
- **Where:** `evals/eval_prompts.py:771` (docstring), `:995`, `:1186-1252`, `:1311`, `:1375`.
- **Problem:** 31 `"priority"` entries are ignored in favour of a hardcoded `drop_order`, and the two orderings disagree. The `diagnosis_refinement` section is never trimmed and sorts after `closing`.
- **Fix:** Drive `drop_order` from `priority` or delete the field. Use a single registry of sections.

#### LLM-21: Medium, Likely. Reported TOON token savings include plain whitespace removal
- **Where:** `evals/toon_llm_encoder.py:139`; `evals/token_counting.py:43`.
- **Problem:** The baseline is `json.dumps(indent=2)`. The "savings" are reported even when no TOON library is available, and the tokenizer silently falls back to `o200k_base` for non-OpenAI models.
- **Fix:** Use minified JSON as the baseline, and report the tokenizer resolution mode.

#### LLM-22: Low, Confirmed. Backoff has no jitter and ignores `Retry-After`
- **Where:** `evals/patchers/openai_compat_patcher.py:5495`, `:122`.

#### LLM-23: Low, Confirmed. `a/` and `b/` prefixes are stripped unconditionally
- **Where:** `evals/patchers/openai_compat_patcher.py:816-818`.
- **Problem:** A real top-level directory named `a` or `b` gets mangled.

#### LLM-24: Low, Confirmed. Unmapped dataset bug types silently collapse into `logic_error`
- **Where:** `evals/category_mapping.py:71`, `:117`, `:174`.
- **Problem:** Only 30 of PyResBugs' 62 ODC acronyms are mapped, and the fallback then maps to snapshot dependency "helpful". See DAT-21.
- **Fix:** Return "unknown" and count those cases separately.

#### LLM-25: Low, Confirmed. The λ-RLM backend writes the API key to a temp file and folds child stderr into exception text
- **Where:** `evals/lambda_rlm_backend.py:264`, `:271`, `:286-287`.
- **Fix:** Pass the key through the environment only, and scrub key values from stderr.

#### LLM-26: Low, Likely. Repository content is put into prompts without structural delimiters, and one parser reads the prompt back
- **Where:** `evals/eval_prompts.py:1070`; `evals/patchers/openai_compat_patcher.py:5999` (`_extract_file_index_paths`).
- **Problem:** A case file containing a line that starts with "Available files" can redirect the forced-patch target list.
- **Fix:** Pass the file index as structured data, and fence untrusted sections.

### LLM structural notes
- **`openai_compat_patcher.py`** is 6,082 lines, about 5× the soft limit. Extraction seams:

  | Target module | Lines | Size | Notes |
  |---------------|-------|------|-------|
  | `openai_transport.py` | 144–390, 462–530, 4705–5375, 5489–5540 | ~1,100 | HTTP transport, client pooling, retries |
  | `openai_probes.py` | 2348–2994 | ~650 | self-contained server probing |
  | `diff_text.py` | 708–885, 5934–6082 | ~320 | pure; where 6 of the 9 confirmed High findings live |
  | `edits_apply.py` | 5642–5812 | ~170 | pure |
  | `openai_response.py` | 5375–5490, 5593–5640, 5821–5930 | ~250 | response accessors |

- **Sync/async twins:** see X-04.
- **Three independent JSON extractors with different robustness:** `openai_compat_patcher.py:5841`, `lambda_rlm_backend.py:114`, and the pydantic path at `lambda_rlm_backend.py:126`.
- **Parameter counts:** `_build_stage_b_prompt_sections` takes 30 parameters, `_propose_patch_via_edits` 20, `_propose_patch_via_diff` 19, `_chat_completion` 18. Twelve transport and compatibility parameters are threaded through identically and could become one `ChatRequestConfig`.
- **`PatcherMeta`** (`payload_types.py`) declares 29 keys, but about 60 more are written and passed through `cast`. Split it into composed per-concern TypedDicts.
- **Prompt section ordering** in `eval_prompts.py` lives in three places that disagree: construction order, two `order_map` dicts, and `drop_order`.
- **Missing test:** no end-to-end round trip from model text through extraction, `_fix_hunk_counts` and `apply_diff` to the final file content. A table-driven corpus of about 15 realistic outputs would have caught LLM-01 through LLM-10.

---

## CAP: core capture and serialization

Scope: `src/llmdebug/` modules `capture.py`, `serialize.py`, `config.py`, `output.py`, `compact_keys.py`, `detail_filter.py`, `error_categories.py`, `hypotheses.py`, `_debug_session.py`, `_snapshot_loader.py`, `_formatting.py`, `_presentation.py`, `_json.py`, `_coerce.py`, `_hashing.py`, `snapshot_types.py`, `snapshot_diff.py`, `export_artifact.py`, `toon_encoder.py`, `redaction_policy.py`, `hooks.py`, `log_capture.py`, `async_context.py`, `_rate_limiter.py`, `__init__.py`.

#### CAP-01: Critical, Verified. A local integer wider than 64 bits aborts the whole capture
- **Where:** `src/llmdebug/_json.py:107`, via `src/llmdebug/output.py:234`. `to_jsonlike` passes ints through unchanged (`src/llmdebug/serialize.py:770`).
- **Problem:** orjson raises `TypeError: Integer exceeds 64-bit range` and doesn't consult `default=`. A frame containing `big = 2**70` produces no snapshot at all; the lead reproduced this through `capture_exception`. Callers swallow the error, so under the pytest plugin nothing is shown unless `debug=True`.
- **Fix:** In `to_jsonlike`, convert ints outside the 64-bit range with `safe_repr`. Add a hostile-input test suite (see CAP-02 and CAP-04).

#### CAP-02: High, Reproduced. An exception whose `__str__` raises aborts the capture
- **Where:** `src/llmdebug/capture.py:94` (`_clean(str(exc))`), `:98` (`str(a)` on the args).
- **Problem:** The standard library's `traceback` module guards this case; llmdebug doesn't.
- **Fix:** Add a `_safe_str` helper and use it for the message, args and notes.

#### CAP-03: High, Verified. `redaction_profile="prod"` and `pii_safe=True` don't redact secret-valued local variables
- **Where:** `src/llmdebug/redaction_policy.py:12` (`PII_PATTERNS`); `src/llmdebug/serialize.py:98` (`redact_jsonlike`).
- **Problem:** The patterns match `key=value` inside a single string, but structured locals keep the key and the value apart. The lead's repro found `password="hunter2"` and the `api_key` values verbatim in a prod-profile snapshot. `redact_keys=True` hides the variable name and keeps the value.
- **Fix:** Add redaction keyed on sensitive names: dict keys and local-variable names matching a sensitive-name pattern get their values replaced. Cover connection strings such as `postgres://user:pw@host`.

#### CAP-04: High, Reproduced. Long snapshot names fail with `ENAMETOOLONG`, losing the snapshot
- **Where:** `src/llmdebug/output.py:228-258`; the pytest plugin derives the name from `item.nodeid` (`src/llmdebug/pytest_plugin.py:373`).
- **Problem:** Parametrized tests with long IDs exceed the 255-byte filename limit.
- **Fix:** Truncate the sanitized name to about 120 characters and append a hash of the full node ID.

#### CAP-05: Medium, Reproduced. `.traceback.txt` sidecar files are never cleaned up for pytest captures
- **Where:** `src/llmdebug/output.py:101` (`with_suffix("").with_suffix(".traceback.txt")`).
- **Problem:** The name is wrong whenever the snapshot name contains a dot, which every `test_foo.py` node does. Six captures with `max_snapshots=2` left six orphaned sidecars. The existing test uses a name without a dot.
- **Fix:** Use `snapshot.with_name(snapshot.stem + ".traceback.txt")`.

#### CAP-06: Medium, Confirmed. `--mode redacted` can produce an unredacted artifact labelled "redacted"
- **Where:** `src/llmdebug/export_artifact.py:298-314`.
- **Problem:** With the `dev` profile or no profile, no redactors are configured, the payload is deep-copied unchanged, and it's stamped `export_mode="redacted"`.
- **Fix:** Reject `dev` and `None` in redacted mode, or record `redaction_applied: false` and warn.

#### CAP-07: Medium, Likely. The one-slot cache in `compile_redactors` is racy
- **Where:** `src/llmdebug/serialize.py:43-66`.
- **Problem:** The cached patterns and the cached result are assigned in two separate statements. Threads using different configs can get an empty redactor list. This is reachable because `threading.excepthook` is installed.
- **Fix:** Assign one `(patterns, compiled)` tuple atomically, or use `functools.lru_cache`.

#### CAP-08: Medium, Reproduced. `get_source_snippet` bypasses `linecache`
- **Where:** `src/llmdebug/capture.py:192-220`.
- **Problem:** The `os.path.exists` check returns nothing for IPython cells, `exec` and zipimport sources. Real files are also re-read once per frame.
- **Fix:** Use `linecache.getlines(filename, frame.f_globals)`.

#### CAP-09: Medium, Reproduced. Function arguments are serialized twice per frame
- **Where:** `src/llmdebug/capture.py:318-330`.
- **Problem:** `locals` already contains the arguments, so array statistics are recomputed and the payload grows.
- **Fix:** Emit only the argument names, or serialize once and share the result.

#### CAP-10: Medium, Reproduced. Set locals are fully rendered and sorted before truncation
- **Where:** `src/llmdebug/serialize.py:726-731`.
- **Problem:** A 200,000-element set takes 40 ms, against 0.017 ms for the equivalent list.
- **Fix:** Use `heapq.nsmallest(max_items + 1, …)` and cap the scan.

#### CAP-11: Medium, Reproduced. `filter_snapshot(detail="crash")` produces a different key order in each process
- **Where:** `src/llmdebug/detail_filter.py:74-78`.
- **Problem:** It iterates a `frozenset`, so the order depends on hash randomization. This breaks diffing and prompt caching.
- **Fix:** Iterate the snapshot and filter by membership, as the `"full"` branch does.

#### CAP-12: Medium, Reproduced. `frame.f_locals` is re-read once per argument
- **Where:** `src/llmdebug/capture.py:254-280`, `:342`.
- **Problem:** On Python 3.10–3.12 each read rebuilds the dict, about 80× slower than reading a cached reference.
- **Fix:** Read `f_locals` once.

#### CAP-13: Medium, Reproduced. The DataFrame preview ignores `max_items` for columns
- **Where:** `src/llmdebug/serialize.py:688-692`.
- **Problem:** With `max_items=5` on a 400-column frame, the preview holds 1,200 cell values. Long strings aren't truncated either.
- **Fix:** Slice columns to `max_items` before `.head()`, and pass cells through `truncate_str`.

#### CAP-14: Medium, Likely. Shared mutable placeholder objects leak into public result fields
- **Where:** `src/llmdebug/snapshot_diff.py:23-32`, `:570-571`.
- **Problem:** The module-level `_EMPTY_DICT` and `_EMPTY_LIST` end up in `SnapshotDiffResult.old_git` and `new_git`. A consumer that mutates them corrupts every later diff.
- **Fix:** Return fresh objects, or use `MappingProxyType`.

#### CAP-15: Low, Confirmed. Retention does a full directory scan on every write while below the limit
- **Where:** `src/llmdebug/output.py:169-188`.
- **Fix:** Distinguish "index healthy" from "over limit", and query with `limit=max_snapshots+1`.

#### CAP-16: Low, Confirmed. Hypothesis de-duplication silently keeps only the first hypothesis per pattern
- **Where:** `src/llmdebug/hypotheses.py:473-478`.
- **Fix:** De-duplicate by `(pattern, name)` and cap the number per pattern.

#### CAP-17: Low, Reproduced. Sixteen `KEY_MAP` entries can never take effect
- **Where:** `src/llmdebug/compact_keys.py:63-79`.
- **Problem:** The array and DataFrame summary keys only occur inside user-data subtrees, which compaction deliberately skips.
- **Fix:** Delete them, or compact those generated keys explicitly.

#### CAP-18: Low, Confirmed. The unraisable hook captures `KeyboardInterrupt` and `SystemExit`, contrary to its docstring
- **Where:** `src/llmdebug/hooks.py:276-292`; the docstring at `:370` says otherwise. Also `if unraisable.object` should be `is not None`.

#### CAP-19: Low, Confirmed. `split_extra_context` always drops `eval_outcome`
- **Where:** `src/llmdebug/_debug_session.py:27`.
- **Problem:** The default argument to `pop` is evaluated eagerly, so `eval_outcome` is removed even when `eval` is present.

#### CAP-20: Low, Confirmed. Dict keys are emitted as reprs
- **Where:** `src/llmdebug/serialize.py:748`.
- **Problem:** A local `{"k": "v"}` is emitted as `{"'k'": "v"}`. This wastes tokens and breaks key-based matching, including the key-aware redaction proposed in CAP-03.
- **Fix:** Emit `str` keys as-is and use `repr` only for other key types. This changes the payload format, so document it.

#### CAP-21: Low, Likely. Capture calls side-effecting dunder methods on arbitrary user objects
- **Where:** `src/llmdebug/serialize.py:803-806`; `summarize_dataframe` at `:667-689`.
- **Problem:** `len()` on a lazy Django `QuerySet` executes the query.
- **Fix:** Restrict `len()` to known-safe types, or put it behind a config flag.

#### CAP-22: Low, Confirmed. Snapshot names aren't sanitized for backslashes or control characters
- **Where:** `src/llmdebug/output.py:228`.
- **Fix:** Allow only `[A-Za-z0-9._-]`.

#### CAP-23: Low, Confirmed. Two parallel array-statistics implementations
- **Where:** `src/llmdebug/serialize.py:275-522`.
- **Problem:** About 130 duplicated lines that have already drifted: anomalies are computed on the up-cast copy, and CuPy gets no anomaly counts.
- **Fix:** Keep the combined implementation and delegate to it.

### CAP structural notes
- **`serialize.py`** (889 lines) mixes redaction, array introspection (twice), pandas summarisation, JSON coercion and the locals bundle. Splitting out `serialize_arrays.py` and `redaction.py` is the natural seam.
- **Snapshot TypedDicts are unused by the producer:** `snapshot_types.py` defines `FrameDict` and `SnapshotDict`, but `capture.py` builds `dict[str, Any]`. The types appear only in `cast(...)` calls at read sites.
- **Duplicated aliases and helpers:**
  - `DetailLevel`: `config.py:12` and `detail_filter.py:17`.
  - `RedactionProfile`: `config.py:13` and `redaction_policy.py:9`.
  - `frame_file_label` vs `snapshot_diff._frame_file_value`.
  - `_truncate` in both `hypotheses.py:95` and `snapshot_diff.py:588`.
- **Heavy `SnapshotConfig`:** four `object.__setattr__` writes on a frozen dataclass, three provenance fields, a hand-written `with_updates` that re-implements `dataclasses.replace`, and a sentinel class passed through `cast`. A `from_inputs` factory would remove most of this.
- **`_json.py`** wraps orjson and then post-processes its output in pure-Python loops (`_apply_separators`, `_escape_non_ascii`). This loses the speed benefit and diverges from stdlib `json`: big ints raise, and NaN becomes `null`.
- **`output.py::write_bundle`** writes the snapshot and sidecar outside the directory lock, but updates the symlink and runs cleanup inside it. So `latest` can point at the older of two concurrent snapshots.
- **Three unrelated channels for capture diagnostics:** `_capture_warnings`, `_warnings.emit_warning` and `warnings.warn`.

---

## RCA: core RCA, persistence and interfaces

Scope: `src/llmdebug/` modules `rca_*.py`, `gates.py`, `state_db.py`, `cli.py`, `_cli_display.py`, `_quickstart.py`, `mcp_*.py`, `pytest_plugin.py`, `jupyter_plugin.py`, `middleware.py`, `git_context.py`, `coverage_context.py`.

#### RCA-01: High, Verified in code (Reproduced by the area reviewer). A stale verification result defeats the strict gate
- **Where:** `src/llmdebug/rca_transitions.py:182-184` (`resolve_attempt_values`); `src/llmdebug/gates.py:110-122`.
- **Problem:** When a request omits `verification_result`, the previous one is carried forward, even when the failure signature has changed. Repro: verify `ValueError:AAA` with `{"status": "pass"}`, then `resolve` a different `TypeError:BBB` with no verification. The result is `allowed=True` and state `resolved`, under `gate_mode=strict`.
- **Fix:** Drop the carried verification result when `same_signature` is false or the action is `change`.

#### RCA-02: High, Reproduced. MCP tools accept an unconfined `out_dir` and create directories and SQLite files anywhere
- **Where:** `src/llmdebug/mcp_server.py:129` (and every tool); `src/llmdebug/state_db.py:120` (`mkdir(parents=True)`); `src/llmdebug/_snapshot_loader.py:65`.
- **Problem:** `handle_rca_status(out_dir="/tmp/x/deep/nested")` created the whole tree plus a database file. `_validate_snapshot_path` uses the same client-supplied root, so it confines nothing. A prompt-injected agent gets directory creation, file creation and JSON/TOON file reads with the IDE's privileges.
- **Fix:** Resolve `out_dir` against a configured root, reject anything outside it, and never create files during read-only operations (decision D11).

#### RCA-03: High, Reproduced. A partially populated snapshot index permanently hides snapshots that exist on disk
- **Where:** `src/llmdebug/_snapshot_loader.py:313-320`. Index writes are best-effort (`src/llmdebug/output.py:260`).
- **Problem:** The filesystem is rescanned only when the index is completely empty. With 3 files on disk and 1 indexed, `find_all_snapshots` returned 1. Plausible under pytest-xdist contention.
- **Fix:** Compare the index count (or newest mtime) with a cheap filesystem check, and rescan on mismatch.

#### RCA-04: High, Verified in code. A denied action resets a `stalled` session to `captured`
- **Where:** `src/llmdebug/rca_transitions.py:218-221`.
- **Problem:** `"stalled"` isn't one of the values in `_STATE_TRANSITIONS`, so the fallback picks `"captured"`. The stall marker, which is the point of the anti-looping machinery, is lost.
- **Fix:** Preserve any valid `RCAState`, using the full literal set.

#### RCA-05: Medium, Reproduced. JSONL-to-SQLite migration never completes and ignores JSONL history when SQLite already has rows
- **Where:** `src/llmdebug/rca_backend.py:174-176`.
- **Problem:** It returns early before rotating the JSONL file, so every later read and write pays for a file lock and an extra count query.
- **Fix:** Merge on migration (by `attempt_id`) and always rotate the file, or warn.

#### RCA-06: Medium, Reproduced. The RCA state machine never validates transitions
- **Where:** `src/llmdebug/gates.py:226` (`_ = current_state`); `src/llmdebug/rca_transitions.py:215`; `src/llmdebug/rca_engine.py:191`.
- **Problem:** `resolve_target_state("captured", "resolve", True)` returns `"resolved"`. `allowed_actions` is written into records but never enforced.
- **Fix:** Enforce `allowed_actions_for_state` with an `illegal_transition` check, or document the transitions as advisory.

#### RCA-07: Medium, Confirmed. The Jupyter `install()` leaves the traceback hook wrapped but `_installed=False` when magic registration fails
- **Where:** `src/llmdebug/jupyter_plugin.py:621-632`.
- **Problem:** `uninstall()` then does nothing, and a retried `install()` wraps the hook twice, so tracebacks print twice and are captured twice.
- **Fix:** Register the magics first, or restore the original hook on failure.

#### RCA-08: Medium, Confirmed. Jupyter HTML output interpolates `line` and `execution_count` without escaping
- **Where:** `src/llmdebug/jupyter_plugin.py:149`, `:194`, `:229`, `:478`.
- **Problem:** A snapshot with `"line": "<img onerror=…>"` from a shared directory runs script in the notebook's origin.
- **Fix:** Route these values through `_esc(str(...))`.

#### RCA-09: Medium, Confirmed. All MCP tools are synchronous and block the server's event loop
- **Where:** `src/llmdebug/mcp_server.py:127-741`.
- **Problem:** Git calls (time budgets of 1.2–2 s), a 30-second SQLite `busy_timeout` and file locks stall every other tool call and the protocol heartbeat.
- **Fix:** Make the tools `async def` and dispatch blocking work with `anyio.to_thread.run_sync`.

#### RCA-10: Medium, Confirmed. `llmdebug_git_context` has no exception guard
- **Where:** `src/llmdebug/mcp_server.py:538`; `src/llmdebug/cli.py:854`.
- **Problem:** Exceptions escape the standard error envelope, and the CLI prints a raw traceback.
- **Fix:** Catch them and map to `GIT_CONTEXT_ERROR`.

#### RCA-11: Medium, Confirmed. The state database has no schema version or migration path
- **Where:** `src/llmdebug/state_db.py:69-113`.
- **Problem:** Tables are created with bare `CREATE TABLE IF NOT EXISTS`. Adding a column later will break inserts on existing directories.
- **Fix:** Track `PRAGMA user_version` with an explicit upgrade sequence.

#### RCA-12: Medium, Likely. Failure signatures aren't normalized, so stall detection rarely fires
- **Where:** `src/llmdebug/rca_identity.py:38-47`.
- **Problem:** Memory addresses, `tmp_path` values and timestamps in messages make every signature unique.
- **Fix:** Strip hex addresses, digits and absolute paths before hashing.

#### RCA-13: Medium, Confirmed. The Rich branch of `_print_frame` is unreachable and returns inside its loop
- **Where:** `src/llmdebug/_cli_display.py:268-305`. Kept alive only by tests that pass a fake console (`tests/test_cli_edge_cases.py:406`, `tests/test_cli.py:1909`).
- **Fix:** Delete the branch and those tests.

#### RCA-14: Medium, Likely. `_run_git_command` can raise `UnicodeDecodeError`
- **Where:** `src/llmdebug/git_context.py:61-76`.
- **Problem:** It uses `text=True` with the locale encoding and strict errors. Capture swallows the error; the CLI and MCP paths crash.
- **Fix:** Pass `encoding="utf-8", errors="replace"`.

#### RCA-15: Low, Reproduced. CLI error paths exit with status 0
- **Where:** `src/llmdebug/cli.py:717-719`; also `git-context` and `frames`.
- **Fix:** Exit with status 1 whenever writing an error to stderr.

#### RCA-16: Low, Confirmed. The result of `PRAGMA journal_mode=WAL` is never checked, and the "already initialized" cache survives database deletion
- **Where:** `src/llmdebug/state_db.py:37-54`.
- **Fix:** Cache only when the returned mode is `"wal"`.

#### RCA-17: Low, Confirmed. `llmdebug_list` doesn't validate `limit`
- **Where:** `src/llmdebug/mcp_snapshot_tools.py:281-307`.
- **Problem:** `limit <= 0` returns everything, while the sibling `rca_history` tool rejects it.
- **Fix:** Apply the same validation.

#### RCA-18: Low, Confirmed. JSONL readers take no lock while the writer truncates and rewrites the file
- **Where:** `src/llmdebug/rca_backend.py:141-162` vs `:68-86` and `:121-138`.
- **Fix:** Lock reads too, or rewrite via a temporary file and atomic rename.

#### RCA-19: Low, Confirmed. `except (AttributeError, Exception)` is redundant and maximally broad
- **Where:** `src/llmdebug/coverage_context.py:210`.

#### RCA-20: Low, Confirmed. `# pragma: no cover` on module-level imports and constants
- **Where:** `src/llmdebug/state_db.py:3-21`.
- **Problem:** These lines are covered whenever the module is imported, and the pragmas would hide genuinely untested module-level code added later.

#### RCA-21: Low, Confirmed. `clean --all` leaves the state database, deletes a lock another process may hold, and races on `stat()`
- **Where:** `src/llmdebug/cli.py:669-679`, `:730`.
- **Fix:** Include the database files (or add `--purge-state`), skip the lock file, and guard the `stat()` call.

#### RCA-22: Low, Confirmed. `RCAState` is declared twice, and RCA records are untyped dicts
- **Where:** `src/llmdebug/gates.py:9-18` vs `src/llmdebug/rca_transitions.py:7-16`. `resolve_previous_values` returns an unnamed six-element tuple.
- **Fix:** Define `RCAState` once, and add an `RCAAttemptRecord` TypedDict.

#### RCA-23: Low, Confirmed. Several functions exceed six named parameters, and there is no signature-budget test
- **Where:**

  | Function | Location | Parameters |
  |----------|----------|-----------:|
  | `apply_stall_rule` | `src/llmdebug/rca_transitions.py:224` | 8 |
  | `resolve_attempt_values` | `src/llmdebug/rca_transitions.py:168` | 8 |
  | `build_transition_history` | `src/llmdebug/rca_transitions.py:250` | 7 |
  | `evaluate_gate` | `src/llmdebug/gates.py:162` | 7 |
  | `handle_frame` | `src/llmdebug/mcp_snapshot_tools.py:378` | 8 |
  | `handle_diff` | `src/llmdebug/mcp_diff_tools.py:33` | 8 |

  See X-06.
- **Fix:** Group these into `TransitionInputs` and `GateRequest` objects.

### RCA structural notes
- **The RCA split holds where it matters:** `rca_engine` and `rca_transitions` are pure, persistence is isolated, and `rca_backend` is a real adapter. The domain objects are dicts, though, so the purity buys less safety than it could.
- **`rca_state.py`** (95 lines) is a facade of one-line delegating wrappers that adds a hop without adding behaviour.
- **`mcp_server.py`** (760 lines) isn't thin:
  - `diagnose`, `show` and `git_context` are implemented inline, while the other tools delegate to handler modules.
  - Every inline tool repeats the same five-step preamble.
  - An 18-line `advance_rca_with_snapshot_refs` block appears five times.
- **Private-name imports:** `cli.py` imports 13 private functions from `_cli_display`, each with a pyright ignore comment. `middleware.py` and `jupyter_plugin.py` reach into private helpers in `hooks.py`. These should be promoted to an explicit internal seam.
- **Snapshot-index ownership is split** across `output.py` (write), `_snapshot_loader.py` (read, prune, rebuild) and `cli.py clean` (delete files but not rows). That split is why RCA-03 is possible; one `snapshot_index` module should own all of it.
- **Every `state_db` call opens a fresh connection** and re-runs four PRAGMAs and five DDL statements. An MCP call with `with_rca=True` does at least three full cycles.
- **`pytest_plugin.py` is well structured.** Setup and teardown failures don't produce snapshots (`report.when != "call"`); that's intentional and worth documenting.

---

## X: cross-cutting

#### X-01: High, Verified. Oversized functions and modules
- **Largest functions** (from radon):

  | Function | Location | Lines | Complexity |
  |----------|----------|------:|-----------:|
  | `run_one_condition_impl` | `evals/eval_attempt_flow.py:1150` | 6,674 | 648 |
  | `main` | `evals/run_eval.py:1079` | 1,607 | 2 |
  | `run_main_flow` | `evals/eval_main_flow.py:64` | 982 | 99 |
  | `_adaptive_rollup_metrics` | `evals/analyze_results.py:1929` | 937 | 319 |
  | `_summarize_bucket` | `evals/analyze_results.py:1185` | 606 | 190 |
  | `_derive_case_condition_rows` | `evals/analyze_results.py:569` | 604 | 43 |
  | `RLMAdapter.run_repair` | `baselines/rlm/adapter.py:258` | 576 | 78 |
  | `_build_stage_b_prompt_sections` | `evals/eval_prompts.py:739` | 441 | 76 |
  | `_adaptive_budget_policy_attempt_metrics` | `evals/analyze_results.py:3582` | 438 | 114 |
  | `LambdaRLMAdapter.run_repair` | `baselines/lambda_rlm/adapter.py:234` | 434 | 45 |
  | `prepare_main_setup` | `evals/eval_main_setup.py:362` | 424 | 25 |
  | `_empty_attempt_metadata` | `evals/eval_recording.py:704` | 361 | 1 |

  Pyright reports `run_one_condition_impl` as "too complex to analyze", so the most important function in the harness gets no type checking.
- **Twelve modules exceed the 1,200-line soft limit:** `eval_attempt_flow.py` (7,823), `run_eval.py` (6,588), `patchers/openai_compat_patcher.py` (6,082), `analyze_results.py` (5,867), `report_run_metrics.py` (2,155), `eval_prompts.py` (1,838), `adaptive_controller_packet.py` (1,745), `adaptive_budget_online_policy.py` (1,450), `eval_types.py` (1,421), `baselines/rlm/adapter.py` (1,388), `eval_recording.py` (1,383) and `baselines/intervenor/adapter.py` (1,374).
- **Fix:** Use the extraction seams listed in the ATT, ORC, STA, LLM and POL structural notes.

#### X-02: High, Verified. "Extracted" modules depend back on `run_eval` through forwarding shims
- **Where:**
  - `evals/eval_attempt_support.py` re-imports about 65 private names from `evals.run_eval`, plus `_legacy_call` forwarding functions (`:78-123`).
  - `evals/eval_main_support.py` imports 19 private names, adds forwarding functions, and ends with a module-level `__getattr__` that delegates to `run_eval`.
  - `evals/eval_attempt_flow.py` imports 68 private names from the support module.
- **Problem:**
  - The cycle `run_eval → eval_*_flow → eval_*_support → run_eval` only works because of function-local imports (`evals/run_eval.py:2683`, `:3064`, `:5549`).
  - The forwarding functions are typed `(*args: Any, **kwargs: Any) -> Any`, so all type information is lost.
  - Reads go to `run_eval` but writes stay on the shim, which caused ORC-01 and ORC-02.
- **Boundary violations elsewhere:**
  - Repo-wide, 393 private names are imported across module boundaries. Most-imported modules: `run_eval` (85), `eval_attempt_support` (68), `eval_policy_features` (31), `eval_main_support` (23), `eval_evidence` (21).
  - `analyze_results`' private helpers serve as the API for four other modules.
  - `baselines/` and `evals/` import each other.
  - `tests/test_architecture_boundaries.py` covers only `src/llmdebug`.
- **Fix:**
  - Pass an explicit run-context object into the flows (or use setter functions).
  - Move shared helpers out of `run_eval` into modules that own them, imported in one direction.
  - Give private helpers that other modules use public names.
  - Add an `evals/` boundary test: no `legacy.<NAME> =` writes, and no new imports from `run_eval` in extracted modules.

#### X-03: Medium, Verified. Records are `dict[str, Any]` from the code that produces them to the analysis
- **Where:**
  - `evals/eval_recording.py:704-1067`: `_empty_attempt_metadata`, a 363-line literal.
  - `:1069`: `_materialize_attempt_metadata_request`, 261 lines.
  - `_build_patcher_config`: 306 lines.
  - `evals/patchers/payload_types.py`: `PatcherMeta` declares 29 keys while about 90 are written, which accounts for most of the 308 pyright errors in `openai_compat_patcher.py`.
  - `evals/analyze_results.py:754-1171`: the derived row has about 170 keys, read via string literals in five modules.
  - `src/llmdebug/snapshot_types.py`: TypedDicts not used by `capture.py`.
  - Positional reward tuples: `evals/eval_policy_features.py:712`, `:1051`.
  - `float(x or 0.0)` appears about 60 times in `_adaptive_rollup_metrics`. It treats "missing" the same as zero, which is the root of STA-01.
- **Fix:**
  - Define TypedDicts or dataclasses for the attempt record, the patch metadata (composed per concern), the derived analysis row, and snapshot frames.
  - As a first step, add a constants module for record keys.
  - Use `None` for missing values.

#### X-04: Medium, Verified. Duplicate implementations that have already drifted apart
- **Sync/async twins in `evals/patchers/openai_compat_patcher.py`:** 11 function pairs are 93–100% identical after normalizing `async` and `await`, about 1,400 lines in total.
  - The async variants of `_propose_patch_via_{diff,edits,full_file}` lack `system_prompt_override`. That's latent today, since the async path is used only by `evals/swebench/generate.py:474` and `evals/swebench_pro/generate.py:529`.
  - `_http_get_json_async` (`:2493`) lacks the `ValueError → bad_request_url` handler that the sync version has (`:2348`).
- **SWE-Bench Lite and Pro:** `evals/swebench/io.py` and `evals/swebench_pro/io.py` are about 98% identical; `generate.py` is about 75% shared; `smoke.py` is near-identical.
- **Policy:** three LinUCB implementations (helper, budget, bootstrap replay).
- **Attempt flow:** five terminal-failure blocks in `eval_attempt_flow.py`, about 500 lines.
- **Baselines:** about 1.2–1.5k lines copy-pasted across adapters.
- **Analysis:**
  - four formatter sets
  - two bootstrap implementations and two exact-binomial ones
  - two `_format_ope_targets`
  - two definitions of `EVAL_RECORD_SCHEMA_VERSION` (`evals/eval_types.py:120`, `evals/analyze_results.py:29`)
- **Patching:** two byte-identical `_build_diff_from_changes` functions, and two forced-patch marker strings that disagree (LLM-13).
- **Core library:** duplicated `DetailLevel`, `RedactionProfile` and `RCAState` aliases, and two array-statistics implementations (CAP-23).
- **Fix:** Keep one implementation of each. For the sync/async twins, derive one side from a shared core, or add a test that checks the signatures and behaviour of every pair match.

#### X-05: Medium, Verified. Type checking covers little of the eval harness
- **Where:**
  - `pyrightconfig.evals.json` includes 14 files.
  - Basic mode over all 129 files in `evals/` and `baselines/` gives 379 errors (see the Repository health snapshot).
  - Undefined annotation names `AdaptiveBudgetControlProfile`, `PolicyDecision`, `BudgetPolicyDecision` and `ContextRequestPayload` at `evals/eval_attempt_flow.py:2624`, `:2705`, `:2718`, `:3558`. Harmless at runtime under `from __future__ import annotations`, but never checked.
  - `baselines/{intervenor,lambda_rlm,ldb}/case_converter.py` return `BaseCaseInput` where a subclass is declared.
  - Optional `.get` at `evals/adaptive_history.py:98-138`.
- **Fix:** Fix the shim and TypedDict problems first (X-02, X-03), then widen the include list module by module as seams are extracted.

#### X-06: Low, Verified. The signature budget in AGENTS.md isn't enforced
- **Where:** there's no signature-budget test in `tests/`. Worst offenders:

  | Function | Parameters |
  |----------|-----------:|
  | `_finalize_adaptive_budget_policy_attempt` | 48 |
  | `_finalize_adaptive_online_attempt` | 44 |
  | `_build_stage_b_prompt_sections` | 30 |
  | `build_attempt_metadata_request` | 23 |
  | `_propose_patch_via_edits` | 20 |
  | `_propose_patch_via_diff` | 19 |
  | `_chat_completion` | 18 |
  | `_build_attempt_outcome_record` | 16 |
  | baseline `_final_metrics` and `_compatibility_failure_result` | 11–14 |
  | `RunConditionInputs.from_components` | 13 |
  | `build_startup_sections` | 12 |
  | `build_export_payload` (core) | 10 |
  | `_build_export_html_cards`, `resolve_redaction_policy` (core) | 9 |
  | RCA functions (RCA-23) | 7–8 |

- **Fix:** Add an AST-based test with an allowlist that can only shrink.

#### X-07: Low, Confirmed. Silently swallowed exceptions hide failures
- **Where:**
  - `evals/eval_attempt_flow.py:7818-7822` (ATT-17)
  - `evals/swebench_pro/score_docker.py:214-215`, `:378-398`
  - `evals/importers/base.py:240-243`
  - `evals/importers/native_repro_helpers.py:139-140`
  - `evals/importers/pybughive_importer.py:134-136`
  - `evals/importers/trickybugs_importer.py:133-137`
  - `src/llmdebug/coverage_context.py:210` (RCA-19)
  - Capture diagnostics use three separate channels (CAP structural notes).
- **Fix:** Record failures as events or fields, and use one logging channel for capture diagnostics.

#### X-08: Low, Confirmed. Documentation contradicts the code
- **Where:**
  - adaptive-policy docs (POL-25)
  - the comment at `evals/configs/openai_adaptive_probe_decay_098.template.toml:3` (POL-01)
  - `docs/baseline-adapters.md:376-388` (BAS-08)
  - the policy event schema version (POL-13)
  - `evals/eval_prompts.py:771` (LLM-20)
  - `evals/eval_patches.py:269-274` (ATT-19)
  - the hooks docstring about `KeyboardInterrupt` (CAP-18)
  - the SWE-Bench README examples without `--snapshot-dir` (DAT-03)
- **Fix:** Generate the defaults tables with `evals.doc_sync`, and fix each listed spot alongside its code fix.
