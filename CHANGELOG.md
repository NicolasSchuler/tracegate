# CHANGELOG

## Unreleased (3.2.0)

### Features

- `python -m evals.reanalysis_audit <jsonl…>` audits existing results against the 2026-10-04 review: headline metrics with the corrected analysis next to the pre-review semantics, plus data audits for test/config edits in successes, evidence fallbacks, policy decay, OPE propensities, deterministic runs, repeated runs, unmatched case sets, missing usage, LDB counts, retry-gate and forced-patch records, schema versions and SWE-Bench prompt policy. Golden-file tests now pin `eval_summary` and `report_run_metrics` output on a fixture with unequal case sets, missing usage and repeated runs, and `eval_summary` prints token coverage.
- Off-policy estimates (IPS, SNIPS, DR) carry 95% case-bootstrap intervals (`ope_ci`, 1,000 seeded resamples, cross-fitted models held fixed), printed next to each estimate; they had no uncertainty at all.
- Add an offline `llmdebug quickstart` command that creates and displays a real synthetic snapshot.
- Add a System-1 decision-model advisor for the adaptive helper policy (`evals/decision_model/`): a SystemOne-compatible endpoint (e.g. Cloudflare clef-flash served by llama.cpp through llama-swap, or the bundled PyTorch sidecar) can run in `shadow`, `policy`, `features` or `prior` role via `--adaptive-decision-model-role`, with versioned question profiles, a redacted bounded decision state, policy-event schema v5, decision-model off-policy targets, and an `--adaptive-decision-model-api-key` (default `$TRACEGATE_DECISION_MODEL_API_KEY`) that `run_eval_docker.sh` passes as a flag and lockfiles redact.

### Bug Fixes

- Jupyter (RCA-07, RCA-08): when magic registration failed, `install()` left `showtraceback` wrapped with `_installed=False`, so `uninstall()` did nothing and a retry wrapped it twice (tracebacks printed and captured twice); the original hook is now restored. Notebook HTML now escapes the crash `line` and the cell `execution_count` like every other value; a snapshot from a shared directory could inject script into the notebook's origin.
- MCP tools no longer block the server's event loop (RCA-09): FastMCP ran the synchronous tools on the loop, so git calls (1.2–2 s budgets), the 30 s SQLite busy timeout and file locks stalled every other call and the protocol heartbeat. Each tool is now registered as an async wrapper that runs it in a worker thread (`asyncio.to_thread`); `advance_rca_attempt` serializes read-then-write RCA updates within the process.
- `llmdebug clean` (RCA-21): `--all` no longer deletes `.llmdebug.lock`, which another process may hold; deleted snapshots' index rows are removed instead of going stale; the new `--purge-state` flag also deletes the state database and the JSONL RCA history, which `--all` keeps; and a snapshot deleted by another process during the run no longer crashes the sort with `FileNotFoundError`.
- A partially populated snapshot index no longer hides snapshots (RCA-03): index writes are best-effort, and listings rescanned the directory only when the index was completely empty, so with three files on disk and one indexed, `llmdebug list`, `latest`/`previous` refs and `llmdebug_list` saw one snapshot until the index was rebuilt by hand. The index is used only while its row count matches the snapshot files on disk (one directory scan); otherwise the listing rescans and repairs it. The snapshot-file rule and the index row builder now live in `llmdebug.snapshot_index`, shared by capture, listing, retention and `llmdebug clean`.
- The JSONL RCA store trims through a temporary file and an atomic rename (RCA-18): readers take no lock, and the writer truncated and rewrote the live file, so a concurrent read could see a partial history.
- The state database is versioned (RCA-11): tables were created with bare `CREATE TABLE IF NOT EXISTS`, so a later column change would have broken writes to existing directories. The schema version now lives in `PRAGMA user_version` (currently 1) with an ordered migration list applied under `BEGIN IMMEDIATE`; unversioned databases upgrade in place, and a database newer than the installed llmdebug is refused. WAL mode is checked on every open instead of being remembered per path, so a deleted and recreated database is switched to WAL again, and a filesystem that refuses WAL gets one warning instead of silent rollback-journal locking (RCA-16).
- `llmdebug_git_context` and `llmdebug git-context` handle failures (RCA-10): an exception while collecting git metadata escaped the MCP error envelope and printed a raw traceback in the CLI. The tool now answers `GIT_CONTEXT_ERROR`; the CLI prints the error and exits 1. `llmdebug_list` rejects `limit <= 0` like `llmdebug_rca_history` instead of listing every snapshot (RCA-17).
- Read-only operations never create files (RCA-02, D11): reading RCA status or history, or listing snapshots, created `out_dir`, the state database and the snapshot index, and the JSONL-to-SQLite migration ran on reads. State-DB read helpers now return empty results when the database doesn't exist, the snapshot index is only rebuilt when it already exists, and the migration runs on the first write; until then, RCA reads merge the pending JSONL rows in memory.
- MCP tools confine `out_dir` to one root (RCA-02, decision D11): any client-supplied `out_dir` was used as-is, so `llmdebug_rca_status(out_dir="/tmp/x/deep/nested")` created that tree and a SQLite file, and snapshot-path validation used the same client-chosen root. Every tool now resolves `out_dir` against `$LLMDEBUG_MCP_ROOT` (default: the server's working directory) and returns `INVALID_ARGUMENT` for anything outside it, symlinks included.
- RCA gate and transition helpers take grouped inputs (RCA-23): `evaluate_gate(GateRequest(...))` replaces seven keyword parameters (the keyword form still works with a `DeprecationWarning`), `apply_stall_rule` takes `TransitionInputs`, `build_transition_history` a `TransitionStep`, and `derive_attempt_progress` returns an `AttemptProgress` NamedTuple. The signature-arity ratchet drops the `gates.py` and `rca_transitions.py` entries.
- RCA types have one definition (RCA-22): `RCAState` was declared twice (in `gates` and `rca_transitions`); it and `RCAAction` now live in `llmdebug.rca_types` and are re-exported. `build_rca_attempt_record` and `advance_rca_attempt` return the new `RCAAttemptRecord` TypedDict instead of `dict[str, Any]`, and `resolve_previous_values` returns a `PreviousAttemptValues` NamedTuple instead of an unnamed six-element tuple (still unpackable).
- The JSONL-to-SQLite RCA migration completes (RCA-05): it stopped as soon as SQLite had any row, so the JSONL history was never merged and the file was never rotated, and every later read and write paid for the migration lock and a count query again. JSONL records are now merged by `attempt_id` (rows already in SQLite win) and the file is always rotated to `rca_attempts.jsonl.bak`; a failed rotation prints a warning and the idempotent merge is retried on the next call.
- RCA failure signatures ignore run-specific values (RCA-12): memory addresses, `tmp_path` directories, UUIDs, long hex ids and numbers in the exception message (or the text-fallback tail, e.g. pytest's `in 0.23s`) made every attempt's signature unique, so repeat counting and stall detection rarely fired. `build_failure_signature(..., normalize=True)` (the new default) replaces them before hashing; `normalize=False` keeps the raw signature, and the eval harness uses it so the adaptive conditions' `failure_signature_repeat` stays comparable with earlier runs. Existing RCA sessions see one signature change, which resets their repeat count once.
- The RCA state machine enforces its documented transitions (RCA-06): `allowed_actions` was written into every record but never checked, so `resolve` from `captured` went straight to `resolved`. A new `illegal_transition` gate check, evaluated last, blocks an action that isn't an edge out of the current state in `strict` mode (the exploratory override still applies) and warns in `soft` mode; a session's first attempt is exempt. `evaluate_gate(current_state=None)` means a first attempt.
- A strict-gate denial no longer resets a `stalled` RCA session to `captured` (RCA-04): the fallback only knew the states that actions lead to, and no action leads to `stalled`, so the anti-looping marker was lost on the next blocked request. A denied action now keeps any valid state.
- A stale verification result no longer defeats the strict RCA gate (RCA-01): when a request omitted `verification_result`, the previous one was carried forward even for a different failure signature, so a `pass` recorded for `ValueError:AAA` let `resolve` succeed for `TypeError:BBB` under `gate_mode="strict"`. The previous result now carries forward only while the signature is unchanged and the action is not `change`.
- The redaction text patterns run in linear time (CAP-03 follow-up): the new secret-name pattern and the existing email pattern backtracked quadratically, so a `ci`/`prod` capture redacting a few hundred KB of captured stdout could take minutes (48k characters took 3.4 s). Repeats are bounded, and the email local part is capped at the RFC 5321 limit of 64 characters. Secret-named values in text also cover escaped quotes, `b''` literals and truncated values; URL userinfo is redacted with or without a password and up to the last `@` before the host, also when truncation cut it, and before the email pattern so the host survives; a `[name, value]` pair with a secret-like name has its value redacted.
- `llmdebug export --mode redacted` no longer writes an unredacted artifact labelled "redacted" (CAP-06): with `--redaction-profile dev` (or no profile through the API) no redactors were configured, the payload was copied unchanged and stamped `export_mode="redacted"`. The CLI now rejects `dev` in redacted mode with a usage error, and `apply_export_redaction` raises `ValueError` when the profile resolves to no redactors.
- `compile_redactors` caches through `functools.lru_cache` (32 pattern sets) instead of a one-slot cache that stored the patterns and the result in two statements, so concurrent captures with different configs (for example `threading.excepthook` next to the main thread) could get each other's redactors (CAP-07). It returns a fresh list on every call.
- `redaction_profile="prod"`/`"ci"` and `pii_safe=True` redact secret-valued variables (CAP-03): the PII patterns only matched `key=value` inside one string, so a local `password = "hunter2"`, an `api_key` argument or a `{"password": …}` dict reached the snapshot verbatim, and `redact_keys=True` hid the name but kept the value. Whenever any redaction pattern is active, the whole value under a secret-like local, argument, attribute or dict-key name (`password`, `secret`, `api_key`, `auth_token`, `authorization`, `cookie`, …; not bare `key` or `token`) is replaced by `[REDACTED]` without being `repr`'d, and its `locals_meta` drops the length. `PII_PATTERNS` adds URL/connection-string credentials (`postgres://[REDACTED]@host`) and secret-named keys inside text, such as the `repr` of a nested dict. No eval case has such names or strings, so `ci`-profile eval snapshots are unchanged.
- **Snapshot format change (schema `2.1`):** string dict keys in locals are emitted as plain strings instead of their `repr` (CAP-20): a local `{"k": "v"}` was written as `{"'k'": "v"}`, which wasted tokens and broke key-based matching, including key-aware redaction. Other key types keep their `repr`, and a collision such as `{1: …, "1": …}` keeps both entries as `"1"` and `"1#2"`. `DebugSession.schema_version` is now `"2.1"` (readers accept any `2.x`), and eval run metadata records it as `llmdebug_snapshot_schema_version`, also part of the settings fingerprint, because snapshot-based conditions quote these locals in prompts.
- `capture_exception(extra=...)` no longer drops `eval_outcome` when `eval` is also given (CAP-19): `eval` wins as the eval payload and `eval_outcome` stays in the context. The docstring named the reserved keys `_trace`/`_eval`; they are `trace` and `eval` (alias `eval_outcome`).
- The `sys.unraisablehook` wrapper no longer captures `KeyboardInterrupt` or `SystemExit`, as `install_hooks` documents and the other hooks already did, and records a falsy object such as `[]` instead of dropping it (CAP-18).
- Snapshot file names are bounded and sanitized (CAP-04, CAP-22): a parametrized pytest node ID longer than the 255-byte filename limit failed with `ENAMETOOLONG` and lost the snapshot, and backslashes and control characters reached the file name. The name part now keeps only `[A-Za-z0-9._-]` and is cut to 120 characters plus an xxh64 hash of the full name; the payload keeps the original name. Non-ASCII characters in snapshot names now become `_` in the file name.
- One hostile local can no longer abort the capture or hide its siblings (Phase 4.1): an object whose `__getattribute__` raises something other than `AttributeError` made `hasattr`/`isinstance` inside the serializer raise, so no snapshot was written. Each value (and each container element) is now serialized independently and becomes `<serialization failed: <ErrorType>>` on failure; the array/DataFrame shape probe in `locals_meta` is guarded the same way. The new hostile-object suite (`tests/test_capture_hostile.py`) runs broken `__str__`, `__repr__`, `__len__` and `__getattr__`, wide ints, recursive containers, lazy objects and 300-character names through `capture_exception`.
- An exception whose `__str__` raises (or whose args or notes can't be printed) no longer aborts the capture (CAP-02): the message, each arg and each note fall back to `<str() failed: <ErrorType>>` individually, as the stdlib `traceback` module does, and error categorization still runs on the exception type. One unprintable arg used to replace the whole `args` list.
- A local integer outside the 64-bit range (`2**70`, `-(2**64)`) no longer aborts the whole capture (CAP-01): orjson can't encode it and raised `TypeError: Integer exceeds 64-bit range`, so no snapshot was written, and the pytest plugin swallowed the error. Such ints, including cells of pandas `head` previews, are now stored as their `repr` string (bounded by `max_str`); ints within the signed/unsigned 64-bit range stay numbers.
- The adaptive-policy docs carry generated defaults tables (POL-25, X-08): `python -m evals.doc_sync --write` renders algorithms, modes, actions, CLI and Python API defaults, reward coefficients and schema versions from the code into `docs/ADAPTIVE_POLICY_QUICK_REFERENCE.md` and `docs/adaptive_online_policy_design.md`, and `just docs-check` fails when they drift. The hand-written values had drifted (8 actions, reward hints 0.25/0.5/0.25/0.5, event schema 3, `alpha` 0.8 and `epsilon` 0.05, warm-up 200, "linucb only"). The `l2`, warm-up and save-interval CLI defaults now come from named constants.
- TEP full mode accepts a prompt update only when a critic scores the candidate surface strictly above the current one (POL-17): the "free" and "nudged" critiques scored the same surface, so about half of all updates were accepted on noise and the candidate was never scored. A critic response without a usable score is marked `critic_parsed: false` instead of reading as a `stable` (converged) critique, the update is skipped, and scores such as `"7/10"` are parsed instead of raising. The TEP state is saved after every learning pass, and a failed save is recorded as `adaptive_tep_save_error` on the attempt instead of being swallowed (ATT-17).
- `bootstrap_policy_state` refuses `percentile_v1` state, whose scaling caps can't be rebuilt from event logs (a replayed state resumed in a different feature space; POL-14), and can keep evaluation cases out of a warm start: `--exclude-case-ids` and `--exclude-run-ids` (comma-separated or `@file`) drop events, and `--target-case-ids` prints a warning and records `target_case_overlap` when replayed cases are also target cases (POL-15). The documented warm-start example now excludes the target cases.
- Policy-event export and bootstrap replay share one loader (`evals/policy_event_log.py`; POL-18, POL-13): events are replayed in chronological order (`created_at_utc`, run, then log position, since timestamps have one-second resolution) and the first occurrence of a duplicate wins. Export used to sort rows by case id and keep the last duplicate, and replay followed glob order, so a bootstrapped state depended on how files were listed. Events with a non-finite reward, propensity, score or feature value are rejected (`records_non_finite`) instead of writing `inf` to the CSV, and unsupported schema versions are counted as `records_unsupported_schema`.
- Budget-policy events log behavior propensity 1.0 in shadow mode (POL-03): the budget controller's `record_outcome` still used the selection's propensity whenever it matched the applied baseline, so agreeing warm-up events carried 0.5 and agreeing post-warm-up events `1 − ε/2`, inflating importance weights; the attempt-metadata path was already fixed. One shared rule (`linucb_core.behavior_propensity`) now serves all four sites, and tests cover shadow mode, on mode and warm-up for both controllers.
- Every budget step above base widens the discrete limits (helper calls, Stage-A requests, compaction calls) by at least 1 (POL-10): on the default ladder with `max_ups_per_case=1` the widening was `floor(2·1/4) = 0`, so the `hierarchical_v1` `context` family left them unchanged and wasn't comparable with `tier_binary`.
- `bwk_v1` budget control uses its dual variable (POL-04): the variable was computed and logged but never read, and its update (`overspend / budget`) stayed 0 until the budget was nearly exhausted, so the only control was the exhaustion mask. It now paces spending to `budget / (cases × k)` tokens per decision with the dual step `λ ← max(0, λ + 0.05 (tokens − target) / target)`, and each gate arm's score is reduced by `λ × ĉ_k / target` using the arm's mean tokens per attempt. Events add `bwk_target_tokens_per_decision` and `bwk_cost_estimates` (budget event schema 7); bootstrap replay applies the new rule to schema 7 events and the old one to earlier logs.
- The helper policy's delayed rescue bonus (up to 0.75) is paid only when evidence the helper added is sent again in the successful attempt's prompt; it was paid to any helper call one or two attempts before a success, a built-in bias toward invoking the helper that `skip_helper` could never earn. Unmatched credits resolve as `local_only_evidence_not_in_success_prompt`, and the never-read `case_difficulty` and `case_snapshot_dependency` inputs are gone (POL-11).
- The budget policy charges an escalation only the cost it added (POL-08): the Stage-B proposal is paid under either arm, but `increase_budget` was charged the whole attempt (about 0.40 of a 1.0 success reward at `lambda_tokens=0.08` and 5k tokens) while `keep_budget` was free. `increase_budget` now pays the share `1 - default/applied` (budget multipliers) of the attempt's tokens and time; events still log the full attempt cost.
- `percentile_v1` scaling fits each count-like feature's cap once, from its first 30 observations, and then freezes it (POL-09): caps were refitted on every decision, so `stagnation_streak=3` scaled to 1.26, then 0.71, then 0.60 within one run and incompatible vectors were summed into one model. Only count-like features (the only ones a cap applies to) collect history and report caps, an all-zero history keeps the `v1` cap instead of amplifying later values with a cap of 1.0, and states saved earlier are pruned to this layout on load. The `v1_interactions` terms are products of `v1`-scaled factors under new names (`ix_attempt_frac_x_stagnation`, `ix_pregate_score_x_snapshot_available`, `ix_helper_calls_used_x_low_yield_ratio`); they multiplied raw values, and `attempt_frac_x_stagnation_streak` was log-scaled again because of its suffix (POL-19).
- A warm-start or resumed policy state trained on features the run no longer produces is discarded at the first decision with a `FeatureSetMismatch` load error, instead of reading those features as 0 under the learned weights; this applies to the helper, budget gate and budget family models, and covers states saved before POL-06. Arms reset to the prior because of a shape mismatch are counted too: policy events carry cumulative `state_resets` (helper event schema 7, budget event schema 6), and bootstrap replay warns with `feature_set_changed_across_events` or `arm_reset_to_prior` (POL-21).
- The helper policy's score biases have one source and are small priors (POL-07, decision D6): the CLI default (`search_only=-1.00,frame_only=-0.50`), the controller default (`-0.50`/`-0.25`) and the docs disagreed, and with a success reward of 1.0 the CLI value acted as a veto (`search_only` got 1.7% of post-warm-up decisions instead of about 10%). The default is now `frame_only=-0.05,search_only=-0.10`, derived from `DEFAULT_ONLINE_ACTION_SCORE_BIAS_OVERRIDES`; `--adaptive-online-action-score-biases` rejects biases beyond ±0.1; and the config templates no longer set the biases, so they use that default.
- The helper policy's features no longer contain an exactly redundant direction (POL-06): the ten `last_helper_action_*` one-hot flags summed to `bias` on every decision, so `skip_helper` is now the reference level and has no flag, and `attempts_remaining_frac` (always `1 - attempt_frac`) is dropped from the helper and budget features. Saved policy states and bootstrapped states from earlier runs carry the old feature names.
- Greedy ties in the adaptive policies are broken uniformly at random with the policy RNG, and the logged propensity accounts for the tie (`ε/K + (1−ε)/|ties|`); at cold start every tie went to `invoke_helper`, the budget gate's default action or the `time` family. Controller state keeps `applied_action_counts` next to `action_counts` (the selected action), so shadow-mode state no longer counts only actions that never ran; online and budget state schema 4, schema 3 still loads, and bootstrap replay restores the applied counts (POL-24).
- The adaptive policies' NaN-feature counter, the only signal of NaN features, is updated under the controller lock; concurrent decisions lost increments (POL-16).
- Scoring a bandit arm no longer changes it: when an arm's matrix is singular, the re-regularized matrix is used for that score only instead of being written back with an extra `l2·I` on every such call (POL-22).
- The adaptive bandits can learn (POL-01, decision D6): LinUCB recency decay defaults to 1.0 for both the helper and budget policies (it was 0.95 and shrank all ten arms on every update, leaving about 20 observations of evidence), and decay below 1 now only discounts the arm being updated. Outcome events carry `effective_evidence` per arm (`trace(A_k) - d·λ`; online event schema 6, budget event schema 5), the shared LinUCB core serves all three controllers (`evals/linucb_core.py`), and run metadata records `policy_defaults_version: 2`.
- Doubly robust off-policy estimates are cross-fitted (5 folds per stratum) instead of fitting the reward model on the samples it corrects, and the OPE summary reports the ridge alphas actually used per action (`ridge_alphas_selected`, `ridge_alpha_fallback`, `dr_crossfit_folds`) instead of the configured fallback value. Very small strata now report DR as unsupported (`insufficient_crossfit_support`) (STA-11).
- Off-policy samples without a logged propensity are dropped and counted (`propensity_missing_dropped`) instead of silently getting propensity 1.0 in IPS, SNIPS and DR; the OPE lines print that count and the number of shadow-mode propensities corrected for POL-03 (STA-12).
- Conditions with different attempt budgets are compared at a common k (STA-13, D12): each condition reports `success_within_common_k`, the share of cases solved within the smallest attempt budget in the bucket (`common_attempt_k`), next to its own-budget `success_rate`; `evals.stats.pass_at_k` adds the unbiased pass@k estimator.
- `baselines/compare_results` pairs cases through the shared case-level analysis: repeated runs are aggregated by majority vote with `runs_per_case_max` reported, instead of the last run in glob order overwriting the others, and per-category N matches the paired N (BAS-06). An unreadable input file prints a warning instead of raising `TypeError` (BAS-09).
- `analyze_results --strict-input` now also governs schema validation: without it a schema-invalid record is skipped and counted as `records_schema_invalid` instead of aborting the whole analysis (STA-25).
- `eval_two_pass_summary` labels its result as the cumulative outcome of two passes, not an "absolute uplift vs baseline": the rescue pass runs only on first-pass failures, so it is about twice the compute and cannot show a regression. It now prints each pass's own token and runtime cost (STA-22).
- `report_run_metrics` accepts `--adaptive-ope-min-action-support` and `--adaptive-ope-propensity-floor`, passes them to the condition summary and prints the effective values; it always used the defaults, so its off-policy numbers could differ from `analyze_results` (STA-21).
- `budget_risk_report` reads attempts through the shared case-outcome rule (no `infra_event` rows, one row per attempt, so retried attempts aren't double-counted) and keys cases by `run_id`, so a case that failed in one run is no longer hidden by its success in another (STA-18).
- Bucket-level adaptive rates (helper invocation, online and budget policy) are computed over the rows with adaptive telemetry only; every non-adaptive condition's cases used to dilute them, so with one adaptive and one native condition a 50% helper rate read 25% (STA-16).
- `report_run_metrics` overhead shares are a ratio of sums over the rows that report both values, zeros included, with the row count printed; one 80 s LLM case among ten 100 s cases read 80% instead of 8% (STA-08). The baseline internal-total, eval and orchestration times, documented and labelled but never accumulated, now appear (STA-17).
- Multiple comparisons are controlled (STA-06, STA-20): `analyze_results --stats` Holm-adjusts the McNemar tests across every condition pair (`mcnemar_p_adjusted`, `holm_rejected`; nine conditions are 36 tests, so about 1.8 spurious significant pairs were expected), and `report_run_metrics` marks the `--primary-slice` test as the pre-registered primary endpoint and Holm-adjusts the three nested exploratory slices.
- Success-rate uplifts compare conditions on the cases both ran (STA-03): raw rates over different case sets were subtracted, so a condition that ran only the 5 easiest of 10 cases showed +30pp where the matched uplift is 0. Pairwise entries carry `n_matched_cases` and `case_sets_match`, conditions report `cases_expected` and `cases_missing`, and both summaries print them.
- `power_report` uses the McNemar power formula on the discordant counts (`|n01 - n10| / sqrt(n_discordant) - z`); it mixed the discordant count with the overall uplift and reported 0.245 where the power is about 0.949 (STA-04). An undefined Cohen's kappa is reported as `None`/`undefined` instead of "almost_perfect" and invalid `NaN` JSON (STA-10).
- Paired statistics use the case as the unit (STA-02, STA-14): repeated runs of a case were independent McNemar pairs and bootstrap draws, so tripling an identical result turned p=0.29 into p=0.02. `analyze_paired_results` now aggregates a case's runs (`run_aggregation`: `majority_vote` by default, `any_success`, or `per_run`, which refuses repeated runs), bootstraps over cases with run-averaged success rates, and reports `unit`, `runs_per_case_max` and `tied_cases`; `analyze_results --aggregation per_run` refuses multi-run input; `report_run_metrics` pairs token, runtime and kappa data per case. A majority-vote tie (k even, half succeeded) is undecided instead of a failure, so k=2 no longer needs both runs to succeed.
- Missing token telemetry is `None`, not 0 tokens (STA-01): a case whose attempts all reported `token_usage_available: false` used to sum to 0, so a baseline that timed out without usage looked cheapest, counted as solved within every fixed token budget and pushed coverage to 100%. Per-attempt token means divide by the attempts that reported tokens (STA-09), and rows and conditions report `token_covered_attempts` and `token_covered_fraction`.
- One definition of a case outcome (`evals/case_outcome.py`) is shared by `analyze_results`, `report_run_metrics`, `select_cases` and `extract_failed_cases`, so a case can no longer succeed in the summary and appear in `<run>.failed.txt`; a `unit_runtime_error` row is a failure instead of vanishing from `select_cases`, and duplicate rows for one attempt resolve the same way regardless of file order (STA-19, DAT-24). Condition names are stripped everywhere and condition families come from one registry (`CONDITION_FAMILIES`), so `eval_summary` no longer treats any name containing "adaptive" as adaptive.
- Eval statistics share one implementation (STA-05, STA-07, STA-15, STA-23, STA-24): McNemar is always the exact binomial test with `alpha` as a parameter (it switched to a chi-square approximation above 25 discordant pairs), Holm rejects at the boundary (`<=`) and derives rejection from the adjusted p, every bootstrap is the seeded percentile method (`report_run_metrics` used a second, `basic` bootstrap that gave a contradictory CI for the same token delta), `analyze_results --bootstrap-seed` (default 42) makes `--stats` reproducible and is recorded in `analysis_config`, and p-values print with three significant digits instead of being rounded to two or four decimals.
- `evals/README.md` documents the case-suite limits the review found: ConDefects cases (69%) are graded by at most five visible sample tests, `snapshot_dependency` is derived from the heuristic category rather than labelled, and 14 committed cases are flagged by the determinism scan; changes are left to decision D12 (DAT-17, DAT-21).
- ConDefects, DebugBench and PyResBugs imports skip a bug whose `source_id` is already present in the output directory, so a changed difficulty estimate (part of the case ID) no longer imports the same bug twice; PyResBugs' old existence check looked for an ID without the difficulty suffix and never matched. Existing case IDs are unchanged (DAT-16).
- DebugBench test generation parses example values as JSON or Python literals and emits them with `repr`, rejecting anything that doesn't round-trip, instead of pasting dataset text into `test_buggy.py` as code that runs under the default local executor; LeetCode `true`/`false`/`null` now become `True`/`False`/`None` (DAT-20). ConDefects already embedded its inputs with `repr`.
- Importer `--validate` now runs every generated case (the baseline must fail, the oracle patch must fix it) in all nine importers; seven of them only checked the schema. The schema check always runs, the importer functions default to `validate=False` like the CLI, and native-repro cases (BugsInPy, PyBugHive, TrickyBugs) are checked for the files their oracle patch changes instead of a `buggy.py` they never have, which made `--validate` reject all of them (DAT-05).
- Importers format code with `sys.executable -m ruff format --isolated` and the repository's line length and target version, so neither the `ruff` on `PATH` nor a `pyproject.toml` in the working directory changes imported case content; a missing ruff or a timeout raises `RuffFormatError` instead of silently skipping formatting (DAT-15).
- `generate_oracle_patch` emits a correct diff when the source lacks a trailing newline (it used to merge the last removed and added lines) and shares the runner's diff builder; `count_patch_lines` splits on `\n` only (DAT-10). The 1,828 committed patches were unaffected because ruff had added the newline; the native and HumanEvalPack importers don't run ruff.
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

### Performance

- LinUCB scoring uses each arm's cached inverse matrix, kept current by Sherman-Morrison rank-1 updates and recomputed exactly every 64 updates, instead of ten linear solves per decision; an entry is dropped whenever its matrix is replaced (decay, feature expansion, arm reset, load). Score plus update with 53 features and 10 arms went from about 0.19 ms to 0.05 ms per decision, and percentile caps are no longer refitted from a copied, re-sorted history on every decision (POL-20, with POL-09).

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
