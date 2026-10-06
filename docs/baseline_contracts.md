# Baseline Contracts

This document defines the explicit llmdebug integration contract for each
external baseline.

## Shared Fields

Every baseline result now reports:

- `contract_version`
- `upstream_commit`
- `fidelity_level`
- `repair_scope_policy`
- `prompt_contract_kind`
- `output_contract`
- `patch_scope_summary`
- `config_validation_mode`

Hybrid adaptive+baseline attempts additionally record:

- `baseline_hybrid_guidance_mode`
- `baseline_hybrid_guidance_chars`
- `baseline_hybrid_guidance_sections`

`baseline_case_context.json` is versioned with `schema_version` and currently
uses `v1`.

Each record's `patch_meta.baseline_repo_state` carries the observed upstream checkout
(BAS-05, BAS-14): `observed_commit`, `dirty` and `tree_patch_sha256`. The tree hash covers
the tracked diff against `HEAD` plus a path/content digest of the untracked files, so editing
an untracked module the upstream patches added (TraceCoder's `src/exact_test_context.py`, for
example) changes it. Untracked files larger than 4 MiB contribute their size instead of a
content hash.

## Case-information contract (`case_info_policy = neutral_v1`)

Defined in `baselines/case_information.py` (review findings ATT-02 and BAS-07,
decision D1). It fixes which case data any arm may see:

- `baseline_case_context.json` carries only `case_id`, `python_args`,
  `timeout_sec` and `case_info_policy`. Dataset-derived labels (`description`,
  `category`, `difficulty`, `snapshot_dependency`) stay in `case.json` for
  analysis and never reach a model. The loader drops them from older context
  files and records `hidden_case_fields_dropped:<fields>` in
  `baseline_case_context_issues`.
- Every arm sees the buggy source and the test source. The native conditions,
  INTERVENOR, TraceCoder, lambda-RLM and RLM see the full `test_buggy.py`
  (`test_source_exposure = "full"`). LDB's converter only accepts plain-assert
  test bodies and passes them as its visible tests, so LDB sees the assertions
  (`test_source_exposure = "asserts"`).
- Each record's `patch_meta` carries `case_info_policy` and
  `case_information` (`case_info_policy`, `test_source_exposure`,
  `prompt_files`), and `run_metadata.patcher_config.case_info_policy` records
  the run-level policy.

## Generation contract (decision D4, BAS-03/04/08/11)

`run_eval` builds one `BaselineGenerationSettings` from the native flags and passes it to
every adapter through `BaselineConfig.generation`: `--openai-temperature`, `--seed`,
`--openai-max-tokens` (one completion cap) and `--openai-transport-max-retries`.
Baseline-specific knobs such as `baseline_rlm_temperature` still win when set; each adapter
declares those in `generation_override_keys`, and `patch_meta.baseline_generation` then reports
the overriding value plus an `overrides` map naming the knob that won.

| Adapter | Enforced | How |
|---|---|---|
| INTERVENOR | yes | request temperature, `max_tokens`, `seed`; client `max_retries` |
| lambda-RLM, RLM | yes | env-client payload `temperature`, `max_tokens`, `seed` and `transport_max_retries`, applied to the wrapped upstream client (`EnvClientGenerationSettings`) |
| LDB, TraceCoder | no | `TRACEGATE_BASELINE_*` environment variables; the upstream patches still hard-code their own sampling until they read them |

Each record's `patch_meta.baseline_generation` holds the effective values and `enforced`.
`run_metadata.patcher_config` records the resolved contract for the run as
`baseline_generation` (plus `baseline_api_timeout_sec`, `baseline_process_timeout_sec`,
`baseline_max_iterations` and the resolved `baseline_extra`), so changing any of them changes
the run-settings fingerprint and resume refuses to append incomparable records. The seed
follows `--deterministic` exactly like native generation: `--seed` alone drives policy
exploration and leaves baseline sampling unseeded.
`process_timeout_sec` is one wall-clock deadline per `run_repair` call, computed once at entry
(`RepairDeadline`). Every blocking step is clamped to what is left of it — the pytest
precheck/postcheck, the generation request (in-process and through the env client) and the
env-client subprocess — and no new step starts once it has expired: INTERVENOR stops before the
learner call and before the postcheck, lambda-RLM and RLM stop before the next attempt, all with
`error="deadline_exceeded"`. Every record carries `baseline_deadline_sec`,
`baseline_run_repair_wall_sec` and `baseline_deadline_exceeded`.
TraceCoder's per-case test timeout falls back to the process timeout like the other adapters
(BAS-12).

## Credentials in the env clients (LLM-25)

The lambda-RLM and RLM env clients receive the API key in `TRACEGATE_ENV_CLIENT_API_KEY`,
never in their request file. `baselines.openai_compat.env_client_api_key` *consumes* that
variable: it is removed from the env client's own environment as it is read, and candidate
subprocesses (the import smoke test and `run_case_tests`) are started from
`candidate_subprocess_env()`, so model-written code can never print the key into
model-visible output or into persisted artifacts.

## Retry context (BAS-02, decision D4)

`BaselinePatcher.propose_patch` ignores the harness repair prompt, so each harness attempt used
to restart cold while the native conditions saw the previous failure. On attempt 2+ the patcher
now injects `case_input["llmdebug_retry_context"]` and appends the same text (the attempt number
and the tail of the failing pytest output) to the adapter's free-text task field
(`retry_context_prompt_field`: INTERVENOR `prompt`, TraceCoder `complete_prompt`,
lambda-RLM/RLM `prompt_context`). LDB's prompt is a function stub with no such field, so
records carry `baseline_retry_context_consumed=false` there.

## Unsupported cases (BAS-01, decision D3)

A baseline that can't represent a case (for example LDB's converter rejects subprocess-driven,
class-based or multi-file cases) records `error="case_incompatible"` and
`baseline_supported=False`. The analysis drops those rows from the baseline's success rate and
from paired statistics, so the comparison runs on the convertible subset and reports
`baseline_unsupported_cases`. `run_metadata.baseline_case_support` records `total_cases`,
`convertible_cases` and `incompatible_reasons` for the run's case selection.

Runs recorded before this contract gave the baselines `case.description`
(often a root-cause statement or bug-type label) while the native conditions
never saw it; see the 2026-10-04 review findings before comparing them.

## Current Integrations

| Baseline | Fidelity | Repair Scope | Prompt Contract |
|---|---|---|---|
| `ldb` | `upstream_patched` | `single_file_only` | upstream native LDB prompt contract via llmdebug lowering |
| `intervenor` | `prompt_asset_port` | `single_file_only` | upstream prompt assets plus adapter-owned repair/output instructions |
| `tracecoder` | `upstream_patched` | `single_file_only` | patched llmdebug TraceCoder dataset/prompt contract |
| `lambda_rlm` | `generic_api_adaptation` | `single_file_only` | adapter-defined code-repair prompt (`canonical`) or solve-first repair-fair prompt (`repair_fair_v1`) |
| `rlm` | `generic_api_adaptation` | `single_file_only` | adapter-defined tool-augmented code-repair prompt (`canonical`) or solve-first repair-fair prompt (`repair_fair_v1`) |

Repair-fair prompt variants are reported as distinct `prompt_contract_kind`
values in run metadata so tuned-baseline runs can be analyzed separately from
canonical runs.

Hybrid augmentation modes are intentionally separate from
`prompt_contract_kind`:

- `tuned_v1` preserves the current baseline-specific llmdebug guidance path
- `fair_v1` uses one shared canonical llmdebug guidance block across all five
  baselines

This lets paper-facing analysis separate:

- raw baseline
- hybrid-canonical (`baseline_hybrid_guidance_mode = fair_v1`)
- hybrid-tuned (`baseline_hybrid_guidance_mode = tuned_v1`)

## Explicit Adapter Knobs

- shared hybrid knob for all five baselines
  - `baseline_hybrid_guidance_mode` (default `tuned_v1`; allowed values:
    `tuned_v1`, `fair_v1`)
- `intervenor`
  - `baseline_intervenor_teacher_max_tokens` (default `3072`)
  - `baseline_intervenor_learner_max_tokens` (default `6144`)
  - `baseline_intervenor_teacher_guidance_max_chars` (default `4000`)
  - `baseline_intervenor_learner_guidance_max_chars` (default `6000`)
  - `baseline_intervenor_append_output_contract` (default `true`)
- `tracecoder`
  - `baseline_tracecoder_controller_guidance_max_chars` (default `2000`)
  - `baseline_tracecoder_max_no_improvement_streak` (default `2`)
- `lambda_rlm`
  - `baseline_lambda_controller_guidance_max_chars` (default `2500`)
  - `baseline_lambda_failure_feedback_max_chars` (default `3500`)
  - `baseline_lambda_failure_feedback_meta_max_chars` (default `2000`)
  - `baseline_lambda_prompt_contract` (default `canonical`; allowed values: `canonical`, `repair_fair_v1`)
- `rlm`
  - `baseline_rlm_controller_guidance_max_chars` (default `2500`)
  - `baseline_rlm_failure_feedback_max_chars` (default `3500`)
  - `baseline_rlm_failure_feedback_meta_max_chars` (default `2000`)
  - `baseline_rlm_prompt_contract` (default `canonical`; allowed values: `canonical`, `repair_fair_v1`)

## Validation Policy

- Eval-mode baseline runs use strict `config.extra` validation by default.
- Unknown baseline-specific config keys fail fast unless the caller explicitly
  sets warn-only mode.
- Baseline-incompatible cases should fail explicitly with a compatibility reason
  instead of being silently reshaped.
