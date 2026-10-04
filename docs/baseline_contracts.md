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
Baseline-specific knobs such as `baseline_rlm_temperature` still win when set.

| Adapter | Enforced | How |
|---|---|---|
| INTERVENOR | yes | request temperature, `max_tokens`, `seed`; client `max_retries` |
| lambda-RLM, RLM | yes | env-client payload `temperature` and `max_tokens` |
| LDB, TraceCoder | no | `TRACEGATE_BASELINE_*` environment variables; the upstream patches still hard-code their own sampling until they read them |

Each record's `patch_meta.baseline_generation` holds the effective values and `enforced`.
`process_timeout_sec` is one wall-clock deadline per `run_repair` call: INTERVENOR stops
iterating when it is reached (`error="deadline_exceeded"`), and every record carries
`baseline_deadline_sec`, `baseline_run_repair_wall_sec` and `baseline_deadline_exceeded`.
TraceCoder's per-case test timeout falls back to the process timeout like the other adapters
(BAS-12).

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
