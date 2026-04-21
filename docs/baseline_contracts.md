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
