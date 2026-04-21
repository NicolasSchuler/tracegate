# RCA Prompt Contract

> **Date:** Unknown | **Status:** Current

This contract is used by `evals/run_eval.py` prompts and optional MCP `with_rca=true`
responses to reduce debugging loops where an LLM repeatedly applies equivalent fixes
without updating the underlying root-cause model.

Evidence tools are evidence-first by default (`with_rca=false`). RCA coaching is an
explicit opt-in channel and should not be mixed into default evidence transport.

## Required internal sequence

1. Observed failure with concrete evidence references.
2. Falsifiable root-cause hypothesis.
3. Why the cited evidence supports the hypothesis.
4. Minimal fix plan.
5. Verification plan.
6. Diff vs previous failed attempt.

## Loop-breaking rule

If the failure signature is unchanged across attempts, the model should not repeat an
equivalent patch unless at least one of these changed:

- new evidence was collected
- hypothesis was revised/refined
- explicit exploratory intent was stated

## Scope and budget guardrails

- Request only the minimum additional evidence needed before patching.
- Focus on behavior-changing fixes; ignore cosmetic CLI/TUI wording or formatting
  tweaks unless failing evidence explicitly requires them.
- Keep prompt sections bounded by prompt-budget limits in `evals/run_eval.py`.

## Verification gate modes

- `off`: no gate checks.
- `soft` (default): emit warnings and next-required steps, but allow progress.
- `strict`: block missing RCA prerequisites unless exploratory override is provided.
