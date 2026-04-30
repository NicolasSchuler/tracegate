# ASE 2026 Demo Workflow

This directory contains the small screencast workflow for the ASE 2026 Tools and
Data Sets submission. It demonstrates TraceGate as a research artifact for
policy-controlled crash evidence in LLM-assisted repair, not as a polished IDE
assistant.

## Demo Story

Run INTERVENOR/Qwen-9B on 10 residual failures with TraceGate's active adaptive
layer. The policy is warm-started from the downloaded evaluation event logs, so
the screencast shows mature policy behavior rather than a cold random start.

The visible flow is:

1. failing residual-repair case list
2. bootstrapped helper and budget policy states
3. active adaptive run over 10 cases
4. captured `snapshot.json`
5. selected evidence and budget actions in policy logs
6. persisted `prompt.txt` and `evidence.json`
7. generated `patch.diff`
8. validation in `pytest_after_combined.txt`
9. logged helper/budget rewards and outcomes

## Prerequisites

- The Zenodo data is extracted under `analysis/data/`.
- An OpenAI-compatible local model endpoint is running.
- External baseline dependencies for INTERVENOR are available locally.

By default the runner expects:

```bash
analysis/data/final-intervenor-adaptive-repair-9b.adaptive_policy.jsonl
analysis/data/final-intervenor-adaptive-repair-9b.adaptive_budget_policy.jsonl
http://127.0.0.1:12000/v1
Qwen3.5-9B-UD-Q6_K_XL
```

## Run

From the repository root:

```bash
demo/ase2026/run_demo.sh
```

Common overrides:

```bash
OPENAI_BASE_URL=http://127.0.0.1:12001/v1 \
OPENAI_MODEL=Qwen3.5-9B-UD-Q6_K_XL \
RUN_ID=ase2026-intervenor9b-demo \
demo/ase2026/run_demo.sh
```

The script sets `HELPER_EPSILON=0.0` and `BUDGET_EPSILON=0.0` by default after
warm-starting, which makes the demo less dependent on exploration randomness.
Set either variable to the paper-run values if you want exploration visible.

The Rich live dashboard is enabled by default for a real terminal by unsetting
`NO_COLOR` and setting a fallback `TERM=xterm-256color`. Disable that setup with
`ENABLE_TUI=0` if you want plain logs:

```bash
ENABLE_TUI=0 demo/ase2026/run_demo.sh
```

To warm-start from a local prewarm run instead of the downloaded logs, point the
runner at that run prefix:

```bash
SOURCE_RUN_PREFIX=evals/results/<prewarm-run-id> demo/ase2026/run_demo.sh
```

You can also override `ONLINE_EVENTS` and `BUDGET_EVENTS` directly.

## Outputs

Primary outputs:

```bash
evals/results/<run-id>.jsonl
evals/results/<run-id>.adaptive_policy.jsonl
evals/results/<run-id>.adaptive_budget_policy.jsonl
evals/artifacts/<run-id>/contexts/<case-id>/layered_adaptive_llm/attempt_*/
evals/artifacts/policies/<run-id>.online.json
evals/artifacts/policies/<run-id>.budget.json
```

The runner records full prompt and evidence artifacts:

```bash
evals/artifacts/<run-id>/contexts/.../prompt.txt
evals/artifacts/<run-id>/contexts/.../evidence.json
evals/artifacts/<run-id>/contexts/.../snapshot.json
evals/artifacts/<run-id>/contexts/.../patch.diff
evals/artifacts/<run-id>/contexts/.../pytest_after_combined.txt
```

To summarize an existing run:

```bash
uv run python demo/ase2026/summarize_demo.py --run-id <run-id>
```

## Screencast Timing

- 0:00-0:30: Show `demo/ase2026/cases.txt` and the Zenodo-backed event logs in
  `analysis/data/`.
- 0:30-1:10: Run `demo/ase2026/run_demo.sh`; point out policy bootstrap and
  active `layered_adaptive_llm` settings.
- 1:10-2:15: Open one case under `evals/cases/` and show the failing test input.
- 2:15-3:20: Inspect an attempt artifact directory: `snapshot.json`,
  `evidence.json`, `prompt.txt`, and `patch.diff`.
- 3:20-4:25: Inspect `*.adaptive_policy.jsonl` and
  `*.adaptive_budget_policy.jsonl` to show selected actions, rewards, and
  budget controls.
- 4:25-5:00: Show validation (`pytest_after_combined.txt`) and the run summary.

## Claims To Make

- TraceGate captures structured crash evidence and integrates it with existing
  repair loops such as INTERVENOR.
- A policy layer, not the patching model alone, controls which evidence and
  budget actions are exposed.
- The demo is a small replayable micro-run linked to the larger Zenodo results,
  not a replacement for the full benchmark.

## Claims To Avoid

- Do not present this as a general human IDE/debugging assistant.
- Do not claim semantic correctness beyond the validation tests.
- Do not imply the 10-case run estimates full benchmark performance.
