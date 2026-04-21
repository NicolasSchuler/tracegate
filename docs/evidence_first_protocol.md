# Evidence-First Protocol

> **Date:** 2026-02-16 | **Status:** Current

`llmdebug` is an evidence transport and retry-state scaffold, not a replacement reasoner.
The LLM does the debugging logic; the tooling should keep evidence compact, structured,
and minimally biased.

## Overview

The evidence-first philosophy inverts the common LLM debugging pattern of
_guess → patch → hope_. Instead, every debugging action must be grounded in
concrete, machine-readable evidence collected at failure time. The tooling
captures snapshots (exception details, stack frames, locals, execution context),
surfaces them through CLI and MCP tools, and enforces — via configurable gate
modes — that hypotheses and patches cite real evidence before proceeding. The
result is shorter debugging loops, fewer equivalent-patch retries, and
reproducible reasoning chains that can be audited after the fact.

## Core Principles

1. **Structured evidence by default.** Evidence tools return JSON
   (`response_format="json"`) so consumers parse data, not prose.
2. **RCA coaching is opt-in.** Root-cause analysis state is activated with
   `with_rca=true` and never mixed into default evidence transport.
3. **Compact payloads by default.** `evidence_schema="summary"` ships the
   minimum useful context; `"full"` is available when explicitly requested.
4. **Delta-first retries.** On each retry, compare the current failure to the
   previous failure _before_ proposing a patch.
5. **Stable evidence IDs.** Every evidence block gets a deterministic
   `evidence_id` and is deduplicated across retries.
6. **Deterministic prompt budgets.** Prompt assembly has per-section caps
   (retry packet, logs, snapshot, evidence) with predictable truncation order.
7. **New-evidence gate.** Repeated failure signatures require new evidence
   before the system accepts another equivalent patch.

## Worked Example

A concrete walk-through of the full protocol, from failure to verified fix.

### 1. Test failure — snapshot is captured automatically

```bash
$ pytest tests/test_parser.py::test_parse_config
FAILED  # llmdebug writes .llmdebug/latest.json
```

### 2. Read crash-level evidence

```bash
$ llmdebug show
# Exception: KeyError: 'timeout'
# Crash frame: src/myapp/parser.py:42  in parse_config
#   code: val = raw["timeout"]
# Locals: raw = {"host": "localhost", "port": 8080}
```

The snapshot tells you _exactly_ what the runtime state was. No guessing.

### 3. Expand context if needed

```bash
$ llmdebug show --detail full     # all frames
$ llmdebug frames -i 1            # inspect the caller frame
$ llmdebug git-context            # blame + recent commits on crash file
```

### 4. Form a falsifiable hypothesis

> **Hypothesis:** `parse_config` assumes the `"timeout"` key is always present
> in the raw config dict, but the test fixture omits it.

This hypothesis is _falsifiable_ — it predicts that adding a default or
making the key optional will turn the test green.

### 5. Plan and apply a targeted, minimal fix

```python
# Before
val = raw["timeout"]

# After — single behaviour-changing line
val = raw.get("timeout", 30)
```

One line. One behaviour change. Directly tied to the evidence.

### 6. Verify

```bash
$ pytest tests/test_parser.py::test_parse_config   # now passes
$ llmdebug diff                                     # confirm snapshot delta
```

`llmdebug diff` shows the before/after snapshot comparison, confirming the
failure signature changed (or disappeared) as the hypothesis predicted.

## Gate Modes

Gate modes control how strictly the protocol enforces evidence-first behaviour
during RCA-driven debugging. Set via the `gate_mode` parameter on MCP tools or
in eval configurations.

| Mode | Behaviour | Use case |
|------|-----------|----------|
| `off` | No checks. All transitions allowed unconditionally. | Exploratory hacking, early prototyping. |
| `soft` (default) | Warns when prerequisites are missing but allows progress. | Day-to-day development — gentle nudges. |
| `strict` | Blocks transitions when required evidence, hypothesis, or plan is missing. | CI evals, audit-quality debugging sessions. |

### What the gates check

The gate system ([`gates.py`](../src/llmdebug/gates.py)) evaluates these
prerequisites before allowing an RCA state transition:

- **`evidence_required`** — actions past `capture` require cited evidence refs.
- **`hypothesis_required`** — planning or patching requires a falsifiable claim.
- **`plan_required`** — code changes require an experiment plan.
- **`verification_required`** — resolving requires a passing verification.
- **`loop_protection`** — repeated failure signatures without hypothesis change
  trigger a warning (and a block in `strict` mode after the exploration budget).

In `strict` mode, a blocked action can be overridden by setting
`exploratory=true` with an `override_reason`, allowing intentional exploration
without disabling the gate system entirely.

### RCA state progression

```
captured → evidence_bound → hypothesized → planned → changed → verified → resolved
                                                       ↓
                                                    stalled  (loop protection)
```

Each arrow requires the gate checks above to pass (in `soft`/`strict` mode).

## MCP Integration

IDE clients (Cursor, VS Code + Continue, etc.) interact with `llmdebug` via
the MCP server. Evidence-first behaviour is the default contract — tools return
compact, structured JSON and never inject RCA coaching unless asked.

Key MCP tools and their evidence-first roles:

| Tool | Role |
|------|------|
| `llmdebug_diagnose` | Structured crash diagnosis: exception, crash frame, locals, repro command. |
| `llmdebug_show` | Snapshot display with layered detail (`crash` → `full` → `context`). |
| `llmdebug_frame` | Inspect a single stack frame — locals, source, position. |
| `llmdebug_list` | List available snapshots with metadata. |
| `llmdebug_rca_advance` | Advance RCA state (gate-checked). |
| `llmdebug_rca_status` | Current RCA state, next-required steps, gate feedback. |
| `llmdebug_rca_history` | Full transition history for audit. |

All tools accept `response_format`, `evidence_schema`, and `gate_mode`.
See the full [MCP Reference](mcp-reference.md) for envelope schemas and
parameter details.

## CLI Workflow

The CLI provides the same evidence-first workflow outside an IDE.
Requires the `cli` extra: `pip install 'llmdebug[cli]'`.

```bash
# 1. Run tests — snapshot captured on failure
pytest

# 2. Read crash evidence (compact by default)
llmdebug show

# 3. Expand when needed
llmdebug show --detail full          # all frames
llmdebug show --detail context       # everything (git, env, repro)
llmdebug frames -i 0                 # zoom into crash frame
llmdebug git-context                 # blame + recent diffs

# 4. After a fix attempt, compare snapshots
pytest
llmdebug diff                        # latest vs previous

# 5. Inspect specific snapshots
llmdebug list                        # see all snapshots
llmdebug show '#2'                   # view snapshot #2
llmdebug diff '#3' '#1'              # compare any two snapshots
```

See the full [CLI Reference](cli-reference.md) for all commands, detail
levels, and snapshot reference syntax.

## Anti-Patterns

Things the evidence-first protocol is designed to prevent.

### ❌ Guess-and-check

Proposing a fix based on the exception _type_ alone without reading locals,
stack frames, or execution context. The snapshot exists — use it.

### ❌ Multi-change patches

Changing three things at once "just in case." Each change should trace back to
a specific piece of evidence and a falsifiable claim about why it helps.
One behaviour change per attempt.

### ❌ Ignoring snapshot data

Re-reading only the traceback string when the snapshot contains locals,
frame-level source, and repro commands. Layered detail (`crash` → `full` →
`context`) exists so you can expand progressively, not skip it.

### ❌ Repeating equivalent patches

Applying the same fix after the same failure signature without collecting new
evidence or revising the hypothesis. The retry gate (`loop_protection`) exists
to catch this — but the habit matters more than the gate.

### ❌ Skipping verification

Marking a bug resolved without running the test again. The `strict` gate mode
blocks this explicitly: `resolve` requires `verification_result.status == "pass"`.

## Eval Packaging Conditions

- `base_repair`: baseline pytest-only context.
- `layered_adaptive_llm`: base repair + structured snapshot with online policy packaging and
  hard retry gate (`new evidence before same-signature patch`), plus delta-only retries
  after attempt 1.

## Retry Gate Policy

When failure signatures repeat:

1. Compute a deterministic evidence fingerprint from retrieved evidence blocks.
2. If fingerprint is unchanged, do not propose a patch.
3. Record `note="retry_gate_no_new_evidence"` and retry-gate metadata.
4. Allow patch proposal only after evidence changes.

This policy keeps loops informative without adding model-side heuristics.

## Prompt Budgeting

Stage-B prompt assembly uses deterministic budgets:

1. Per-section caps for retry packet, pytest output, snapshot summary, and Stage-A evidence.
2. Optional total prompt cap with deterministic truncation/drop order.
3. Attempt metadata records budget usage and dropped sections for analysis.

## Reporting

Analyzer outputs include:

1. Legacy uplift (`layered_adaptive_llm - base_repair`) for continuity.
2. Structured uplift metrics and retry-gate block rate.
3. Pairwise uplift matrix across all observed conditions.
4. Prompt-size and evidence-ID telemetry for protocol-level debugging.
