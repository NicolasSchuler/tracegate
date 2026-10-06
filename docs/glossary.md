# Glossary

Alphabetical reference of key terms used across `llmdebug` documentation.

---

**Capture Pipeline** — The chain from unhandled exception → stack-frame extraction → local-variable serialization → snapshot file output. See: [Configuration Reference](configuration.md#capture-entry-points).

**Compact Keys** — Abbreviated JSON key names (e.g., `_exc` for `exception`) used by the `json_compact` output format to reduce token usage by ~40%; `get_latest_snapshot()` auto-expands them on read. See: [Configuration Reference](configuration.md#output-formats).

**Condition** *(eval)* — An experimental arm in the llmdebug eval harness (`base_repair`, `cot_repair`, `layered_adaptive_gated`, `layered_adaptive_llm`, `ldb`, `intervenor`, `tracecoder`, `lambda_rlm`, `rlm`). See: [Eval Framework](../evals/README.md).

**Crash Frame** — The stack frame where the exception originated, identified by the `crash_frame_index` field in the snapshot. The hypothesis engine, detail filter, and failure-signature builder all key off this frame for root-cause analysis.

**Crash Frame Index** — The zero-based integer index into the `frames` array pointing to the crash frame. When the `crash` detail level is selected, only this frame is retained; all other frames are stripped. See: [`detail_filter.py`](../src/llmdebug/detail_filter.py).

**Debug Session** — The versioned JSON envelope (`kind: "llmdebug.debug_session"`, `schema_version: "2.1"`) that wraps every snapshot with session metadata; the on-disk contract for all consumers. See: [README § Output](../README.md#output).

**Debug Snapshot** — A structured JSON capture of an exception event containing the exception chain, prioritized stack frames, typed locals, and optional context (git, env, coverage). See: [README § Features](../README.md#features).

**Detail Level** — The three-tier disclosure filter (`crash` ~2 K tokens, `full` ~5 K, `context` ~10 K) controlling how much of a snapshot is exposed to a consumer. See: [CLI Reference § Detail Levels](cli-reference.md#detail-levels).

**Error Category** — A pattern-based classification assigned to an exception by matching its type and message against built-in rules (e.g., `shape_mismatch`, `key_error`, `import_error`). Each category includes an actionable suggestion surfaced in the snapshot. See: [`error_categories.py`](../src/llmdebug/error_categories.py).

**Evidence-First Protocol** — The debugging methodology requiring the LLM to read snapshot evidence before proposing a patch; RCA coaching is opt-in (`with_rca=true`), not default. See: [Evidence-First Protocol](evidence_first_protocol.md).

**Evidence Refs** — A deduplicated, order-stable list of string references (e.g., crash frame locations, traceback lines, diff hunks) cited during an RCA session. Merged across attempts with a cap of 16 items. See: [RCA Prompt Contract](rca_prompt_contract.md).

**Exploration Budget** — The number of repeated-signature attempts allowed before the gate system's loop-protection check triggers. Defaults to 2; exceeding it without hypothesis advancement produces a warning (`soft`) or a block (`strict`). See: [`gates.py`](../src/llmdebug/gates.py).

**Failure Signature** — A stable, hash-based identifier (format `ExceptionType:<xxh64_prefix>`) derived from the exception type, message, crash-frame location, and pytest node ID. Used for stall detection and retry-gate deduplication. See: [`rca_identity.py`](../src/llmdebug/rca_identity.py).

**Gate Feedback** — A frozen dataclass returned by `evaluate_gate()` containing the gate mode, allow/deny decision, severity level, warnings, next-required coaching steps, and the list of triggered checks. Serialized into RCA advance responses. See: [`gates.py`](../src/llmdebug/gates.py).

**Gate Mode** — RCA verification-gate enforcement level: `off` (no checks), `soft` (warnings but allows progress), `strict` (blocks unless evidence prerequisites are met or exploratory override is set). See: [RCA Prompt Contract § Verification gate modes](rca_prompt_contract.md#verification-gate-modes).

**Gate Severity** — The urgency level attached to a `GateFeedback` result: `info` (no issues), `warning` (prerequisites missing but allowed), or `error` (blocked in `strict` mode). Consumers use this to surface appropriate UI indicators.

**GitDirtyMode** — Controls how git dirty-file detection runs when `include_git=True`: `always` (full `git status`), `cached` (index-only check, faster), or `off` (skip dirty detection entirely). See: [Configuration Reference](configuration.md#parameter-reference).

**Hypothesis Advancement** — A boolean progression signal indicating the current hypothesis meaningfully changed compared to the previous attempt (classified as `added`, `refined`, or `replaced`). When `False` across repeated failure signatures, stall detection may trigger. See: [`rca_transitions.py`](../src/llmdebug/rca_transitions.py).

**Hypothesis Engine** — The module (`llmdebug.hypotheses`) that auto-generates ranked debugging hypotheses from snapshot patterns, each with a confidence score and actionable suggestion. See: [Configuration Reference § API Reference](configuration.md#api-reference).

**Hypothesis Pattern** — The machine-readable label assigned to a generated hypothesis (e.g., `empty_array`, `shape_mismatch`, `none_dereference`, `off_by_one`, `nan_inf_anomaly`). Used for deduplication within a session and for analytics across debugging sessions. See: [`hypotheses.py`](../src/llmdebug/hypotheses.py).

**Importer** *(eval)* — A module under `evals/importers/` that converts cases from an external benchmark (DebugBench, ConDefects, BugsInPy, etc.) into the eval `cases/<case_id>/` format. See: [Eval Framework § Case contract](../evals/README.md).

**LinUCB** — Linear Upper Confidence Bound — the contextual-bandit algorithm used by the adaptive online policy to select per-attempt helper-invocation strategies. Balances exploitation of known-good actions with exploration of alternatives using feature-weighted confidence intervals. See: [`adaptive_online_policy.py`](../evals/adaptive_online_policy.py).

**LocalsMode** — Controls how local variables are serialized in stack frames: `safe` (serialize values with truncation), `meta` (type and size metadata only, no values), or `none` (omit locals entirely). See: [Configuration Reference](configuration.md#parameter-reference).

**MCP** — Model Context Protocol — the stdio-based server (`llmdebug-mcp`) exposing 9 tools for IDE integration with Claude Code, Cursor, and other MCP-capable editors. See: [MCP Reference](mcp-reference.md).

**Online Policy** — The adaptive contextual-bandit controller (`AdaptiveOnlinePolicyController`) that decides per-attempt whether to invoke, skip, or route helper calls through a specific strategy (e.g., `frame_only`, `snapshot_lite`, `compact_then_patch`). Supports `off`, `shadow`, and `on` modes. See: [`adaptive_online_policy.py`](../evals/adaptive_online_policy.py).

**Online Policy Action** *(eval)* — One of eight strategy choices available to the online policy: `skip_helper`, `invoke_helper`, `frame_only`, `file_only`, `search_only`, `snapshot_lite`, `compact_then_patch`, or `retry_helper_strict_json`. Each routes the helper call differently to optimize cost/quality trade-offs.

**Output Format** — One of three serialization modes: `json` (pretty, human-readable), `json_compact` (abbreviated keys, default), or `toon` (maximum token savings at ~50% reduction). See: [Configuration Reference § Output Formats](configuration.md#output-formats).

**Patcher** *(eval)* — The backend that generates fix patches during an eval run: `command` (local CLI), `openai` (API-based), `heuristic` (snapshot-aware demo), or `null` (no-op for smoke tests). See: [Eval Framework](../evals/README.md).

**Production Hooks** — The `install_hooks()` API that wires `sys.excepthook`, `threading.excepthook`, and `sys.unraisablehook` for automatic crash capture with rate limiting and PII redaction. See: [Configuration Reference § Production Hooks](configuration.md#production-hooks).

**Rate Limiter** — A sliding-window, per-error-signature throttle used by production hooks to prevent snapshot floods. Keyed on `(exception_type, file, line)` and configured via `max_captures_per_minute`; emits a stderr warning when a signature is first suppressed. See: [`hooks.py`](../src/llmdebug/hooks.py).

**RCA** — Root Cause Analysis — a structured debugging workflow (observe → hypothesize → cite evidence → fix plan → verify → diff vs previous) enforced via prompt contract and optional MCP state machine. See: [RCA Prompt Contract](rca_prompt_contract.md).

**RCA Action** — One of seven transitions that advance the RCA state machine: `capture`, `bind_evidence`, `hypothesize`, `plan`, `change`, `verify`, or `resolve`. Each action maps to a target state and is subject to gate checks in `soft`/`strict` mode. See: [RCA Prompt Contract](rca_prompt_contract.md).

**RCA State** — One of eight discrete states in the RCA state machine: `captured` (initial snapshot taken), `evidence_bound` (evidence cited), `hypothesized` (root-cause claim stated), `planned` (experiment designed), `changed` (patch applied), `verified` (tests re-run), `resolved` (fix confirmed), or `stalled` (loop protection triggered). The current state determines which actions are allowed next. See: [RCA Prompt Contract](rca_prompt_contract.md).

**Redaction Profile** — A privacy preset (`dev`, `ci`, `prod`) that configures default redaction behavior; profiles are additive and explicit `redact`/`redact_*` options always take precedence. See: [Configuration Reference § Redaction Profiles](configuration.md#redaction-profiles).

**Schema Version** — The semantic version string (currently `"2.1"`) embedded in every `DebugSession` envelope, defining the on-disk contract for snapshot consumers. The major part changes on breaking envelope changes; readers accept any `2.x`. `2.1` emits string dict keys in locals as plain strings instead of their `repr`, and counts array `anomalies` on the original array instead of the up-cast float32 copy (so large `float64` values no longer report false `inf`, and CuPy arrays are covered). See: [README § Output](../README.md#output).

**Session ID** — A stable string identifier (`<resolved_out_dir>::<scope>`) derived from the output directory and pytest node ID (or snapshot name). Used to correlate RCA attempts across retries for the same test or code region. See: [`rca_identity.py`](../src/llmdebug/rca_identity.py).

**Snapshot Config** — The frozen dataclass (`SnapshotConfig`) holding all capture parameters (frames, locals mode, redaction, output format, etc.) passed to `@debug_snapshot()` or `snapshot_section()`. See: [Configuration Reference § Parameter Reference](configuration.md#parameter-reference).

**Snapshot Reference** — The uniform addressing syntax (`latest`, `previous`, `#N`, or a file path) used by all CLI and MCP inspection commands to select a snapshot. See: [CLI Reference § Snapshot References](cli-reference.md#snapshot-references).

**Snapshot Section** — A named subsection created via the `snapshot_section()` context manager for targeted instrumentation of a specific code region within a larger run. See: [Configuration Reference § Context Manager](configuration.md#context-manager).

**SourceMode** — Controls which stack frames receive inline source-context lines: `all` (every frame), `crash_only` (only the crash frame), or `none` (no source context). Reduces snapshot size in production captures. See: [Configuration Reference](configuration.md#parameter-reference).

**Stall Detection** — A safety rule that marks the RCA state as `stalled` when the repair streak for one failure signature reaches the stall threshold. The streak counts consecutive `bind_evidence`/`change`/`verify` attempts that bring no new evidence, no advanced hypothesis or plan, and no passing verification, so ordinary workflow steps never trip it. Forces the debugger to gather new evidence or revise the hypothesis before retrying. See: [`rca_transitions.py`](../src/llmdebug/rca_transitions.py).

**Token Budget** — Deterministic per-section caps (retry packet, pytest output, snapshot summary, evidence blocks) governing how much of each content type is included in a prompt. Sections are truncated or dropped in a fixed order when the total cap is exceeded. See: [Evidence-First Protocol § Prompt Budgeting](evidence_first_protocol.md#prompt-budgeting).

**TOON** — A compact, text-oriented output format for snapshots (~50% smaller than pretty JSON) optimized for maximum LLM context-window savings; requires the `toon` install extra. See: [Configuration Reference § Output Formats](configuration.md#output-formats).

**Transition History** — A capped (max 32 entries) chronological log of RCA state transitions, recording the timestamp, action, from/to states, gate decision, and gate mode for each step. Used for audit trails and debugging workflow analysis. See: [`rca_transitions.py`](../src/llmdebug/rca_transitions.py).
