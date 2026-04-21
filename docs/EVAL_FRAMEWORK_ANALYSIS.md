# Evaluation Framework: Metrics/Measurement Capabilities Analysis

## 1. EVAL RUNNER (`evals/run_eval.py`)

### Overview
The eval runner orchestrates case discovery, condition execution, patch proposal via pluggable patchers, pytest verification, and JSONL result recording. It supports local and Docker-based execution, TOML configs, resume functionality, adaptive gating policies, online contextual-bandit policy controllers, and budget-aware resource management.

### Key Metrics Tracked

#### Timing Metrics:
- **`before_duration_sec`** (line 3671): Duration of initial pytest run before patch application
- **`attempt_wall_clock_sec`** (line 3673): Total wall-clock time for the entire attempt
- **`after_duration_sec`** (line 3678): Duration of pytest run after patch application
- **`propose_duration_sec`** (line 8969): Outer patcher wall time for the proposal/repair step
- **`request_context_duration_sec`** (openai_compat_patcher.py:737, 874): Duration of stage A (context request) call
- **`adaptive_online_local_wall_sec`** (line 4840): Wall time spent in adaptive online policy
- **`adaptive_budget_local_wall_sec`** (line 5546): Wall time spent in budget policy

#### Timing Breakdown Semantics:
- **`stage_b_llm_sec`**: Pure Stage-B LLM time. For baseline patchers, this comes from baseline-internal telemetry rather than `propose_duration_sec`.
- **`baseline_internal_total_sec`**: Total time spent inside the baseline repair loop.
- **`baseline_eval_sec`**: Time spent in baseline-internal evaluation/test execution.
- **`baseline_orchestration_sec`**: Baseline-internal orchestration that is neither LLM time nor internal evaluation.
- **`llm_total_sec`**: Pure LLM time across stage A, stage B, and compaction.
- **`overhead_sec`**: Residual framework overhead after subtracting pytest, pure LLM time, and baseline-internal eval/orchestration.

#### Token Metrics (from attempt_metadata):
- **`tokens_prompt`** (eval_recording.py:553): Prompt tokens used in LLM calls
- **`tokens_completion`** (eval_recording.py:554): Completion tokens used in LLM calls
- **`tokens_total`** (eval_recording.py:555): Total tokens used across all calls
- **`llm_call_count`** (eval_recording.py:552): Number of LLM API calls made

#### Adaptive Policy Tracking (adaptive_online_local_* / adaptive_budget_local_*):
- **`adaptive_online_local_llm_call_count`** (line 4836): LLM calls in this policy decision
- **`adaptive_online_local_tokens_prompt`** (line 4837): Prompt tokens for this decision
- **`adaptive_online_local_tokens_completion`** (line 4838): Completion tokens for this decision
- **`adaptive_online_local_tokens_total`** (line 4839): Total tokens for this decision
- **`adaptive_online_local_wall_sec`** (line 4840): Wall time for this decision
- Similarly for `adaptive_budget_*` variants (lines 5542-5546)

### Result Recording Infrastructure

#### File: `evals/eval_recording.py`
- **`_SynchronizedJsonlWriter`** (line 137): Thread-safe JSONL writer with lazy file open
- **`_RunResultsLock`** (line 48): File-based cross-process advisory lock for results
- **`_build_base_record()`** (line 501): Constructs skeleton JSONL record with all metadata
- **`_empty_attempt_metadata()`** (line 535): Initial metadata dict with 300+ fields
- **`_materialize_attempt_metadata()`** (line 803): Fills in attempt metadata post-execution

#### Results Output
- **Location**: `evals/results/<run_id>.jsonl` (line 1360)
- **Schema Version**: `EVAL_RECORD_SCHEMA_VERSION = 6` (eval_types.py:72)
- **Per-Record Structure**:
  - `schema_version`, `run_id`, `case_id`, `condition`, `attempt`
  - `run_metadata`, `attempt_metadata`, `model_metadata`
  - Success flag, timing fields, patch metadata
  - Adaptive policy fields (online and budget)

### How Tool Calls Are Made

1. **Patcher Selection** (run_eval.py:280-600):
   - Pluggable patcher architecture via `Patcher` base class
   - Patchers: `openai_compat_patcher`, `command_patcher`, `heuristic_patcher`, `null_patcher`

2. **Stage A (Context Request)**:
   - Called via `request_context_fn()` at line 7405-7416
   - Invokes LLM to request needed context (files, frames, search results, snapshots)
   - Returns `ContextRequestResult` with `request_json` payload and metadata
   - Timing tracked: `request_context_duration_sec` in metadata

3. **Patch Proposal**:
   - Called via `_propose_patch_with_snapshot_context()` at line 8862
   - Invokes LLM with full evidence/snapshot context
   - Returns `PatchResult` with diff and metadata including token usage
   - Timing tracked: `propose_duration_sec` (line 8969, 8871)

4. **Test Execution**:
   - Via pytest CLI execution in `run_pytest()` (line 10535)
   - Timing tracked in `before_duration_sec` and `after_duration_sec`

### Timeline of Execution

```
attempt_start (time.perf_counter() at line 8741, 9453)
  ↓
Before pytest: measure via run_pytest() → before_duration_sec
  ↓
Stage A (if enabled): request_context_fn() → request_context_duration_sec
  ↓
Patch proposal: _propose_patch_with_snapshot_context() → propose_duration_sec (patch_duration_sec)
  ↓
Patch application: git apply
  ↓
After pytest: run_pytest() → after_duration_sec
  ↓
attempt_wall_clock_sec = time.perf_counter() - attempt_start (line 3691-3692)
```

---

## 2. PATCHER BACKENDS (`evals/patchers/`)

### Base Class: `Patcher` (base.py)

```python
@dataclass(frozen=True)
class PatchResult:
    diff: str | None
    meta: PatcherMeta | None = None  # Contains timing and token metrics
```

### Available Patchers

| Patcher | File | Description | Timing? | Token Tracking? |
|---------|------|-------------|---------|-----------------|
| **OpenAICompat** | `openai_compat_patcher.py` | LLM-based via OpenAI-compatible API | ✅ Yes | ✅ Yes |
| **Command** | `command_patcher.py` | Executes local command/CLI | ✅ Yes (timeout) | ❌ No |
| **Heuristic** | `heuristic_patcher.py` | Hard-coded demo patcher | ❌ No | ❌ No |
| **Null** | `null_patcher.py` | Returns None (no-op) | ❌ No | ❌ No |

### OpenAICompatPatcher Implementation

#### File: `evals/patchers/openai_compat_patcher.py` (3963 lines)

**Key Methods:**
- `propose_patch()` (line 1043): Main synchronous entry point
- `apropose_patch()` (line 1171): Async entry point
- `request_context()` (line 629): Request context from helper
- `arequest_context()` (line 766): Async context request

**Timing Tracking:**
- `request_context_duration_sec` (line 737, 874): Total duration of context request
- `request_context_attempts`: Number of retry attempts
- `request_context_retry_count`: Retries after initial failure
- `request_context_transport_retry_count`: HTTP-level retries
- `request_context_strict_retry_count`: JSON parsing retries
- `max_propose_sec`: Deadline for patch proposal (line 1052-1054)
- `propose_duration_sec`: Duration of patch proposal call

**Token Tracking:**

Two levels of tracking:

1. **Per-Call Usage** (line 333-345: `_usage_counts_from_payload()`):
   - `prompt_tokens`: From response usage object
   - `completion_tokens`: From response usage object
   - `total_tokens`: Calculated or from response
   - Falls back to `input_tokens`/`output_tokens` if needed

2. **Accumulated Tracking** (line 375-406: `_accumulate_llm_totals()`, `_meta_with_llm_totals()`):
   - Tracks `llm_call_count`: Number of API calls
   - Accumulates across retries and fallback modes
   - Supports diff mode (default) and edits mode (fallback)
   - Merges metadata when fallback occurs

**Metadata Structure** (`payload_types.py:33-62`):

```python
class PatcherMeta(TypedDict, total=False):
    # Execution
    error: str
    returncode: int
    status: int
    
    # Timing
    timeout_sec: float
    max_propose_sec: float
    request_context_duration_sec: float
    
    # Token Usage
    llm_call_count: int
    usage: UsagePayload  # {prompt_tokens, completion_tokens, total_tokens}
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    
    # LLM Config
    base_url: str
    model: str
    response_mode: str
    temperature: float
    max_tokens: int
    api_compat_seed_requested: bool
    api_compat_seed_acknowledged: bool
```

### CommandPatcher Implementation

File: `evals/patchers/command_patcher.py`

- Runs external command with prompt as stdin
- **Timeout Tracking** (line 24, 88-100):
  - `timeout_sec`: Configured timeout
  - `timed_out`: Boolean flag if timeout occurred
- **No Token Tracking**: Command patchers don't measure tokens
- **Metadata** (line 123-133):
  - `returncode`: Command exit code
  - `stdout`/`stderr`: Command output

---

## 3. RESULTS FORMAT

### Output JSONL Schema

**File**: `evals/results/<run_id>.jsonl`
**Schema Version**: 6 (eval_types.py:72, analyze_results.py:29)

#### Record Structure (high-level):
```json
{
  "schema_version": 6,
  "run_id": "uuid",
  "case_id": "string",
  "condition": "base_repair|cot_repair|layered_adaptive_gated|layered_adaptive_llm|ldb|intervenor|tracecoder",
  "attempt": 1,
  "success": true/false,
  "category": "string",
  "difficulty": "easy|medium|hard",
  
  "run_metadata": { ... },
  "attempt_metadata": { ... },
  "model_metadata": { ... },
  
  "before_duration_sec": 1.23,
  "after_duration_sec": 2.45,
  "attempt_wall_clock_sec": 10.5,
  "cumulative_runtime_sec": 45.0,
  
  "patch_meta": { ... },
  "patch_touches_tests": true,
  "patch_file_count": 2,
  
  "note": "string"
}
```

### Nested Metadata Objects

#### run_metadata (eval_recording.py:166):
```python
{
  "run_id": str,
  "started_at_utc": str,
  "deterministic": bool,
  "seed": int | None,
  "python_version": str,
  "platform": str,
  "llmdebug_version": str,
  "patcher_name": str,
  "patcher_config": { ... },  # Extensive config dict
  "patcher_config_fingerprint": str,
}
```

#### attempt_metadata (eval_recording.py:535-800):
**300+ fields** including:
- `llm_call_count`: Number of LLM API calls
- `tokens_prompt`, `tokens_completion`, `tokens_total`: Token usage
- `stage_a_prompt_chars`, `stage_a_requests_count`: Context request metrics
- `request_context_duration_sec`: Helper call duration
- `adaptive_online_local_*`: Local token/timing for online policy
- `adaptive_budget_local_*`: Local token/timing for budget policy
- `patch_verifier_*`: Patch verification results
- `context_compaction_*`: Context compression metrics

#### model_metadata (eval_recording.py:290+):
```python
{
  "backend": "openai|command|heuristic|null",
  "name": str,
  # For OpenAI:
  "base_url": str,
  "model": str,
  "temperature": float,
  "max_tokens": int,
  "response_mode": "diff|edits",
  "api_compat_seed_requested": bool | None,
  "api_compat_seed_acknowledged": bool | None,
}
```

### Example with Real Metrics

Based on code analysis, a typical record would include:
```json
{
  "attempt_metadata": {
    "llm_call_count": 2,
    "tokens_prompt": 3500,
    "tokens_completion": 250,
    "tokens_total": 3750,
    "stage_a_prompt_chars": 4200,
    "stage_a_requests_count": 3,
    "request_context_duration_sec": 2.1,
    "adaptive_online_local_llm_call_count": 1,
    "adaptive_online_local_tokens_total": 1200,
    "adaptive_online_local_wall_sec": 1.8,
    "adaptive_online_reward": 0.65
  },
  "before_duration_sec": 1.2,
  "after_duration_sec": 1.5,
  "attempt_wall_clock_sec": 8.3,
  "patch_meta": {
    "llm_call_count": 1,
    "tokens_prompt": 2300,
    "tokens_completion": 250,
    "tokens_total": 2550,
    "propose_duration_sec": 3.2,
    "usage": {
      "prompt_tokens": 2300,
      "completion_tokens": 250,
      "total_tokens": 2550
    }
  }
}
```

### Validation & Analysis

**File**: `evals/analyze_results.py`

**Required Fields Validation** (lines 40-89):
```python
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version", "run_id", "case_id", "condition", "attempt", "success",
    "category", "difficulty", "snapshot_dependency", "run_metadata",
    "attempt_metadata", "model_metadata"
)

REQUIRED_ATTEMPT_METADATA_FIELDS = (
    "attempt", "max_attempts", "attempt_ordinal_label", "llm_call_count",
    "tokens_prompt", "tokens_completion", "tokens_total",
    "retry_history_budget_chars", "retry_history_budget_applied",
    "prompt_xxh3_128", "prompt_path", "evidence_xxh3_128", "evidence_path"
)
```

**Token Extraction** (analyze_results.py:line ~800):
```python
def _token_usage_breakdown(ordered: list) -> tuple[float, float, float]:
    prompt_total = completion_total = total_total = 0.0
    for attempt_meta in [r.get("attempt_metadata") or {} for r in ordered]:
        prompt_total += float(attempt_meta.get("tokens_prompt", 0))
        completion_total += float(attempt_meta.get("tokens_completion", 0))
        total_total += float(attempt_meta.get("tokens_total", 0))
    return prompt_total, completion_total, total_total
```

**Summary Stats Computed** (analyze_results.py):
- Per-condition success rates
- Token usage (mean, total)
- Runtime (mean, total)
- Local policy accounting (tokens, wall time, rewards)

---

## 4. ADAPTIVE POLICIES

### Online Policy (`evals/adaptive_online_policy.py`)

**Config**: `OnlinePolicyConfig` (line 67)
- `mode`: "off|shadow|on"
- `algorithm`: "linucb"
- `alpha`, `epsilon`, `l2`: LinUCB hyperparameters
- `warmup_decisions`: Number of uniform sampling decisions
- `state_path`, `events_path`: Persistent storage

**Decision Output**: `PolicyDecision` (line 135)
```python
@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    created_at_utc: str
    selected_action: str
    applied_action: str
    default_action: str
    propensity: float
    decision_strategy: str
    scores: dict[str, float]  # Action → score mapping
    feature_values: dict[str, float]
```

**Reward Calculation** (run_eval.py:4430-4500):
```python
def _adaptive_online_reward_signal(
    success: bool,
    progress_score: float,
    local_tokens_total: int,
    local_wall_sec: float,
    lambda_tokens: float = 0.08,
    mu_latency: float = 0.05,
    token_scale: float = 1000.0,
    latency_scale: float = 10.0,
) -> float:
    # reward = success - lambda*(tokens/token_scale) - mu*(latency/latency_scale)
    base = float(success)
    token_penalty = (lambda_tokens * float(local_tokens_total)) / max(1.0, token_scale)
    latency_penalty = (mu_latency * float(local_wall_sec)) / max(1.0, latency_scale)
    return base - token_penalty - latency_penalty
```

**Tracked in attempt_metadata** (eval_recording.py:704-744):
- `adaptive_online_policy_mode`: "off|shadow|on"
- `adaptive_online_decision_id`: UUID for the decision
- `adaptive_online_action_selected`: Chosen action
- `adaptive_online_action_applied`: Actually executed action
- `adaptive_online_propensity`: Propensity score (IPS/MAGIC weighting)
- `adaptive_online_feature_vector_raw`: Raw feature dict
- `adaptive_online_feature_vector`: Scaled feature dict
- `adaptive_online_reward`: Computed reward signal
- `adaptive_online_reward_success_hint`: 1.0 if success, 0.0 else
- `adaptive_online_reward_progress_score`: Progress component
- `adaptive_online_reward_token_penalty`: Token cost
- `adaptive_online_reward_latency_penalty`: Latency cost
- `adaptive_online_local_*`: Local accounting (lines 737-741)

### Budget Policy (`evals/adaptive_budget_online_policy.py`)

**Config**: `BudgetPolicyConfig` (line 50)
- Similar structure to online policy
- Controls request/token budgets via tier multipliers

**Decision Output**: `BudgetPolicyDecision` (line 118)
- Similar structure to `PolicyDecision`

**Applied Controls** (run_eval.py:5459-5546):
```python
"adaptive_budget_applied_control_values": {
    "helper_allowance": int,
    "stage_a_requests": int,
    "stage_a_chars": int,
    "retry_history_budget": int,
    "compaction_allowance": int,
    "response_mode": str,
    "openai_max_tokens": int,
    "propose_budget": float,
}
"adaptive_budget_local_*": {
    "llm_call_count": int,
    "tokens_prompt": int,
    "tokens_completion": int,
    "tokens_total": int,
    "wall_sec": float,
}
```

**Reward Calculation** (run_eval.py:4650-4720):
```python
def _adaptive_budget_reward_signal(
    success: bool,
    progress_score: float,
    request_budget_exceeded: bool,
    local_tokens_total: int,
    local_wall_sec: float,
    rho_budget_exceeded: float = 0.5,
    # ... other params
) -> float:
    base = float(success)
    budget_penalty = (
        float(rho_budget_exceeded) * float(request_budget_exceeded)
    )
    token_penalty = (lambda_tokens * float(local_tokens_total)) / max(1.0, token_scale)
    latency_penalty = (mu_latency * float(local_wall_sec)) / max(1.0, latency_scale)
    return base - budget_penalty - token_penalty - latency_penalty
```

**Do they track timing?** ✅ **YES** via `local_wall_sec`

---

## 5. MCP TOOLS & LLMDEBUG INTEGRATION

### How LLM Tools Are Invoked

#### Direct Function Calls (No MCP Protocol)

The eval framework invokes llmdebug tools via **direct Python imports**, not MCP:

**Evidence**: run_eval.py lines 11098-11109
```python
from llmdebug.output import get_latest_snapshot
from llmdebug.rca_state import build_failure_signature
from llmdebug.snapshot_diff import diff_snapshots, structured_diff_summary
```

**Tool Invocation Points**:

1. **Snapshot Extraction** (run_eval.py:~11100):
   ```python
   snapshot = get_latest_snapshot(workdir)
   ```

2. **Failure Signature** (run_eval.py:~11109):
   ```python
   signature = build_failure_signature(snapshot)
   ```

3. **Snapshot Diffing** (run_eval.py:~11145):
   ```python
   diff = diff_snapshots(before_snapshot, after_snapshot)
   summary = structured_diff_summary(...)
   ```

#### Stage A Tool Requests (LLM-Directed)

The LLM doesn't directly invoke tools via MCP. Instead:

1. **LLM requests context** via `request_context()` → returns `ContextRequestPayload`
2. **Payload contains request specifications** (payload_types.py:21-30):
   ```python
   {
       "diagnosis": str,
       "missing_fact": str | None,
       "need_snapshot": bool,
       "need_compact_context": bool,
       "requests": [
           {"type": "file|frame|search|index", "path": str, "pattern": str, ...}
       ]
   }
   ```
3. **Executor interprets payload** and calls appropriate tools:
   - File reader (`fsutil.read_file()`)
   - Frame extractor (snapshot frame access)
   - Search/grep tools
   - File indexing

**No native MCP integration in current eval framework** - tools are invoked as Python functions internally.

---

## 6. EXISTING TIMING/TOKEN TRACKING

### Comprehensive Audit

| Metric | Location | Type | Granularity | Updated |
|--------|----------|------|-------------|---------|
| **Core Timing** | | | | |
| `before_duration_sec` | run_eval.py:3671, eval_recording.py | pytest before | Per attempt | ✅ Always |
| `after_duration_sec` | run_eval.py:3678 | pytest after | Per attempt | ✅ If after run |
| `attempt_wall_clock_sec` | run_eval.py:3673 | Wall time | Per attempt | ✅ Always |
| `propose_duration_sec` | run_eval.py:8871, 8969 | LLM patch | Per patch call | ✅ When patched |
| `request_context_duration_sec` | openai_compat_patcher.py:737, 874 | LLM context | Per context call | ✅ When called |
| | | | | |
| **Token Usage** | | | | |
| `tokens_prompt` | eval_recording.py:553 | Tokens | Cumulative | ✅ Always |
| `tokens_completion` | eval_recording.py:554 | Tokens | Cumulative | ✅ Always |
| `tokens_total` | eval_recording.py:555 | Tokens | Cumulative | ✅ Always |
| `llm_call_count` | eval_recording.py:552 | Count | Cumulative | ✅ Always |
| `usage` (dict) | openai_compat_patcher.py:348-360 | Usage obj | Per LLM call | ✅ On response |
| | | | | |
| **Adaptive Policy** | | | | |
| `adaptive_online_local_*` | run_eval.py:4836-4840 | Local | Per decision | ✅ When active |
| `adaptive_budget_local_*` | run_eval.py:5542-5546 | Local | Per decision | ✅ When active |
| `adaptive_online_reward` | run_eval.py:4840+ | Float | Per attempt | ✅ If active |
| `adaptive_budget_reward` | run_eval.py:5546+ | Float | Per attempt | ✅ If active |
| `adaptive_online_reward_*_penalty` | eval_recording.py:730-731 | Breakdown | Per attempt | ✅ If active |
| | | | | |
| **Context Operations** | | | | |
| `stage_a_prompt_chars` | eval_recording.py:546 | Chars | Per attempt | ✅ Always |
| `stage_a_requests_count` | eval_recording.py:547 | Count | Per attempt | ✅ Always |
| `stage_a_chars_returned` | eval_recording.py:548 | Chars | Per attempt | ✅ Always |
| `context_compaction_*` | eval_recording.py:661-668 | Various | Per attempt | ✅ If compacted |
| | | | | |
| **Retry Tracking** | | | | |
| `request_context_retry_count` | openai_compat_patcher.py:739 | Count | Per context | ✅ Always |
| `request_context_transport_retry_count` | openai_compat_patcher.py:740 | Count | Per context | ✅ Always |
| `request_context_strict_retry_count` | openai_compat_patcher.py:741 | Count | Per context | ✅ Always |

### Implementation Details

**Where Timing is Measured**:

1. **Before pytest** (run_eval.py:6623):
   ```python
   "before_duration_sec": current.duration_sec,
   ```

2. **Patch proposal** (run_eval.py:8861-8871):
   ```python
   patch_start = time.perf_counter()
   patch = _propose_patch_with_snapshot_context(...)
   patch_duration_sec = time.perf_counter() - patch_start
   ```

3. **Context request** (openai_compat_patcher.py:656+, 793+):
   ```python
   start = time.time()
   result = _request_context_via_json(...)
   duration_sec = time.time() - start
   result_meta["request_context_duration_sec"] = duration_sec
   ```

4. **After pytest** (run_eval.py:8770+):
   ```python
   attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
   ```

5. **Adaptive policy wall time** (run_eval.py:7357, 7383, 7544):
   ```python
   helper_local_started_at = time.perf_counter() if run_stage_a else None
   # ... later ...
   helper_local_wall_sec = time.perf_counter() - helper_local_started_at
   ```

**Where Tokens Are Extracted**:

1. **From OpenAI response** (openai_compat_patcher.py:333-345):
   ```python
   def _usage_counts_from_payload(usage):
       prompt_tokens = usage.get("prompt_tokens", 0)
       completion_tokens = usage.get("completion_tokens", 0)
       total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
       return prompt_tokens, completion_tokens, total_tokens
   ```

2. **Accumulated across calls** (openai_compat_patcher.py:375-406):
   ```python
   def _accumulate_llm_totals(totals, meta, default_calls):
       calls, prompt, completion, total = totals
       usage_counts = _usage_counts_from_payload(meta.get("usage"))
       return (calls + llm_calls, prompt + usage_counts[0], ...)
   ```

3. **Stored in attempt_metadata** (run_eval.py:3247-3259):
   ```python
   def _sync_tokens(attempt_meta, llm_call_count, tokens_prompt, tokens_completion, tokens_total):
       attempt_meta["llm_call_count"] = _nonnegative_int(llm_call_count)
       attempt_meta["tokens_prompt"] = _nonnegative_int(tokens_prompt)
       attempt_meta["tokens_completion"] = _nonnegative_int(tokens_completion)
       attempt_meta["tokens_total"] = _nonnegative_int(tokens_total)
   ```

---

## Summary: Where to Add Instrumentation

### High-Priority Additions:

1. **Patcher-Level Latency Breakdown**:
   - Currently: `propose_duration_sec` is total time
   - Add: `{request_body_prepare, api_call, response_parse}_sec` breakdown
   - Location: `openai_compat_patcher.py` around `_propose_patch_via_diff()`

2. **Helper Cost Attribution**:
   - Currently: `adaptive_online_local_*` tracks helpers
   - Issue: Doesn't distinguish request_context vs. other operations
   - Add: Split metrics for `helper_stage_a_*` vs. `helper_snapshot_*` vs. `helper_compact_*`
   - Location: `run_eval.py` lines 7357-7544 (helper invocation block)

3. **Test Execution Details**:
   - Currently: `before_duration_sec` and `after_duration_sec` only
   - Add: pytest collection time, setup time, teardown time, per-test time
   - Location: `run_eval.py` around `run_pytest()` call (line 10535)

4. **Policy Decision Overhead**:
   - Currently: No timing for policy decision computation
   - Add: `{online_policy_compute, budget_policy_compute}_sec`
   - Location: `adaptive_online_policy.py` and `adaptive_budget_online_policy.py`

5. **Context Compaction Latency**:
   - Currently: Only boolean `context_compaction_invoked`
   - Add: `context_compaction_duration_sec`
   - Location: `run_eval.py` around context compaction invocation (~line 7700+)

### File Path Summary for Instrumentation:

```
/Users/nicolas/Documents/PhD/llmdebug/
├── evals/
│   ├── run_eval.py                          # Main orchestrator (11384 lines)
│   │   ├── Lines 1667-2150: Run timing setup & progress
│   │   ├── Lines 7357-7544: Helper (Stage A) invocation & wall time
│   │   ├── Lines 8800-9500: Patch proposal & timing
│   │   ├── Lines 6600-6800: Test execution and timing
│   │   └── Lines 4400-5700: Reward computation (tokens + latency)
│   │
│   ├── eval_recording.py                    # Metadata assembly
│   │   ├── Lines 500-532: Record building
│   │   ├── Lines 535-800: Attempt metadata template
│   │   └── Lines 803-926: Metadata materialization
│   │
│   ├── eval_types.py                        # Data structures
│   │   ├── Lines 447-457: AdaptivePolicyLocalAccounting
│   │   └── Lines 72: EVAL_RECORD_SCHEMA_VERSION
│   │
│   ├── patchers/
│   │   ├── base.py                          # Patcher interface (lines 15-97)
│   │   ├── openai_compat_patcher.py          # OpenAI implementation (3963 lines)
│   │   │   ├── Lines 59-180: HTTP client setup
│   │   │   ├── Lines 333-406: Usage extraction & accumulation
│   │   │   ├── Lines 629-765: request_context() method
│   │   │   ├── Lines 766-900: arequest_context() method
│   │   │   └── Lines 1043-1240: propose_patch() method
│   │   ├── command_patcher.py               # Command execution
│   │   │   └── Lines 23-133: propose_patch() with timeout
│   │   └── payload_types.py                 # Schema (lines 1-74)
│   │
│   ├── adaptive_online_policy.py             # Online contextual-bandit policy
│   │   ├── Lines 66-132: OnlinePolicyConfig
│   │   └── Lines 134-150: PolicyDecision
│   │
│   ├── adaptive_budget_online_policy.py      # Budget escalation policy
│   │   ├── Lines 50-115: BudgetPolicyConfig
│   │   └── Lines 118-135: BudgetPolicyDecision
│   │
│   ├── analyze_results.py                    # Results validation & summary
│   │   ├── Lines 40-89: Schema validation
│   │   └── Lines ~800: Token breakdown
│   │
│   └── eval_summary.py                       # Per-run summary
│       └── Lines 81-150: Summary output
```
