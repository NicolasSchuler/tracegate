"""Names and shapes shared by the code that writes eval records and the code that reads them.

Attempt notes and the record keys that several modules spell out live here, so a
rename can't leave one reader behind (ORC-23). Older keys are still written as bare
strings; move them here when you touch them. :class:`AttemptRecord` types the keys the
record builders write (X-03).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, get_args

# The terminal ``note`` of an attempt record.
AttemptNote = Literal[
    "success",
    "tests_still_failing",
    "patch_apply_failed",
    "syntax_preflight_failed",
    "patch_verifier_rejected",
    "no_patch_proposed",
    "forced_noop_patch",
    "truncated_response",
    "patcher_request_failed",
    "infra_transport_failure",
    "retry_gate_no_new_evidence",
    "no_patch_needed",
    "unit_runtime_error",
    "patch_cycle_incomplete",
]
ATTEMPT_NOTES: frozenset[str] = frozenset(get_args(AttemptNote))

NOTE_SUCCESS: AttemptNote = "success"
NOTE_TESTS_STILL_FAILING: AttemptNote = "tests_still_failing"
NOTE_PATCH_APPLY_FAILED: AttemptNote = "patch_apply_failed"
NOTE_SYNTAX_PREFLIGHT_FAILED: AttemptNote = "syntax_preflight_failed"
NOTE_PATCH_VERIFIER_REJECTED: AttemptNote = "patch_verifier_rejected"
NOTE_NO_PATCH_PROPOSED: AttemptNote = "no_patch_proposed"
NOTE_FORCED_NOOP_PATCH: AttemptNote = "forced_noop_patch"
NOTE_TRUNCATED_RESPONSE: AttemptNote = "truncated_response"
NOTE_PATCHER_REQUEST_FAILED: AttemptNote = "patcher_request_failed"
NOTE_INFRA_TRANSPORT_FAILURE: AttemptNote = "infra_transport_failure"
NOTE_RETRY_GATE_NO_NEW_EVIDENCE: AttemptNote = "retry_gate_no_new_evidence"
NOTE_NO_PATCH_NEEDED: AttemptNote = "no_patch_needed"
NOTE_UNIT_RUNTIME_ERROR: AttemptNote = "unit_runtime_error"
NOTE_PATCH_CYCLE_INCOMPLETE: AttemptNote = "patch_cycle_incomplete"

# Keys read outside the module that writes them.
KEY_INFRA_RESUME_ELIGIBLE = "infra_resume_eligible"
KEY_ADAPTIVE_PROGRESS_STOP_NOTE = "adaptive_progress_stop_note"
KEY_PROMPT_CHARS_TOTAL = "prompt_chars_total"


class TimeoutFields(TypedDict, total=False):
    """Present on a record only when its test run timed out."""

    timed_out: bool
    timeout_sec: int | None


class AttemptRecord(TimeoutFields, total=False):
    """An attempt (or infra-event) row of ``results.jsonl``, as the builders write it (X-03).

    The attempt flow adds condition-specific keys after the builders return, so
    readers should still use ``.get`` for anything not listed here.
    """

    # The skeleton from eval_recording._build_base_record.
    schema_version: int
    run_id: str
    case_id: str
    category: str
    difficulty: str
    snapshot_dependency: str
    expected_exception: str
    tags: list[str]
    bug_source: str | None
    contamination_risk: str | None
    condition: str
    attempt: int
    patcher: str
    context_protocol: str
    run_metadata: dict[str, Any]
    attempt_metadata: dict[str, Any]
    model_metadata: dict[str, Any]
    # The outcome, from attempt_records.
    record_type: str
    success: bool
    note: AttemptNote
    before_returncode: int
    before_duration_sec: float
    after_returncode: int
    after_duration_sec: float
    cumulative_runtime_sec: float
    attempt_wall_clock_sec: float
    patch_meta: dict[str, Any]
    patch_touches_tests: bool
    patch_test_file_count: int
    patch_file_count: int
