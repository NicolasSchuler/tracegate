"""Names shared by the code that writes eval records and the code that reads them (ORC-23).

Attempt notes and the record keys that several modules spell out live here, so a
rename can't leave one reader behind. Older keys are still written as bare strings;
move them here when you touch them.
"""

from __future__ import annotations

from typing import Literal, get_args

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

# Keys read outside the module that writes them.
KEY_INFRA_RESUME_ELIGIBLE = "infra_resume_eligible"
KEY_ADAPTIVE_PROGRESS_STOP_NOTE = "adaptive_progress_stop_note"
KEY_PROMPT_CHARS_TOTAL = "prompt_chars_total"
