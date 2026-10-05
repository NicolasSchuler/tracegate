"""The shared attempt-note names stay in step with each other (ORC-23)."""

from __future__ import annotations

from typing import get_args

from evals import record_keys
from evals.eval_patch_constants import TRUNCATED_RESPONSE_ERROR


def test_every_note_constant_is_an_attempt_note_and_every_note_has_a_constant() -> None:
    constants = {value for name, value in vars(record_keys).items() if name.startswith("NOTE_")}
    assert constants == set(get_args(record_keys.AttemptNote))
    assert constants == record_keys.ATTEMPT_NOTES


def test_truncated_response_note_matches_the_patcher_error_code() -> None:
    assert record_keys.NOTE_TRUNCATED_RESPONSE == TRUNCATED_RESPONSE_ERROR
