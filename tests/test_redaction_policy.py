"""Tests for redaction policy resolution."""

from __future__ import annotations

import re

import pytest

from llmdebug.redaction_policy import (
    PII_PATTERNS,
    normalize_redaction_profile,
    resolve_redaction_policy,
)


def test_normalize_redaction_profile() -> None:
    assert normalize_redaction_profile(None) is None
    assert normalize_redaction_profile("") is None
    assert normalize_redaction_profile(" DEV ") == "dev"
    assert normalize_redaction_profile("ci") == "ci"
    assert normalize_redaction_profile("prod") == "prod"


def test_normalize_redaction_profile_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid redaction profile"):
        normalize_redaction_profile("staging")


def test_profile_defaults_ci() -> None:
    policy = resolve_redaction_policy(redaction_profile="ci")
    assert policy.profile == "ci"
    assert tuple(policy.patterns) == PII_PATTERNS
    assert policy.redact_keys is False
    assert policy.redact_traceback is False
    assert policy.redact_exception_strings is True


def test_profile_defaults_prod() -> None:
    policy = resolve_redaction_policy(redaction_profile="prod")
    assert policy.profile == "prod"
    assert tuple(policy.patterns) == PII_PATTERNS
    assert policy.redact_traceback is True
    assert policy.redact_exception_strings is True


def test_explicit_patterns_override_profile() -> None:
    custom = (r"token=\w+", re.compile(r"secret=\w+"))
    policy = resolve_redaction_policy(
        redaction_profile="prod",
        redact=custom,
    )
    assert policy.profile == "prod"
    assert policy.patterns == custom


def test_explicit_booleans_override_profile() -> None:
    policy = resolve_redaction_policy(
        redaction_profile="prod",
        redact_traceback=False,
        redact_exception_strings=False,
        redact_keys=True,
    )
    assert policy.redact_traceback is False
    assert policy.redact_exception_strings is False
    assert policy.redact_keys is True


def test_legacy_fallback_applies_without_profile_or_explicits() -> None:
    policy = resolve_redaction_policy(
        legacy_redact=(r"password",),
        legacy_redact_traceback=False,
        legacy_redact_exception_strings=False,
    )
    assert policy.profile is None
    assert policy.patterns == (r"password",)
    assert policy.redact_traceback is False
    assert policy.redact_exception_strings is False


def test_hard_defaults_when_nothing_provided() -> None:
    policy = resolve_redaction_policy()
    assert policy.profile is None
    assert policy.patterns == ()
    assert policy.redact_keys is False
    assert policy.redact_traceback is False
    assert policy.redact_exception_strings is False


@pytest.mark.parametrize("field", ["redact_keys", "redact_traceback", "redact_exception_strings"])
def test_explicit_boolean_overrides_reject_non_bool_values(field: str) -> None:
    kwargs = {field: "yes"}
    with pytest.raises(ValueError, match="Expected bool\\|None"):
        resolve_redaction_policy(**kwargs)  # type: ignore[arg-type]
