"""Redaction profile resolution for all snapshot capture surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from re import Pattern
from typing import Any, Literal, cast

RedactionProfile = Literal["dev", "ci", "prod"]

# CAP-03: names that mark their value as a secret, as a regex alternation. Bare "key" and
# "token" are left out: they are too common in ordinary code (dict keys, tokenizers).
SENSITIVE_NAMES = (
    r"password|passwd|passphrase|secret|credential|private[_-]?key"
    r"|(?:api|access)[_-]?key|(?:access|auth|refresh|bearer|session|csrf|api)[_-]?token"
    r"|authorization|cookie"
)

# Secret-name suffix after one of SENSITIVE_NAMES, e.g. the "-key" of "secret-key". Hyphens
# are included so text matching accepts the same names as `is_sensitive_name`. Bounded
# because the suffix class overlaps the names themselves.
_NAME_SUFFIX = r"[\w-]{0,128}"

# A Python/JSON string literal with escapes and an optional b/r/u prefix; the closing quote is
# optional so a value cut by truncation is still covered to its end.
_QUOTED_VALUE = r"""[bru]{0,2}(?:'(?:[^'\\]|\\.)*'?|"(?:[^"\\]|\\.)*"?)"""

# One level of bracketed container, so a composite value is consumed whole instead of up to
# its first comma: {'password': ['alpha', 'beta']} must not keep "beta". Each inner class
# excludes its own brackets and is bounded, so matching stays linear.
_CONTAINER_VALUE = r"\[[^\[\]\r\n]{0,1024}\]|\{[^{}\r\n]{0,1024}\}|\([^()\r\n]{0,1024}\)"

# A bare value, up to the delimiter that ends it in a repr or JSON rendering.
_BARE_VALUE = r"[^\s,;}\])]+"

_SECRET_VALUE = f"(?:{_QUOTED_VALUE}|{_CONTAINER_VALUE}|{_BARE_VALUE})"

# Header fields whose value is whitespace- or ";"-separated, so one token is not the whole
# secret: "Authorization: Basic dXNlcjpwYXNz", "Cookie: a=1; b=2". Their value is consumed up
# to the structural delimiters of a surrounding repr/JSON object, which stay readable.
_AUTH_FIELD_NAMES = r"(?:proxy-)?authorization|(?:set-)?cookie"

# Built-in PII-focused patterns used by safer profiles and production paths. Every repeat
# that can follow a match start is bounded, so redacting long text (captured stdout, big
# tracebacks) stays linear. The order matters: `redact_text` applies them in sequence, and a
# pattern that consumes a longer value must run before one that would stop inside it.
PII_PATTERNS: tuple[str, ...] = (
    # Authorization/Cookie fields first: the generic assignment pattern below would stop at
    # the space in "Basic dXNlcjpwYXNz" and leave the credential behind.
    r"(?i)\b(?:" + _AUTH_FIELD_NAMES + r")\b['\"]?\s*[:=]\s*[^\r\n,}\])]{0,4096}",
    # A secret-named name/value assignment, in a log line, an exception message or the repr
    # of a nested dict: password="a b", {'secret-key': 'hunter2'}, api_key: ['a', 'b'].
    # This one pattern replaces the former `password\s*[:=]\s*\S+` pair, which matched first
    # and cut quoted multi-word values off after their first word.
    r"(?i)(?:" + SENSITIVE_NAMES + r")" + _NAME_SUFFIX + r"['\"]?\s*[:=]\s*" + _SECRET_VALUE,
    # A quoted secret name followed by a quoted value: the repr of a header or form pair,
    # ['Authorization', 'Basic dXNlcjpwYXNz'], which has no ":"/"=" separator.
    r"(?i)['\"](?:" + SENSITIVE_NAMES + r")" + _NAME_SUFFIX + r"['\"]\s*,\s*" + _SECRET_VALUE,
    r"(?i)bearer\s+\S+",
    # URL userinfo, up to the last "@" before the host (passwords may contain "@"), with or
    # without a password: postgres://u:p@ss@db -> //[REDACTED]@db, https://TOKEN@host ->
    # //[REDACTED]@host. An authority cut by truncation ("...[TRUNC]") is redacted whole,
    # since it may be userinfo, and so is an authority longer than any real host (fail
    # closed). Runs before the email pattern, which would otherwise take "TOKEN@github.com"
    # together with the host.
    # Branch order is the fail-closed order: a truncated or over-long authority must win over
    # ordinary userinfo matching, which would otherwise stop at an "@" inside the password
    # ("postgres://admin:p@ss...[TRUNC]") or before the real host of an over-long authority.
    r"(?<=://)[^\s/]{513,}"
    r"|(?<=://)[^\s/]{1,512}(?=\.\.\.\[TRUNC\])"
    r"|(?<=://)[^\s/]{1,512}(?=@)",
    # RFC 5321 caps the local part at 64 characters; the bound also keeps this linear.
    r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,63}\b",
    r"(?i)(?:sk|pk|token|key)[_-][a-zA-Z0-9_-]{20,}",
)

_VALID_PROFILES = {"dev", "ci", "prod"}


@dataclass(frozen=True)
class RedactionPolicy:
    """Resolved redaction behavior for a capture call/site."""

    profile: RedactionProfile | None
    patterns: tuple[str | Pattern[str], ...]
    redact_keys: bool
    redact_traceback: bool
    redact_exception_strings: bool


_PROFILE_DEFAULTS: dict[RedactionProfile, RedactionPolicy] = {
    "dev": RedactionPolicy(
        profile="dev",
        patterns=(),
        redact_keys=False,
        redact_traceback=False,
        redact_exception_strings=False,
    ),
    "ci": RedactionPolicy(
        profile="ci",
        patterns=PII_PATTERNS,
        redact_keys=False,
        redact_traceback=False,
        redact_exception_strings=True,
    ),
    "prod": RedactionPolicy(
        profile="prod",
        patterns=PII_PATTERNS,
        redact_keys=False,
        redact_traceback=True,
        redact_exception_strings=True,
    ),
}


def normalize_redaction_profile(value: str | None) -> RedactionProfile | None:
    """Normalize redaction profile names; return None for empty input."""
    if value is None:
        return None
    candidate = value.strip().lower()
    if not candidate:
        return None
    if candidate not in _VALID_PROFILES:
        raise ValueError(
            f"Invalid redaction profile {value!r}. Expected one of: {sorted(_VALID_PROFILES)}"
        )
    return cast(RedactionProfile, candidate)


def _as_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected bool|None, got {type(value).__name__}")


def resolve_redaction_policy(
    *,
    redaction_profile: RedactionProfile | None = None,
    redact: tuple[str | Pattern[str], ...] | None = None,
    redact_keys: bool | None = None,
    redact_traceback: bool | None = None,
    redact_exception_strings: bool | None = None,
    legacy_redact: tuple[str | Pattern[str], ...] | None = None,
    legacy_redact_keys: bool = False,
    legacy_redact_traceback: bool = False,
    legacy_redact_exception_strings: bool = False,
) -> RedactionPolicy:
    """Resolve final redaction behavior with explicit precedence.

    Precedence:
    1) Explicit overrides (`redact`, `redact_keys`, `redact_traceback`,
       `redact_exception_strings`)
    2) Explicit `redaction_profile`
    3) Legacy fallback values for a given call surface
    4) Hard defaults (empty patterns / False booleans)
    """
    explicit_redact_keys = _as_optional_bool(redact_keys)
    explicit_redact_traceback = _as_optional_bool(redact_traceback)
    explicit_redact_exception_strings = _as_optional_bool(redact_exception_strings)

    profile_defaults = (
        _PROFILE_DEFAULTS[redaction_profile] if redaction_profile is not None else None
    )

    patterns: tuple[str | Pattern[str], ...]
    if redact is not None:
        patterns = tuple(redact)
    elif profile_defaults is not None:
        patterns = profile_defaults.patterns
    elif legacy_redact is not None:
        patterns = tuple(legacy_redact)
    else:
        patterns = ()

    if explicit_redact_keys is not None:
        resolved_redact_keys = explicit_redact_keys
    elif profile_defaults is not None:
        resolved_redact_keys = profile_defaults.redact_keys
    else:
        resolved_redact_keys = legacy_redact_keys

    if explicit_redact_traceback is not None:
        resolved_redact_traceback = explicit_redact_traceback
    elif profile_defaults is not None:
        resolved_redact_traceback = profile_defaults.redact_traceback
    else:
        resolved_redact_traceback = legacy_redact_traceback

    if explicit_redact_exception_strings is not None:
        resolved_redact_exception_strings = explicit_redact_exception_strings
    elif profile_defaults is not None:
        resolved_redact_exception_strings = profile_defaults.redact_exception_strings
    else:
        resolved_redact_exception_strings = legacy_redact_exception_strings

    return RedactionPolicy(
        profile=redaction_profile,
        patterns=patterns,
        redact_keys=resolved_redact_keys,
        redact_traceback=resolved_redact_traceback,
        redact_exception_strings=resolved_redact_exception_strings,
    )
