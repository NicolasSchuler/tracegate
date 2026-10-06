"""CAP-03: secret-named values are redacted whenever any redaction pattern is active.

The PII patterns match ``key=value`` inside one string, but a structured local keeps the
name and the value apart, so ``password = "hunter2"`` used to reach a ``prod`` snapshot
verbatim. These tests go end-to-end through ``capture_exception``.
"""

from __future__ import annotations

import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from llmdebug import get_latest_snapshot
from llmdebug.capture import capture_exception
from llmdebug.config import SnapshotConfig
from llmdebug.serialize import compile_redactors, is_sensitive_name, redact_text

# Secret literals live here so that captured source lines and tracebacks never contain
# them; only the runtime values can leak, which is what CAP-03 is about.
_S = {
    "password": "hunter2",
    "api_key": "abc123",
    "attr": "attr-secret",
    "nested": "pw1",
    "dsn": "postgres://admin:pw2@db:5432/app",
    "login": "pw3",
    "session": "tok-0123",
    "header": "k1",
    "ctx": "ctx-secret",
    # Values used by the nested-container, truncated-URL and header-field regressions.
    "deep_list": "beta-pw",
    "basic": "dXNlcjpwYXNz",
    "cookie": "sid-0ead",
    "at_dsn": "postgres://admin:p@ssword-0123456789abcdef@db:5432/app",
    "hyphen": "hunter-9",
    "quoted": "alpha beta gamma",
    "long_key": "hunter-long",
    "tuple_field": "nt-secret",
    "bare": "ab,cd;ef)gh",
}


class _Settings:
    def __init__(self) -> None:
        self.password = _S["attr"]
        self.host = "db.local"


def _login(user: str, password: str) -> None:
    session_token = _S["session"]  # noqa: F841 - captured local
    raise PermissionError(f"login failed for {user}")


def _capture(tmp_path: Path, profile: str | None, **extra_cfg: Any) -> tuple[dict[str, Any], str]:
    def boom() -> None:
        password = _S["password"]  # noqa: F841 - captured local
        api_key = _S["api_key"]  # noqa: F841 - captured local
        settings = _Settings()  # noqa: F841 - captured local
        config = {"db": {"Password": _S["nested"], "host": "h"}, "x-api-key": [_S["header"]]}  # noqa: F841
        dsn = _S["dsn"]  # noqa: F841 - captured local
        tokens = ["a", "b"]  # noqa: F841 - ordinary name, kept
        key = "row-7"  # noqa: F841 - ordinary name, kept
        _login("alice", _S["login"])

    try:
        boom()
    except PermissionError as exc:
        cfg = SnapshotConfig(
            out_dir=str(tmp_path),
            redaction_profile=profile,  # type: ignore[arg-type]
            include_git=False,
            output_format="json",
            **extra_cfg,
        )
        path = capture_exception(
            "secrets", exc, exc.__traceback__, cfg, extra={"api_key": _S["ctx"], "run": 1}
        )
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    return snapshot, path.read_text(encoding="utf-8")


@pytest.mark.parametrize("profile", ["prod", "ci"])
def test_secret_values_never_reach_the_snapshot(tmp_path: Path, profile: str) -> None:
    snapshot, raw = _capture(tmp_path, profile)

    for name, secret in _S.items():
        if name != "dsn":
            assert secret not in raw, name
    assert "pw2" not in raw

    boom_frame = next(f for f in snapshot["frames"] if f["function"] == "boom")
    local_vars = boom_frame["locals"]
    assert local_vars["password"] == "[REDACTED]"
    assert local_vars["api_key"] == "[REDACTED]"
    assert local_vars["settings"]["password"] == "[REDACTED]"
    assert local_vars["settings"]["host"] == "db.local"
    assert local_vars["config"]["x-api-key"] == "[REDACTED]"
    assert local_vars["dsn"] == "postgres://[REDACTED]@db:5432/app"
    assert local_vars["tokens"] == ["a", "b"]
    assert local_vars["key"] == "row-7"
    # The metadata keeps the type but not the length of a secret.
    assert boom_frame["locals_meta"]["password"] == {"type": "builtins.str"}

    login_frame = snapshot["frames"][0]
    assert login_frame["args"]["password"] == "[REDACTED]"
    assert login_frame["args"]["user"] == "alice"
    assert snapshot["api_key"] == "[REDACTED]"
    assert snapshot["run"] == 1


def test_nested_secret_key_inside_config_dict(tmp_path: Path) -> None:
    snapshot, _ = _capture(tmp_path, "prod")

    boom_frame = next(f for f in snapshot["frames"] if f["function"] == "boom")
    assert boom_frame["locals"]["config"]["db"]["Password"] == "[REDACTED]"
    assert boom_frame["locals"]["config"]["db"]["host"] == "h"


def test_secret_key_inside_a_repr_string(tmp_path: Path) -> None:
    """Values two levels down are repr strings, so only the text pattern can see the key."""

    def boom() -> None:
        deep = {"a": {"b": {"password": _S["nested"] + " tail", "host": "h"}}}  # noqa: F841
        raise ValueError("deep")

    try:
        boom()
    except ValueError as exc:
        cfg = SnapshotConfig(out_dir=str(tmp_path), redaction_profile="prod", include_git=False)
        capture_exception("deep", exc, exc.__traceback__, cfg)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    rendered = snapshot["frames"][0]["locals"]["deep"]["a"]["b"]
    assert isinstance(rendered, str)
    assert "pw1" not in rendered
    assert "tail" not in rendered
    assert "'host': 'h'" in rendered


@pytest.mark.parametrize("profile", ["dev", None])
def test_no_redaction_without_patterns(tmp_path: Path, profile: str | None) -> None:
    snapshot, raw = _capture(tmp_path, profile)

    assert "hunter2" in raw
    boom_frame = next(f for f in snapshot["frames"] if f["function"] == "boom")
    assert boom_frame["locals"]["dsn"] == "postgres://admin:pw2@db:5432/app"


def test_explicit_empty_patterns_disable_name_redaction(tmp_path: Path) -> None:
    _, raw = _capture(tmp_path, "prod", redact=())

    assert "hunter2" in raw


@pytest.mark.parametrize(
    "name",
    [
        "password",
        "db_password",
        "PASSWD",
        "passphrase",
        "client_secret",
        "SECRET_KEY",
        "credentials",
        "private_key",
        "api_key",
        "apiKey",
        "x-api-key",
        "access_key",
        "accessToken",
        "auth_token",
        "refresh_token",
        "csrf-token",
        "Authorization",
        "set-cookie",
    ],
)
def test_sensitive_names(name: str) -> None:
    assert is_sensitive_name(name)


@pytest.mark.parametrize(
    "name", ["token", "tokens", "key", "keys", "monkey", "author", "user_id", "host", "pwd"]
)
def test_ordinary_names(name: str) -> None:
    assert not is_sensitive_name(name)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("postgres://u:pw@host/db", "postgres://[REDACTED]@host/db"),
        ("redis://:pw@cache:6379/0", "redis://[REDACTED]@cache:6379/0"),
        ("see https://example.com:8080/a@b", "see https://example.com:8080/a@b"),
        ("https://host/path", "https://host/path"),
        ("postgres://user:p@ss@db:5432/x", "postgres://[REDACTED]@db:5432/x"),
        ("https://ghp_abc123@github.com/o/r", "https://[REDACTED]@github.com/o/r"),
        ("https://medium.com/@user/post", "https://medium.com/@user/post"),
        ("postgres://user:pw...[TRUNC]", "postgres://[REDACTED]...[TRUNC]"),
        ("{'password': 'a b', 'host': 'h'}", "{'[REDACTED], 'host': 'h'}"),
        ("{'password': 'it\\'s \"x\" y', 'host': 'h'}", "{'[REDACTED], 'host': 'h'}"),
        ("{'password': b'a b', 'n': 1}", "{'[REDACTED], 'n': 1}"),
        ("{'api_key': 'abc...[TRUNC]", "{'[REDACTED]"),
        ("mail bob@example.com now", "mail [REDACTED] now"),
        ("https://" + "u" * 600 + ":pw@host/x", "https://[REDACTED]/x"),
        ("https://u@" + "p" * 600 + "@host/x", "https://[REDACTED]/x"),
        ("password" + "_" * 100 + "x" + " " * 30 + "= hunter2", "[REDACTED]"),
        ('{"apiKey": "abc"}', '{"[REDACTED]}'),
        ("SECRET_KEY=xyz rest", "[REDACTED] rest"),
        ("tokens = [1]; key: 2", "tokens = [1]; key: 2"),
        # Quoted multi-word values: the whole literal goes, not just its first word.
        ('password="alpha beta gamma"', "[REDACTED]"),
        ("api_key='alpha beta gamma'", "[REDACTED]"),
        # Composite values inside a repr: one level of container is consumed whole.
        ("{'password': ['alpha', 'beta']}", "{'[REDACTED]}"),
        ("{'api_key': {'v': 'alpha'}, 'n': 1}", "{'[REDACTED], 'n': 1}"),
        # Hyphenated secret names, as `is_sensitive_name` already accepted them.
        ("{'secret-key': 'hunter2'}", "{'[REDACTED]}"),
        ('{"secret-key": "hunter2"}', '{"[REDACTED]}'),
        ("secret-key: hunter2 rest", "[REDACTED] rest"),
        # Header fields whose value contains whitespace or ";"-separated pairs.
        ("Authorization: Basic dXNlcjpwYXNz", "[REDACTED]"),
        ("Cookie: first=alpha; second=beta", "[REDACTED]"),
        ("{'Authorization': 'Basic abc', 'Accept': 'json'}", "{'[REDACTED], 'Accept': 'json'}"),
        # A quoted name/value pair, as in the repr of a header list.
        ("['Authorization', 'Basic abc']", "[[REDACTED]]"),
        ("('password', 'alpha beta')", "([REDACTED])"),
        # Unquoted values run to the next whitespace, as the legacy `\\S+` patterns did.
        ("password=ab,cd", "[REDACTED]"),
        ("api_key=x;y)z rest", "[REDACTED] rest"),
        ("password: hunter2;next rest", "[REDACTED] rest"),
        # A quoted header value is taken whole, commas and brackets included.
        ("{'Authorization': 'Basic a,b', 'Accept': 'json'}", "{'[REDACTED], 'Accept': 'json'}"),
        ('Cookie: "a=1, b=2" rest', "[REDACTED] rest"),
        # Brackets inside a string do not end a container value.
        ("{'password': ['a]b', 'c'], 'host': 'h'}", "{'[REDACTED], 'host': 'h'}"),
        # A nested, unterminated or over-long container fails closed to the end of the line.
        ("{'password': ['a', ['b']], 'host': 'h'}", "{'[REDACTED]"),
        ("{'password': ['a', 'b'", "{'[REDACTED]"),
        ("{'password': [" + "9, " * 500 + "0]}", "{'[REDACTED]"),
        ("password=(a, (b))\nnext", "[REDACTED]\nnext"),
        # A truncated authority is redacted whole even when the password contains "@".
        ("postgres://admin:p@ss-secret...[TRUNC]", "postgres://[REDACTED]...[TRUNC]"),
        (
            "{'dsn': 'postgres://admin:p@ss...[TRUNC]'}",
            "{'dsn': 'postgres://[REDACTED]...[TRUNC]'}",
        ),
        ("https://host/a...[TRUNC]", "https://host/a...[TRUNC]"),
    ],
)
def test_text_patterns(text: str, expected: str) -> None:
    from llmdebug.redaction_policy import PII_PATTERNS

    assert redact_text(text, compile_redactors(PII_PATTERNS)) == expected


@pytest.mark.parametrize(
    "text",
    ["secret" * 100_000, "a." * 300_000, "://" + "a@" * 200_000, "password: '" + "x" * 500_000],
)
def test_text_patterns_stay_linear_on_long_input(text: str) -> None:
    """Captured stdout can be megabytes; the quadratic patterns took minutes per capture."""
    from llmdebug.redaction_policy import PII_PATTERNS

    redactors = compile_redactors(PII_PATTERNS)
    started = time.perf_counter()
    redact_text(text, redactors)

    assert time.perf_counter() - started < 2.0


def test_secret_name_value_pairs(tmp_path: Path) -> None:
    def boom() -> None:
        form = [("username", "bob"), ("password", _S["password"])]  # noqa: F841
        headers = [["Authorization", _S["api_key"]], ["Accept", "json"]]  # noqa: F841
        raise ValueError("pairs")

    try:
        boom()
    except ValueError as exc:
        cfg = SnapshotConfig(out_dir=str(tmp_path), redaction_profile="ci", include_git=False)
        capture_exception("pairs", exc, exc.__traceback__, cfg)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    local_vars = snapshot["frames"][0]["locals"]
    assert local_vars["form"] == [["username", "bob"], ["password", "[REDACTED]"]]
    assert local_vars["headers"] == [["Authorization", "[REDACTED]"], ["Accept", "json"]]


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: " + "a" * 100_000,
        "cookie=" + "a;" * 50_000,
        "{'secret-key': '" + "x" * 100_000,
        "['api_key', '" + "x" * 100_000,
        "password: [" + "x" * 100_000,
        "password: {" + "x" * 100_000,
        "://" + "a" * 100_000 + "...[TRUNC]",
        "://" + "a...[TRUNC]" * 10_000,
        "'secret" + "-k" * 50_000,
        # Quoted container items must not make the container repeat ambiguous.
        "password: [" + "b'x'" * 30_000,
        "password: [" + "'a', " * 20_000,
        "password: " + "[" * 50_000,
        "password=" + "x" * 100_000,
        "Authorization: '" + "x" * 100_000,
        "{'password': '" + "a\\'" * 50_000,
    ],
)
def test_new_patterns_stay_linear_on_adversarial_input(text: str) -> None:
    """The header-field, container-value and pair patterns must stay linear-time."""
    from llmdebug.redaction_policy import PII_PATTERNS

    redactors = compile_redactors(PII_PATTERNS)
    started = time.perf_counter()
    redact_text(text, redactors)

    assert time.perf_counter() - started < 1.0


def _capture_locals(
    tmp_path: Path, body: Any, profile: str | None = "prod", **extra_cfg: Any
) -> tuple[dict[str, Any], str]:
    """Capture one frame through ``capture_exception`` and return its locals plus raw text."""
    try:
        body()
    except ValueError as exc:
        cfg = SnapshotConfig(
            out_dir=str(tmp_path),
            redaction_profile=profile,  # type: ignore[arg-type]
            include_git=False,
            output_format="json",
            **extra_cfg,
        )
        path = capture_exception("deep", exc, exc.__traceback__, cfg)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    return snapshot["frames"][0]["locals"], path.read_text(encoding="utf-8")


def test_nested_container_secret_values_past_the_depth_limit(tmp_path: Path) -> None:
    """A composite value under a secret name used to survive the depth-to-repr fallback."""

    def boom() -> None:
        config = {"a": {"b": {"password": ["alpha", _S["deep_list"]], "host": "h"}}}  # noqa: F841
        headers = {"a": {"b": [["Authorization", "Basic " + _S["basic"]], ["Accept", "json"]]}}  # noqa: F841
        jar = {"a": {"b": {"Cookie": f"sid={_S['cookie']}; theme=dark"}}}  # noqa: F841
        raise ValueError("deep")

    local_vars, raw = _capture_locals(tmp_path, boom)

    for name in ("deep_list", "basic", "cookie"):
        assert _S[name] not in raw, name
    config_repr = local_vars["config"]["a"]["b"]
    assert isinstance(config_repr, str)
    assert "alpha" not in config_repr
    assert "'host': 'h'" in config_repr
    assert "'Accept', 'json'" in local_vars["headers"]["a"]["b"]


def test_truncated_url_authority_with_at_sign_in_password(tmp_path: Path) -> None:
    """Truncation can remove the host separator, leaving an "@" inside the password."""

    def boom() -> None:
        dsn = _S["at_dsn"]  # noqa: F841 - ordinary name, only the text pattern sees it
        raise ValueError("dsn")

    local_vars, raw = _capture_locals(tmp_path, boom, max_str=40)

    # "password" itself appears in the echoed config patterns, so match the secret suffix.
    assert "ssword-0" not in raw
    assert local_vars["dsn"] == "postgres://[REDACTED]...[TRUNC]"


def test_hyphenated_sensitive_key_in_serialized_text(tmp_path: Path) -> None:
    """``secret-key`` is sensitive structurally; the text matcher used to reject the hyphen."""

    def boom() -> None:
        deep = {"a": {"b": {"secret-key": _S["hyphen"], "host": "h"}}}  # noqa: F841
        payload = '{"secret-key": "' + _S["hyphen"] + '"}'  # noqa: F841
        shallow = {"secret-key": _S["hyphen"]}  # noqa: F841
        raise ValueError("hyphen")

    local_vars, raw = _capture_locals(tmp_path, boom)

    assert _S["hyphen"] not in raw
    assert local_vars["shallow"]["secret-key"] == "[REDACTED]"
    assert local_vars["payload"] == '{"[REDACTED]}'


def test_quoted_and_header_field_values_in_an_exception_message(tmp_path: Path) -> None:
    """Exception strings are redacted in ``prod``; the legacy patterns cut values short."""

    def boom() -> None:
        raise ValueError(
            f'password="{_S["quoted"]}" Authorization: Basic {_S["basic"]} '
            f"Cookie: sid={_S['cookie']}; theme=dark"
        )

    try:
        boom()
    except ValueError as exc:
        cfg = SnapshotConfig(
            out_dir=str(tmp_path),
            redaction_profile="prod",
            include_git=False,
            source_mode="none",
            output_format="json",
        )
        path = capture_exception("msg", exc, exc.__traceback__, cfg)
    raw = path.read_text(encoding="utf-8")

    for name in ("quoted", "basic", "cookie"):
        assert _S[name] not in raw, name
    assert "beta gamma" not in raw


def test_sensitive_dict_key_longer_than_max_str(tmp_path: Path) -> None:
    """Name-aware redaction ran after key truncation, so the secret suffix was lost."""

    def boom() -> None:
        creds = {"database_connection_password": _S["long_key"]}  # noqa: F841
        raise ValueError("long key")

    local_vars, raw = _capture_locals(tmp_path, boom, max_str=20)

    assert _S["long_key"] not in raw
    assert local_vars["creds"] == {"database...[TRUNC]": "[REDACTED]"}


def test_bytes_dict_key_is_sensitive_through_its_repr(tmp_path: Path) -> None:
    def boom() -> None:
        creds = {b"password": _S["long_key"]}  # noqa: F841
        raise ValueError("bytes key")

    local_vars, raw = _capture_locals(tmp_path, boom)

    assert _S["long_key"] not in raw
    assert local_vars["creds"] == {"b'password'": "[REDACTED]"}


def test_namedtuple_fields_keep_their_names_past_the_depth_limit(tmp_path: Path) -> None:
    """Rebuilding a namedtuple as a plain tuple would drop the names the patterns need."""
    creds = namedtuple("creds", ["user", "password"])

    def boom() -> None:
        deep = {"a": {"b": creds("bob", _S["tuple_field"])}}  # noqa: F841
        raise ValueError("namedtuple")

    local_vars, raw = _capture_locals(tmp_path, boom)

    assert _S["tuple_field"] not in raw
    # The type and the non-secret field names survive, so the text patterns still apply.
    assert local_vars["deep"]["a"]["b"] == "creds(user='bob', [REDACTED])"


def test_unquoted_secret_value_with_structural_delimiters(tmp_path: Path) -> None:
    """The delimiter-bounded value class must not apply to plain `name=value` text."""

    def boom() -> None:
        raise ValueError("connect failed: password=" + _S["bare"] + " host=db")

    try:
        boom()
    except ValueError as exc:
        cfg = SnapshotConfig(
            out_dir=str(tmp_path),
            redaction_profile="prod",
            include_git=False,
            source_mode="none",
            output_format="json",
        )
        path = capture_exception("bare", exc, exc.__traceback__, cfg)
    raw = path.read_text(encoding="utf-8")

    assert _S["bare"] not in raw
    assert "cd;ef" not in raw
