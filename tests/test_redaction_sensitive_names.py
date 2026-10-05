"""CAP-03: secret-named values are redacted whenever any redaction pattern is active.

The PII patterns match ``key=value`` inside one string, but a structured local keeps the
name and the value apart, so ``password = "hunter2"`` used to reach a ``prod`` snapshot
verbatim. These tests go end-to-end through ``capture_exception``.
"""

from __future__ import annotations

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
        ("{'password': 'a b', 'host': 'h'}", "{'[REDACTED], 'host': 'h'}"),
        ('{"apiKey": "abc"}', '{"[REDACTED]}'),
        ("SECRET_KEY=xyz rest", "[REDACTED] rest"),
        ("tokens = [1]; key: 2", "tokens = [1]; key: 2"),
    ],
)
def test_text_patterns(text: str, expected: str) -> None:
    from llmdebug.redaction_policy import PII_PATTERNS

    assert redact_text(text, compile_redactors(PII_PATTERNS)) == expected
