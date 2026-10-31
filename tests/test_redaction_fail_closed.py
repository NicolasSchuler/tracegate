"""Redaction must fail closed and must not leak secret sizes in metadata-only mode.

Regressions found during the Google-docstring pass:

* ``_redact_context`` used to write the whole context unredacted when ``redact_jsonlike``
  raised; the reserved ``trace`` and ``eval`` payloads from ``extra`` bypassed redaction
  entirely.
* ``redact_jsonlike(..., redact_keys=True)`` raised on a non-string dict key.
* ``locals_metadata`` applied no name-based redaction, so ``locals_mode="meta"`` recorded
  the ``len`` of a local or argument named ``password`` even though
  ``docs/configuration.md`` promises the metadata keeps the type but not the length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import llmdebug.capture as capture_module
from llmdebug import get_latest_snapshot
from llmdebug.capture import _redact_context, capture_exception
from llmdebug.config import SnapshotConfig
from llmdebug.serialize import compile_redactors, locals_metadata, redact_jsonlike

_SECRET = "ctx-secret-0123"
_PASSWORD = "hunter2-long-enough"


def _capture(
    tmp_path: Path, cfg_kwargs: dict[str, Any], extra: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    def login(user: str, password: str) -> None:
        raise RuntimeError(f"boom for {user}")

    def boom() -> None:
        password = _PASSWORD  # noqa: F841 - captured local
        items = [1, 2, 3]  # noqa: F841 - captured local
        login("alice", _PASSWORD)

    try:
        boom()
    except RuntimeError as exc:
        cfg = SnapshotConfig(
            out_dir=str(tmp_path), include_git=False, output_format="json", **cfg_kwargs
        )
        path = capture_exception("fail_closed", exc, exc.__traceback__, cfg, extra=extra)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    return snapshot, path.read_text(encoding="utf-8")


def _break_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("redaction crashed")

    monkeypatch.setattr(capture_module, "redact_jsonlike", _boom)


def test_redact_context_drops_context_when_redaction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _break_redaction(monkeypatch)
    cfg = SnapshotConfig(redact=("secret",))
    redactors = compile_redactors(cfg.redact)
    warnings: list[str] = []

    out = _redact_context({"payload": {"value": _SECRET}}, redactors, cfg, warnings)

    assert out == {}
    assert len(warnings) == 1
    assert "context redaction failed" in warnings[0]
    assert "context omitted" in warnings[0]


def test_redact_context_passes_through_without_active_redaction() -> None:
    cfg = SnapshotConfig(redact=())
    payload = {"payload": {1: "plain"}}

    assert _redact_context(payload, [], cfg, []) is payload


def test_redact_jsonlike_stringifies_non_string_keys_when_redacting_keys() -> None:
    redactors = compile_redactors(("secret",))

    out = redact_jsonlike({1: "my-secret", "k": {2.5: "x"}}, redactors, redact_keys=True)

    assert out == {"1": "my-[REDACTED]", "k": {"2.5": "x"}}
    # Without key redaction the original keys are kept untouched.
    assert redact_jsonlike({1: "my-secret"}, redactors) == {1: "my-[REDACTED]"}


def test_non_string_extra_keys_no_longer_drop_the_context(tmp_path: Path) -> None:
    snapshot, raw = _capture(
        tmp_path, {"redact": ("secret",), "redact_keys": True}, {"payload": {1: _SECRET}}
    )

    assert _SECRET not in raw
    assert snapshot["payload"] == {"1": "ctx-[REDACTED]-0123"}
    assert "env" in snapshot
    assert not any("redaction failed" in w for w in snapshot.get("_capture_warnings", []))


def test_snapshot_never_contains_unredacted_context_after_redaction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_redaction(monkeypatch)
    snapshot, raw = _capture(tmp_path, {"redact": ("secret",)}, {"payload": {"value": _SECRET}})

    assert _SECRET not in raw
    # get_latest_snapshot flattens ``context`` into the top level; ``env`` is captured by
    # default, so its absence proves the whole context was dropped, not merely redacted.
    assert "payload" not in snapshot
    assert "env" not in snapshot
    assert any("context redaction failed" in w for w in snapshot.get("_capture_warnings", []))


def test_trace_and_eval_payloads_are_redacted(tmp_path: Path) -> None:
    snapshot, raw = _capture(
        tmp_path,
        {"redaction_profile": "prod"},
        {
            "trace": {"password": _SECRET, "spans": [1]},
            "eval_outcome": {"note": f"api_key={_SECRET}", "status": "ok"},
        },
    )

    assert _SECRET not in raw
    assert snapshot["trace"] == {"password": "[REDACTED]", "spans": [1]}
    assert snapshot["eval"]["status"] == "ok"
    assert _SECRET not in snapshot["eval"]["note"]


def test_trace_and_eval_payloads_are_omitted_when_redaction_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_redaction(monkeypatch)
    snapshot, raw = _capture(
        tmp_path,
        {"redact": ("secret",)},
        {"trace": {"value": _SECRET}, "eval": {"value": _SECRET}},
    )

    assert _SECRET not in raw
    assert "trace" not in snapshot
    assert "eval" not in snapshot
    warnings = snapshot.get("_capture_warnings", [])
    assert any("trace redaction failed" in w and "trace omitted" in w for w in warnings)
    assert any("eval redaction failed" in w and "eval omitted" in w for w in warnings)


def test_trace_and_eval_payloads_pass_through_without_redaction(tmp_path: Path) -> None:
    snapshot, _raw = _capture(
        tmp_path, {"redact": ()}, {"trace": {"spans": [1]}, "eval": {"status": "ok"}}
    )

    assert snapshot["trace"] == {"spans": [1]}
    assert snapshot["eval"] == {"status": "ok"}


def test_locals_metadata_withholds_len_for_secret_names() -> None:
    cfg = SnapshotConfig(redact=("secret",))
    redactors = compile_redactors(cfg.redact)
    local_vars = {"password": _PASSWORD, "items": [1, 2, 3]}

    meta = locals_metadata(local_vars, cfg, redactors=redactors)

    assert meta["password"] == {"type": "builtins.str"}
    assert meta["items"] == {"type": "builtins.list", "len": 3}
    # Without active redaction the size is still reported (documented behaviour).
    assert locals_metadata(local_vars, cfg)["password"]["len"] == len(_PASSWORD)


def test_meta_locals_mode_hides_secret_length_end_to_end(tmp_path: Path) -> None:
    snapshot, _raw = _capture(tmp_path, {"redaction_profile": "prod", "locals_mode": "meta"}, {})

    boom_frame = next(f for f in snapshot["frames"] if f["function"] == "boom")
    assert boom_frame["locals_meta"]["password"] == {"type": "builtins.str"}
    assert boom_frame["locals_meta"]["items"] == {"type": "builtins.list", "len": 3}

    # Function arguments go through ``args_meta`` and must not leak the size either.
    login_frame = next(f for f in snapshot["frames"] if f["function"] == "login")
    assert login_frame["args_meta"]["password"] == {"type": "builtins.str"}
    assert login_frame["args_meta"]["user"] == {"type": "builtins.str", "len": 5}
