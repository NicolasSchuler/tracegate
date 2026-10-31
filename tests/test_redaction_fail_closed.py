"""Redaction must fail closed and must not leak secret sizes in metadata-only mode.

Two regressions found during the Google-docstring pass:

* ``_redact_context`` used to write the whole context unredacted when ``redact_jsonlike``
  raised (for example ``redact_keys=True`` with a non-string dict key in ``extra``).
* ``locals_metadata`` applied no name-based redaction, so ``locals_mode="meta"`` recorded
  the ``len`` of a local named ``password`` even though ``docs/configuration.md`` promises
  the metadata keeps the type but not the length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llmdebug import get_latest_snapshot
from llmdebug.capture import _redact_context, capture_exception
from llmdebug.config import SnapshotConfig
from llmdebug.serialize import compile_redactors, locals_metadata

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


def test_redact_context_drops_context_when_redaction_raises() -> None:
    cfg = SnapshotConfig(redact=("secret",), redact_keys=True)
    redactors = compile_redactors(cfg.redact)
    warnings: list[str] = []

    out = _redact_context({"payload": {1: _SECRET}}, redactors, cfg, warnings)

    assert out == {}
    assert len(warnings) == 1
    assert "context redaction failed" in warnings[0]
    assert "context omitted" in warnings[0]


def test_redact_context_passes_through_without_active_redaction() -> None:
    cfg = SnapshotConfig(redact=())
    payload = {"payload": {1: "plain"}}

    assert _redact_context(payload, [], cfg, []) is payload


def test_snapshot_never_contains_unredacted_context_after_redaction_failure(
    tmp_path: Path,
) -> None:
    snapshot, raw = _capture(
        tmp_path,
        {"redact": ("secret",), "redact_keys": True},
        {"payload": {1: _SECRET}},
    )

    assert _SECRET not in raw
    # get_latest_snapshot flattens ``context`` into the top level; ``env`` is captured by
    # default, so its absence proves the whole context was dropped, not merely redacted.
    assert "payload" not in snapshot
    assert "env" not in snapshot
    assert any("context redaction failed" in w for w in snapshot.get("_capture_warnings", []))


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
