"""Evidence-enforcement regressions from the 2026-10-04 review (ATT-03, ATT-20, ATT-21)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from evals import eval_evidence
from evals.eval_evidence import (
    RESTRICTED_ACTION_NO_VALID_REQUESTS,
    build_fallback_context_bundle,
    deny_fallback_for_restricted_action,
    execute_context_requests,
)
from evals.eval_types import (
    ContextRequestPayload,
    _filter_context_request_payload_for_action,
)


def _workdir(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "buggy.py").write_text("import helper\n\ndef f():\n    return 1\n", "utf-8")
    (tmp_path / "helper.py").write_text("def value():\n    return 1\n", "utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "from buggy import f\n\ndef test_f():\n    assert f() == 2\n", "utf-8"
    )
    return tmp_path


_SNAPSHOT = {"frames": [{"file_rel": "buggy.py", "line": 4, "function": "f", "locals": {}}]}
_INDEX = ["buggy.py", "helper.py", "test_buggy.py"]


def _execute(root: Path, payload: dict[str, Any], *, max_requests: int = 5):
    return execute_context_requests(
        request_payload=cast(ContextRequestPayload, payload),
        root=root,
        snapshot=_SNAPSHOT,
        file_index=_INDEX,
        stdout="test_buggy.py::test_f FAILED",
        stderr="",
        max_requests=max_requests,
        max_total_chars=10_000,
    )


def test_att03_filtered_restricted_request_does_not_fall_back_to_whole_files(
    tmp_path: Path,
) -> None:
    root = _workdir(tmp_path)
    payload = {"need_more_context": True, "requests": [{"type": "file", "path": "buggy.py"}]}
    filtered = _filter_context_request_payload_for_action(
        action="search_only", request_payload=cast(ContextRequestPayload, payload)
    )
    assert filtered is not None and filtered["requests"] == []

    unrestricted = _execute(root, dict(filtered))
    assert unrestricted.used_fallback is True  # the leak the review found
    restricted = deny_fallback_for_restricted_action(unrestricted, restricted=True)
    assert restricted.used_fallback is False
    assert restricted.text == ""
    assert restricted.requests_count == 0
    assert restricted.chars_returned == 0
    assert restricted.reason == RESTRICTED_ACTION_NO_VALID_REQUESTS
    # Unrestricted actions keep the fallback behaviour.
    assert deny_fallback_for_restricted_action(unrestricted, restricted=False) is unrestricted


def test_att21_request_budget_zero_executes_no_request(tmp_path: Path) -> None:
    root = _workdir(tmp_path)
    payload = {"need_more_context": True, "requests": [{"type": "file", "path": "buggy.py"}]}
    data = _execute(root, payload, max_requests=0)
    assert data.requests_count == 0
    assert data.text == ""
    assert data.used_fallback is False
    assert data.reason == "request_budget_zero"


def test_att20_fallback_bundle_ignores_paths_outside_the_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workdir(tmp_path / "work")
    secret = tmp_path / "secret.py"
    secret.write_text("TOKEN = 'leak'\n", "utf-8")

    read_paths: list[Path] = []
    real_read = eval_evidence._read_text_limited

    def _spy_read(path: Path, *, max_chars: int) -> str | None:
        read_paths.append(path)
        return real_read(path, max_chars=max_chars)

    monkeypatch.setattr(eval_evidence, "_read_text_limited", _spy_read)

    for bad in ("../secret.py", str(secret), "/etc/passwd"):
        data = build_fallback_context_bundle(
            root=root,
            snapshot={"frames": [{"file_rel": bad}]},
            stdout="",
            stderr="",
            max_total_chars=10_000,
            reason="test",
            file_index_lines=_INDEX,
        )
        assert "leak" not in data.text
        assert all(".." not in str(block.get("path")) for block in data.blocks)
    # ATT-20: neighbor discovery used to open the crash path before validating it, so
    # assert the outside file was never read rather than merely absent from the result.
    outside = [path for path in read_paths if not path.resolve().is_relative_to(root.resolve())]
    assert not outside, outside
