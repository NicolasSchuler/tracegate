"""Regression tests for Phase 1G (LLM output handling) of the 2026-10-04 review.

The diff extraction corpus lives in ``test_evals_diff_corpus.py``; this file covers the
patcher parse paths around it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from evals.patchers import openai_compat_patcher as openai_mod

_DIFF = "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
_EDITS = '{"edits":[{"path":"buggy.py","find":"return 1","replace":"return 2","count":1}]}'
_CALL: dict[str, Any] = {
    "url": "http://127.0.0.1:1/v1/chat/completions",
    "api_key": "dummy",
    "model": "dummy",
    "prompt": "Pytest output:\nfailed\n",
    "temperature": 0.0,
    "max_tokens": 32,
    "timeout_sec": 5.0,
    "seed": None,
    "http2": False,
}


def _response(content: str, finish_reason: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def _scripted(
    monkeypatch: pytest.MonkeyPatch, responses: list[tuple[dict[str, Any] | None, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Serve *responses* from both chat-completion entry points; return the call log."""
    calls: list[dict[str, Any]] = []

    def _next(kwargs: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        calls.append(kwargs)
        raw, meta = responses[min(len(calls), len(responses)) - 1]
        return raw, {"llm_call_count": 1, **meta}

    async def _next_async(**kwargs: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return _next(kwargs)

    monkeypatch.setattr(openai_mod, "_chat_completion", lambda **kwargs: _next(kwargs))
    monkeypatch.setattr(openai_mod, "_chat_completion_async", _next_async)
    return calls


def _buggy(tmp_path: Path) -> Path:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


# --- LLM-02: finish_reason == "length" is a terminal truncated_response ------------------


@pytest.mark.parametrize("use_async", [False, True])
def test_llm02_truncated_diff_is_not_repaired_into_a_valid_patch(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    # The cut-off hunk drops the "+" line; _fix_hunk_counts would make it a valid deletion.
    truncated = _DIFF.rsplit("+", 1)[0]
    calls = _scripted(monkeypatch, [(_response(truncated, "length"), {"finish_reason": "length"})])

    if use_async:
        result = asyncio.run(openai_mod._propose_patch_via_diff_async(**_CALL))
    else:
        result = openai_mod._propose_patch_via_diff(**_CALL)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["error"] == "truncated_response"
    assert result.meta["finish_reason"] == "length"
    assert len(calls) == 1  # terminal: the same budget would truncate again


@pytest.mark.parametrize("use_async", [False, True])
def test_llm02_truncated_edits_payload_is_terminal_without_diff_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    calls = _scripted(
        monkeypatch, [(_response(_EDITS[:-3], "length"), {"finish_reason": "length"})]
    )
    call = {**_CALL, "cwd": _buggy(tmp_path)}

    if use_async:
        result = asyncio.run(openai_mod._propose_patch_via_edits_async(**call))
    else:
        result = openai_mod._propose_patch_via_edits(**call)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["error"] == "truncated_response"  # was no_json_payload
    assert result.meta["structured_output_attempts"][-1]["parse_error"] == "truncated_response"
    assert len(calls) == 1


def test_llm02_truncated_but_parseable_edits_are_not_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A complete-looking JSON object can still be a truncated answer (more edits were coming).
    _scripted(monkeypatch, [(_response(_EDITS, "length"), {"finish_reason": "length"})])

    result = openai_mod._propose_patch_via_edits(**_CALL, cwd=_buggy(tmp_path))

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["error"] == "truncated_response"


def test_llm02_generate_json_reports_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted(monkeypatch, [(_response('{"a": 1}', "length"), {"finish_reason": "length"})])
    patcher = openai_mod.OpenAICompatPatcher(
        base_url="http://127.0.0.1:1", model="dummy", http2=False
    )

    result = patcher.generate_json(
        system_prompt="s",
        prompt="p",
        context_dir=tmp_path,
        cwd=tmp_path,
        schema_name="probe",
        schema={"type": "object"},
    )

    assert result.payload is None
    assert result.meta["error"] == "truncated_response"


def test_llm02_truncated_context_request_and_compaction_payloads_are_rejected() -> None:
    response = _response('{"requests": [], "summary": "partial"}', "length")
    assert openai_mod._normalize_context_request_payload(response)[0] is None
    assert openai_mod._normalize_context_compaction_payload(response)[0] is None


def test_llm02_complete_diff_still_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    _scripted(monkeypatch, [(_response(_DIFF, "stop"), {"finish_reason": "stop"})])

    result = openai_mod._propose_patch_via_diff(**_CALL)

    assert result.diff is not None


# --- LLM-14: the specific failure reason wins over a stale transport error ---------------


def test_llm14_parse_failure_is_not_overwritten_by_a_stale_transport_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stage 1 answers with prose (no JSON); stages 2 and 3 hit transport errors.
    _scripted(
        monkeypatch,
        [
            (_response("I would change the return value.", "stop"), {}),
            (None, {"error": "http_error", "status": 503}),
        ],
    )

    result = openai_mod._propose_patch_via_edits(
        **_CALL, cwd=_buggy(tmp_path), allow_diff_fallback=False
    )

    assert result.meta is not None
    assert result.meta["error"] == "no_json_payload"
    assert result.meta["transport_error"] == "http_error"


def test_llm14_error_meta_keeps_the_reason_and_the_earlier_error() -> None:
    meta = openai_mod._error_meta({"error": "request_failed", "status": 502}, "file_not_found")
    assert meta == {
        "error": "file_not_found",
        "status": 502,
        "transport_error": "request_failed",
    }
    assert openai_mod._error_meta({"error": "x"}, "x") == {"error": "x"}
