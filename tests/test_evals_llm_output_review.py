"""Regression tests for Phase 1G (LLM output handling) of the 2026-10-04 review.

The diff extraction corpus lives in ``test_evals_diff_corpus.py``; this file covers the
patcher parse paths around it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from evals import eval_patches
from evals.eval_patches import apply_diff
from evals.patchers import openai_compat_patcher as openai_mod
from tests._openai_stub_server import OpenAIStubServer

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


# --- LLM-05: schema-legal find="" and count=0 no longer corrupt the file -----------------


def test_llm05_schema_requires_nonempty_find_and_positive_count() -> None:
    item = openai_mod.PATCH_EDITS_JSON_SCHEMA["properties"]["edits"]["items"]["properties"]
    assert item["find"]["minLength"] == 1
    assert item["count"]["minimum"] == 1


def test_llm05_empty_find_is_rejected_not_inserted_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edits = '{"edits":[{"path":"buggy.py","find":"","replace":"# ","count":0}]}'
    _scripted(monkeypatch, [(_response(edits, "stop"), {})])

    result = openai_mod._propose_patch_via_edits(
        **_CALL, cwd=_buggy(tmp_path), allow_diff_fallback=False
    )

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["error"] == "empty_find"


def test_llm05_count_zero_replaces_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "m.py").write_text("x = 1\ny = 1\n", encoding="utf-8")
    edits = '{"edits":[{"path":"m.py","find":"1","replace":"2","count":0}]}'
    _scripted(monkeypatch, [(_response(edits, "stop"), {})])

    result = openai_mod._propose_patch_via_edits(**_CALL, cwd=tmp_path)

    assert result.diff is not None
    ok, message = apply_diff(tmp_path, result.diff)
    assert ok, message
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "x = 2\ny = 1\n"


# --- LLM-09: CRLF files keep their line endings in edits and full-file mode -------------


def test_llm09_edit_to_crlf_file_changes_one_line_and_keeps_crlf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "buggy.py"
    target.write_bytes(b"def f():\r\n    return 1\r\n\r\n\r\ndef g():\r\n    return 3\r\n")
    _scripted(monkeypatch, [(_response(_EDITS, "stop"), {})])

    result = openai_mod._propose_patch_via_edits(**_CALL, cwd=tmp_path)

    assert result.diff is not None
    changed = [line for line in result.diff.splitlines() if line[:1] in "+-"]
    assert changed == ["--- a/buggy.py", "+++ b/buggy.py", "-    return 1", "+    return 2"]
    ok, message = apply_diff(tmp_path, result.diff)
    assert ok, message
    assert target.read_bytes() == (
        b"def f():\r\n    return 2\r\n\r\n\r\ndef g():\r\n    return 3\r\n"
    )


def test_llm09_full_file_answer_for_crlf_file_keeps_crlf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "buggy.py"
    target.write_bytes(b"def f():\r\n    return 1\r\n")
    (tmp_path / "test_buggy.py").write_text("def test(): pass\n", encoding="utf-8")
    _scripted(monkeypatch, [(_response("def f():\n    return 2\n", "stop"), {})])

    result = openai_mod._propose_patch_via_full_file(**_CALL, cwd=tmp_path)

    assert result.diff is not None
    ok, message = apply_diff(tmp_path, result.diff)
    assert ok, message
    assert target.read_bytes() == b"def f():\r\n    return 2\r\n"


def test_llm09_chained_edits_to_one_file_give_one_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "m.py").write_text("a = 1\nb = 1\n", encoding="utf-8")
    edits = (
        '{"edits":[{"path":"m.py","find":"a = 1","replace":"a = 2"},'
        '{"path":"m.py","find":"b = 1","replace":"b = 2"}]}'
    )
    _scripted(monkeypatch, [(_response(edits, "stop"), {})])

    result = openai_mod._propose_patch_via_edits(**_CALL, cwd=tmp_path)

    assert result.diff is not None
    assert result.diff.count("diff --git") == 1
    ok, message = apply_diff(tmp_path, result.diff)
    assert ok, message
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "a = 2\nb = 2\n"


# --- LLM-15: diffs are built from "\n"-split lines, with the no-newline marker ----------


def test_llm15_patcher_and_runner_share_one_diff_builder() -> None:
    assert openai_mod._build_diff_from_changes is eval_patches._build_diff_from_changes


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("x = 1\n# page\x0cbreak\ny = 1\n", "x = 2\n# page\x0cbreak\ny = 1\n"),  # form feed
        ('s = "a\u2028b"\nx = 1\n', 's = "a\u2028b"\nx = 2\n'),  # U+2028 in a literal
        ("x = 1\nlast = 0", "x = 2\nlast = 0"),  # no final newline, unchanged
        ("x = 1\n", "x = 1"),  # final newline removed
        ("x = 1", "x = 1\ny = 2\n"),  # final newline added
    ],
)
def test_llm15_built_diff_reproduces_after_exactly(tmp_path: Path, before: str, after: str) -> None:
    target = tmp_path / "m.py"
    target.write_bytes(before.encode())

    diff = eval_patches._build_diff_from_changes([("m.py", before, after)])
    ok, message = apply_diff(tmp_path, diff)

    assert ok, message
    assert message == ""  # strict apply, not the relaxed fallback
    assert target.read_bytes() == after.encode()


def test_llm15_missing_final_newline_gets_the_marker() -> None:
    diff = eval_patches._build_diff_from_changes([("m.py", "x = 1", "x = 2")])
    assert diff.endswith(
        "-x = 1\n\\ No newline at end of file\n+x = 2\n\\ No newline at end of file\n"
    )


# --- LLM-19: payload-shape failures all take the diff fallback; bad JSON is retried -----


@pytest.mark.parametrize(
    ("edits", "reason"),
    [
        ('{"edits":["not an object"]}', "bad_edit_entry"),
        ('{"edits":[{"path":"","find":"a","replace":"b"}]}', "bad_path"),
        ('{"edits":[{"path":"buggy.py","find":1,"replace":"b"}]}', "bad_find_replace"),
        ('{"edits":[{"path":"missing.py","find":"a","replace":"b"}]}', "file_not_found"),
        ('{"edits":[{"path":"../x.py","find":"a","replace":"b"}]}', "path_outside_case"),
        ('{"edits":[{"path":"buggy.py","find":"nope","replace":"b"}]}', "find_not_found"),
    ],
)
def test_llm19_payload_shape_failures_fall_back_to_diff_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edits: str, reason: str
) -> None:
    _scripted(monkeypatch, [(_response(edits, "stop"), {}), (_response(_DIFF, "stop"), {})])

    result = openai_mod._propose_patch_via_edits(**_CALL, cwd=_buggy(tmp_path))

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta["fallback_from"] == "edits"
    assert result.meta["fallback_reason"] == reason


def test_llm19_bad_json_response_body_is_retried() -> None:
    with OpenAIStubServer() as server:
        server.enqueue_text(method="POST", path="/v1/chat/completions", body="<html>502</html>")
        server.enqueue_json(
            method="POST", path="/v1/chat/completions", body=_response(_DIFF, "stop")
        )

        raw, meta = openai_mod._chat_completion(
            url=f"{server.base_url_v1}/chat/completions",
            api_key="local",
            model="dummy",
            messages=[{"role": "user", "content": "fix"}],
            temperature=0.0,
            max_tokens=32,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            transport_retry_backoff_base_sec=0.0,
            transport_retry_backoff_max_sec=0.0,
        )

    assert raw is not None
    assert meta["transport_retry_count"] == 1
    assert meta["transport_retry_attempts"][0]["error"] == "bad_json_response"


# --- LLM-06: a degenerate prompt budget is a configuration error at startup -------------


def test_llm06_run_aborts_before_any_case_when_max_tokens_fills_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from evals import run_eval
    from evals.native_stage_b_flow import OpenAIPromptBudgetError

    monkeypatch.setattr(run_eval, "_REPO_ROOT", tmp_path)
    case_dir = tmp_path / "cases" / "c1"
    case_dir.mkdir(parents=True)
    (case_dir / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    case = run_eval.CaseSpec(
        case_id="c1",
        path=case_dir,
        category="logic",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(
        run_eval, "make_patcher", lambda _args: SimpleNamespace(name="openai", max_tokens=8192)
    )
    monkeypatch.setattr(
        run_eval,
        "run_one_condition",
        lambda **_kwargs: pytest.fail("no case may run with a degenerate prompt budget"),
    )

    with pytest.raises(OpenAIPromptBudgetError, match="--openai-max-tokens 8192"):
        run_eval.main(
            [
                "--run-id",
                "llm06",
                "--patcher",
                "null",
                "--conditions",
                "base_repair",
                "--openai-context-window-tokens",
                "8192",
                "--executor",
                "local",
            ]
        )
