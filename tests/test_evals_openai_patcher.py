"""Tests for OpenAI patcher diff building, URL parsing, and response handling."""

from __future__ import annotations

import argparse
import json
import threading

import pytest

from evals import run_eval
from evals.patchers import openai_compat_patcher as openai_mod
from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import (
    OpenAICompatPatcher,
    _base_url_from_chat_url,
    _build_diff_from_changes,
)


def test_openai_patcher_default_max_propose_sec() -> None:
    patcher = OpenAICompatPatcher(base_url="http://127.0.0.1:1234", model="dummy")

    assert patcher.max_propose_sec == 300.0


def test_openai_force_patch_attempt_builds_fallback_diff(tmp_path, monkeypatch) -> None:
    """Verifies that when both diff and edits strategies return no patch and force_patch_attempt=True, a synthetic fallback diff targeting the first non-test source file is generated and is apply-compatible."""
    module_path = "evals.patchers.openai_compat_patcher._propose_patch_via_diff"
    monkeypatch.setattr(
        module_path,
        lambda **kwargs: PatchResult(diff=None, meta={"error": "no_diff_found"}),
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_edits",
        lambda **kwargs: PatchResult(diff=None, meta={"error": "no_json_payload"}),
    )

    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text("def test_f():\n    assert True\n", encoding="utf-8")

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=True,
    )
    prompt = (
        "Available files (use exact relative paths):\n"
        "test_buggy.py\n"
        "buggy.py\n\n"
        "Pytest output (stdout):\nfailed\n"
    )
    result = patcher.propose_patch(prompt=prompt, context_dir=tmp_path, cwd=tmp_path)

    assert result.diff is not None
    assert "diff --git a/buggy.py b/buggy.py" in result.diff
    assert "llmdebug-force-attempt" in result.diff
    assert result.meta is not None
    assert result.meta["error"] == "no_json_payload"
    assert result.meta.get("fallback_from") == "diff"
    assert result.meta["forced_patch_attempt"] is True
    assert result.meta["forced_patch_target"] == "buggy.py"

    applied, _ = run_eval.apply_diff(tmp_path, result.diff)
    assert applied


def test_openai_force_patch_attempt_disabled_preserves_none(tmp_path, monkeypatch) -> None:
    """Verifies that with force_patch_attempt=False the result diff remains None and the meta records fallback_from, response_mode, and api_compat_seed_requested correctly."""
    module_path = "evals.patchers.openai_compat_patcher._propose_patch_via_diff"
    monkeypatch.setattr(
        module_path,
        lambda **kwargs: PatchResult(diff=None, meta={"error": "no_diff_found"}),
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_edits",
        lambda **kwargs: PatchResult(diff=None, meta={"error": "no_json_payload"}),
    )

    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=False,
    )
    prompt = (
        "Available files (use exact relative paths):\nbuggy.py\n\nPytest output (stdout):\nfailed\n"
    )
    result = patcher.propose_patch(prompt=prompt, context_dir=tmp_path, cwd=tmp_path)

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["error"] == "no_json_payload"
    assert result.meta.get("fallback_from") == "diff"
    assert result.meta["response_mode"] == "diff"
    assert result.meta["api_compat_seed_requested"] is None


def test_openai_force_patch_attempt_skips_on_transport_failure(tmp_path, monkeypatch) -> None:
    """Verifies that force_patch_attempt is suppressed when the underlying failure is a transport/request error, recording forced_patch_skipped and the specific skip detail."""
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        lambda **kwargs: PatchResult(
            diff=None,
            meta={"error": "request_failed", "exc": "TimeoutError: timed out patcher"},
        ),
    )

    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=True,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["forced_patch_skipped"] is True
    assert result.meta["forced_patch_skip_reason"] == "transport_or_context_failure"
    assert result.meta["forced_patch_skip_detail"] == "request_failed"
    assert "forced_patch_attempt" not in result.meta


def test_openai_force_patch_attempt_skips_on_context_limit_failure(tmp_path, monkeypatch) -> None:
    """Verifies that force_patch_attempt is suppressed when the failure is a context-limit HTTP error, setting the skip detail to "context_limit"."""
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        lambda **kwargs: PatchResult(
            diff=None,
            meta={
                "error": "http_error",
                "status": 400,
                "body": "Prompt is too long and exceeds the available context size.",
            },
        ),
    )

    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=True,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["forced_patch_skipped"] is True
    assert result.meta["forced_patch_skip_reason"] == "transport_or_context_failure"
    assert result.meta["forced_patch_skip_detail"] == "context_limit"
    assert "forced_patch_attempt" not in result.meta


def test_chat_completion_applies_qwen_compat_request_profile(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Client:
        last_http_version = "HTTP/1.1"

        def post_json(self, *, target, body, headers, timeout_sec):
            del target, headers, timeout_sec
            captured["payload"] = json.loads(body.decode("utf-8"))
            return 200, json.dumps({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(
        openai_mod,
        "_get_thread_http_client",
        lambda **_kwargs: _Client(),
    )

    payload, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="dummy",
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=32,
        compat_profile="qwen_openai_raw",
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert payload is not None
    sent_payload = captured["payload"]
    assert isinstance(sent_payload, dict)
    assert sent_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in sent_payload
    assert meta["compat_profile_applied"] == "qwen_openai_raw"
    assert meta["direct_output_requested"] is True


def test_chat_completion_applies_qwen_compat_request_profile_with_thinking_enabled(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        last_http_version = "HTTP/1.1"

        def post_json(self, *, target, body, headers, timeout_sec):
            del target, headers, timeout_sec
            captured["payload"] = json.loads(body.decode("utf-8"))
            return 200, json.dumps({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(
        openai_mod,
        "_get_thread_http_client",
        lambda **_kwargs: _Client(),
    )

    payload, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="dummy",
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=32,
        compat_profile="qwen_openai_raw",
        allow_reasoning_fallback=False,
        enable_thinking=True,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert payload is not None
    sent_payload = captured["payload"]
    assert isinstance(sent_payload, dict)
    assert sent_payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert meta["compat_profile_applied"] == "qwen_openai_raw"
    assert meta["allow_reasoning_fallback"] is True
    assert meta["enable_thinking"] is True
    assert meta["direct_output_requested"] is False


def test_chat_completion_reasoning_only_response_returns_compat_error(monkeypatch) -> None:
    class _Client:
        last_http_version = "HTTP/1.1"

        def post_json(self, *, target, body, headers, timeout_sec):
            del target, body, headers, timeout_sec
            return (
                200,
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "reasoning": "step by step",
                                }
                            }
                        ]
                    }
                ),
            )

    monkeypatch.setattr(
        openai_mod,
        "_get_thread_http_client",
        lambda **_kwargs: _Client(),
    )

    payload, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="dummy",
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=32,
        compat_profile="qwen_openai_raw",
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert payload is None
    assert meta["error"] == "openai_compat_reasoning_only"
    assert meta["empty_content_with_reasoning"] is True
    assert meta["content_present"] is False
    assert meta["reasoning_present"] is True
    assert "compat_message" in meta


def test_chat_completion_reasoning_only_response_uses_reasoning_fallback_when_enabled(
    monkeypatch,
) -> None:
    class _Client:
        last_http_version = "HTTP/1.1"

        def post_json(self, *, target, body, headers, timeout_sec):
            del target, body, headers, timeout_sec
            return (
                200,
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "reasoning": "step by step",
                                }
                            }
                        ]
                    }
                ),
            )

    monkeypatch.setattr(
        openai_mod,
        "_get_thread_http_client",
        lambda **_kwargs: _Client(),
    )

    payload, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="dummy",
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=32,
        compat_profile="qwen_openai_raw",
        enable_thinking=True,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert payload is not None
    assert meta["used_reasoning_fallback"] is True
    assert meta["allow_reasoning_fallback"] is True
    assert meta["enable_thinking"] is True
    assert openai_mod._get_first_choice_content(payload) == "step by step"


def test_run_eval_make_patcher_defaults_tolerate_reasoning_only_local_responses(
    monkeypatch,
) -> None:
    class _Client:
        last_http_version = "HTTP/1.1"

        def post_json(self, *, target, body, headers, timeout_sec):
            del target, body, headers, timeout_sec
            return (
                200,
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "reasoning": "step by step",
                                }
                            }
                        ]
                    }
                ),
            )

    monkeypatch.setattr(
        openai_mod,
        "_get_thread_http_client",
        lambda **_kwargs: _Client(),
    )

    patcher = run_eval.make_patcher(
        argparse.Namespace(
            patcher="openai",
            openai_model="dummy",
            openai_api_key="dummy",
            openai_base_url="http://127.0.0.1:1234/v1",
            openai_timeout_sec=30.0,
            openai_http2=True,
            openai_transport_max_retries=3,
            openai_transport_retry_backoff_base_sec=1.0,
            openai_transport_retry_backoff_max_sec=8.0,
            openai_max_propose_sec=120.0,
            openai_max_tokens=32,
            openai_temperature=0.0,
            openai_compat_profile="lmstudio_model_yaml",
            openai_allow_reasoning_fallback=True,
            openai_enable_thinking=True,
            openai_response_mode="edits",
            openai_force_patch_attempt=False,
            deterministic=False,
            seed=None,
        )
    )

    payload, meta = openai_mod._chat_completion(
        url=str(patcher.base_url) + "/chat/completions",
        api_key=str(patcher.api_key or ""),
        model=str(patcher.model),
        messages=[{"role": "user", "content": "hi"}],
        temperature=float(patcher.temperature),
        max_tokens=int(patcher.max_tokens),
        compat_profile=str(patcher.compat_profile),
        allow_reasoning_fallback=bool(patcher.allow_reasoning_fallback),
        enable_thinking=bool(patcher.enable_thinking),
        timeout_sec=float(patcher.timeout_sec),
        seed=None,
        http2=bool(patcher.http2),
    )

    assert payload is not None
    assert meta["used_reasoning_fallback"] is True
    assert meta["allow_reasoning_fallback"] is True
    assert meta["compat_profile_applied"] == "lmstudio_model_yaml"
    assert meta["enable_thinking"] is True
    assert openai_mod._get_first_choice_content(payload) == "step by step"


def test_openai_edits_short_circuits_on_reasoning_only_error(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            None,
            {
                "error": "openai_compat_reasoning_only",
                "compat_message": "reasoning-only response",
                "empty_content_with_reasoning": True,
                "reasoning_present": True,
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat_completion)
    monkeypatch.setattr(
        openai_mod,
        "_propose_patch_via_diff",
        lambda **_kwargs: pytest.fail("diff fallback should not run on compat reasoning-only"),
    )

    result = openai_mod._propose_patch_via_edits(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="dummy",
        model="dummy",
        prompt="Pytest output:\nfailed\n",
        cwd=tmp_path,
        temperature=0.0,
        max_tokens=32,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["error"] == "openai_compat_reasoning_only"
    attempts = result.meta["structured_output_attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 1
    assert attempts[0]["terminal"] is True


def test_openai_force_patch_attempt_skips_on_compat_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        lambda **_kwargs: PatchResult(
            diff=None,
            meta={
                "error": "openai_compat_reasoning_only",
                "compat_message": "reasoning-only response",
            },
        ),
    )

    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=True,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is None
    assert result.meta is not None
    assert result.meta["forced_patch_skipped"] is True
    assert result.meta["forced_patch_skip_reason"] == "compatibility_failure"
    assert result.meta["forced_patch_skip_detail"] == "openai_compat_reasoning_only"


def test_openai_plan_patch_preserves_compat_reasoning_only_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        openai_mod,
        "_chat_completion",
        lambda **_kwargs: (
            None,
            {
                "error": "openai_compat_reasoning_only",
                "compat_message": "reasoning-only response",
                "empty_content_with_reasoning": True,
                "reasoning_present": True,
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        ),
    )

    patcher = OpenAICompatPatcher(base_url="http://127.0.0.1:1234", model="dummy")
    result = patcher.plan_patch(prompt="plan", context_dir=tmp_path, cwd=tmp_path)

    assert result is not None
    assert result.plan_text is None
    assert result.meta["error"] == "openai_compat_reasoning_only"
    assert result.meta["compat_message"] == "reasoning-only response"


def test_build_diff_from_changes_is_patch_apply_compatible(tmp_path) -> None:
    target = tmp_path / "buggy.py"
    before = "def f():\n\n    return 1\n"
    after = "def f():\n\n    return 2\n"
    target.write_text(before, encoding="utf-8")

    diff = _build_diff_from_changes([("buggy.py", before, after)])
    assert "diff --git a/buggy.py b/buggy.py" in diff

    applied, message = run_eval.apply_diff(tmp_path, diff)
    assert applied, message
    assert target.read_text(encoding="utf-8") == after


def test_openai_full_file_mode_builds_local_diff_from_single_source_file(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {"choices": [{"message": {"content": "def f() -> int:\n    return 2\n"}}]},
            {"llm_call_count": 1},
        )

    monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat_completion)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="full_file",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

    assert result.diff is not None
    assert "diff --git a/buggy.py b/buggy.py" in result.diff
    assert "+    return 2" in result.diff
    assert result.meta is not None
    assert result.meta["response_mode"] == "full_file"
    assert result.meta["full_file_target_path"] == "buggy.py"
    assert result.meta["full_file_supported"] is True
    assert result.meta["full_file_support_reason"] == "single_non_test_python_file"
    assert result.meta["full_file_output_chars"] == len("def f() -> int:\n    return 2\n")
    assert result.meta["full_file_code_fence_stripped"] is False


def test_openai_full_file_mode_extracts_fenced_code_with_preamble(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Here is the fixed file:\n"
                                "```python\n"
                                "def f() -> int:\n"
                                "    return 2\n"
                                "```\n"
                                "This is the full file."
                            )
                        }
                    }
                ]
            },
            {"llm_call_count": 1},
        )

    monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat_completion)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="full_file",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

    assert result.diff is not None
    assert "+    return 2" in result.diff
    assert "Here is the fixed file" not in result.diff
    assert result.meta is not None
    assert result.meta["full_file_code_fence_stripped"] is True


def test_openai_full_file_mode_plain_unified_diff_falls_back_to_diff(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )
    plain_diff = (
        "--- buggy.py\n"
        "+++ buggy.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {"choices": [{"message": {"content": plain_diff}}]},
            {"llm_call_count": 1},
        )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={"llm_call_count": 1},
        )

    monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat_completion)
    monkeypatch.setattr(openai_mod, "_propose_patch_via_diff", _fake_diff)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="full_file",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta["fallback_from"] == "full_file"
    assert result.meta["fallback_reason"] == "returned_diff_instead_of_file"
    assert "+--- buggy.py" not in result.diff


def test_openai_full_file_mode_truncated_output_falls_back_to_diff(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text(
        "def f() -> int:\n    first = 1\n    second = 2\n    return first + second\n",
        encoding="utf-8",
    )
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )
    calls = {"count": 0}

    def _fake_chat_completion(**kwargs):
        del kwargs
        calls["count"] += 1
        return (
            {
                "choices": [
                    {
                        "message": {"content": "def f() -> int:\n    first = 1\n    second = 2\n"},
                        "finish_reason": "length",
                    }
                ]
            },
            {"llm_call_count": 1},
        )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,4 +1,4 @@\n"
                " def f() -> int:\n"
                "     first = 1\n"
                "     second = 2\n"
                "-    return first + second\n"
                "+    return second + second\n"
            ),
            meta={"llm_call_count": 1},
        )

    monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat_completion)
    monkeypatch.setattr(openai_mod, "_propose_patch_via_diff", _fake_diff)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="full_file",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

    assert calls["count"] == 1
    assert result.diff is not None
    assert result.meta is not None
    assert result.meta["fallback_from"] == "full_file"
    assert result.meta["fallback_reason"] == "truncated_full_file_response"
    assert result.meta["fallback_trigger"]["finish_reason"] == "length"


def test_openai_full_file_mode_identical_output_falls_back_to_diff(tmp_path, monkeypatch) -> None:
    before = "def f() -> int:\n    return 1\n"
    (tmp_path / "buggy.py").write_text(before, encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {"choices": [{"message": {"content": before}}]},
            {"llm_call_count": 1},
        )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={"llm_call_count": 1},
        )

    monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat_completion)
    monkeypatch.setattr(openai_mod, "_propose_patch_via_diff", _fake_diff)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="full_file",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta["fallback_from"] == "full_file"
    assert result.meta["fallback_reason"] == "no_effective_changes"
    assert result.meta["full_file_supported"] is True
    assert result.meta["full_file_target_path"] == "buggy.py"
    assert result.meta["full_file_output_chars"] == len(before)


def test_openai_full_file_mode_unsupported_case_shape_falls_back_to_diff(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/helper.py b/helper.py\n"
                "--- a/helper.py\n"
                "+++ b/helper.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def value() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={"llm_call_count": 1},
        )

    monkeypatch.setattr(openai_mod, "_propose_patch_via_diff", _fake_diff)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="full_file",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(prompt="fix helper.py", context_dir=tmp_path, cwd=tmp_path)

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta["fallback_from"] == "full_file"
    assert result.meta["fallback_reason"] == "unsupported_case_shape"
    assert result.meta["full_file_supported"] is False
    assert result.meta["full_file_support_reason"] == "multiple_non_test_python_files"
    assert result.meta["full_file_target_path"] == ""


@pytest.mark.asyncio
async def test_openai_full_file_mode_async_truncated_output_falls_back_to_diff(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "buggy.py").write_text(
        "def f() -> int:\n    first = 1\n    second = 2\n    return first + second\n",
        encoding="utf-8",
    )
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )
    calls = {"count": 0}

    async def _fake_chat_completion_async(**kwargs):
        del kwargs
        calls["count"] += 1
        return (
            {
                "choices": [
                    {
                        "message": {"content": "def f() -> int:\n    first = 1\n    second = 2\n"},
                        "finish_reason": "length",
                    }
                ]
            },
            {"llm_call_count": 1},
        )

    async def _fake_diff_async(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,4 +1,4 @@\n"
                " def f() -> int:\n"
                "     first = 1\n"
                "     second = 2\n"
                "-    return first + second\n"
                "+    return second + second\n"
            ),
            meta={"llm_call_count": 1},
        )

    monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat_completion_async)
    monkeypatch.setattr(openai_mod, "_propose_patch_via_diff_async", _fake_diff_async)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="full_file",
        force_patch_attempt=False,
    )
    result = await patcher.apropose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

    assert calls["count"] == 1
    assert result.diff is not None
    assert result.meta is not None
    assert result.meta["fallback_from"] == "full_file"
    assert result.meta["fallback_reason"] == "truncated_full_file_response"
    assert result.meta["fallback_trigger"]["finish_reason"] == "length"


def test_base_url_from_chat_url_supports_non_v1_paths() -> None:
    assert _base_url_from_chat_url("https://api.z.ai/api/coding/paas/v4/chat/completions") == (
        "https://api.z.ai/api/coding/paas/v4"
    )
    assert _base_url_from_chat_url("http://127.0.0.1:1234/v1/chat/completions") == (
        "http://127.0.0.1:1234/v1"
    )


def test_context_exceeded_http_error_detects_lmstudio_phrase() -> None:
    assert openai_mod._is_context_exceeded_http_error(
        {
            "error": "http_error",
            "body": (
                "The number of tokens to keep from the initial prompt is greater than "
                "the context length."
            ),
        }
    )


def test_context_exceeded_http_error_detects_generic_context_length_phrase() -> None:
    assert openai_mod._is_context_exceeded_http_error(
        {
            "error": "http_error",
            "body": "Requested prompt is greater than the context length for this model.",
        }
    )


def test_probe_context_window_tokens_prefers_lmstudio_v1_loaded_instance(monkeypatch) -> None:
    """Verifies that when an LM Studio v1 loaded instance is present, its context_length is returned in preference to the model's max_context_length, with the correct provider and source fields in meta."""

    def _fake_get_json(*, url: str, api_key: str, timeout_sec: float, http2: bool):
        del api_key, timeout_sec, http2
        if url.endswith("/api/v1/models"):
            return (
                {
                    "models": [
                        {
                            "key": "dummy",
                            "loaded_instances": [
                                {"id": "dummy", "config": {"context_length": 32768}}
                            ],
                            "max_context_length": 65536,
                        }
                    ]
                },
                {},
            )
        raise AssertionError(f"unexpected probe url: {url}")

    monkeypatch.setattr(openai_mod, "_http_get_json", _fake_get_json)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
    )
    tokens, meta = patcher.probe_context_window_tokens()

    assert tokens == 32768
    assert meta["context_window_probe_provider"] == "lmstudio_api_v1"
    assert meta["context_window_source"] == "loaded_instance.context_length"
    assert meta["context_window_probe_instance_id"] == "dummy"


def test_probe_context_window_tokens_falls_back_to_lmstudio_v0(monkeypatch) -> None:
    """Verifies that a 404 from the v1 endpoint causes a fallback to v0, and that the failed v1 attempt is recorded in the probe attempt history list."""

    def _fake_get_json(*, url: str, api_key: str, timeout_sec: float, http2: bool):
        del api_key, timeout_sec, http2
        if url.endswith("/api/v1/models"):
            return None, {"error": "http_error", "status": 404, "body": "missing"}
        if url.endswith("/api/v0/models"):
            return (
                {
                    "data": [
                        {
                            "id": "dummy",
                            "max_context_length": 16384,
                        }
                    ]
                },
                {},
            )
        raise AssertionError(f"unexpected probe url: {url}")

    monkeypatch.setattr(openai_mod, "_http_get_json", _fake_get_json)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
    )
    tokens, meta = patcher.probe_context_window_tokens()

    assert tokens == 16384
    assert meta["context_window_probe_provider"] == "lmstudio_api_v0"
    assert meta["context_window_source"] == "max_context_length"
    attempts = meta["context_window_probe_attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["provider"] == "lmstudio_api_v1"
    assert attempts[0]["error"] == "http_error"


def test_probe_context_window_tokens_reports_probe_failure(monkeypatch) -> None:
    """Verifies that when all probe endpoints fail, tokens is None and the error detail encodes each attempted provider's failure reason in the probe error string."""

    def _fake_get_json(*, url: str, api_key: str, timeout_sec: float, http2: bool):
        del api_key, timeout_sec, http2
        return None, {"error": "request_failed", "exc": f"failed:{url}"}

    monkeypatch.setattr(openai_mod, "_http_get_json", _fake_get_json)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
    )
    tokens, meta = patcher.probe_context_window_tokens()

    assert tokens is None
    assert meta["error"] == "context_window_probe_failed"
    assert "lmstudio_api_v1:request_failed" in str(meta["context_window_probe_error"])


def test_probe_context_window_tokens_skips_duplicate_openai_models_url(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_get_json(*, url: str, api_key: str, timeout_sec: float, http2: bool):
        del api_key, timeout_sec, http2
        calls.append(url)
        return None, {"error": "request_failed"}

    monkeypatch.setattr(openai_mod, "_http_get_json", _fake_get_json)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234/api/v1",
        model="dummy",
    )
    tokens, meta = patcher.probe_context_window_tokens()

    assert tokens is None
    assert calls == [
        "http://127.0.0.1:1234/api/v1/models",
        "http://127.0.0.1:1234/api/v0/models",
    ]
    assert meta["error"] == "context_window_probe_failed"


def test_http_get_json_request_failures_return_none(monkeypatch) -> None:
    import httpx

    class _BoomTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise RuntimeError("boom")

    _original_client = httpx.Client

    def _boom_client(**kwargs: object) -> httpx.Client:
        kwargs.pop("http2", None)
        return _original_client(transport=_BoomTransport(), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(openai_mod.httpx, "Client", _boom_client)

    payload, meta = openai_mod._http_get_json(
        url="http://127.0.0.1:1234/v1/models",
        api_key="dummy",
        timeout_sec=1.0,
        http2=False,
    )

    assert payload is None
    assert meta["error"] == "request_failed"


def test_probe_server_ready_prefers_health(monkeypatch) -> None:
    def _fake_probe_ready(*, url: str, api_key: str, timeout_sec: float, http2: bool):
        del api_key, timeout_sec, http2
        if url.endswith("/health"):
            return True, {"status": 200, "transport_http_version": "HTTP/1.1"}
        raise AssertionError(f"unexpected probe url: {url}")

    monkeypatch.setattr(openai_mod, "_http_probe_ready", _fake_probe_ready)

    patcher = OpenAICompatPatcher(base_url="http://127.0.0.1:1234/v1", model="dummy")
    ready, meta = patcher.probe_server_ready()

    assert ready is True
    assert meta["server_ready"] is True
    assert meta["server_ready_probe_provider"] == "health"


def test_probe_server_ready_falls_back_to_models(monkeypatch) -> None:
    def _fake_probe_ready(*, url: str, api_key: str, timeout_sec: float, http2: bool):
        del api_key, timeout_sec, http2
        if url.endswith("/health"):
            return False, {"error": "http_error", "status": 503}
        if url.endswith("/v1/models"):
            return True, {"status": 200}
        raise AssertionError(f"unexpected probe url: {url}")

    monkeypatch.setattr(openai_mod, "_http_probe_ready", _fake_probe_ready)

    patcher = OpenAICompatPatcher(base_url="http://127.0.0.1:1234/v1", model="dummy")
    ready, meta = patcher.probe_server_ready()

    assert ready is True
    assert meta["server_ready_probe_provider"] == "openai_models"
    assert meta["server_ready_probe_attempts"] == [
        {
            "provider": "health",
            "url": "http://127.0.0.1:1234/health",
            "error": "http_error",
            "status": 503,
        }
    ]


def test_probe_server_ready_reports_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        openai_mod,
        "_http_probe_ready",
        lambda **_kwargs: (False, {"error": "request_failed", "status": None}),
    )

    patcher = OpenAICompatPatcher(base_url="http://127.0.0.1:1234/v1", model="dummy")
    ready, meta = patcher.probe_server_ready()

    assert ready is False
    assert meta["error"] == "server_ready_probe_failed"
    assert "health:request_failed" in str(meta["server_ready_probe_error"])


def test_normalize_unified_blocks_preserves_paths_with_spaces() -> None:
    """Verifies that tab-delimited timestamp suffixes on --- / +++ headers are stripped and a correct git-style diff --git header is produced, even when file paths contain spaces."""
    lines = [
        "--- a/buggy file.py\t2026-02-23 20:00:00 +0000",
        "+++ b/buggy file.py\t2026-02-23 20:00:00 +0000",
        "@@ -1 +1 @@",
        "-return 1",
        "+return 2",
    ]

    normalized = openai_mod._normalize_unified_blocks(lines)

    assert normalized is not None
    assert "diff --git a/buggy file.py b/buggy file.py" in normalized
    assert "--- a/buggy file.py" in normalized
    assert "+++ b/buggy file.py" in normalized


def test_openai_patcher_prefers_z_api_env_for_missing_key(tmp_path, monkeypatch) -> None:
    """Verifies that Z_API is used as the fallback API key when no explicit key is provided and neither OPENAI_API_KEY nor LLMDEBUG_EVAL_OPENAI_API_KEY is set in the environment."""
    captured: dict[str, str] = {}

    def _fake_diff(**kwargs) -> PatchResult:
        captured["api_key"] = str(kwargs.get("api_key") or "")
        return PatchResult(diff=None, meta={"error": "no_diff_found"})

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        _fake_diff,
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_edits",
        lambda **_kwargs: PatchResult(diff=None, meta={"error": "no_json_payload"}),
    )
    monkeypatch.setenv("Z_API", "z-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMDEBUG_EVAL_OPENAI_API_KEY", raising=False)

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=False,
        api_key=None,
    )
    _ = patcher.propose_patch(
        prompt="Pytest output (stdout):\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )
    assert captured["api_key"] == "z-key"


def test_openai_edits_fallback_aggregates_call_and_token_usage(tmp_path, monkeypatch) -> None:
    """Verifies that when the edits strategy fails and falls back to diff, the llm_call_count and all token-usage counters from both strategies are summed in the final result meta."""

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {"choices": [{"message": {"content": "not json"}}]},
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            ),
            meta={
                "llm_call_count": 2,
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        _fake_diff,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output (stdout):\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert int(result.meta["llm_call_count"]) == 5
    usage = result.meta.get("usage")
    assert isinstance(usage, dict)
    assert int(usage.get("prompt_tokens") or 0) == 26
    assert int(usage.get("completion_tokens") or 0) == 10
    assert int(usage.get("total_tokens") or 0) == 36


def test_request_context_retry_aggregates_usage_when_response_format_is_rejected(
    tmp_path, monkeypatch
) -> None:
    """Verifies that a response_format-rejected first attempt triggers a format-free retry that succeeds, and that token usage from both calls is summed in the final meta."""
    calls = {"count": 0}

    def _fake_chat_completion(**kwargs):
        del kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                None,
                {
                    "error": "http_error",
                    "body": "response_format not supported",
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_more_context":false,"requests":[],'
                                + '"sufficiency_rationale":"ok"}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = patcher.request_context(prompt="x", context_dir=tmp_path, cwd=tmp_path)
    assert result is not None
    assert result.meta is not None
    assert int(result.meta["llm_call_count"]) == 2
    usage = result.meta.get("usage")
    assert isinstance(usage, dict)
    assert int(usage.get("prompt_tokens") or 0) == 5
    assert int(usage.get("completion_tokens") or 0) == 1
    assert int(usage.get("total_tokens") or 0) == 6


def test_request_context_transport_reissue_records_attempt_history(tmp_path, monkeypatch) -> None:
    """Verifies that a transient 503 backend error triggers a transport retry, and that the attempt list captures each attempt's outcome and failure_kind in order."""
    calls = {"count": 0}

    def _fake_chat_completion(**kwargs):
        del kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                None,
                {
                    "error": "http_error",
                    "status": 503,
                    "body": "backend temporarily unavailable",
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_more_context":false,"requests":[],'
                                '"sufficiency_rationale":"ok"}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = patcher.request_context(prompt="x", context_dir=tmp_path, cwd=tmp_path)

    assert result is not None
    assert result.request_json is not None
    assert result.meta is not None
    assert result.meta["request_context_outcome"] == "valid_payload"
    assert result.meta["request_context_effective_success"] is True
    assert int(result.meta["request_context_retry_count"]) == 1
    assert int(result.meta["request_context_transport_retry_count"]) == 1
    attempts = result.meta.get("request_context_attempts")
    assert isinstance(attempts, list)
    assert attempts[0]["outcome"] == "unavailable_backend"
    assert attempts[0]["failure_kind"] == "backend_unavailable"
    assert attempts[1]["outcome"] == "valid_payload"


def test_request_context_strict_json_retry_marks_invalid_payload_attempt(
    tmp_path, monkeypatch
) -> None:
    """Verifies that a non-JSON first response is recorded as an invalid_payload attempt and the subsequent strict-JSON retry uses an augmented system prompt containing the strict-JSON instruction."""
    calls: list[dict[str, object]] = []

    def _fake_chat_completion(**kwargs):
        messages = kwargs.get("messages")
        system_prompt = ""
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                system_prompt = str(first.get("content") or "")
        calls.append({"system_prompt": system_prompt})
        if len(calls) == 1:
            return (
                {"choices": [{"message": {"content": "not-json"}}]},
                {
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_more_context":false,"requests":[],'
                                '"sufficiency_rationale":"ok"}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = patcher.request_context(prompt="x", context_dir=tmp_path, cwd=tmp_path)

    assert result is not None
    assert result.request_json is not None
    assert result.meta is not None
    assert result.meta["request_context_outcome"] == "valid_payload"
    assert result.meta["request_context_effective_success"] is True
    assert int(result.meta["request_context_retry_count"]) == 1
    assert int(result.meta["request_context_strict_retry_count"]) == 1
    attempts = result.meta.get("request_context_attempts")
    assert isinstance(attempts, list)
    assert attempts[0]["outcome"] == "invalid_payload"
    assert attempts[0]["failure_kind"] == "invalid_payload"
    assert attempts[1]["strict_json"] is True
    assert "If unsure, return a valid object" in str(calls[1]["system_prompt"])


def test_request_context_strict_json_first_starts_in_strict_mode(tmp_path, monkeypatch) -> None:
    """Verifies that when strict_json_first=True the very first call already uses the strict-JSON system prompt, with zero strict retry count and strict_json=True on the initial attempt record."""
    calls: list[dict[str, object]] = []

    def _fake_chat_completion(**kwargs):
        messages = kwargs.get("messages")
        system_prompt = ""
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                system_prompt = str(first.get("content") or "")
        calls.append({"system_prompt": system_prompt})
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_more_context":false,"requests":[],'
                                '"sufficiency_rationale":"ok"}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = patcher.request_context(
        prompt="x",
        context_dir=tmp_path,
        cwd=tmp_path,
        strict_json_first=True,
    )

    assert result is not None
    assert result.request_json is not None
    assert result.meta is not None
    assert result.meta["request_context_strict_json_first"] is True
    assert int(result.meta["request_context_strict_retry_count"]) == 0
    attempts = result.meta.get("request_context_attempts")
    assert isinstance(attempts, list)
    assert attempts[0]["strict_json"] is True
    assert "If unsure, return a valid object" in str(calls[0]["system_prompt"])


@pytest.mark.asyncio
async def test_arequest_context_transport_reissue_records_attempt_history(
    tmp_path, monkeypatch
) -> None:
    calls = {"count": 0}

    async def _fake_chat_completion_async(**kwargs):
        del kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                None,
                {
                    "error": "http_error",
                    "status": 503,
                    "body": "backend temporarily unavailable",
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_more_context":false,"requests":[],'
                                '"sufficiency_rationale":"ok"}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion_async",
        _fake_chat_completion_async,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = await patcher.arequest_context(prompt="x", context_dir=tmp_path, cwd=tmp_path)

    assert result is not None
    assert result.request_json is not None
    assert result.meta is not None
    assert result.meta["request_context_outcome"] == "valid_payload"
    assert result.meta["request_context_effective_success"] is True
    assert int(result.meta["request_context_retry_count"]) == 1
    assert int(result.meta["request_context_transport_retry_count"]) == 1
    attempts = result.meta.get("request_context_attempts")
    assert isinstance(attempts, list)
    assert attempts[0]["outcome"] == "unavailable_backend"
    assert attempts[1]["outcome"] == "valid_payload"


@pytest.mark.asyncio
async def test_arequest_context_strict_json_retry_marks_invalid_payload_attempt(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    async def _fake_chat_completion_async(**kwargs):
        messages = kwargs.get("messages")
        system_prompt = ""
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                system_prompt = str(first.get("content") or "")
        calls.append({"system_prompt": system_prompt})
        if len(calls) == 1:
            return (
                {"choices": [{"message": {"content": "not-json"}}]},
                {
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_more_context":false,"requests":[],'
                                '"sufficiency_rationale":"ok"}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion_async",
        _fake_chat_completion_async,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = await patcher.arequest_context(prompt="x", context_dir=tmp_path, cwd=tmp_path)

    assert result is not None
    assert result.request_json is not None
    assert result.meta is not None
    assert result.meta["request_context_outcome"] == "valid_payload"
    assert result.meta["request_context_effective_success"] is True
    assert int(result.meta["request_context_retry_count"]) == 1
    assert int(result.meta["request_context_transport_retry_count"]) == 0
    assert int(result.meta["request_context_strict_retry_count"]) == 1
    attempts = result.meta.get("request_context_attempts")
    assert isinstance(attempts, list)
    assert attempts[0]["outcome"] == "invalid_payload"
    assert attempts[0]["failure_kind"] == "invalid_payload"
    assert attempts[1]["strict_json"] is True
    assert "If unsure, return a valid object" in str(calls[1]["system_prompt"])


@pytest.mark.asyncio
async def test_arequest_context_strict_json_first_starts_in_strict_mode(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    async def _fake_chat_completion_async(**kwargs):
        messages = kwargs.get("messages")
        system_prompt = ""
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                system_prompt = str(first.get("content") or "")
        calls.append({"system_prompt": system_prompt})
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"need_more_context":false,"requests":[],'
                                '"sufficiency_rationale":"ok"}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion_async",
        _fake_chat_completion_async,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = await patcher.arequest_context(
        prompt="x",
        context_dir=tmp_path,
        cwd=tmp_path,
        strict_json_first=True,
    )

    assert result is not None
    assert result.request_json is not None
    assert result.meta is not None
    assert result.meta["request_context_strict_json_first"] is True
    assert int(result.meta["request_context_strict_retry_count"]) == 0
    attempts = result.meta.get("request_context_attempts")
    assert isinstance(attempts, list)
    assert attempts[0]["strict_json"] is True
    assert "If unsure, return a valid object" in str(calls[0]["system_prompt"])


def test_compact_context_retry_aggregates_usage_when_response_format_is_rejected(
    tmp_path, monkeypatch
) -> None:
    calls = {"count": 0}

    def _fake_chat_completion(**kwargs):
        del kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return (
                None,
                {
                    "error": "http_error",
                    "body": "response_format not supported",
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": ('{"compacted_context":"A compact summary with key facts"}')
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
    )
    result = patcher.compact_context(
        prompt="compact",
        context_text="very long context",
        context_dir=tmp_path,
        cwd=tmp_path,
        target_chars=20,
    )
    assert result is not None
    assert result.compacted_text == "A compact summary wi"
    assert result.meta is not None
    assert int(result.meta["llm_call_count"]) == 2
    usage = result.meta.get("usage")
    assert isinstance(usage, dict)
    assert int(usage.get("prompt_tokens") or 0) == 6
    assert int(usage.get("completion_tokens") or 0) == 1
    assert int(usage.get("total_tokens") or 0) == 7


def test_normalize_context_compaction_payload_uses_fallback_fields() -> None:
    compacted, raw = openai_mod._normalize_context_compaction_payload(
        {
            "choices": [
                {"message": {"content": '{"summary":"keep traceback file and assertion text"}'}}
            ]
        }
    )
    assert raw is not None
    assert compacted == "keep traceback file and assertion text"


def test_normalize_context_request_payload_parses_string_false() -> None:
    payload, raw = openai_mod._normalize_context_request_payload(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"need_more_context":"false","requests":[],'
                            '"sufficiency_rationale":"enough"}'
                        )
                    }
                }
            ]
        }
    )
    assert raw is not None
    assert payload is not None
    assert payload["need_more_context"] is False


def test_normalize_context_request_payload_preserves_diagnosis_and_missing_fact() -> None:
    payload, raw = openai_mod._normalize_context_request_payload(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"diagnosis":"root cause",'
                            '"missing_fact":null,'
                            '"need_more_context":false,'
                            '"requests":[]}'
                        )
                    }
                }
            ]
        }
    )
    assert raw is not None
    assert payload is not None
    assert payload["diagnosis"] == "root cause"
    assert "missing_fact" in payload
    assert payload["missing_fact"] is None
    assert payload["need_more_context"] is False


def test_normalize_context_request_payload_preserves_zero_index_requests() -> None:
    payload, raw = openai_mod._normalize_context_request_payload(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"need_more_context":true,'
                            '"requests":[{"type":"frame","index":0},{"type":"local_var","frame":0,"name":"value"}]}'
                        )
                    }
                }
            ]
        }
    )

    assert raw is not None
    assert payload is not None
    assert payload["requests"] == [
        {"type": "frame", "index": 0},
        {"type": "local_var", "frame": 0, "name": "value"},
    ]


def test_normalize_context_request_payload_salvages_valid_requests() -> None:
    payload, raw = openai_mod._normalize_context_request_payload(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"need_more_context":true,'
                            '"requests":['
                            '{"type":"unknown","path":"buggy.py"},'
                            '{"type":"frame","index":0}'
                            "]}"
                        )
                    }
                }
            ]
        }
    )

    assert raw is not None
    assert payload is not None
    assert payload["requests"] == [{"type": "frame", "index": 0}]


def test_normalize_context_request_payload_fails_closed_and_drops_invalid_entries() -> None:
    payload, raw = openai_mod._normalize_context_request_payload(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"need_more_context":"maybe",'
                            '"need_snapshot":"off",'
                            '"need_compact_context":"off",'
                            '"requests":['
                            '{"type":"frame","index":"abc"},'
                            '{"type":"local_var","frame":"bad","name":"value"},'
                            '{"type":"file","path":"buggy.py"}'
                            "],"
                            '"sufficiency_rationale":"ok"}'
                        )
                    }
                }
            ]
        }
    )

    assert raw is not None
    assert payload is not None
    assert payload["need_more_context"] is False
    assert payload["need_snapshot"] is False
    assert payload["need_compact_context"] is False
    assert payload["requests"] == [{"type": "file", "path": "buggy.py"}]


def test_openai_diff_retry_does_not_prefill_assistant(tmp_path, monkeypatch) -> None:
    captured_messages: list[list[dict[str, str]]] = []

    def _fake_chat_completion(**kwargs):
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            captured_messages.append(messages)
        return (
            None,
            {
                "error": "http_error",
                "body": "upstream transient",
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is None
    assert len(captured_messages) == 2
    assert all(
        all(isinstance(message, dict) and message.get("role") != "assistant" for message in call)
        for call in captured_messages
    )
    second_system = captured_messages[1][0]
    assert "Start your response with 'diff --git a/'." in str(second_system.get("content"))


def test_openai_diff_no_diff_found_falls_back_to_edits(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=None,
            meta={
                "error": "no_diff_found",
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    def _fake_edits(**kwargs) -> PatchResult:
        assert kwargs.get("allow_diff_fallback") is False
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f():\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={
                "structured_output_mode": "json_schema",
                "llm_call_count": 2,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    monkeypatch.setattr("evals.patchers.openai_compat_patcher._propose_patch_via_diff", _fake_diff)
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_edits", _fake_edits
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="diff",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("fallback_from") == "diff"
    assert result.meta.get("fallback_reason") == "no_diff_found"
    fallback_meta = result.meta.get("fallback_meta")
    assert isinstance(fallback_meta, dict)
    assert fallback_meta.get("error") == "no_diff_found"
    assert int(result.meta.get("llm_call_count") or 0) == 3
    usage = result.meta.get("usage")
    assert isinstance(usage, dict)
    assert int(usage.get("prompt_tokens") or 0) == 8
    assert int(usage.get("completion_tokens") or 0) == 3
    assert int(usage.get("total_tokens") or 0) == 11


def test_openai_edits_fallback_sequence_json_schema_json_object_plain(
    tmp_path, monkeypatch
) -> None:
    call_extras: list[object] = []
    calls = {"count": 0}

    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        calls["count"] += 1
        call_extras.append(kwargs.get("extra_payload"))
        if calls["count"] == 1:
            return (
                None,
                {
                    "error": "http_error",
                    "body": "json_schema is not supported",
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
                },
            )
        if calls["count"] == 2:
            return (
                None,
                {
                    "error": "http_error",
                    "body": "response_format not supported",
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"return 1",'
                                '"replace":"return 2","count":1}]}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert len(call_extras) == 3
    assert isinstance(call_extras[0], dict)
    assert call_extras[0]["response_format"]["type"] == "json_schema"
    assert isinstance(call_extras[0]["response_format"]["json_schema"], dict)
    assert call_extras[0]["response_format"]["json_schema"]["name"] == "patch_edits"
    assert isinstance(call_extras[1], dict)
    assert call_extras[1]["response_format"]["type"] == "json_object"
    assert call_extras[2] is None
    assert result.meta is not None
    assert result.meta.get("structured_output_mode") == "plain"
    assert int(result.meta.get("llm_call_count") or 0) == 3
    attempts = result.meta.get("structured_output_attempts")
    assert isinstance(attempts, list)
    assert [entry.get("mode") for entry in attempts] == ["json_schema", "json_object", "plain"]

    applied, message = run_eval.apply_diff(tmp_path, result.diff)
    assert applied, message
    assert (tmp_path / "buggy.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_openai_edits_records_reasoning_content_metadata(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"return 1",'
                                '"replace":"return 2","count":1}]}'
                            ),
                            "reasoning_content": "step by step reasoning",
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("reasoning_content_present") is True
    assert int(result.meta.get("reasoning_content_chars") or 0) > 0


def test_openai_edits_content_parts_ignore_reasoning_chunks(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "reasoning", "text": "internal thoughts"},
                                {
                                    "type": "output_text",
                                    "text": (
                                        '{"edits":[{"path":"buggy.py","find":"return 1",'
                                        '"replace":"return 2","count":1}]}'
                                    ),
                                },
                            ],
                            "reasoning_content": "internal thoughts",
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    applied, message = run_eval.apply_diff(tmp_path, result.diff)
    assert applied, message
    assert (tmp_path / "buggy.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_openai_edits_parse_failure_advances_to_next_stage(tmp_path, monkeypatch) -> None:
    call_extras: list[object] = []
    calls = {"count": 0}
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        calls["count"] += 1
        call_extras.append(kwargs.get("extra_payload"))
        if calls["count"] == 1:
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"edits": [ {"path":"buggy.py","find":"return 1"',
                            }
                        }
                    ]
                },
                {
                    "llm_call_count": 1,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                },
            )
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"return 1",'
                                '"replace":"return 2","count":1}]}'
                            ),
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert len(call_extras) >= 2
    assert isinstance(call_extras[0], dict)
    assert call_extras[0]["response_format"]["type"] == "json_schema"
    assert isinstance(call_extras[0]["response_format"]["json_schema"], dict)
    assert call_extras[0]["response_format"]["json_schema"]["name"] == "patch_edits"
    assert isinstance(call_extras[1], dict)
    assert call_extras[1]["response_format"]["type"] == "json_object"
    assert result.meta is not None
    assert result.meta.get("structured_output_mode") == "json_object"
    attempts = result.meta.get("structured_output_attempts")
    assert isinstance(attempts, list)
    assert attempts[0].get("parse_error") == "no_json_payload"


def test_openai_edits_accepts_single_edit_root_shape(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"path":"buggy.py","find":"return 1","replace":"return 2","count":1}'
                            ),
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("edits_payload_shape") == "single_edit_root"
    applied, message = run_eval.apply_diff(tmp_path, result.diff)
    assert applied, message
    assert (tmp_path / "buggy.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_openai_edits_find_recovery_normalized_whitespace(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_bytes(b"def f():\r\n    value = 1   \r\n    return value\r\n")

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"def f():\\n    value = 1\\n'
                                '    return value\\n","replace":"def f():\\n    value = 2\\n'
                                '    return value\\n","count":1}]}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("find_recovery_applied") is True
    assert result.meta.get("find_recovery_mode") in {"normalized_newlines", "normalized_whitespace"}
    applied, message = run_eval.apply_diff(tmp_path, result.diff)
    assert applied, message
    assert "value = 2" in (tmp_path / "buggy.py").read_text(encoding="utf-8")


def test_openai_edits_find_recovery_fuzzy_block(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text(
        "def f(res, n):\n    return res // (n * (n - 1))\n",
        encoding="utf-8",
    )

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"return res // n * (n - 1)",'
                                '"replace":"return res // (n * (n + 1))","count":1}]}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    def _unexpected_diff(**kwargs) -> PatchResult:
        del kwargs
        raise AssertionError("diff fallback should not be used when fuzzy recovery succeeds")

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        _unexpected_diff,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("find_recovery_applied") is True
    assert result.meta.get("find_recovery_mode") == "fuzzy_block"
    applied, message = run_eval.apply_diff(tmp_path, result.diff)
    assert applied, message
    assert "return res // (n * (n + 1))" in (tmp_path / "buggy.py").read_text(encoding="utf-8")


def test_openai_edits_fuzzy_recovery_skips_replace_all_count(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text(
        "def f(res, n):\n    return res // (n * (n - 1))\n",
        encoding="utf-8",
    )

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"return res // n * (n - 1)",'
                                '"replace":"return res // (n * (n + 1))","count":0}]}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f(res, n):\n"
                "-    return res // (n * (n - 1))\n"
                "+    return res // (n * (n + 1))\n"
            ),
            meta={
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        _fake_diff,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("fallback_reason") == "find_not_found"
    applied, message = run_eval.apply_diff(tmp_path, result.diff)
    assert applied, message
    assert "return res // (n * (n + 1))" in (tmp_path / "buggy.py").read_text(encoding="utf-8")


def test_openai_edits_find_not_found_falls_back_to_diff(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"does_not_exist",'
                                '"replace":"return 2","count":1}]}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f():\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        _fake_diff,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("fallback_reason") == "find_not_found"


def test_openai_edits_no_effective_changes_falls_back_to_diff(tmp_path, monkeypatch) -> None:
    (tmp_path / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _fake_chat_completion(**kwargs):
        del kwargs
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"edits":[{"path":"buggy.py","find":"return 1",'
                                '"replace":"return 1","count":1}]}'
                            )
                        }
                    }
                ]
            },
            {
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    def _fake_diff(**kwargs) -> PatchResult:
        del kwargs
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f():\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._chat_completion",
        _fake_chat_completion,
    )
    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_diff",
        _fake_diff,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n", context_dir=tmp_path, cwd=tmp_path
    )

    assert result.diff is not None
    assert result.meta is not None
    assert result.meta.get("fallback_reason") == "no_effective_changes"


def test_chat_http_client_cache_is_keyed_by_http2(monkeypatch) -> None:
    monkeypatch.setattr(openai_mod, "_THREAD_HTTP_CLIENTS", threading.local())

    client_h2_a = openai_mod._get_thread_http_client(
        url="http://127.0.0.1:1234/v1/chat/completions",
        http2=True,
    )
    client_h2_b = openai_mod._get_thread_http_client(
        url="http://127.0.0.1:1234/v1/chat/completions",
        http2=True,
    )
    client_h1 = openai_mod._get_thread_http_client(
        url="http://127.0.0.1:1234/v1/chat/completions",
        http2=False,
    )

    assert client_h2_a is client_h2_b
    assert client_h2_a is not client_h1
    assert client_h2_a.http2 is True
    assert client_h1.http2 is False


def test_chat_completion_falls_back_to_http1_when_http2_protocol_errors(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, *, http2: bool) -> None:
            self.http2 = http2
            self.last_http_version = "HTTP/2" if http2 else "HTTP/1.1"

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            if self.http2:
                raise openai_mod.httpx.RemoteProtocolError("HTTP/2 stream reset")
            return (
                200,
                '{"choices":[{"message":{"content":"{\\"ok\\":true}"}}],'
                '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
            )

    calls: list[bool] = []

    def _fake_get_client(*, url: str, http2: bool) -> _FakeClient:
        del url
        calls.append(bool(http2))
        return _FakeClient(http2=http2)

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", _fake_get_client)

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert raw is not None
    assert calls == [True, False]
    assert meta.get("transport_http2_requested") is True
    assert meta.get("transport_http2_effective") is False
    assert meta.get("transport_http2_fallback_used") is True
    assert meta.get("transport_http_version") == "HTTP/1.1"


def test_openai_plan_patch_emits_retry_and_backoff_transport_progress(
    tmp_path, monkeypatch
) -> None:
    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0
            self.last_http_version = "HTTP/1.1"

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return (
                200,
                '{"choices":[{"message":{"content":"Minimal edit plan"}}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
            )

    client = _FlakyClient()
    sleep_calls: list[float] = []
    events: list[tuple[str, dict[str, object] | None]] = []

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", lambda *, url, http2: client)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    patcher = OpenAICompatPatcher(base_url="http://127.0.0.1:1234", model="dummy")
    patcher.set_progress_callback(lambda event, payload: events.append((event, payload)))

    result = patcher.plan_patch(prompt="fix bug", context_dir=tmp_path, cwd=tmp_path)

    assert result is not None
    assert result.plan_text == "Minimal edit plan"
    states = [
        str(payload["transport_state"])
        for event, payload in events
        if event == "openai_transport" and isinstance(payload, dict)
    ]
    assert states == [
        "request_started",
        "retry_scheduled",
        "backoff_sleep",
        "request_started",
    ]
    retry_payload = next(
        payload
        for event, payload in events
        if event == "openai_transport"
        and isinstance(payload, dict)
        and payload.get("transport_state") == "retry_scheduled"
    )
    assert retry_payload["owner"] == "openai"
    assert retry_payload["source"] == "transport"
    assert retry_payload["waiting_for"] == "openai_server"
    assert retry_payload["external_operation"] == "plan_patch"
    assert retry_payload["phase_label"] == "openai: plan patch"
    assert retry_payload["error"] == "request_failed"
    assert retry_payload["transport_retry_count"] == 1
    assert sleep_calls and sleep_calls[0] > 0.0


def test_openai_plan_patch_emits_http2_fallback_progress_once(tmp_path, monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, *, http2: bool) -> None:
            self.http2 = http2
            self.last_http_version = "HTTP/2" if http2 else "HTTP/1.1"

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            if self.http2:
                raise openai_mod.httpx.RemoteProtocolError("HTTP/2 stream reset")
            return (
                200,
                '{"choices":[{"message":{"content":"Minimal edit plan"}}],'
                '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
            )

    events: list[tuple[str, dict[str, object] | None]] = []

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(
        openai_mod,
        "_get_thread_http_client",
        lambda *, url, http2: _FakeClient(http2=http2),
    )

    patcher = OpenAICompatPatcher(base_url="http://127.0.0.1:1234", model="dummy")
    patcher.set_progress_callback(lambda event, payload: events.append((event, payload)))

    result = patcher.plan_patch(prompt="fix bug", context_dir=tmp_path, cwd=tmp_path)

    assert result is not None
    assert result.meta["transport_http2_fallback_used"] is True
    fallback_events = [
        payload
        for event, payload in events
        if event == "openai_transport"
        and isinstance(payload, dict)
        and payload.get("transport_state") == "http2_fallback"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0]["owner"] == "openai"
    assert fallback_events[0]["source"] == "transport"
    assert fallback_events[0]["external_operation"] == "plan_patch"


def test_chat_completion_emits_request_budget_exhausted_progress() -> None:
    events: list[tuple[str, dict[str, object] | None]] = []
    callback_token = openai_mod._OPENAI_PROGRESS_CALLBACK.set(
        lambda event, payload: events.append((event, payload))
    )
    payload_token = openai_mod._OPENAI_PROGRESS_BASE_PAYLOAD.set(
        {
            "owner": "openai",
            "source": "transport",
            "waiting_for": "openai_server",
            "external_operation": "plan_patch",
            "phase_label": "openai: plan patch",
        }
    )
    try:
        raw, meta = openai_mod._chat_completion(
            url="http://127.0.0.1:1234/v1/chat/completions",
            api_key="local",
            model="dummy",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_tokens=64,
            timeout_sec=30.0,
            seed=None,
            http2=True,
            deadline_monotonic=openai_mod.time.monotonic() - 1.0,
        )
    finally:
        openai_mod._OPENAI_PROGRESS_BASE_PAYLOAD.reset(payload_token)
        openai_mod._OPENAI_PROGRESS_CALLBACK.reset(callback_token)

    assert raw is None
    assert meta["error"] == "request_budget_exceeded"
    assert events == [
        (
            "openai_transport",
            {
                "owner": "openai",
                "source": "transport",
                "waiting_for": "openai_server",
                "external_operation": "plan_patch",
                "phase_label": "openai: plan patch",
                "note": "request budget exhausted",
                "transport_state": "request_budget_exhausted",
                "transport_attempt": 0,
                "transport_retry_count": 0,
            },
        )
    ]


def test_chat_completion_uses_proxy_httpx_transport_when_proxy_enabled(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, *, status_code: int, text: str, http_version: str) -> None:
            self.status_code = status_code
            self.text = text
            self.http_version = http_version

    class _FakeHTTPXClient:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.http2 = bool(kwargs.get("http2"))

        def __enter__(self) -> _FakeHTTPXClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def post(self, url: str, content: bytes, headers: dict[str, str]) -> _FakeResponse:
            del url, content, headers
            return _FakeResponse(
                status_code=200,
                text=(
                    '{"choices":[{"message":{"content":"{\\"ok\\":true}"}}],'
                    '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
                ),
                http_version="HTTP/2" if self.http2 else "HTTP/1.1",
            )

    def _unexpected_client(*, url: str, http2: bool):
        raise AssertionError(
            "persistent HTTP client should not be used when proxy transport is enabled "
            f"(url={url}, http2={http2})"
        )

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", _unexpected_client)
    monkeypatch.setattr(openai_mod.httpx, "Client", _FakeHTTPXClient)

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert raw is not None
    assert meta.get("error") is None
    assert meta.get("transport_http2_effective") is True
    assert meta.get("transport_http_version") == "HTTP/2"


def test_chat_completion_proxy_transport_retries_request_failed_then_succeeds(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, *, status_code: int, text: str, http_version: str = "HTTP/1.1") -> None:
            self.status_code = status_code
            self.text = text
            self.http_version = http_version

    state = {"calls": 0}
    sleep_calls: list[float] = []

    class _FlakyHTTPXClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self) -> _FlakyHTTPXClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def post(self, url: str, content: bytes, headers: dict[str, str]) -> _FakeResponse:
            del url, content, headers
            state["calls"] += 1
            if state["calls"] == 1:
                raise TimeoutError("proxy timeout")
            return _FakeResponse(
                status_code=200,
                text=(
                    '{"choices":[{"message":{"content":"{\\"ok\\":true}"}}],'
                    '"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}'
                ),
            )

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)
    monkeypatch.setattr(openai_mod.httpx, "Client", _FlakyHTTPXClient)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert raw is not None
    assert meta.get("error") is None
    assert int(meta.get("llm_call_count") or 0) == 2
    assert int(meta.get("transport_retry_count") or 0) == 1
    retry_attempts = meta.get("transport_retry_attempts")
    assert isinstance(retry_attempts, list)
    assert retry_attempts[0].get("error") == "request_failed"
    assert sleep_calls and sleep_calls[0] > 0.0


def test_chat_completion_proxy_transport_retries_http_503_then_succeeds(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, *, status_code: int, text: str, http_version: str = "HTTP/1.1") -> None:
            self.status_code = status_code
            self.text = text
            self.http_version = http_version

    state = {"calls": 0}
    sleep_calls: list[float] = []

    class _FlakyHTTPXClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self) -> _FlakyHTTPXClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

        def post(self, url: str, content: bytes, headers: dict[str, str]) -> _FakeResponse:
            del url, content, headers
            state["calls"] += 1
            if state["calls"] == 1:
                return _FakeResponse(status_code=503, text="upstream overloaded")
            return _FakeResponse(
                status_code=200,
                text=(
                    '{"choices":[{"message":{"content":"{\\"ok\\":true}"}}],'
                    '"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}'
                ),
            )

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)
    monkeypatch.setattr(openai_mod.httpx, "Client", _FlakyHTTPXClient)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert raw is not None
    assert meta.get("error") is None
    assert int(meta.get("llm_call_count") or 0) == 2
    assert int(meta.get("transport_retry_count") or 0) == 1
    retry_attempts = meta.get("transport_retry_attempts")
    assert isinstance(retry_attempts, list)
    assert retry_attempts[0].get("error") == "http_error"
    assert int(retry_attempts[0].get("status") or 0) == 503
    assert sleep_calls and sleep_calls[0] > 0.0


def test_chat_completion_retries_request_failed_then_succeeds(monkeypatch) -> None:
    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return (
                200,
                '{"choices":[{"message":{"content":"{\\"ok\\":true}"}}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
            )

    client = _FlakyClient()
    sleep_calls: list[float] = []

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", lambda *, url, http2: client)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert raw is not None
    assert meta.get("error") is None
    assert int(meta.get("llm_call_count") or 0) == 2
    assert int(meta.get("transport_retry_count") or 0) == 1
    retry_attempts = meta.get("transport_retry_attempts")
    assert isinstance(retry_attempts, list)
    assert retry_attempts[0].get("error") == "request_failed"
    assert sleep_calls and sleep_calls[0] > 0.0


def test_chat_completion_retries_retryable_http_error_then_succeeds(monkeypatch) -> None:
    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            self.calls += 1
            if self.calls == 1:
                return 503, "upstream overloaded"
            return (
                200,
                '{"choices":[{"message":{"content":"{\\"ok\\":true}"}}],'
                '"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
            )

    client = _FlakyClient()
    sleep_calls: list[float] = []

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", lambda *, url, http2: client)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert raw is not None
    assert meta.get("error") is None
    assert int(meta.get("llm_call_count") or 0) == 2
    assert int(meta.get("transport_retry_count") or 0) == 1
    retry_attempts = meta.get("transport_retry_attempts")
    assert isinstance(retry_attempts, list)
    assert retry_attempts[0].get("error") == "http_error"
    assert int(retry_attempts[0].get("status") or 0) == 503
    assert sleep_calls and sleep_calls[0] > 0.0


def test_chat_completion_does_not_retry_non_retryable_http_error(monkeypatch) -> None:
    class _BadRequestClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            self.calls += 1
            return 400, "prompt exceeds available context size"

    client = _BadRequestClient()
    sleep_calls: list[float] = []

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", lambda *, url, http2: client)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        http2=True,
    )

    assert raw is None
    assert meta.get("error") == "http_error"
    assert int(meta.get("status") or 0) == 400
    assert int(meta.get("llm_call_count") or 0) == 1
    assert int(meta.get("transport_retry_count") or 0) == 0
    assert int(meta.get("transport_retry_limit_effective") or 0) == 6
    assert float(meta.get("transport_retry_backoff_base_sec") or 0.0) == 1.0
    assert float(meta.get("transport_retry_backoff_max_sec") or 0.0) == 8.0
    assert "transport_retry_attempts" not in meta
    assert sleep_calls == []


def test_chat_completion_stops_after_transport_retry_limit(monkeypatch) -> None:
    class _AlwaysFlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            self.calls += 1
            raise TimeoutError("still down")

    client = _AlwaysFlakyClient()
    sleep_calls: list[float] = []

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", lambda *, url, http2: client)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        transport_max_retries=2,
        http2=True,
    )

    assert raw is None
    assert meta.get("error") == "request_failed"
    assert int(meta.get("llm_call_count") or 0) == 3
    assert int(meta.get("transport_retry_count") or 0) == 2
    assert int(meta.get("transport_retry_limit_effective") or 0) == 2
    retry_attempts = meta.get("transport_retry_attempts")
    assert isinstance(retry_attempts, list)
    assert len(retry_attempts) == 2
    assert all(item.get("error") == "request_failed" for item in retry_attempts)
    assert len(sleep_calls) == 2


def test_chat_completion_respects_custom_transport_retry_profile(monkeypatch) -> None:
    class _AlwaysFlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self,
            *,
            target: str,
            body: bytes,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> tuple[int, str]:
            del target, body, headers, timeout_sec
            self.calls += 1
            raise TimeoutError("still down")

    client = _AlwaysFlakyClient()
    sleep_calls: list[float] = []

    monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: False)
    monkeypatch.setattr(openai_mod, "_get_thread_http_client", lambda *, url, http2: client)
    monkeypatch.setattr(openai_mod.time, "sleep", lambda sec: sleep_calls.append(float(sec)))

    raw, meta = openai_mod._chat_completion(
        url="http://127.0.0.1:1234/v1/chat/completions",
        api_key="local",
        model="dummy",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=64,
        timeout_sec=30.0,
        seed=None,
        transport_max_retries=4,
        transport_retry_backoff_base_sec=1.0,
        transport_retry_backoff_max_sec=8.0,
        http2=True,
    )

    assert raw is None
    assert meta.get("error") == "request_failed"
    assert int(meta.get("llm_call_count") or 0) == 5
    assert int(meta.get("transport_retry_count") or 0) == 4
    assert int(meta.get("transport_retry_limit_effective") or 0) == 4
    assert float(meta.get("transport_retry_backoff_base_sec") or 0.0) == 1.0
    assert float(meta.get("transport_retry_backoff_max_sec") or 0.0) == 8.0
    assert sleep_calls == [1.0, 2.0, 4.0, 8.0]


def test_openai_patcher_propagates_transport_retry_settings_to_edits_call(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_edits(**kwargs) -> PatchResult:
        captured.update(kwargs)
        return PatchResult(diff=None, meta={"error": "request_failed"})

    monkeypatch.setattr(
        "evals.patchers.openai_compat_patcher._propose_patch_via_edits",
        _fake_edits,
    )

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="dummy",
        response_mode="edits",
        force_patch_attempt=False,
        transport_max_retries=7,
        transport_retry_backoff_base_sec=1.25,
        transport_retry_backoff_max_sec=9.0,
    )
    result = patcher.propose_patch(
        prompt="Pytest output:\nfailed\n",
        context_dir=tmp_path,
        cwd=tmp_path,
    )

    assert result.diff is None
    assert int(captured.get("transport_max_retries") or 0) == 7
    assert float(captured.get("transport_retry_backoff_base_sec") or 0.0) == 1.25
    assert float(captured.get("transport_retry_backoff_max_sec") or 0.0) == 9.0
