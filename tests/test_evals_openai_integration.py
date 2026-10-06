"""Integration tests for the OpenAI-compatible patcher against a stub HTTP server."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from evals import eval_attempt_flow, run_eval
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from llmdebug import _json as json

from ._openai_stub_server import OpenAIStubServer

pytestmark = pytest.mark.integration


def _make_case(tmp_path: Path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "import helper\n\n\ndef f() -> int:\n    return helper.value()\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="openai_integration_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def _chat_json_response(content: str, *, finish_reason: str | None = None) -> dict[str, Any]:
    choice: dict[str, Any] = {"message": {"content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _diff_response_body() -> dict[str, Any]:
    diff = (
        "diff --git a/helper.py b/helper.py\n"
        "--- a/helper.py\n"
        "+++ b/helper.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    return _chat_json_response(diff)


def _full_file_response_body(content: str, *, finish_reason: str | None = None) -> dict[str, Any]:
    return _chat_json_response(content, finish_reason=finish_reason)


def test_openai_request_context_request_contract(tmp_path) -> None:
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response('{"need_more_context":false,"requests":[]}'),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            temperature=0.2,
            max_tokens=321,
        )

        result = patcher.request_context(
            prompt="Need more context?", context_dir=tmp_path, cwd=tmp_path
        )

        assert result is not None
        assert result.request_json is not None
        assert result.request_json["need_more_context"] is False

        req = server.last_request(method="POST", path="/v1/chat/completions")
        assert req is not None
        assert req["headers"]["authorization"] == "Bearer local"
        payload = req["json"]
        assert isinstance(payload, dict)
        assert payload["model"] == "demo-model"
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 321
        assert payload["response_format"]["type"] == "json_object"
        messages = payload["messages"]
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        system_content = str(messages[0]["content"])
        assert '"diagnosis"' in system_content
        assert '"missing_fact"' in system_content
        assert '"need_snapshot"' in system_content
        assert '"snapshot_justification"' in system_content
        assert messages[1]["role"] == "user"


def test_openai_request_context_retries_invalid_payload_with_strict_json(tmp_path) -> None:
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response("not valid json"),
        )
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response('{"need_more_context":false,"requests":[]}'),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            max_tokens=1024,
        )

        result = patcher.request_context(
            prompt="Need more context?", context_dir=tmp_path, cwd=tmp_path
        )

        assert result is not None
        assert result.request_json is not None
        assert result.request_json["need_more_context"] is False
        assert result.meta is not None
        assert result.meta["request_context_retry_count"] == 1
        assert result.meta["request_context_transport_retry_count"] == 0
        assert result.meta["request_context_strict_retry_count"] == 1
        assert result.meta["request_context_failure_kind"] == ""

        reqs = server.requests_for(method="POST", path="/v1/chat/completions")
        assert len(reqs) == 2
        first_payload = reqs[0]["json"]
        second_payload = reqs[1]["json"]
        assert isinstance(first_payload, dict)
        assert isinstance(second_payload, dict)
        assert first_payload["max_tokens"] == 1024
        assert second_payload["max_tokens"] == 768
        second_messages = second_payload["messages"]
        assert isinstance(second_messages, list)
        assert "If unsure, return a valid object" in str(second_messages[0]["content"])


def test_openai_compact_context_request_contract(tmp_path) -> None:
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response('{"compacted_context":"key traceback facts"}'),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            temperature=0.0,
            max_tokens=222,
        )

        result = patcher.compact_context(
            prompt="compact evidence",
            context_text="stdout ...\nsnapshot ...",
            context_dir=tmp_path,
            cwd=tmp_path,
            target_chars=32,
        )

        assert result is not None
        assert result.compacted_text == "key traceback facts"

        req = server.last_request(method="POST", path="/v1/chat/completions")
        assert req is not None
        payload = req["json"]
        assert isinstance(payload, dict)
        assert payload["model"] == "demo-model"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 222
        assert payload["response_format"]["type"] == "json_object"
        messages = payload["messages"]
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert "<= 32 characters" in str(messages[0]["content"])
        assert messages[1]["role"] == "user"
        assert "Context to compact:" in str(messages[1]["content"])


def test_openai_edits_mode_response_format_fallback_contract(tmp_path) -> None:
    (tmp_path / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            status=400,
            body={"error": {"message": "response_format not supported"}},
        )
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(
                json.dumps(
                    {
                        "edits": [
                            {
                                "path": "helper.py",
                                "find": "return 1",
                                "replace": "return 2",
                                "count": 1,
                            }
                        ]
                    }
                )
            ),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="edits",
        )

        result = patcher.propose_patch(prompt="fix helper", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is not None
        assert "diff --git a/helper.py b/helper.py" in result.diff

        reqs = server.requests_for(method="POST", path="/v1/chat/completions")
        assert len(reqs) == 2
        first_payload = reqs[0]["json"]
        second_payload = reqs[1]["json"]
        assert isinstance(first_payload, dict)
        assert isinstance(second_payload, dict)
        assert first_payload["response_format"]["type"] == "json_schema"
        assert second_payload["response_format"]["type"] == "json_object"


def test_openai_diff_mode_request_has_no_response_format(tmp_path) -> None:
    with OpenAIStubServer() as server:
        server.enqueue_json(method="POST", path="/v1/chat/completions", body=_diff_response_body())
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="diff",
        )

        result = patcher.propose_patch(
            prompt="return unified diff", context_dir=tmp_path, cwd=tmp_path
        )

        assert result.diff is not None
        req = server.last_request(method="POST", path="/v1/chat/completions")
        assert req is not None
        payload = req["json"]
        assert isinstance(payload, dict)
        assert "response_format" not in payload


def test_openai_full_file_mode_request_has_no_response_format(tmp_path) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_full_file_response_body("def f() -> int:\n    return 2\n"),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="full_file",
        )

        result = patcher.propose_patch(
            prompt="return repaired file", context_dir=tmp_path, cwd=tmp_path
        )

        assert result.diff is not None
        req = server.last_request(method="POST", path="/v1/chat/completions")
        assert req is not None
        payload = req["json"]
        assert isinstance(payload, dict)
        assert "response_format" not in payload
        messages = payload["messages"]
        assert isinstance(messages, list)
        assert "complete repaired contents of buggy.py" in str(messages[0]["content"])


def test_openai_full_file_mode_builds_local_diff_for_single_source_case(tmp_path) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_full_file_response_body("def f() -> int:\n    return 2\n"),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="full_file",
        )

        result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is not None
        assert "diff --git a/buggy.py b/buggy.py" in result.diff
        assert "+    return 2" in result.diff
        assert result.meta is not None
        assert result.meta["full_file_target_path"] == "buggy.py"
        assert result.meta["full_file_supported"] is True
        assert result.meta["full_file_support_reason"] == "single_non_test_python_file"


def test_openai_full_file_mode_extracts_fenced_code_with_preamble(tmp_path) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_full_file_response_body(
                "Here is the fixed file:\n"
                "```python\n"
                "def f() -> int:\n"
                "    return 2\n"
                "```\n"
                "This is the only change."
            ),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="full_file",
        )

        result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is not None
        assert "+    return 2" in result.diff
        assert "Here is the fixed file" not in result.diff
        assert result.meta is not None
        assert result.meta["full_file_code_fence_stripped"] is True


def test_openai_full_file_mode_plain_unified_diff_falls_back_to_diff(tmp_path) -> None:
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
    fallback_diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_full_file_response_body(plain_diff),
        )
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(fallback_diff),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="full_file",
        )

        result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is not None
        assert result.meta is not None
        assert result.meta["fallback_from"] == "full_file"
        assert result.meta["fallback_reason"] == "returned_diff_instead_of_file"
        assert "+--- buggy.py" not in result.diff
        requests = server.requests_for(method="POST", path="/v1/chat/completions")
        assert len(requests) == 2


def test_openai_full_file_mode_truncated_output_falls_back_to_diff(tmp_path) -> None:
    (tmp_path / "buggy.py").write_text(
        "def f() -> int:\n    first = 1\n    second = 2\n    return first + second\n",
        encoding="utf-8",
    )
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )
    fallback_diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,4 +1,4 @@\n"
        " def f() -> int:\n"
        "     first = 1\n"
        "     second = 2\n"
        "-    return first + second\n"
        "+    return second + second\n"
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_full_file_response_body(
                "def f() -> int:\n    first = 1\n    second = 2\n",
                finish_reason="length",
            ),
        )
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(fallback_diff),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="full_file",
        )

        result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is not None
        assert result.meta is not None
        assert result.meta["fallback_from"] == "full_file"
        assert result.meta["fallback_reason"] == "truncated_full_file_response"
        assert result.meta["fallback_trigger"]["finish_reason"] == "length"
        requests = server.requests_for(method="POST", path="/v1/chat/completions")
        assert len(requests) == 2


def test_openai_full_file_mode_identical_output_falls_back_to_diff(tmp_path) -> None:
    before = "def f() -> int:\n    return 1\n"
    (tmp_path / "buggy.py").write_text(before, encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )

    diff = (
        "diff --git a/buggy.py b/buggy.py\n"
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_full_file_response_body(before),
        )
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(diff),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="full_file",
        )

        result = patcher.propose_patch(prompt="fix buggy.py", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is not None
        assert result.meta is not None
        assert result.meta["fallback_from"] == "full_file"
        assert result.meta["fallback_reason"] == "no_effective_changes"
        assert result.meta["full_file_output_chars"] == len(before)


def test_openai_full_file_mode_unsupported_case_shape_falls_back_to_diff(tmp_path) -> None:
    (tmp_path / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_buggy.py").write_text(
        "def test_f() -> None:\n    assert True\n", encoding="utf-8"
    )

    diff = (
        "diff --git a/helper.py b/helper.py\n"
        "--- a/helper.py\n"
        "+++ b/helper.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def value() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(diff),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url,
            model="demo-model",
            api_key="local",
            response_mode="full_file",
        )

        result = patcher.propose_patch(prompt="fix helper.py", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is not None
        assert result.meta is not None
        assert result.meta["fallback_from"] == "full_file"
        assert result.meta["fallback_reason"] == "unsupported_case_shape"
        assert result.meta["full_file_supported"] is False
        assert result.meta["full_file_support_reason"] == "multiple_non_test_python_files"
        requests = server.requests_for(method="POST", path="/v1/chat/completions")
        assert len(requests) == 1


def test_openai_probe_context_window_endpoint_order(tmp_path) -> None:
    del tmp_path
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="GET",
            path="/api/v1/models",
            status=404,
            body={"error": {"message": "not found"}},
        )
        server.enqueue_json(
            method="GET",
            path="/api/v0/models",
            body={"data": [{"id": "demo-model", "max_context_length": 16384}]},
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url_v1,
            model="demo-model",
            api_key="local",
        )

        tokens, meta = patcher.probe_context_window_tokens()

        assert tokens == 16384
        assert meta["context_window_probe_provider"] == "lmstudio_api_v0"
        assert meta["context_window_source"] == "max_context_length"
        request_paths = [item["path"] for item in server.requests if item["method"] == "GET"]
        assert request_paths == ["/api/v1/models", "/api/v0/models"]


def test_openai_probe_context_window_falls_back_to_openai_models_endpoint(tmp_path) -> None:
    del tmp_path
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="GET",
            path="/api/v1/models",
            status=404,
            body={"error": {"message": "not found"}},
        )
        server.enqueue_json(
            method="GET",
            path="/api/v0/models",
            status=404,
            body={"error": {"message": "not found"}},
        )
        server.enqueue_json(
            method="GET",
            path="/v1/models",
            body={"data": [{"id": "demo-model", "context_length": 8192}]},
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url_v1,
            model="demo-model",
            api_key="local",
        )

        tokens, meta = patcher.probe_context_window_tokens()

        assert tokens == 8192
        assert meta["context_window_probe_provider"] == "openai_models"
        assert meta["context_window_source"] == "context_length"
        request_paths = [item["path"] for item in server.requests if item["method"] == "GET"]
        assert request_paths == ["/api/v1/models", "/api/v0/models", "/v1/models"]


def test_openai_edits_rejects_path_traversal_without_mutating_parent_file(tmp_path) -> None:
    outside_file = tmp_path.parent / "outside_sensitive_file.py"
    outside_file.write_text("FLAG = 'original'\n", encoding="utf-8")
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(
                json.dumps(
                    {
                        "edits": [
                            {
                                "path": "../outside_sensitive_file.py",
                                "find": "original",
                                "replace": "modified",
                                "count": 1,
                            }
                        ]
                    }
                )
            ),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url_v1,
            model="demo-model",
            api_key="local",
            response_mode="edits",
        )

        result = patcher.propose_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta["error"] == "path_outside_case"
        assert outside_file.read_text(encoding="utf-8") == "FLAG = 'original'\n"


def test_openai_edits_rejects_directory_target(tmp_path) -> None:
    (tmp_path / "target_dir").mkdir(parents=True, exist_ok=True)
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(
                json.dumps(
                    {
                        "edits": [
                            {
                                "path": "target_dir",
                                "find": "x",
                                "replace": "y",
                                "count": 1,
                            }
                        ]
                    }
                )
            ),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url_v1,
            model="demo-model",
            api_key="local",
            response_mode="edits",
        )

        result = patcher.propose_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta["error"] == "path_not_file"
        assert result.meta["path"] == "target_dir"


def test_openai_edits_surfaces_file_read_error(tmp_path, monkeypatch) -> None:
    target_file = tmp_path / "helper.py"
    target_file.write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    # The edits path reads bytes to keep CRLF files intact (LLM-09).
    original_read_bytes = Path.read_bytes

    def _raising_read_bytes(self: Path) -> bytes:
        if self.resolve() == target_file.resolve():
            raise OSError("simulated read failure")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _raising_read_bytes)
    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(
                json.dumps(
                    {
                        "edits": [
                            {
                                "path": "helper.py",
                                "find": "return 1",
                                "replace": "return 2",
                                "count": 1,
                            }
                        ]
                    }
                )
            ),
        )
        patcher = OpenAICompatPatcher(
            base_url=server.base_url_v1,
            model="demo-model",
            api_key="local",
            response_mode="edits",
        )

        result = patcher.propose_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)

        assert result.diff is None
        assert result.meta is not None
        assert result.meta["error"] == "file_read_error"
        assert result.meta["path"] == "helper.py"
        assert "simulated read failure" in str(result.meta.get("exc"))


def test_stage_a_outside_file_request_uses_fallback_and_does_not_leak_secret(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    secret_marker = "TOP_SECRET_SENTINEL_4731"
    outside_secret_path = tmp_path / "secret.txt"
    outside_secret_path.write_text(secret_marker, encoding="utf-8")
    workdir = tmp_path / "workdir_openai_stage_a"
    context_dir = tmp_path / "context_openai_stage_a"
    shutil.copytree(case.path, workdir)

    with OpenAIStubServer() as server:
        # Attempt 1 (no stagnation): monolithic patch
        server.enqueue_json(method="POST", path="/v1/chat/completions", body=_diff_response_body())
        # Attempt 2 (stagnation triggers Stage-A): outside-file request → fallback
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(
                json.dumps(
                    {
                        "need_more_context": True,
                        "requests": [{"type": "file", "path": "../secret.txt"}],
                        "sufficiency_rationale": "need more context",
                    }
                )
            ),
        )
        server.enqueue_json(method="POST", path="/v1/chat/completions", body=_diff_response_body())
        patcher = OpenAICompatPatcher(
            base_url=server.base_url_v1,
            model="demo-model",
            api_key="local",
            response_mode="diff",
            force_patch_attempt=False,
        )

        call_counter = {"count": 0}
        failure_output = "test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n"

        def _fake_run_pytest(*args, **kwargs) -> run_eval.RunResult:
            del args, kwargs
            call_counter["count"] += 1
            if call_counter["count"] <= 2:
                return run_eval.RunResult(
                    returncode=1,
                    duration_sec=0.01,
                    stdout=failure_output,
                    stderr="",
                )
            return run_eval.RunResult(
                returncode=0,
                duration_sec=0.01,
                stdout="1 passed\n",
                stderr="",
            )

        monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
        monkeypatch.setattr(
            eval_attempt_flow,
            "read_latest_snapshot",
            lambda *_args, **_kwargs: (None, "missing snapshot"),
        )

        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=2,
            output_format="json_compact",
            run_id="run-openai-stage-a-fallback",
            include_files_context=False,
            force_patch_attempt=True,
            context_protocol="staged",
        )

        assert len(records) >= 1
        # Verify the secret never appears in any request sent to the LLM
        requests = server.requests_for(method="POST", path="/v1/chat/completions")
        assert len(requests) >= 1
        for req in requests:
            payload = req.get("json", {})
            messages = payload.get("messages", [])
            for msg in messages:
                content = str(msg.get("content", ""))
                assert secret_marker not in content, "Secret leaked into LLM request"
                assert "../secret.txt" not in content, "Outside path leaked into LLM request"


def test_attempt_usage_equals_sum_of_served_llm_calls(tmp_path) -> None:
    """Phase 1F acceptance: every LLM call the attempt made is in its recorded total."""
    import shutil

    from evals.eval_types import Condition

    case = _make_case(tmp_path)
    workdir = tmp_path / "work"
    shutil.copytree(case.path, workdir)
    no_diff_body = {
        "choices": [{"message": {"content": "I am not sure yet."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    patch_body = _diff_response_body()
    patch_body["usage"] = {"prompt_tokens": 23, "completion_tokens": 5, "total_tokens": 28}
    bodies = [no_diff_body, patch_body]
    with OpenAIStubServer() as server:
        for body in bodies:
            server.enqueue_json(method="POST", path="/v1/chat/completions", body=body)
        patcher = OpenAICompatPatcher(base_url=server.base_url, model="demo", api_key="local")
        previous = run_eval.set_active_execution_config(run_eval.ExecutionConfig(executor="local"))
        try:
            records = run_eval.run_one_condition(
                case=case,
                condition=Condition.BASE_REPAIR,
                workdir=workdir,
                context_dir_base=tmp_path / "ctx",
                patcher=patcher,
                k=1,
                output_format="json_compact",
                run_id="usage-acceptance",
                include_files_context=True,
                force_patch_attempt=False,
                patch_verifier_scope="disabled",
            )
        finally:
            run_eval.set_active_execution_config(previous)
        served = len(server.requests)

    # Requests beyond the enqueued bodies get stub 404s (no usage); the patcher retries
    # them, and those calls must be counted too.
    attempt = next(r for r in records if int(r.get("attempt", 0)) == 1)
    meta = attempt["attempt_metadata"]
    served_bodies = bodies[:served]
    assert served >= 2
    assert meta["llm_call_count"] == served
    assert meta["tokens_total"] == sum(b["usage"]["total_tokens"] for b in served_bodies)
    assert meta["tokens_prompt"] == sum(b["usage"]["prompt_tokens"] for b in served_bodies)
