from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from baselines.controller_packet import parse_controller_packet
from evals import lambda_rlm_backend
from llmdebug import _json as json


def _minimal_controller_packet(**overrides):
    """Build a minimal controller packet with optional field overrides.

    Supports overrides for nested dicts: ``evidence_bundle``, ``history_bundle``,
    ``verifier_report``, and ``solver_hints`` are merged (not replaced) so callers
    only need to specify the keys they care about.
    """
    evidence_bundle = {
        "items": [],
        **(overrides.pop("evidence_bundle", None) or {}),
    }
    history_bundle = overrides.pop("history_bundle", None) or {}
    verifier_report = {
        "status": "",
        "reasons": [],
        **(overrides.pop("verifier_report", None) or {}),
    }
    solver_hints = {
        "tool_routes": [],
        **(overrides.pop("solver_hints", None) or {}),
    }
    raw = {
        "schema_version": "v1",
        "case_id": overrides.pop("case_id", "test_case"),
        "condition": overrides.pop("condition", "test_cond"),
        "attempt": overrides.pop("attempt", 1),
        "controller_state": overrides.pop("controller_state", {}),
        "evidence_bundle": evidence_bundle,
        "history_bundle": history_bundle,
        "verifier_report": verifier_report,
        "solver_hints": solver_hints,
        **overrides,
    }
    packet = parse_controller_packet(raw)
    assert packet is not None
    return packet


def _controller_packet():
    packet = parse_controller_packet(
        {
            "schema_version": "v1",
            "case_id": "adaptive_case",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "controller_state": {
                "helper_action_executed": "compact_then_patch",
                "helper_decision_reason": "policy_action_compact_then_patch",
            },
            "evidence_bundle": {
                "summary": {
                    "crash_anchor": "buggy.py:4 in f",
                    "top_facts": [
                        "Crash is localized to buggy.py",
                        "The return expression is wrong",
                    ],
                },
                "ast_context": {
                    "path": "buggy.py",
                    "line_span": [4, 4],
                    "symbol_name": "f",
                    "node_type": "Return",
                    "snippet": "return helper.value()",
                },
                "items": [
                    {"id": "e1", "kind": "frame", "summary": "AssertionError on return value"},
                    {
                        "id": "e2",
                        "kind": "source_span",
                        "summary": "Return expression at buggy.py:4",
                    },
                ],
            },
            "history_bundle": {
                "lessons": [
                    {
                        "source": "attempt_history",
                        "attempt": 1,
                        "lesson": "Previous patch missed buggy.py",
                    }
                ],
                "repo_history": {
                    "crash_file_blame_summary": "recent edit introduced the offset bug",
                    "recent_commits": [{"summary": "simplify helper math"}],
                },
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["patch missed crash file"],
            },
            "solver_hints": {
                "tool_routes": ["retry_plan", "history_lessons", "evidence_summary"],
                "retry_plan": "Keep the edit local to the return expression.",
            },
        }
    )
    assert packet is not None
    return packet


def test_parse_lambda_rlm_decomposition_text_filters_unknown_routes_and_evidence() -> None:
    result = lambda_rlm_backend.parse_lambda_rlm_decomposition_text(
        json.dumps(
            {
                "summary": "Localize the failing expression before editing.",
                "steps": [
                    {
                        "title": "Inspect Return Path",
                        "question": "Inspect buggy.py:4 before widening the patch.",
                        "supporting_routes": ["retry_plan", "unknown_route"],
                        "evidence_ids": ["e1", "missing"],
                    }
                ],
            }
        ),
        allowed_routes={"retry_plan", "history_lessons"},
        allowed_evidence_ids={"e1", "e2"},
    )

    assert result.summary == "Localize the failing expression before editing."
    assert result.steps == [
        {
            "title": "inspect_return_path",
            "question": "Inspect buggy.py:4 before widening the patch.",
            "supporting_routes": ["retry_plan"],
            "evidence_ids": ["e1"],
        }
    ]


def test_invoke_lambda_rlm_decomposition_reads_env_client_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _controller_packet()
    observed: dict[str, object] = {}

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        cwd: str,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        request_path = Path(cmd[2])
        response_path = Path(cmd[3])
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        observed["prompt"] = request_payload["prompt"]
        observed["cwd"] = cwd
        response_path.write_text(
            json.dumps(
                {
                    "root_model": "test-root-model",
                    "response": json.dumps(
                        {
                            "summary": "First localize the return path, then keep the edit minimal.",
                            "steps": [
                                {
                                    "title": "constrain_edit",
                                    "question": "Inspect buggy.py:4 and keep the edit local.",
                                    "supporting_routes": ["retry_plan", "history_lessons"],
                                    "evidence_ids": ["e1", "e2"],
                                }
                            ],
                        }
                    ),
                    "execution_time": 0.25,
                    "usage_summary": {
                        "total_input_tokens": 11,
                        "total_output_tokens": 7,
                        "model_usage_summaries": {"root": {"total_calls": 1}},
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(lambda_rlm_backend.subprocess, "run", _fake_run)

    result = lambda_rlm_backend.invoke_lambda_rlm_decomposition(
        controller_packet=packet,
        model="test-model",
        base_url="http://127.0.0.1:1234",
        api_key="sk-test",
        repo_dir=tmp_path / "lambda-RLM",
        python_executable=tmp_path / "lambda-python",
        context_window_chars=1000,
        timeout_sec=30,
    )

    assert observed["cwd"] == str(tmp_path / "lambda-RLM")
    assert "Case: adaptive_case" in str(observed["prompt"])
    assert "Allowed routes: retry_plan, history_lessons, evidence_summary" in str(
        observed["prompt"]
    )
    assert result.root_model == "test-root-model"
    assert result.execution_time_sec == pytest.approx(0.25)
    assert result.usage_summary["total_input_tokens"] == 11
    assert result.steps[0]["title"] == "constrain_edit"


def test_invoke_lambda_rlm_decomposition_uses_env_fallback_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _controller_packet()
    observed: dict[str, object] = {}

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        cwd: str,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        request_path = Path(cmd[2])
        response_path = Path(cmd[3])
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        observed["api_key"] = request_payload["api_key"]
        response_path.write_text(
            json.dumps(
                {
                    "root_model": "test-root-model",
                    "response": json.dumps({"summary": "ok", "steps": []}),
                    "execution_time": 0.1,
                    "usage_summary": {},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv("Z_API", raising=False)
    monkeypatch.delenv("LLMDEBUG_EVAL_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setattr(lambda_rlm_backend.subprocess, "run", _fake_run)

    lambda_rlm_backend.invoke_lambda_rlm_decomposition(
        controller_packet=packet,
        model="test-model",
        base_url="http://127.0.0.1:1234",
        api_key="",
        repo_dir=tmp_path / "lambda-RLM",
        python_executable=tmp_path / "lambda-python",
        context_window_chars=1000,
        timeout_sec=30,
    )

    assert observed["api_key"] == "env-openai-key"
    assert observed["api_key"] != "sk-placeholder"


def test_invoke_lambda_rlm_decomposition_defaults_to_local_when_no_api_key_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _controller_packet()
    observed: dict[str, object] = {}

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        cwd: str,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        request_path = Path(cmd[2])
        response_path = Path(cmd[3])
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        observed["api_key"] = request_payload["api_key"]
        response_path.write_text(
            json.dumps(
                {
                    "root_model": "test-root-model",
                    "response": json.dumps({"summary": "ok", "steps": []}),
                    "execution_time": 0.1,
                    "usage_summary": {},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv("Z_API", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMDEBUG_EVAL_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(lambda_rlm_backend.subprocess, "run", _fake_run)

    lambda_rlm_backend.invoke_lambda_rlm_decomposition(
        controller_packet=packet,
        model="test-model",
        base_url="http://127.0.0.1:1234",
        api_key="",
        repo_dir=tmp_path / "lambda-RLM",
        python_executable=tmp_path / "lambda-python",
        context_window_chars=1000,
        timeout_sec=30,
    )

    assert observed["api_key"] == "local"


# ---------- _truncate edge cases ----------


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        ("hello", 3, "hel"),
        ("abcdef", 2, "ab"),
        ("abcdef", 0, ""),
        ("abcdef", 1, "a"),
    ],
)
def test_truncate_small_max_chars(text: str, max_chars: int, expected: str) -> None:
    assert lambda_rlm_backend._truncate(text, max_chars=max_chars) == expected


# ---------- default path helpers ----------


def test_default_lambda_rlm_repo_dir() -> None:
    result = lambda_rlm_backend.default_lambda_rlm_repo_dir()
    assert result.name == "lambda-RLM"
    assert "baselines" in str(result)


def test_default_lambda_rlm_python() -> None:
    result = lambda_rlm_backend.default_lambda_rlm_python()
    assert result.name == "python"
    assert "lambda_rlm" in str(result)


# ---------- _json_object_candidates edge cases ----------


def test_json_object_candidates_empty_input() -> None:
    assert lambda_rlm_backend._json_object_candidates("") == []
    assert lambda_rlm_backend._json_object_candidates("  ") == []


def test_json_object_candidates_no_braces() -> None:
    """Text with no curly braces returns only the stripped text itself."""
    result = lambda_rlm_backend._json_object_candidates("hello world")
    assert result == ["hello world"]


# ---------- parse_lambda_rlm_decomposition_text error branches ----------


def test_parse_decomposition_empty_text_raises() -> None:
    """Empty text produces no parse candidates and raises ``ValueError``."""
    with pytest.raises(ValueError, match="parse failed"):
        lambda_rlm_backend.parse_lambda_rlm_decomposition_text(
            "", allowed_routes=set(), allowed_evidence_ids=set()
        )


def test_parse_decomposition_invalid_json_raises() -> None:
    """Malformed JSON is rejected as a parse failure."""
    with pytest.raises(ValueError, match="parse failed"):
        lambda_rlm_backend.parse_lambda_rlm_decomposition_text(
            "{not json!}", allowed_routes=set(), allowed_evidence_ids=set()
        )


def test_parse_decomposition_non_dict_payload_raises() -> None:
    """Non-dict JSON payloads are rejected."""
    with pytest.raises(ValueError, match="invalid_payload_type"):
        lambda_rlm_backend.parse_lambda_rlm_decomposition_text(
            json.dumps([1, 2, 3]), allowed_routes=set(), allowed_evidence_ids=set()
        )


def test_parse_decomposition_validation_error_raises() -> None:
    """A dict with invalid pydantic fields triggers lines 146-148.

    LambdaRLMDecompositionPayloadModel has ``steps`` that expects a list.
    Passing a non-list value that isn't coercible triggers ValidationError.
    """
    # steps validator expects list; integer should fail model_validate
    text = json.dumps({"summary": 123, "steps": "not-a-list"})
    with pytest.raises(ValueError, match="parse failed"):
        lambda_rlm_backend.parse_lambda_rlm_decomposition_text(
            text, allowed_routes=set(), allowed_evidence_ids=set()
        )


# ---------- build_lambda_rlm_decomposition_prompt branch coverage ----------


def test_build_prompt_with_failure_signature() -> None:
    """Failure signatures are surfaced when the controller packet provides one."""
    packet = _minimal_controller_packet(
        evidence_bundle={
            "failure_signature": "AssertionError at test_foo.py:10",
            "items": [{"id": "e1", "kind": "frame", "summary": "crash"}],
        },
    )
    prompt = lambda_rlm_backend.build_lambda_rlm_decomposition_prompt(packet)
    assert "Failure signature: AssertionError at test_foo.py:10" in prompt


def test_build_prompt_minimal_packet_no_optional_sections() -> None:
    """Minimal packet with no evidence summary, AST, history, or verifier."""
    packet = _minimal_controller_packet()
    prompt = lambda_rlm_backend.build_lambda_rlm_decomposition_prompt(packet)
    assert "Case: test_case" in prompt
    # Optional sections should be absent.
    assert "Crash anchor" not in prompt
    assert "AST crash snippet" not in prompt
    assert "History lessons" not in prompt
    assert "Retry plan" not in prompt


def test_build_prompt_no_crash_anchor_no_facts() -> None:
    """Empty summary fields are omitted from the rendered prompt."""
    packet = _minimal_controller_packet(
        evidence_bundle={
            "summary": {"crash_anchor": "", "top_facts": []},
            "items": [],
        },
    )
    prompt = lambda_rlm_backend.build_lambda_rlm_decomposition_prompt(packet)
    assert "Crash anchor" not in prompt
    assert "Evidence facts" not in prompt


def test_build_prompt_verifier_status_none_suppressed() -> None:
    """Verifier status ``none`` is intentionally suppressed in the prompt."""
    packet = _minimal_controller_packet(
        verifier_report={"status": "none", "reasons": []},
    )
    prompt = lambda_rlm_backend.build_lambda_rlm_decomposition_prompt(packet)
    assert "Verifier status" not in prompt


# ---------- invoke_lambda_rlm_decomposition error paths ----------


def test_invoke_raises_on_nonzero_returncode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-zero subprocess exit codes become runtime errors."""
    packet = _controller_packet()

    def _failing_run(cmd, *, capture_output, text, timeout, cwd, env):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="oops: crash")

    monkeypatch.setattr(lambda_rlm_backend.subprocess, "run", _failing_run)

    with pytest.raises(RuntimeError, match=r"lambda-RLM env client failed.*oops"):
        lambda_rlm_backend.invoke_lambda_rlm_decomposition(
            controller_packet=packet,
            model="m",
            base_url="http://localhost",
            api_key="sk-test",
            repo_dir=tmp_path,
            python_executable=tmp_path / "py",
            context_window_chars=100,
            timeout_sec=5,
        )


def test_invoke_raises_on_nonzero_returncode_stderr_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stdout becomes the error detail when stderr is empty."""
    packet = _controller_packet()

    def _failing_run(cmd, *, capture_output, text, timeout, cwd, env):
        return subprocess.CompletedProcess(cmd, 1, stdout="stdout-error", stderr="")

    monkeypatch.setattr(lambda_rlm_backend.subprocess, "run", _failing_run)

    with pytest.raises(RuntimeError, match="stdout-error"):
        lambda_rlm_backend.invoke_lambda_rlm_decomposition(
            controller_packet=packet,
            model="m",
            base_url="http://localhost",
            api_key="sk-test",
            repo_dir=tmp_path,
            python_executable=tmp_path / "py",
            context_window_chars=100,
            timeout_sec=5,
        )


def test_invoke_raises_when_response_is_not_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-dict response payloads are rejected."""
    packet = _controller_packet()

    def _list_response_run(cmd, *, capture_output, text, timeout, cwd, env):
        response_path = Path(cmd[3])
        response_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(lambda_rlm_backend.subprocess, "run", _list_response_run)

    with pytest.raises(RuntimeError, match="returned list"):
        lambda_rlm_backend.invoke_lambda_rlm_decomposition(
            controller_packet=packet,
            model="m",
            base_url="http://localhost",
            api_key="sk-test",
            repo_dir=tmp_path,
            python_executable=tmp_path / "py",
            context_window_chars=100,
            timeout_sec=5,
        )


def test_invoke_with_no_usage_summary_returns_empty_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing or invalid usage summaries fall back to an empty dict."""
    packet = _controller_packet()

    def _ok_run(cmd, *, capture_output, text, timeout, cwd, env):
        response_path = Path(cmd[3])
        response_path.write_text(
            json.dumps(
                {
                    "response": json.dumps(
                        {
                            "summary": "s",
                            "steps": [
                                {
                                    "title": "t",
                                    "question": "q",
                                    "supporting_routes": [],
                                    "evidence_ids": [],
                                }
                            ],
                        }
                    ),
                    "usage_summary": "not-a-dict",
                    "execution_time": 0.0,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(lambda_rlm_backend.subprocess, "run", _ok_run)

    result = lambda_rlm_backend.invoke_lambda_rlm_decomposition(
        controller_packet=packet,
        model="m",
        base_url="http://localhost",
        api_key="sk-test",
        repo_dir=tmp_path,
        python_executable=tmp_path / "py",
        context_window_chars=100,
        timeout_sec=5,
    )
    assert result.usage_summary == {}
