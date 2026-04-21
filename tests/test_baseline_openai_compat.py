"""Unit tests for shared OpenAI-compatible baseline response handling."""

from __future__ import annotations

import pytest

from baselines.openai_compat import (
    BaselineOpenAICompatConfig,
    OpenAICompatResponseError,
    apply_compat_request_profile,
    build_request_validation_retry_plan,
    normalize_chat_completion_response,
    request_validation_compat_error_from_exception,
    require_final_content,
)


def test_normalize_response_with_plain_string_content() -> None:
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {"content": "patched"},
                    "finish_reason": "stop",
                }
            ]
        },
        compat_config=BaselineOpenAICompatConfig(),
        base_url="http://127.0.0.1:1234",
    )

    assert normalized.text == "patched"
    assert normalized.content_present is True
    assert normalized.reasoning_present is False
    assert normalized.empty_content_with_reasoning is False


def test_normalize_response_ignores_reasoning_content_parts_when_output_exists() -> None:
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "reasoning_text", "text": "internal thought"},
                            {"type": "output_text", "text": "final patch"},
                        ],
                        "reasoning_content": "separate reasoning",
                        "reasoning": "more reasoning",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
        compat_config=BaselineOpenAICompatConfig(),
        base_url="http://127.0.0.1:1234",
    )

    assert normalized.text == "final patch"
    assert normalized.reasoning_present is True
    assert normalized.empty_content_with_reasoning is False


def test_normalize_response_strips_think_tags_only_from_final_text() -> None:
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>hidden</think>\npatched",
                        "reasoning_content": "hidden",
                    }
                }
            ]
        },
        compat_config=BaselineOpenAICompatConfig(),
        base_url="http://127.0.0.1:1234",
    )

    assert normalized.text == "patched"
    assert normalized.reasoning_content == "hidden"


def test_require_final_content_rejects_reasoning_content_only_response() -> None:
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "reasoning only",
                    }
                }
            ]
        },
        compat_config=BaselineOpenAICompatConfig(
            allow_reasoning_fallback=False,
            enable_thinking=False,
        ),
        base_url="http://127.0.0.1:1234",
    )

    with pytest.raises(OpenAICompatResponseError, match="reasoning but no final answer"):
        require_final_content(normalized, base_url="http://127.0.0.1:1234")


def test_require_final_content_rejects_reasoning_only_response() -> None:
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "reasoning only",
                    }
                }
            ]
        },
        compat_config=BaselineOpenAICompatConfig(
            allow_reasoning_fallback=False,
            enable_thinking=False,
        ),
        base_url="http://127.0.0.1:1234",
    )

    with pytest.raises(OpenAICompatResponseError, match="reasoning but no final answer"):
        require_final_content(normalized, base_url="http://127.0.0.1:1234")


def test_require_final_content_rejects_think_only_response() -> None:
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>reasoning only</think>",
                    }
                }
            ]
        },
        compat_config=BaselineOpenAICompatConfig(
            allow_reasoning_fallback=False,
            enable_thinking=False,
        ),
        base_url="http://127.0.0.1:1234",
    )

    assert normalized.content_present is False
    assert normalized.reasoning_present is True
    assert normalized.empty_content_with_reasoning is True
    with pytest.raises(OpenAICompatResponseError, match="reasoning but no final answer"):
        require_final_content(normalized, base_url="http://127.0.0.1:1234")


def test_require_final_content_uses_think_reasoning_fallback_when_enabled() -> None:
    normalized = normalize_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>reasoning only</think>",
                    }
                }
            ]
        },
        compat_config=BaselineOpenAICompatConfig(allow_reasoning_fallback=True),
        base_url="http://127.0.0.1:1234",
    )

    assert normalized.used_reasoning_fallback is True
    assert require_final_content(normalized, base_url="http://127.0.0.1:1234") == "reasoning only"


def test_qwen_openai_raw_request_profile_disables_thinking() -> None:
    request_kwargs, request_meta = apply_compat_request_profile(
        {"model": "test", "messages": [], "extra_body": {"usage": {"include": True}}},
        compat_config=BaselineOpenAICompatConfig(profile="qwen_openai_raw", enable_thinking=False),
        base_url="http://127.0.0.1:1234/v1",
    )

    assert request_kwargs["extra_body"]["usage"] == {"include": True}
    assert request_kwargs["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert request_meta.compat_profile_applied == "qwen_openai_raw"
    assert request_meta.direct_output_requested is True


def test_qwen_dashscope_request_profile_disables_thinking() -> None:
    request_kwargs, request_meta = apply_compat_request_profile(
        {"model": "test", "messages": []},
        compat_config=BaselineOpenAICompatConfig(profile="qwen_dashscope", enable_thinking=False),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert request_kwargs["enable_thinking"] is False
    assert request_meta.compat_profile_applied == "qwen_dashscope"
    assert request_meta.direct_output_requested is True


def test_auto_profile_stays_conservative_for_unknown_localhost_server() -> None:
    request_kwargs, request_meta = apply_compat_request_profile(
        {"model": "test", "messages": []},
        compat_config=BaselineOpenAICompatConfig(profile="auto"),
        base_url="http://127.0.0.1:1234/v1",
    )

    assert "enable_thinking" not in request_kwargs
    assert "extra_body" not in request_kwargs
    assert request_meta.compat_profile_applied == "auto"
    assert request_meta.direct_output_requested is False


def test_request_validation_helper_classifies_prefill_enable_thinking_conflict() -> None:
    _, request_meta = apply_compat_request_profile(
        {"model": "test", "messages": []},
        compat_config=BaselineOpenAICompatConfig(profile="auto"),
        base_url="http://127.0.0.1:1234/v1",
    )

    compat_exc = request_validation_compat_error_from_exception(
        RuntimeError(
            "BadRequestError: Error code: 400 - {'error': {'code': 400, "
            "'message': 'Assistant response prefill is incompatible with "
            "enable_thinking.', 'type': 'invalid_request_error'}}"
        ),
        request_meta=request_meta,
        base_url="http://127.0.0.1:1234/v1",
    )

    assert compat_exc is not None
    assert compat_exc.meta["request_validation_error"] is True
    assert compat_exc.meta["request_validation_failure_kind"] == (
        "assistant_prefill_enable_thinking_conflict"
    )
    assert compat_exc.meta["compat_profile_applied"] == "auto"
    assert "qwen_openai_raw" in str(compat_exc)
    assert "disable thinking in the server or model template" in str(compat_exc)


def test_request_validation_helper_ignores_unrelated_errors() -> None:
    _, request_meta = apply_compat_request_profile(
        {"model": "test", "messages": []},
        compat_config=BaselineOpenAICompatConfig(profile="auto"),
        base_url="http://127.0.0.1:1234/v1",
    )

    compat_exc = request_validation_compat_error_from_exception(
        RuntimeError("BadRequestError: unsupported parameter"),
        request_meta=request_meta,
        base_url="http://127.0.0.1:1234/v1",
    )

    assert compat_exc is None


def test_request_validation_retry_plan_switches_auto_to_qwen_openai_raw() -> None:
    base_request_kwargs = {"model": "test", "messages": []}
    compat_config = BaselineOpenAICompatConfig(profile="auto")
    _, request_meta = apply_compat_request_profile(
        base_request_kwargs,
        compat_config=compat_config,
        base_url="http://127.0.0.1:1234/v1",
    )

    retry_plan = build_request_validation_retry_plan(
        base_request_kwargs,
        compat_config=compat_config,
        request_meta=request_meta,
        exc=RuntimeError(
            "BadRequestError: Error code: 400 - {'error': {'code': 400, "
            "'message': 'Assistant response prefill is incompatible with "
            "enable_thinking.', 'type': 'invalid_request_error'}}"
        ),
        base_url="http://127.0.0.1:1234/v1",
    )

    assert retry_plan is not None
    assert retry_plan.reason == "retry_switch_profile_qwen_openai_raw"
    assert retry_plan.request_meta.compat_profile_applied == "qwen_openai_raw"
    assert retry_plan.request_meta.enable_thinking is False
    assert retry_plan.request_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_request_validation_retry_plan_does_not_switch_lmstudio_profile() -> None:
    base_request_kwargs = {"model": "test", "messages": []}
    compat_config = BaselineOpenAICompatConfig(profile="lmstudio_model_yaml")
    _, request_meta = apply_compat_request_profile(
        base_request_kwargs,
        compat_config=compat_config,
        base_url="http://127.0.0.1:1234/v1",
    )

    retry_plan = build_request_validation_retry_plan(
        base_request_kwargs,
        compat_config=compat_config,
        request_meta=request_meta,
        exc=RuntimeError(
            "BadRequestError: Error code: 400 - {'error': {'code': 400, "
            "'message': 'Assistant response prefill is incompatible with "
            "enable_thinking.', 'type': 'invalid_request_error'}}"
        ),
        base_url="http://127.0.0.1:1234/v1",
    )

    assert retry_plan is None


def test_request_validation_retry_plan_disables_thinking_for_qwen_profile() -> None:
    base_request_kwargs = {"model": "test", "messages": []}
    compat_config = BaselineOpenAICompatConfig(profile="qwen_openai_raw", enable_thinking=True)
    _, request_meta = apply_compat_request_profile(
        base_request_kwargs,
        compat_config=compat_config,
        base_url="http://127.0.0.1:1234/v1",
    )

    retry_plan = build_request_validation_retry_plan(
        base_request_kwargs,
        compat_config=compat_config,
        request_meta=request_meta,
        exc=RuntimeError(
            "BadRequestError: Error code: 400 - {'error': {'code': 400, "
            "'message': 'Assistant response prefill is incompatible with "
            "enable_thinking.', 'type': 'invalid_request_error'}}"
        ),
        base_url="http://127.0.0.1:1234/v1",
    )

    assert retry_plan is not None
    assert retry_plan.reason == "retry_disable_thinking_qwen_openai_raw"
    assert retry_plan.request_meta.compat_profile_applied == "qwen_openai_raw"
    assert retry_plan.request_meta.enable_thinking is False
    assert retry_plan.request_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


# ---------------------------------------------------------------------------
# enable_thinking option tests
# ---------------------------------------------------------------------------


class TestEnableThinkingOption:
    """Tests for the enable_thinking config option across profiles."""

    def test_qwen_openai_raw_enables_thinking_when_config_true(self) -> None:
        request_kwargs, request_meta = apply_compat_request_profile(
            {"model": "test", "messages": []},
            compat_config=BaselineOpenAICompatConfig(
                profile="qwen_openai_raw", enable_thinking=True
            ),
            base_url="http://127.0.0.1:1234/v1",
        )

        assert request_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert request_meta.enable_thinking is True
        assert request_meta.direct_output_requested is False

    def test_qwen_dashscope_enables_thinking_when_config_true(self) -> None:
        request_kwargs, request_meta = apply_compat_request_profile(
            {"model": "test", "messages": []},
            compat_config=BaselineOpenAICompatConfig(
                profile="qwen_dashscope", enable_thinking=True
            ),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        assert request_kwargs["enable_thinking"] is True
        assert request_meta.enable_thinking is True
        assert request_meta.direct_output_requested is False

    def test_enable_thinking_false_preserves_current_behavior(self) -> None:
        request_kwargs, request_meta = apply_compat_request_profile(
            {"model": "test", "messages": []},
            compat_config=BaselineOpenAICompatConfig(
                profile="qwen_openai_raw", enable_thinking=False
            ),
            base_url="http://127.0.0.1:1234/v1",
        )

        assert request_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        assert request_meta.enable_thinking is False
        assert request_meta.direct_output_requested is True

    def test_enable_thinking_auto_enables_reasoning_fallback(self) -> None:
        from baselines.openai_compat import compat_config_from_extra

        cfg = compat_config_from_extra({"baseline_openai_enable_thinking": True})
        assert cfg.enable_thinking is True
        assert cfg.allow_reasoning_fallback is True

    def test_enable_thinking_false_does_not_force_reasoning_fallback(self) -> None:
        from baselines.openai_compat import compat_config_from_extra

        cfg = compat_config_from_extra(
            {
                "baseline_openai_enable_thinking": False,
                "baseline_openai_allow_reasoning_fallback": False,
            }
        )
        assert cfg.enable_thinking is False
        assert cfg.allow_reasoning_fallback is False

    def test_normalize_response_with_think_tags_and_content(self) -> None:
        """Thinking model produces <think> tags PLUS final content."""
        normalized = normalize_chat_completion_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>Let me analyze...</think>\ndef multiply(a, b):\n    return a * b",
                        }
                    }
                ]
            },
            compat_config=BaselineOpenAICompatConfig(enable_thinking=True),
            base_url="http://127.0.0.1:1234",
        )

        assert "Let me analyze" not in normalized.text
        assert "def multiply" in normalized.text
        assert normalized.content_present is True
        assert normalized.reasoning_present is True

    def test_normalize_response_thinking_only_uses_fallback(self) -> None:
        """Thinking model produces only <think> tags — fallback recovers."""
        normalized = normalize_chat_completion_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>my reasoning about the fix</think>",
                        }
                    }
                ]
            },
            compat_config=BaselineOpenAICompatConfig(
                enable_thinking=True, allow_reasoning_fallback=True
            ),
            base_url="http://127.0.0.1:1234",
        )

        assert normalized.used_reasoning_fallback is True
        text = require_final_content(normalized, base_url="http://127.0.0.1:1234")
        assert "my reasoning about the fix" in text

    def test_normalize_response_reasoning_content_field_uses_fallback(self) -> None:
        """Model returns reasoning_content but empty content — fallback recovers."""
        normalized = normalize_chat_completion_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "def multiply(a, b):\n    return a * b",
                        }
                    }
                ]
            },
            compat_config=BaselineOpenAICompatConfig(
                enable_thinking=True, allow_reasoning_fallback=True
            ),
            base_url="http://127.0.0.1:1234",
        )

        assert normalized.used_reasoning_fallback is True
        text = require_final_content(normalized, base_url="http://127.0.0.1:1234")
        assert "def multiply" in text

    def test_auto_profile_with_enable_thinking_preserves_setting(self) -> None:
        """auto profile doesn't apply qwen shaping but preserves enable_thinking meta."""
        _, request_meta = apply_compat_request_profile(
            {"model": "test", "messages": []},
            compat_config=BaselineOpenAICompatConfig(profile="auto", enable_thinking=True),
            base_url="http://127.0.0.1:1234/v1",
        )

        assert request_meta.enable_thinking is True
