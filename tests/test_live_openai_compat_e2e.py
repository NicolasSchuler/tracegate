"""Live smoke tests for raw OpenAI-compatible response normalization."""

from __future__ import annotations

import json
import urllib.request

import pytest

from baselines.openai_compat import (
    BaselineOpenAICompatConfig,
    OpenAICompatResponseError,
    normalize_chat_completion_response,
    require_final_content,
)

from . import _live_e2e_support as live_support

pytestmark = pytest.mark.integration

live_openai_base_url = live_support.live_openai_base_url
live_openai_model = live_support.live_openai_model
_skip_if_live_server_unreachable = live_support._skip_if_live_server_unreachable


def _post_chat_completion(*, base_url: str, model: str) -> dict[str, object]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Return ONLY the word OK."}],
                "max_tokens": 32,
                "temperature": 0,
            }
        ).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-placeholder",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_live_openai_compat_normalizes_chat_completion_response(
    live_openai_base_url: str,
    live_openai_model: str,
) -> None:
    payload = _post_chat_completion(
        base_url=live_openai_base_url,
        model=live_openai_model,
    )

    without_fallback = normalize_chat_completion_response(
        payload,
        compat_config=BaselineOpenAICompatConfig(allow_reasoning_fallback=False),
        base_url=live_openai_base_url,
    )
    assert without_fallback.content_present or without_fallback.reasoning_present

    if without_fallback.empty_content_with_reasoning:
        with pytest.raises(OpenAICompatResponseError):
            require_final_content(without_fallback, base_url=live_openai_base_url)
        with_fallback = normalize_chat_completion_response(
            payload,
            compat_config=BaselineOpenAICompatConfig(allow_reasoning_fallback=True),
            base_url=live_openai_base_url,
        )
        assert require_final_content(with_fallback, base_url=live_openai_base_url).strip()
        return

    assert require_final_content(without_fallback, base_url=live_openai_base_url).strip()
