"""OpenAI chat client shim executed inside an isolated INTERVENOR env."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.openai_compat import (  # noqa: E402
    OpenAICompatResponseError,
    apply_compat_request_profile,
    compat_config_from_extra,
    normalize_chat_completion_response,
    require_final_content,
)


def _usage_int(source: dict[str, Any], *keys: str) -> int | None:
    fallback_zero: int | None = None
    for key in keys:
        value = source.get(key)
        if isinstance(value, int):
            if value > 0:
                return value
            fallback_zero = 0
    return fallback_zero


def _usage_extra_body() -> dict[str, Any]:
    return {"usage": {"include": True}}


def _usage_payload(usage: object) -> dict[str, int]:
    if usage is None:
        return {}
    source: dict[str, Any]
    if isinstance(usage, dict):
        source = cast(dict[str, Any], usage)
    else:
        source = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "total_input_tokens": getattr(usage, "total_input_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_output_tokens": getattr(usage, "total_output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    prompt_tokens = _usage_int(
        source,
        "prompt_tokens",
        "input_tokens",
        "total_input_tokens",
    )
    completion_tokens = _usage_int(
        source,
        "completion_tokens",
        "output_tokens",
        "total_output_tokens",
    )
    total_tokens = _usage_int(source, "total_tokens")

    result: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        result["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        result["completion_tokens"] = completion_tokens
    if isinstance(total_tokens, int):
        result["total_tokens"] = total_tokens
    elif isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        result["total_tokens"] = prompt_tokens + completion_tokens
    return result


def _finish_reason_payload(response: object) -> dict[str, str]:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        first_choice = cast(object, choices[0])
        finish_reason = str(getattr(first_choice, "finish_reason", "") or "").strip()
        if finish_reason:
            return {"finish_reason": finish_reason}

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            dumped_mapping = cast(dict[str, Any], dumped)
            raw_choices = dumped_mapping.get("choices")
            if isinstance(raw_choices, list) and raw_choices:
                choice = cast(object, raw_choices[0])
                if isinstance(choice, dict):
                    choice_mapping = cast(dict[str, Any], choice)
                    finish_reason = str(choice_mapping.get("finish_reason") or "").strip()
                    if finish_reason:
                        return {"finish_reason": finish_reason}
    return {}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: env_client.py <request.json> <response.json>")

    request_path = Path(argv[1])
    response_path = Path(argv[2])
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    openai_module = import_module("openai")
    openai_factory = getattr(openai_module, "OpenAI", None)
    if openai_factory is None:
        raise RuntimeError("openai.OpenAI is required in the isolated INTERVENOR env")

    base_url = str(payload.get("base_url") or "")
    if base_url and not base_url.endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"

    client = openai_factory(
        base_url=base_url or None,
        api_key=str(payload.get("api_key") or "sk-placeholder"),
        timeout=float(payload.get("timeout_sec") or 0.0) or None,
        max_retries=int(payload.get("max_retries") or 0),
    )
    compat_config = compat_config_from_extra(cast(dict[str, Any], payload))
    request_kwargs, request_meta = apply_compat_request_profile(
        {
            "model": str(payload.get("model") or ""),
            "messages": [{"role": "user", "content": str(payload.get("prompt") or "")}],
            "temperature": float(payload.get("temperature") or 0.0),
            "max_tokens": int(payload.get("max_tokens") or 1024),
            **({"seed": int(payload["seed"])} if payload.get("seed") is not None else {}),
            "extra_body": _usage_extra_body(),
        },
        compat_config=compat_config,
        base_url=base_url,
    )
    response = client.chat.completions.create(**request_kwargs)
    normalized = normalize_chat_completion_response(
        response,
        compat_config=compat_config,
        base_url=base_url,
        request_meta=request_meta,
    )
    try:
        message = require_final_content(normalized, base_url=base_url)
    except OpenAICompatResponseError as exc:
        response_path.write_text(
            json.dumps(
                {
                    "error": "openai_compat_reasoning_only",
                    "compat_message": str(exc),
                    "compat_meta": exc.meta,
                    "meta": _finish_reason_payload(response),
                    "usage": _usage_payload(getattr(response, "usage", None)),
                }
            ),
            encoding="utf-8",
        )
        return 0
    usage_payload = _usage_payload(getattr(response, "usage", None))
    response_path.write_text(
        json.dumps(
            {
                "text": message,
                "usage": usage_payload,
                "meta": _finish_reason_payload(response),
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
