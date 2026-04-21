"""Optional token-counting helpers for eval-only telemetry.

These helpers intentionally degrade to "estimate unavailable" when
``tiktoken`` is not installed or the tokenizer cannot be resolved.
"""

from __future__ import annotations

from typing import Any

_FALLBACK_ENCODINGS = ("o200k_base", "cl100k_base")


def estimate_text_token_count(text: str, *, model: str = "") -> dict[str, Any]:
    """Estimate token count for *text* using ``tiktoken`` when available."""
    normalized_text = str(text or "")
    normalized_model = str(model or "").strip()
    result: dict[str, Any] = {
        "available": False,
        "token_count": None,
        "model": normalized_model,
        "tokenizer": "",
        "resolution_mode": "",
        "error": "",
    }
    try:
        import tiktoken  # pyright: ignore[reportMissingImports]
    except ImportError:
        result["error"] = "tiktoken_unavailable"
        return result

    encoding: Any = None
    resolution_mode = ""
    resolution_error = ""
    if normalized_model:
        try:
            encoding = tiktoken.encoding_for_model(normalized_model)
            resolution_mode = "model"
        except Exception as exc:  # pragma: no cover - exact exception type is library-defined
            resolution_error = f"encoding_for_model_failed:{type(exc).__name__}"

    if encoding is None:
        for encoding_name in _FALLBACK_ENCODINGS:
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                resolution_mode = f"fallback:{encoding_name}"
                break
            except Exception:  # pragma: no cover - exact exception type is library-defined
                continue

    if encoding is None:
        result["error"] = resolution_error or "tokenizer_resolution_failed"
        return result

    try:
        token_count = len(encoding.encode(normalized_text))
    except Exception as exc:  # pragma: no cover - exact exception type is library-defined
        result["tokenizer"] = str(getattr(encoding, "name", "") or "")
        result["resolution_mode"] = resolution_mode
        result["error"] = f"token_count_failed:{type(exc).__name__}"
        return result

    result["available"] = True
    result["token_count"] = int(token_count)
    result["tokenizer"] = str(getattr(encoding, "name", "") or "")
    result["resolution_mode"] = resolution_mode
    if resolution_error and resolution_mode.startswith("fallback:"):
        result["error"] = resolution_error
    return result
