"""TOON-for-LLM encoder: compact snapshot payloads for LLM consumption.

Bridges ``llmdebug.toon_encoder`` and ``llmdebug.compact_keys`` into a single
encode-for-LLM path.  The encoding pipeline is:

  1. Optionally apply compact keys (``compact_keys()``) to shorten structural
     key names while preserving user-data keys (locals, args, params).
  2. Pre-serialize complex nested values inside user-data keys (``locals``,
     ``args``, etc.) as JSON strings so that TOON's tabular format does not
     flatten them into the parent dict.
  3. If ``use_toon=True`` **and** the ``toons`` package is installed, encode
     via TOON (tabular format for uniform arrays like frames).  Otherwise
     fall back to minified JSON.
  4. All LLM-bound output uses minified separators --- no ``indent=2``.

This module is feature-gated: callers in the eval harness should only enable
TOON encoding when the active condition is adaptive (``layered_adaptive_gated``
or ``layered_adaptive_llm``).  Non-adaptive conditions use standard JSON to
preserve baseline reproducibility.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any

logger = logging.getLogger(__name__)

_TOON_FALLBACK_WARNING_EMITTED = False

# Keys whose values are user-data dicts/lists that TOON would flatten.
# We pre-serialize these as JSON strings before TOON encoding and restore
# them after decoding to preserve nested structure.
_USER_DATA_KEYS_FULL = {"locals", "locals_meta", "args", "args_meta", "params"}
_USER_DATA_KEYS_COMPACT = {"_loc", "_lm", "_args", "_am", "_p"}
_USER_DATA_KEYS = _USER_DATA_KEYS_FULL | _USER_DATA_KEYS_COMPACT


def encode_for_llm(
    payload: dict[str, Any],
    *,
    use_toon: bool = False,
    use_compact_keys: bool = True,
) -> str:
    """Encode a snapshot / evidence payload for LLM consumption.

    Args:
        payload: Snapshot or evidence dictionary.
        use_toon: When *True* and the ``toons`` package is available, encode
            as TOON.  Falls back to compact JSON when ``toons`` is missing.
        use_compact_keys: Apply ``compact_keys()`` to shorten structural keys
            before encoding.  User-data keys (locals, args) are preserved.

    Returns:
        Encoded string (TOON or minified JSON).
    """
    data: dict[str, Any] | Any = payload

    if use_compact_keys:
        from llmdebug.compact_keys import compact_keys

        data = compact_keys(payload)

    if use_toon:
        prepared = _prepare_for_toon(data)
        toon_result = _try_toon_encode(prepared)
        if toon_result is not None:
            return toon_result
        # Fallback: compact JSON (toons unavailable or encoding error)

    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)


def decode_for_llm(
    encoded: str,
    *,
    expand: bool = True,
) -> dict[str, Any]:
    """Decode an LLM-encoded payload back to a full-key dictionary.

    Handles both TOON and JSON input.  When *expand* is True, applies
    ``expand_keys()`` to restore original key names.

    Args:
        encoded: TOON or JSON encoded string.
        expand: Restore compact keys to full names.

    Returns:
        Decoded dictionary with optionally expanded keys.
    """
    # Try JSON first --- minified JSON is unambiguous and fast to detect.
    data: dict[str, Any] | None = None
    try:
        parsed = json.loads(encoded)
        if isinstance(parsed, dict):
            data = parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # If JSON did not work, try TOON.
    if data is None:
        data = _try_toon_decode(encoded)

    if data is None:
        msg = "Could not decode payload as JSON or TOON"
        raise ValueError(msg)

    # Restore pre-serialized user-data keys from TOON encoding.
    data = _restore_from_toon(data)

    if expand:
        from llmdebug.compact_keys import expand_keys

        expanded = expand_keys(data)
        if isinstance(expanded, dict):
            return expanded

    return data


def estimate_token_savings(
    payload: dict[str, Any],
    *,
    use_toon: bool = True,
    use_compact_keys: bool = True,
    model: str = "",
) -> dict[str, Any]:
    """Estimate token savings from TOON + compact-key encoding.

    Returns a dict with keys:
        - ``json_chars``: character count of pretty-printed JSON baseline.
        - ``encoded_chars``: character count of the encoded output.
        - ``savings_chars``: absolute character savings.
        - ``savings_frac``: fractional savings (0.0-1.0).
        - ``encoding``: ``"toon"`` or ``"json_compact"``.
    """
    json_baseline = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    encoded = encode_for_llm(payload, use_toon=use_toon, use_compact_keys=use_compact_keys)

    json_chars = len(json_baseline)
    encoded_chars = len(encoded)
    savings_chars = max(0, json_chars - encoded_chars)
    savings_frac = savings_chars / json_chars if json_chars > 0 else 0.0

    from evals.token_counting import estimate_text_token_count

    is_toon = use_toon and _toons_available()
    json_token_estimate = estimate_text_token_count(json_baseline, model=model)
    encoded_token_estimate = estimate_text_token_count(encoded, model=model)
    token_estimate_available = bool(
        json_token_estimate.get("available") and encoded_token_estimate.get("available")
    )
    json_tokens_estimate = (
        int(json_token_estimate.get("token_count") or 0) if token_estimate_available else None
    )
    encoded_tokens_estimate = (
        int(encoded_token_estimate.get("token_count") or 0) if token_estimate_available else None
    )
    savings_tokens_estimate = (
        max(0, json_tokens_estimate - encoded_tokens_estimate)
        if isinstance(json_tokens_estimate, int) and isinstance(encoded_tokens_estimate, int)
        else None
    )
    savings_tokens_frac = (
        (savings_tokens_estimate / json_tokens_estimate)
        if isinstance(savings_tokens_estimate, int) and json_tokens_estimate not in (None, 0)
        else None
    )
    token_estimate_error = ""
    if not token_estimate_available:
        token_estimate_error = str(
            json_token_estimate.get("error") or encoded_token_estimate.get("error") or ""
        )
    token_estimate_tokenizer = str(
        encoded_token_estimate.get("tokenizer") or json_token_estimate.get("tokenizer") or ""
    )

    return {
        "json_chars": json_chars,
        "encoded_chars": encoded_chars,
        "savings_chars": savings_chars,
        "savings_frac": savings_frac,
        "encoding": "toon" if is_toon else "json_compact",
        "token_estimate_available": token_estimate_available,
        "token_estimate_error": token_estimate_error,
        "token_estimate_model": str(model or ""),
        "token_estimate_tokenizer": token_estimate_tokenizer,
        "json_tokens_estimate": json_tokens_estimate,
        "encoded_tokens_estimate": encoded_tokens_estimate,
        "savings_tokens_estimate": savings_tokens_estimate,
        "savings_tokens_frac": savings_tokens_frac,
    }


# ---------------------------------------------------------------------------
# TOON preparation: pre-serialize user-data dicts that would be flattened
# ---------------------------------------------------------------------------


def _prepare_for_toon(data: Any) -> Any:
    """Pre-serialize user-data dict/list values as JSON strings.

    TOON's tabular format flattens nested dicts inside array items into the
    parent dict.  We prevent this by serializing user-data values (locals,
    args, etc.) as minified JSON strings before TOON encoding.
    """
    if isinstance(data, dict):
        result: dict[str, Any] = {}
        for k, v in data.items():
            if k in _USER_DATA_KEYS and isinstance(v, (dict, list)):
                result[k] = json.dumps(v, separators=(",", ":"), ensure_ascii=False, default=str)
            else:
                result[k] = _prepare_for_toon(v)
        return result
    if isinstance(data, list):
        return [_prepare_for_toon(item) for item in data]
    return data


def _restore_from_toon(data: Any) -> Any:
    """Restore pre-serialized user-data values from JSON strings."""
    if isinstance(data, dict):
        result: dict[str, Any] = {}
        for k, v in data.items():
            if k in _USER_DATA_KEYS and isinstance(v, str):
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    result[k] = v
            else:
                result[k] = _restore_from_toon(v)
        return result
    if isinstance(data, list):
        return [_restore_from_toon(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# TOON availability and encode/decode helpers
# ---------------------------------------------------------------------------


def _toons_available() -> bool:
    """Check whether the ``toons`` package is importable."""
    try:
        import toons  # pyright: ignore[reportMissingImports]  # noqa: F401

        return True
    except ImportError:
        return False


def _try_toon_encode(data: Any) -> str | None:
    """Attempt TOON encoding; return *None* on failure or missing package."""
    global _TOON_FALLBACK_WARNING_EMITTED
    try:
        import toons  # pyright: ignore[reportMissingImports]

        return toons.dumps(data)  # type: ignore[no-any-return]
    except ImportError:
        if not _TOON_FALLBACK_WARNING_EMITTED:
            warnings.warn(
                "toons package not installed; falling back to compact JSON. "
                "Install with: pip install 'llmdebug[toon]'",
                stacklevel=2,
            )
            _TOON_FALLBACK_WARNING_EMITTED = True
        return None
    except Exception:
        logger.debug("TOON encoding failed, falling back to compact JSON", exc_info=True)
        return None


def _try_toon_decode(encoded: str) -> dict[str, Any] | None:
    """Attempt TOON decoding; return *None* on failure or non-TOON input."""
    try:
        import toons  # pyright: ignore[reportMissingImports]

        result = toons.loads(encoded)
        if isinstance(result, dict):
            return result
        return None
    except ImportError:
        return None
    except Exception:
        return None
