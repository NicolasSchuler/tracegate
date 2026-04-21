"""Tests for evals/token_counting.py coverage gaps.

Covers:
- tiktoken unavailable path (ImportError)
- tokenizer_resolution_failed path (all fallback encodings fail)
- successful estimation with fallback encoding
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from evals.token_counting import estimate_text_token_count


class TestEstimateTextTokenCountNoTiktoken:
    """Returns 'tiktoken_unavailable' when tiktoken is not installed."""

    def test_returns_unavailable_when_tiktoken_missing(self) -> None:
        # Simulate tiktoken not being importable
        with patch.dict(sys.modules, {"tiktoken": None}):
            result = estimate_text_token_count("hello world")
        assert result["available"] is False
        assert result["error"] == "tiktoken_unavailable"
        assert result["token_count"] is None


class TestEstimateTextTokenCountResolutionFailed:
    """Returns 'tokenizer_resolution_failed' when all fallback encodings fail."""

    def test_returns_resolution_failed_when_all_fallbacks_fail(self) -> None:
        fake_tiktoken = MagicMock()
        # encoding_for_model: not called because model is blank
        # get_encoding: always raises
        fake_tiktoken.get_encoding.side_effect = Exception("no encoding")

        with patch.dict(sys.modules, {"tiktoken": fake_tiktoken}):
            result = estimate_text_token_count("hello world", model="")

        assert result["available"] is False
        assert result["error"] == "tokenizer_resolution_failed"
        assert result["token_count"] is None


class TestEstimateTextTokenCountSuccessWithFallback:
    """Returns token count via fallback encoding when model name is not specified."""

    def test_successful_estimation_via_fallback_encoding(self) -> None:
        fake_encoding = MagicMock()
        fake_encoding.name = "o200k_base"
        fake_encoding.encode.return_value = [1, 2, 3]  # 3 tokens

        fake_tiktoken = MagicMock()
        fake_tiktoken.get_encoding.return_value = fake_encoding

        with patch.dict(sys.modules, {"tiktoken": fake_tiktoken}):
            result = estimate_text_token_count("hello world", model="")

        assert result["available"] is True
        assert result["token_count"] == 3
        assert result["tokenizer"] == "o200k_base"
        assert result["resolution_mode"].startswith("fallback:")
        assert result["error"] == ""


class TestEstimateTextTokenCountFallbackWithResolutionError:
    """Returns error from model lookup when model name fails but fallback succeeds (line 68)."""

    def test_resolution_error_propagated_when_model_fails_but_fallback_succeeds(self) -> None:
        fake_encoding = MagicMock()
        fake_encoding.name = "o200k_base"
        fake_encoding.encode.return_value = [1, 2, 3]

        fake_tiktoken = MagicMock()
        # encoding_for_model raises (model specified but fails)
        fake_tiktoken.encoding_for_model.side_effect = Exception("unknown model")
        # fallback get_encoding succeeds
        fake_tiktoken.get_encoding.return_value = fake_encoding

        with patch.dict(sys.modules, {"tiktoken": fake_tiktoken}):
            result = estimate_text_token_count("hello world", model="unknown-model")

        assert result["available"] is True
        assert result["token_count"] == 3
        assert result["resolution_mode"].startswith("fallback:")
        # resolution_error was set from the model lookup failure, carried into result
        assert "encoding_for_model_failed" in result["error"]
