from __future__ import annotations

import os

from _repo_env_defaults import DEFAULT_OPENAI_API_KEY, ensure_default_openai_api_key


def test_ensure_default_openai_api_key_sets_local_when_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    value = ensure_default_openai_api_key()
    assert value == DEFAULT_OPENAI_API_KEY
    assert os.environ["OPENAI_API_KEY"] == DEFAULT_OPENAI_API_KEY


def test_ensure_default_openai_api_key_preserves_existing_value(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real-key")
    value = ensure_default_openai_api_key()
    assert value == "real-key"
    assert os.environ["OPENAI_API_KEY"] == "real-key"
