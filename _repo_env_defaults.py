"""Repo-local environment defaults for developer workflows.

These defaults are intentionally scoped to the repository checkout rather than
the distributed ``llmdebug`` package. They keep local OpenAI-compatible runs
from failing early when a key is irrelevant but some code path still expects
the environment variable to exist.
"""

from __future__ import annotations

import os

DEFAULT_OPENAI_API_KEY = "local"


def ensure_default_openai_api_key() -> str:
    """Populate ``OPENAI_API_KEY`` with the repo default when blank or unset."""
    current = os.environ.get("OPENAI_API_KEY", "")
    if isinstance(current, str) and current.strip():
        return current
    os.environ["OPENAI_API_KEY"] = DEFAULT_OPENAI_API_KEY
    return DEFAULT_OPENAI_API_KEY
