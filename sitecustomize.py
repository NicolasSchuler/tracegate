"""Repo-local Python startup customizations.

Python imports ``sitecustomize`` automatically when the repository root is on
``sys.path``. We use that hook to supply a harmless default API key for local
OpenAI-compatible servers that ignore authentication.
"""

from __future__ import annotations

from _repo_env_defaults import ensure_default_openai_api_key

ensure_default_openai_api_key()
