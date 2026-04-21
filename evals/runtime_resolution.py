"""Runtime resolution helpers for env-backed eval defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_CONTAINER_IMAGE_FALLBACK = "llmdebug-eval-runner:latest"
DEFAULT_OPENAI_API_KEY = "local"
OPENAI_API_KEY_ENV_FALLBACKS = ("Z_API", "OPENAI_API_KEY", "LLMDEBUG_EVAL_OPENAI_API_KEY")


@dataclass(frozen=True)
class ResolvedContainerImage:
    """Resolved container image value with provenance."""

    value: str
    source: str
    implicit_latest: bool = False


@dataclass(frozen=True)
class ResolvedApiKey:
    """Resolved API key metadata for eval providers."""

    value: str
    source: str
    env_key: str = ""


def ensure_default_openai_api_key() -> str:
    """Populate ``OPENAI_API_KEY`` with the eval-safe default when blank or unset.

    This helper intentionally lives in ``evals`` rather than depending on the
    repo-root ``_repo_env_defaults`` module. Some containerized eval flows ship
    the ``evals`` package without that repo-local helper on ``sys.path``.
    """
    current = os.environ.get("OPENAI_API_KEY", "")
    if isinstance(current, str) and current.strip():
        return current
    os.environ["OPENAI_API_KEY"] = DEFAULT_OPENAI_API_KEY
    return DEFAULT_OPENAI_API_KEY


def first_nonempty_env(*keys: str) -> ResolvedApiKey:
    """Return the first non-empty environment variable value, if any."""
    for key in keys:
        value = os.getenv(str(key), "")
        if isinstance(value, str) and value.strip():
            return ResolvedApiKey(value=value, source="env", env_key=str(key))
    return ResolvedApiKey(value="", source="missing")


def resolve_default_container_image() -> ResolvedContainerImage:
    """Resolve the default eval container image from environment or fallback."""
    raw = os.getenv("LLMDEBUG_EVAL_CONTAINER_IMAGE", "")
    if isinstance(raw, str) and raw.strip():
        return ResolvedContainerImage(
            value=raw.strip(),
            source="env:LLMDEBUG_EVAL_CONTAINER_IMAGE",
            implicit_latest=False,
        )
    return ResolvedContainerImage(
        value=DEFAULT_CONTAINER_IMAGE_FALLBACK,
        source="implicit_fallback",
        implicit_latest=True,
    )


def resolve_openai_api_key_from_env() -> ResolvedApiKey:
    """Resolve an OpenAI-compatible API key from the standard fallback chain."""
    resolved = first_nonempty_env(*OPENAI_API_KEY_ENV_FALLBACKS)
    if resolved.value:
        return resolved
    ensure_default_openai_api_key()
    return ResolvedApiKey(
        value=DEFAULT_OPENAI_API_KEY,
        source="implicit_default",
        env_key="OPENAI_API_KEY",
    )
