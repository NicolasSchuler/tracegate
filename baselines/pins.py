"""Pinned baseline repo commits and env-client compatibility helpers.

This module is the shared source of truth for:

- baseline repo commit pins referenced by ``baselines/setup_repos.sh``
- env-client compatibility declarations for the lambda-RLM and RLM shims
- lightweight adapter setup checks that verify the checked-out repo still
  matches the env-client contract
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedRepo:
    """Pinned metadata for one external baseline repository."""

    name: str
    clone_dir: str
    repo_url: str
    commit: str
    env_client_label: str | None = None


PINNED_REPOS: dict[str, PinnedRepo] = {
    "ldb": PinnedRepo(
        name="LDB",
        clone_dir="LLMDebugger",
        repo_url="https://github.com/FloridSleeves/LLMDebugger.git",
        commit="49ac191f181d47911cf38e5b9944fbbe6d4a6e60",
    ),
    "intervenor": PinnedRepo(
        name="INTERVENOR",
        clone_dir="INTERVENOR",
        repo_url="https://github.com/NEUIR/INTERVENOR.git",
        commit="45c3e461f1c169baf9e412733ec8548f58ab1713",
    ),
    "tracecoder": PinnedRepo(
        name="TraceCoder",
        clone_dir="TraceCoder",
        repo_url="https://github.com/CSMA-Research-Group/TraceCoder.git",
        commit="912e3bcbe14b5435674dc00a008e146a3a32b49c",
    ),
    "lambda_rlm": PinnedRepo(
        name="lambda-RLM",
        clone_dir="lambda-RLM",
        repo_url="https://github.com/lambda-calculus-LLM/lambda-RLM.git",
        commit="c70c6db2cf8498eaf05fa6999dc41088827158f4",
        env_client_label="lambda-RLM",
    ),
    "rlm": PinnedRepo(
        name="RLM",
        clone_dir="rlm",
        repo_url="https://github.com/alexzhang13/rlm.git",
        commit="4eba6ec7f2f0d5bec94d1f507ffcd79ad4ee7015",
        env_client_label="rlm",
    ),
}


def get_pinned_repo(key: str) -> PinnedRepo:
    """Return pinned repo metadata for *key*."""
    normalized_key = str(key or "").strip()
    try:
        return PINNED_REPOS[normalized_key]
    except KeyError as exc:
        raise KeyError(f"unknown pinned baseline repo: {normalized_key!r}") from exc


def get_pinned_repo_commit(key: str) -> str:
    """Return the pinned commit hash for *key*."""
    return get_pinned_repo(key).commit


def get_expected_env_client_compatibility(key: str) -> str:
    """Return the canonical ``_COMPATIBLE_WITH`` string for an env-client shim."""
    repo = get_pinned_repo(key)
    label = repo.env_client_label
    if not label:
        raise KeyError(f"baseline {key!r} does not declare an env-client compatibility target")
    return f"{label}@{repo.commit}"
