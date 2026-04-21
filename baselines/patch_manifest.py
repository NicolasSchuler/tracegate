"""Explicit inventory of llmdebug baseline patches and runtime shims."""

from __future__ import annotations

from dataclasses import dataclass

from baselines.pins import get_expected_env_client_compatibility, get_pinned_repo_commit


@dataclass(frozen=True)
class BaselinePatchEntry:
    baseline_name: str
    kind: str
    relative_path: str
    scope: tuple[str, ...]
    behavioral_impact: str
    upstream_commit: str
    rationale: str


PATCH_MANIFEST: tuple[BaselinePatchEntry, ...] = (
    BaselinePatchEntry(
        baseline_name="ldb",
        kind="repo_patch",
        relative_path="baselines/ldb/patches/0001-add-llmdebug-dataset-support.patch",
        scope=("compat", "telemetry", "behavioral_adaptation"),
        behavioral_impact="Adds llmdebug dataset compatibility, OpenAI compat controls, and metrics sidecars.",
        upstream_commit=get_pinned_repo_commit("ldb"),
        rationale="Run upstream LDB on llmdebug eval cases while preserving explicit compatibility boundaries.",
    ),
    BaselinePatchEntry(
        baseline_name="tracecoder",
        kind="repo_patch",
        relative_path="baselines/tracecoder/patches/0001-add-llmdebug-repair-mode.patch",
        scope=("compat", "telemetry", "behavioral_adaptation"),
        behavioral_impact="Adds llmdebug dataset support, repair-seeded execution, metrics sidecars, and compat controls.",
        upstream_commit=get_pinned_repo_commit("tracecoder"),
        rationale="Run TraceCoder against llmdebug cases in an explicit patched integration mode.",
    ),
    BaselinePatchEntry(
        baseline_name="intervenor",
        kind="env_shim",
        relative_path="baselines/intervenor/env_client.py",
        scope=("compat", "telemetry"),
        behavioral_impact="Normalizes OpenAI-compatible request/response handling inside isolated INTERVENOR envs.",
        upstream_commit=get_pinned_repo_commit("intervenor"),
        rationale="Preserve request compatibility and usage capture when running official prompt assets through local endpoints.",
    ),
    BaselinePatchEntry(
        baseline_name="lambda_rlm",
        kind="env_shim",
        relative_path="baselines/lambda_rlm/env_client.py",
        scope=("env_shim", "telemetry"),
        behavioral_impact="Monkey-patches upstream client resolution and plan/task tracking for llmdebug runtime capture.",
        upstream_commit=get_expected_env_client_compatibility("lambda_rlm"),
        rationale="Capture usage/char telemetry and compat behavior without modifying the pinned upstream repo checkout.",
    ),
    BaselinePatchEntry(
        baseline_name="rlm",
        kind="env_shim",
        relative_path="baselines/rlm/env_client.py",
        scope=("env_shim", "telemetry"),
        behavioral_impact="Monkey-patches upstream client resolution and injects bounded workspace tools for llmdebug repair runs.",
        upstream_commit=get_expected_env_client_compatibility("rlm"),
        rationale="Adapt generic upstream RLM to explicit single-file repair while capturing telemetry and compatibility diagnostics.",
    ),
)
