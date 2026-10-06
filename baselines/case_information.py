"""Case-information contract shared by the harness and every baseline adapter.

Defines which case data may reach a model-facing input (review findings ATT-02 and
BAS-07, decision D1). Dataset-derived case metadata (``description``, ``category``,
``difficulty``, ``snapshot_dependency``) stays in ``case.json`` for analysis and is
never written to ``baseline_case_context.json``, so no arm can see it. Each converter
records which case artifacts entered its prompt via :func:`case_information_record`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from baselines.contracts import BASELINE_CONTRACT_VERSION

CASE_INFO_POLICY = "neutral_v1"
"""Identifier recorded in run metadata and per-record baseline metadata."""

MODEL_HIDDEN_CASE_FIELDS = ("description", "category", "difficulty", "snapshot_dependency")
"""``case.json`` fields that are analysis-only and never reach a model."""

TEST_SOURCE_FULL = "full"
"""The full ``test_buggy.py`` source is in the prompt (the native conditions' exposure)."""

TEST_SOURCE_ASSERTS = "asserts"
"""Only the test assertions are in the prompt; used by LDB, whose converter accepts only
plain-assert test bodies, so the assertions carry the whole test content."""

TEST_SOURCE_NONE = "none"
"""No test source reached the prompt. A native arm whose evidence staging or prompt budget
left the test file out records this, so BAS-07 comparisons have both halves."""


def build_baseline_case_context_payload(
    *,
    case_id: str,
    python_args: Sequence[str],
    timeout_sec: int,
) -> dict[str, Any]:
    """Return the ``baseline_case_context.json`` payload under :data:`CASE_INFO_POLICY`."""
    return {
        "schema_version": BASELINE_CONTRACT_VERSION,
        "case_id": str(case_id),
        "python_args": [str(arg) for arg in python_args],
        "timeout_sec": int(timeout_sec),
        "case_info_policy": CASE_INFO_POLICY,
    }


def case_information_record(
    *,
    test_source_exposure: str,
    prompt_files: Sequence[str],
) -> dict[str, Any]:
    """Describe the case artifacts a converter put into its model-facing prompt."""
    return {
        "case_info_policy": CASE_INFO_POLICY,
        "test_source_exposure": str(test_source_exposure),
        "prompt_files": sorted({str(path) for path in prompt_files}),
    }


REFERENCE_TESTS_HEADER = "Reference tests (test_buggy.py):"
"""Header of the full-test-source prompt section; prompt compaction must not cut it off."""


def render_reference_tests(test_code: str) -> str:
    """Render the full test source as a prompt section (``TEST_SOURCE_FULL``)."""
    return REFERENCE_TESTS_HEADER + "\n" + str(test_code).rstrip() + "\n"
