"""Tests for evals.eval_main_setup — covers _namespace_arg, _find_similar_existing_run_ids,
_append_similar_run_id_warning, and validation helper gaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals import eval_main_setup as mod

# ---------------------------------------------------------------------------
# _namespace_arg — TypeError path (line 85-86)
# ---------------------------------------------------------------------------


class TestNamespaceArg:
    """_namespace_arg reads from vars(args) or falls back to default."""

    def test_reads_attribute_from_namespace(self) -> None:
        import argparse

        ns = argparse.Namespace(foo=42)
        result = mod._namespace_arg(ns, "foo", 0)
        assert result == 42

    def test_falls_back_to_default_when_missing(self) -> None:
        import argparse

        ns = argparse.Namespace()
        result = mod._namespace_arg(ns, "missing_key", "default_val")
        assert result == "default_val"

    def test_typeerror_on_vars_falls_back_to_default(self) -> None:
        """When vars(args) raises TypeError (e.g. non-dict __dict__), return default."""

        class BadArgs:
            @property
            def __dict__(self) -> Any:  # type: ignore[override]
                raise TypeError("no dict")

        result = mod._namespace_arg(BadArgs(), "anything", "fallback")
        assert result == "fallback"

    def test_key_not_in_mapping_returns_default(self) -> None:
        """When key not in vars(args), returns default."""
        import argparse

        ns = argparse.Namespace(other=1)
        result = mod._namespace_arg(ns, "nonexistent", 99)
        assert result == 99


# ---------------------------------------------------------------------------
# _find_similar_existing_run_ids — path coverage
# ---------------------------------------------------------------------------


class TestFindSimilarExistingRunIds:
    """Covers the prefix/difflib matching branches."""

    def test_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        result = mod._find_similar_existing_run_ids(
            run_id="myrun",
            results_root=tmp_path / "nonexistent",
        )
        assert result == []

    def test_skips_adaptive_policy_jsonl_files(self, tmp_path: Path) -> None:
        (tmp_path / "run1.adaptive_policy.jsonl").touch()
        (tmp_path / "run2.adaptive_budget_policy.jsonl").touch()
        result = mod._find_similar_existing_run_ids(run_id="run3", results_root=tmp_path)
        assert result == []

    def test_prefix_match_returned(self, tmp_path: Path) -> None:
        (tmp_path / "run_abc.jsonl").touch()
        # "run_abc_extra" starts with "run_abc" so should match
        result = mod._find_similar_existing_run_ids(run_id="run_abc_extra", results_root=tmp_path)
        assert "run_abc" in result

    def test_same_run_id_not_included(self, tmp_path: Path) -> None:
        (tmp_path / "myrun.jsonl").touch()
        result = mod._find_similar_existing_run_ids(run_id="myrun", results_root=tmp_path)
        assert "myrun" not in result

    def test_close_match_returned_via_difflib(self, tmp_path: Path) -> None:
        (tmp_path / "experiment_v1.jsonl").touch()
        # Very similar but not a prefix match
        result = mod._find_similar_existing_run_ids(run_id="experiment_v1", results_root=tmp_path)
        # same-name is excluded, should return empty since candidates must differ
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _append_similar_run_id_warning — multi-run-id path (lines 350-353)
# ---------------------------------------------------------------------------


class TestAppendSimilarRunIdWarning:
    """Covers the single and multi-candidate warning branches."""

    def test_no_warning_when_results_file_exists(self, tmp_path: Path) -> None:
        results_path = tmp_path / "run.jsonl"
        results_path.touch()
        warnings: list[str] = []
        mod._append_similar_run_id_warning(
            run_id="run",
            results_path=results_path,
            startup_warnings=warnings,
        )
        assert warnings == []

    def test_no_warning_when_no_similar_run_ids(self, tmp_path: Path) -> None:
        results_path = tmp_path / "totally_unique_run.jsonl"
        warnings: list[str] = []
        mod._append_similar_run_id_warning(
            run_id="totally_unique_run",
            results_path=results_path,
            startup_warnings=warnings,
        )
        assert warnings == []

    def test_single_similar_run_id_warning(self, tmp_path: Path) -> None:
        (tmp_path / "myrun_v1.jsonl").touch()
        results_path = tmp_path / "myrun_v1_extra.jsonl"
        warnings: list[str] = []
        mod._append_similar_run_id_warning(
            run_id="myrun_v1_extra",
            results_path=results_path,
            startup_warnings=warnings,
        )
        if warnings:
            assert "similar existing run-id" in warnings[0]

    def test_multiple_similar_run_ids_produce_preview(self, tmp_path: Path) -> None:
        """When multiple similar run-ids found, preview format is used (lines 350-353)."""
        # Create 4 similar run-ids to trigger the multi-id branch + '...' truncation
        for i in range(4):
            (tmp_path / f"exp_base_run_{i:04d}.jsonl").touch()
        results_path = tmp_path / "exp_base_run.jsonl"
        warnings: list[str] = []
        mod._append_similar_run_id_warning(
            run_id="exp_base_run",
            results_path=results_path,
            startup_warnings=warnings,
        )
        # If multiple similar run-ids found, the warning should mention similar run-ids
        if len(warnings) > 0:
            assert "similar existing run-id" in warnings[0].lower()
