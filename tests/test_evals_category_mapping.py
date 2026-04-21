"""Tests for evals.category_mapping — external dataset type mappings."""

from __future__ import annotations

from evals.category_mapping import (
    infer_snapshot_dependency,
    map_condefects_type,
    map_debugbench_type,
    map_humanevalpack_type,
    map_mutmut_operator,
    map_pyresbugs_type,
)

# ---------------------------------------------------------------------------
# DebugBench mapping
# ---------------------------------------------------------------------------


class TestDebugBenchMapping:
    def test_syntax_error(self) -> None:
        assert map_debugbench_type("syntax error") == "syntax_error"

    def test_reference_error(self) -> None:
        assert map_debugbench_type("reference error") == "reference_error"

    def test_logic_error(self) -> None:
        assert map_debugbench_type("logic error") == "logic_error"

    def test_multiple_error(self) -> None:
        assert map_debugbench_type("multiple error") == "logic_error"

    def test_case_insensitive(self) -> None:
        assert map_debugbench_type("Syntax Error") == "syntax_error"
        assert map_debugbench_type("LOGIC ERROR") == "logic_error"

    def test_whitespace_tolerance(self) -> None:
        assert map_debugbench_type("  syntax error  ") == "syntax_error"

    def test_unknown_type_falls_back(self) -> None:
        assert map_debugbench_type("totally_unknown") == "logic_error"


# ---------------------------------------------------------------------------
# HumanEvalPack mapping
# ---------------------------------------------------------------------------


class TestHumanEvalPackMapping:
    def test_missing_logic(self) -> None:
        assert map_humanevalpack_type("missing logic") == "logic_error"

    def test_excess_logic(self) -> None:
        assert map_humanevalpack_type("excess logic") == "logic_error"

    def test_value_misuse(self) -> None:
        assert map_humanevalpack_type("value misuse") == "parse_value"

    def test_operator_misuse(self) -> None:
        assert map_humanevalpack_type("operator misuse") == "logic_error"

    def test_variable_misuse(self) -> None:
        assert map_humanevalpack_type("variable misuse") == "reference_error"

    def test_function_misuse(self) -> None:
        assert map_humanevalpack_type("function misuse") == "cross_file_contract"

    def test_unknown_falls_back(self) -> None:
        assert map_humanevalpack_type("something_else") == "logic_error"


# ---------------------------------------------------------------------------
# ConDefects mapping
# ---------------------------------------------------------------------------


class TestConDefectsMapping:
    def test_wrong_answer(self) -> None:
        assert map_condefects_type("wrong answer") == "logic_error"

    def test_runtime_error(self) -> None:
        assert map_condefects_type("runtime error") == "reference_error"

    def test_time_limit_exceeded(self) -> None:
        assert map_condefects_type("time limit exceeded") == "logic_error"

    def test_unknown_falls_back(self) -> None:
        assert map_condefects_type("compilation error") == "logic_error"


# ---------------------------------------------------------------------------
# mutmut operator mapping
# ---------------------------------------------------------------------------


class TestMutmutOperatorMapping:
    def test_arithmetic(self) -> None:
        assert map_mutmut_operator("arithmetic") == "logic_error"

    def test_comparison(self) -> None:
        assert map_mutmut_operator("comparison") == "logic_error"

    def test_removed_call(self) -> None:
        assert map_mutmut_operator("removed_call") == "cross_file_contract"

    def test_changed_return(self) -> None:
        assert map_mutmut_operator("changed_return") == "hidden_state"

    def test_changed_default(self) -> None:
        assert map_mutmut_operator("changed_default") == "mutability_aliasing"

    def test_string(self) -> None:
        assert map_mutmut_operator("string") == "parse_value"


# ---------------------------------------------------------------------------
# PyResBugs acronym mapping
# ---------------------------------------------------------------------------


class TestPyResBugsAcronymMapping:
    def test_known_acronym(self) -> None:
        assert map_pyresbugs_type("MIA") == "path_import_env"

    def test_case_insensitive_acronym(self) -> None:
        assert map_pyresbugs_type("mia") == "path_import_env"

    def test_unknown_acronym_falls_back(self) -> None:
        assert map_pyresbugs_type("UNKNOWN") == "logic_error"


# ---------------------------------------------------------------------------
# Snapshot dependency inference
# ---------------------------------------------------------------------------


class TestSnapshotDependencyInference:
    def test_syntax_error_is_none(self) -> None:
        assert infer_snapshot_dependency("syntax_error") == "none"

    def test_reference_error_is_none(self) -> None:
        assert infer_snapshot_dependency("reference_error") == "none"

    def test_logic_error_is_helpful(self) -> None:
        assert infer_snapshot_dependency("logic_error") == "helpful"

    def test_hidden_state_is_required(self) -> None:
        assert infer_snapshot_dependency("hidden_state") == "required"

    def test_async_temporal_is_required(self) -> None:
        assert infer_snapshot_dependency("async_temporal") == "required"

    def test_path_import_env_is_none(self) -> None:
        assert infer_snapshot_dependency("path_import_env") == "none"

    def test_traceback_controls_is_none(self) -> None:
        assert infer_snapshot_dependency("traceback_controls") == "none"

    def test_unknown_category_defaults_to_helpful(self) -> None:
        assert infer_snapshot_dependency("something_unknown") == "helpful"
