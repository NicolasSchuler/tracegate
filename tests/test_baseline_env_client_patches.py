"""Tests for the env_client monkey-patch contract.

Validates that ``_PATCH_POINTS`` and ``_COMPATIBLE_WITH`` sentinel
constants in each env_client module stay in sync with the actual
patches applied in the ``_install_tracking()`` and ``main()`` functions.

These tests inspect the source code (AST) of the env_client modules to
verify that every declared patch point corresponds to an actual
assignment in the patching functions, and vice versa.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from baselines.lambda_rlm import env_client as lambda_rlm_env_client
from baselines.pins import get_expected_env_client_compatibility
from baselines.rlm import env_client as rlm_env_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_module_patches(source: str) -> set[str]:
    """Extract ``module.attr = ...`` assignment targets from source.

    Returns dotted strings like ``"clients_mod.get_client"`` for every
    module-level attribute assignment found in the source.
    """
    tree = ast.parse(textwrap.dedent(source))
    patches: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    patches.add(f"{target.value.id}.{target.attr}")
    return patches


def _extract_instance_patches(source: str) -> set[str]:
    """Extract ``instance.attr = ...`` patches on specific variables.

    Looks for patterns like ``rlm._parse_task_type = ...`` where the
    variable name suggests an instance patch rather than a module patch.
    """
    tree = ast.parse(textwrap.dedent(source))
    patches: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    var_name = target.value.id
                    attr_name = target.attr
                    # Skip well-known module aliases and non-patch assignments
                    if var_name in {"self", "cls", "os", "sys", "json", "time"}:
                        continue
                    # Instance patches: variable is not a module alias (lowercase,
                    # not ending with _mod) and attribute starts with _
                    if not var_name.endswith("_mod") and attr_name.startswith("_"):
                        patches.add(f"{var_name}.{attr_name}")
    return patches


def _normalize_patch_point(point: str) -> str:
    """Normalize a _PATCH_POINTS entry to the local form used in source.

    ``"rlm.clients.get_client"`` -> ``"clients_mod.get_client"``
    ``"rlm.lambda_rlm.get_client"`` -> ``"lambda_mod.get_client"``
    ``"rlm.core.rlm.get_client"`` -> ``"rlm_core_mod.get_client"``
    ``"LambdaRLM._parse_task_type"`` -> ``"rlm._parse_task_type"``
    ``"LambdaRLM._plan"`` -> ``"rlm._plan"``
    ``"RLM._default_answer"`` -> ``"RLM._default_answer"``

    These normalizations match the local alias conventions in the env_client
    modules.
    """
    # Instance method patches on LambdaRLM -> local variable "rlm"
    if point.startswith("LambdaRLM."):
        return f"rlm.{point.split('.', 1)[1]}"

    # Module patches: map dotted module paths to local aliases
    _MODULE_ALIASES = {
        "rlm.clients.get_client": "clients_mod.get_client",
        "rlm.lambda_rlm.get_client": "lambda_mod.get_client",
        "rlm.core.rlm.get_client": "rlm_core_mod.get_client",
    }
    return _MODULE_ALIASES.get(point, point)


def _extract_string_constant_assignment(source: str, name: str) -> str | None:
    """Return the assigned literal string value for a module constant, if any."""
    tree = ast.parse(textwrap.dedent(source))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id != name:
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            return None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        return value.value
                    return None
    return None


# ---------------------------------------------------------------------------
# lambda-RLM env_client tests
# ---------------------------------------------------------------------------


class TestLambdaRLMEnvClientContract:
    """Verify lambda-RLM env_client patch contract constants."""

    def test_patch_points_constant_exists(self) -> None:
        assert hasattr(lambda_rlm_env_client, "_PATCH_POINTS")
        assert isinstance(lambda_rlm_env_client._PATCH_POINTS, tuple)
        assert len(lambda_rlm_env_client._PATCH_POINTS) > 0

    def test_compatible_with_constant_exists(self) -> None:
        assert hasattr(lambda_rlm_env_client, "_COMPATIBLE_WITH")
        assert isinstance(lambda_rlm_env_client._COMPATIBLE_WITH, str)
        assert lambda_rlm_env_client._COMPATIBLE_WITH.strip()

    def test_patch_points_are_strings(self) -> None:
        for point in lambda_rlm_env_client._PATCH_POINTS:
            assert isinstance(point, str), f"Expected str, got {type(point)}"
            assert "." in point, f"Patch point should be dotted: {point!r}"

    def test_module_patches_match_source(self) -> None:
        """Every module-level patch in _PATCH_POINTS appears in source."""
        source = inspect.getsource(lambda_rlm_env_client._install_tracking)
        actual_module_patches = _extract_module_patches(source)

        expected_module_points = {
            _normalize_patch_point(p)
            for p in lambda_rlm_env_client._PATCH_POINTS
            if not p.startswith("LambdaRLM.")
        }

        for expected in expected_module_points:
            assert expected in actual_module_patches, (
                f"Declared patch point {expected!r} not found in "
                f"_install_tracking(). Actual patches: {sorted(actual_module_patches)}"
            )

    def test_instance_patches_match_source(self) -> None:
        """Every instance-level patch in _PATCH_POINTS appears in main()."""
        source = inspect.getsource(lambda_rlm_env_client.main)
        actual_instance_patches = _extract_instance_patches(source)

        expected_instance_points = {
            _normalize_patch_point(p)
            for p in lambda_rlm_env_client._PATCH_POINTS
            if p.startswith("LambdaRLM.")
        }

        for expected in expected_instance_points:
            assert expected in actual_instance_patches, (
                f"Declared instance patch point {expected!r} not found in "
                f"main(). Actual instance patches: {sorted(actual_instance_patches)}"
            )

    def test_no_undeclared_module_patches(self) -> None:
        """No module-level ``get_client`` patches exist outside _PATCH_POINTS."""
        source = inspect.getsource(lambda_rlm_env_client._install_tracking)
        actual_module_patches = _extract_module_patches(source)
        # Filter to only get_client patches (the contract-relevant ones)
        actual_get_client = {p for p in actual_module_patches if "get_client" in p}

        declared = {
            _normalize_patch_point(p)
            for p in lambda_rlm_env_client._PATCH_POINTS
            if not p.startswith("LambdaRLM.")
        }

        undeclared = actual_get_client - declared
        assert not undeclared, (
            f"Undeclared module patches found in _install_tracking(): "
            f"{sorted(undeclared)}. Add them to _PATCH_POINTS."
        )

    def test_expected_patch_count(self) -> None:
        """lambda-RLM has exactly 4 patch points (2 module + 2 instance)."""
        assert len(lambda_rlm_env_client._PATCH_POINTS) == 4

    def test_compatible_with_is_literal_string_constant(self) -> None:
        source = inspect.getsource(lambda_rlm_env_client)
        literal = _extract_string_constant_assignment(source, "_COMPATIBLE_WITH")
        assert literal == lambda_rlm_env_client._COMPATIBLE_WITH

    def test_compatible_with_format(self) -> None:
        """_COMPATIBLE_WITH should match the shared lambda-RLM pin."""
        assert get_expected_env_client_compatibility("lambda_rlm") == (
            lambda_rlm_env_client._COMPATIBLE_WITH
        )


# ---------------------------------------------------------------------------
# RLM env_client tests
# ---------------------------------------------------------------------------


class TestRLMEnvClientContract:
    """Verify RLM env_client patch contract constants."""

    def test_patch_points_constant_exists(self) -> None:
        assert hasattr(rlm_env_client, "_PATCH_POINTS")
        assert isinstance(rlm_env_client._PATCH_POINTS, tuple)
        assert len(rlm_env_client._PATCH_POINTS) > 0

    def test_compatible_with_constant_exists(self) -> None:
        assert hasattr(rlm_env_client, "_COMPATIBLE_WITH")
        assert isinstance(rlm_env_client._COMPATIBLE_WITH, str)
        assert rlm_env_client._COMPATIBLE_WITH.strip()

    def test_patch_points_are_strings(self) -> None:
        for point in rlm_env_client._PATCH_POINTS:
            assert isinstance(point, str), f"Expected str, got {type(point)}"
            assert "." in point, f"Patch point should be dotted: {point!r}"

    def test_module_patches_match_source(self) -> None:
        """Every module-level patch in _PATCH_POINTS appears in source."""
        source = inspect.getsource(rlm_env_client._install_tracking)
        actual_module_patches = _extract_module_patches(source)

        expected_module_points = {
            _normalize_patch_point(p)
            for p in rlm_env_client._PATCH_POINTS
            if not p.startswith("RLM.")
        }

        for expected in expected_module_points:
            assert expected in actual_module_patches, (
                f"Declared patch point {expected!r} not found in "
                f"_install_tracking(). Actual patches: {sorted(actual_module_patches)}"
            )

    def test_instance_patches_match_source(self) -> None:
        """Every instance-level patch in _PATCH_POINTS appears in main()."""
        source = inspect.getsource(rlm_env_client.main)
        actual_instance_patches = _extract_instance_patches(source)

        expected_instance_points = {
            _normalize_patch_point(p) for p in rlm_env_client._PATCH_POINTS if p.startswith("RLM.")
        }

        for expected in expected_instance_points:
            assert expected in actual_instance_patches, (
                f"Declared instance patch point {expected!r} not found in "
                f"main(). Actual instance patches: {sorted(actual_instance_patches)}"
            )

    def test_no_undeclared_module_patches(self) -> None:
        """No module-level ``get_client`` patches exist outside _PATCH_POINTS."""
        source = inspect.getsource(rlm_env_client._install_tracking)
        actual_module_patches = _extract_module_patches(source)
        actual_get_client = {p for p in actual_module_patches if "get_client" in p}

        declared = {
            _normalize_patch_point(p)
            for p in rlm_env_client._PATCH_POINTS
            if not p.startswith("RLM.")
        }

        undeclared = actual_get_client - declared
        assert not undeclared, (
            f"Undeclared module patches found in _install_tracking(): "
            f"{sorted(undeclared)}. Add them to _PATCH_POINTS."
        )

    def test_expected_patch_count(self) -> None:
        """RLM has exactly 3 patch points (2 module + 1 instance)."""
        assert len(rlm_env_client._PATCH_POINTS) == 3

    def test_compatible_with_is_literal_string_constant(self) -> None:
        source = inspect.getsource(rlm_env_client)
        literal = _extract_string_constant_assignment(source, "_COMPATIBLE_WITH")
        assert literal == rlm_env_client._COMPATIBLE_WITH

    def test_compatible_with_format(self) -> None:
        """_COMPATIBLE_WITH should match the shared RLM pin."""
        assert get_expected_env_client_compatibility("rlm") == rlm_env_client._COMPATIBLE_WITH

    def test_expected_instance_patch(self) -> None:
        """RLM env_client declares the repair-specific default-answer patch."""
        instance_points = [
            p
            for p in rlm_env_client._PATCH_POINTS
            if p[0].isupper()  # Instance patches start with class name
        ]
        assert instance_points == ["RLM._default_answer"]


# ---------------------------------------------------------------------------
# Cross-adapter contract consistency
# ---------------------------------------------------------------------------


class TestEnvClientContractConsistency:
    """Verify structural consistency between the two env_client modules."""

    def test_both_modules_have_patch_points(self) -> None:
        assert hasattr(lambda_rlm_env_client, "_PATCH_POINTS")
        assert hasattr(rlm_env_client, "_PATCH_POINTS")

    def test_both_modules_have_compatible_with(self) -> None:
        assert hasattr(lambda_rlm_env_client, "_COMPATIBLE_WITH")
        assert hasattr(rlm_env_client, "_COMPATIBLE_WITH")

    def test_both_compatible_with_values_come_from_shared_pins(self) -> None:
        assert get_expected_env_client_compatibility("lambda_rlm") == (
            lambda_rlm_env_client._COMPATIBLE_WITH
        )
        assert get_expected_env_client_compatibility("rlm") == rlm_env_client._COMPATIBLE_WITH

    def test_patch_points_type_consistent(self) -> None:
        assert type(lambda_rlm_env_client._PATCH_POINTS) is type(rlm_env_client._PATCH_POINTS)

    def test_shared_get_client_patch(self) -> None:
        """Both adapters patch rlm.clients.get_client."""
        assert "rlm.clients.get_client" in lambda_rlm_env_client._PATCH_POINTS
        assert "rlm.clients.get_client" in rlm_env_client._PATCH_POINTS

    def test_distinct_secondary_patches(self) -> None:
        """Each adapter patches a different secondary module for get_client."""
        lambda_points = set(lambda_rlm_env_client._PATCH_POINTS)
        rlm_points = set(rlm_env_client._PATCH_POINTS)

        # lambda-RLM patches rlm.lambda_rlm, RLM patches rlm.core.rlm
        assert "rlm.lambda_rlm.get_client" in lambda_points
        assert "rlm.lambda_rlm.get_client" not in rlm_points
        assert "rlm.core.rlm.get_client" in rlm_points
        assert "rlm.core.rlm.get_client" not in lambda_points

    def test_both_have_main_function(self) -> None:
        """Both env_client modules expose a main() entry point."""
        assert callable(getattr(lambda_rlm_env_client, "main", None))
        assert callable(getattr(rlm_env_client, "main", None))

    def test_both_have_install_tracking(self) -> None:
        """Both env_client modules expose _install_tracking()."""
        assert callable(getattr(lambda_rlm_env_client, "_install_tracking", None))
        assert callable(getattr(rlm_env_client, "_install_tracking", None))

    def test_both_have_char_summary(self) -> None:
        """Both env_client modules expose _char_summary_from_clients()."""
        assert callable(getattr(lambda_rlm_env_client, "_char_summary_from_clients", None))
        assert callable(getattr(rlm_env_client, "_char_summary_from_clients", None))
