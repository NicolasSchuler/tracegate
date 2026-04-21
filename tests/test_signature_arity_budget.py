"""Regression guard for refactored function signature arity."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIGNATURE_BUDGET = 6
_FUNCTIONS_UNDER_BUDGET = {
    "src/llmdebug/__init__.py": (
        "debug_snapshot",
        "snapshot_section",
    ),
    "src/llmdebug/mcp_rca.py": (
        "resolve_rca_advance_context",
        "advance_rca",
    ),
    "src/llmdebug/rca_engine.py": ("build_rca_attempt_record",),
    "src/llmdebug/rca_state.py": ("advance_rca_attempt",),
    "evals/run_eval.py": (
        "_execute_eval_unit",
        "run_one_condition",
    ),
}


def _named_parameter_count(function_node: ast.FunctionDef) -> int:
    parameters = (
        list(function_node.args.posonlyargs)
        + list(function_node.args.args)
        + list(function_node.args.kwonlyargs)
    )
    return sum(arg.arg not in {"self", "cls"} for arg in parameters)


def _find_function_node(module_path: Path, function_name: str) -> ast.FunctionDef:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Could not find function {function_name!r} in {module_path}")


def test_refactored_functions_stay_within_signature_budget() -> None:
    for relative_path, function_names in _FUNCTIONS_UNDER_BUDGET.items():
        module_path = _REPO_ROOT / relative_path
        for function_name in function_names:
            function_node = _find_function_node(module_path, function_name)
            named_parameter_count = _named_parameter_count(function_node)
            assert named_parameter_count <= _SIGNATURE_BUDGET, (
                f"{relative_path}:{function_name} has {named_parameter_count} named parameters; "
                f"budget is {_SIGNATURE_BUDGET}"
            )
