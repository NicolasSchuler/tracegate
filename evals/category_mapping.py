"""Maps external dataset bug types to the llmdebug eval taxonomy.

Provides mappings for DebugBench, HumanEvalPack, ConDefects, mutmut
mutation operators, PyResBugs fault types, and QuixBugs, plus a heuristic
for inferring snapshot dependency from category.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DebugBench bug types -> llmdebug categories
# ---------------------------------------------------------------------------

DEBUGBENCH_MAP: dict[str, str] = {
    "syntax error": "syntax_error",
    "reference error": "reference_error",
    "logic error": "logic_error",
    "multiple error": "logic_error",
}


# ---------------------------------------------------------------------------
# HumanEvalPack bug types -> llmdebug categories
# ---------------------------------------------------------------------------

HUMANEVALPACK_MAP: dict[str, str] = {
    "missing logic": "logic_error",
    "excess logic": "logic_error",
    "value misuse": "parse_value",
    "operator misuse": "logic_error",
    "variable misuse": "reference_error",
    "function misuse": "cross_file_contract",
}


# ---------------------------------------------------------------------------
# ConDefects — typically logic errors from competitive programming
# ---------------------------------------------------------------------------

CONDEFECTS_MAP: dict[str, str] = {
    "wrong answer": "logic_error",
    "runtime error": "reference_error",
    "time limit exceeded": "logic_error",
}


# ---------------------------------------------------------------------------
# mutmut mutation operator types -> llmdebug categories
# ---------------------------------------------------------------------------

MUTMUT_OPERATOR_MAP: dict[str, str] = {
    "arithmetic": "logic_error",
    "comparison": "logic_error",
    "boolean": "logic_error",
    "number": "logic_error",
    "string": "parse_value",
    "keyword": "logic_error",
    "removed_call": "cross_file_contract",
    "changed_return": "hidden_state",
    "changed_default": "mutability_aliasing",
}


# ---------------------------------------------------------------------------
# PyResBugs Bug_Type -> llmdebug categories
# Uses the 13 high-level Bug_Type from the ODC fault taxonomy.
# ---------------------------------------------------------------------------

# PyResBugs uses Fault_Acronym (62 ODC categories) as the primary classifier.
# Bug_Type is None for ~94% of rows, so we map by acronym instead.
PYRESBUGS_ACRONYM_MAP: dict[str, str] = {
    # Wrong/missing function call → cross_file_contract
    "MPFC": "cross_file_contract",  # Missing Parameter in Function Call
    "WFCD": "cross_file_contract",  # Wrong Function Called, Different params
    "WFCS": "cross_file_contract",  # Wrong Function Called, Same params
    "MFC": "cross_file_contract",  # Missing Function Call
    "MFCT": "cross_file_contract",  # Missing Function Call (Type)
    "MFCE": "cross_file_contract",  # Missing Function Call Expression
    "EFC": "cross_file_contract",  # Extra Function Call
    # Wrong variable in parameter → reference_error
    "WPFV": "reference_error",  # Wrong Parameter - Variable
    "WVIV": "reference_error",  # Wrong Variable In Value
    # Wrong/missing variable assignment → hidden_state
    "WVAV": "hidden_state",  # Wrong Variable Assigned a Value
    "MVAV": "hidden_state",  # Missing Variable Assignment (Value)
    "MVAE": "hidden_state",  # Missing Variable Assignment (Expression)
    "WVAE": "hidden_state",  # Wrong Variable Assignment Expression
    # Wrong literal/constant → parse_value
    "WPFL": "parse_value",  # Wrong Parameter - Literal
    "WVAL": "parse_value",  # Wrong Variable Assignment Literal
    # Wrong algorithm → logic_error
    "WALL": "logic_error",  # Wrong Algorithm (Large)
    "WALD": "logic_error",  # Wrong Algorithm (Small)
    "WALR": "logic_error",  # Wrong Algorithm (Replacement)
    "WSUIT": "logic_error",  # Wrong Statement Used Instead of
    "WRV": "logic_error",  # Wrong Return Value
    "WPFO": "logic_error",  # Wrong Parameter - Operator
    # Control flow → logic_error
    "MIFS": "logic_error",  # Missing If Statement
    "WLEC": "logic_error",  # Wrong Logical Expression in Control
    "MIES": "logic_error",  # Missing If Exit Statement
    "MIEB": "logic_error",  # Missing If-Else Block
    # Missing statements → logic_error
    "MLPL": "logic_error",  # Missing Loop
    "MLPS": "logic_error",  # Missing Loop Statement
    "MLAC": "logic_error",  # Missing Logic/Assignment/Call
    "MLPA": "logic_error",  # Missing Loop/Path Assignment
    # Import → path_import_env
    "MIA": "path_import_env",  # Missing Import
}


# ---------------------------------------------------------------------------
# Public mapping functions
# ---------------------------------------------------------------------------

_FALLBACK_CATEGORY = "logic_error"


def map_debugbench_type(bug_type: str) -> str:
    """Map a DebugBench bug type to llmdebug category."""
    return DEBUGBENCH_MAP.get(bug_type.lower().strip(), _FALLBACK_CATEGORY)


def map_humanevalpack_type(bug_type: str) -> str:
    """Map a HumanEvalPack bug_type to llmdebug category."""
    return HUMANEVALPACK_MAP.get(bug_type.lower().strip(), _FALLBACK_CATEGORY)


def map_condefects_type(verdict: str) -> str:
    """Map a ConDefects verdict to llmdebug category."""
    return CONDEFECTS_MAP.get(verdict.lower().strip(), _FALLBACK_CATEGORY)


def map_mutmut_operator(operator: str) -> str:
    """Map a mutmut mutation operator type to llmdebug category."""
    return MUTMUT_OPERATOR_MAP.get(operator.lower().strip(), _FALLBACK_CATEGORY)


def map_pyresbugs_type(fault_acronym: str) -> str:
    """Map a PyResBugs Fault_Acronym to llmdebug category."""
    return PYRESBUGS_ACRONYM_MAP.get(fault_acronym.strip().upper(), _FALLBACK_CATEGORY)


def map_quixbugs_type(program_name: str) -> str:
    """Map a QuixBugs program to llmdebug category.

    All QuixBugs bugs are single-line logic errors in classic algorithms.
    """
    del program_name  # All programs map to the same category.
    return "logic_error"


# ---------------------------------------------------------------------------
# Snapshot dependency inference
# ---------------------------------------------------------------------------

_SNAPSHOT_DEPENDENCY_BY_CATEGORY: dict[str, str] = {
    "syntax_error": "none",
    "reference_error": "none",
    "logic_error": "helpful",
    "type_coercion": "helpful",
    "hidden_state": "required",
    "mutability_aliasing": "required",
    "async_temporal": "required",
    "cross_file_contract": "helpful",
    "data_shape_runtime": "required",
    "parse_value": "helpful",
    "path_import_env": "none",
    "traceback_controls": "none",
}


def infer_snapshot_dependency(category: str) -> str:
    """Heuristic: infer snapshot_dependency from category.

    - syntax_error, reference_error, path_import_env -> "none"
      (traceback alone typically sufficient)
    - logic_error, type_coercion, parse_value, cross_file_contract -> "helpful"
    - hidden_state, mutability_aliasing, async_temporal, data_shape_runtime -> "required"
    """
    return _SNAPSHOT_DEPENDENCY_BY_CATEGORY.get(category, "helpful")
