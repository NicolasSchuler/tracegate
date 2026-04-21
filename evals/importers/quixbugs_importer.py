"""QuixBugs dataset importer for llmdebug eval framework.

Source: GitHub ``jkoppel/QuixBugs`` (MIT license).
40 classic algorithm programs, each with a single-line bug.

Each program has a buggy version in ``python_programs/`` and a corrected
version in ``correct_python_programs/``. Test inputs/outputs are stored
as JSON lines in ``json_testcases/``. Graph-based algorithms use inline
pytest test files in ``python_testcases/`` with a ``Node`` helper class.

contamination_risk = "high" (extensively studied in APR literature).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency, map_quixbugs_type
from evals.importers.base import (
    count_patch_lines,
    generate_oracle_patch,
    has_external_deps,
    ruff_format,
    validate_generated_case,
    write_case_files,
)

from .. import _json as json

# ---------------------------------------------------------------------------
# QuixBugs structure constants
# ---------------------------------------------------------------------------

GRAPH_BASED_PROGRAMS = frozenset(
    {
        "breadth_first_search",
        "depth_first_search",
        "detect_cycle",
        "minimum_spanning_tree",
        "reverse_linked_list",
        "shortest_path_length",
        "shortest_path_lengths",
        "shortest_paths",
        "topological_ordering",
    }
)

MAX_CODE_LINES = 200
MAX_FIX_LINES = 10

# The Node class used by graph-based programs, embedded for self-containment.
_NODE_CLASS = textwrap.dedent("""\
    class Node:
        def __init__(
            self,
            value=None,
            successor=None,
            successors=None,
            predecessors=None,
            incoming_nodes=None,
            outgoing_nodes=None,
        ):
            self.value = value
            self.successor = successor
            self.successors = successors if successors is not None else []
            self.predecessors = predecessors if predecessors is not None else []
            self.incoming_nodes = incoming_nodes if incoming_nodes is not None else []
            self.outgoing_nodes = outgoing_nodes if outgoing_nodes is not None else []

        def __repr__(self):
            return f"Node({self.value!r})"

        def __hash__(self):
            return hash(id(self))

""")


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------


def _generate_json_test(program_name: str, test_cases: list[tuple[list[Any], Any]]) -> str:
    """Generate test_buggy.py from JSON test cases (non-graph programs)."""
    lines = [
        f"from buggy import {program_name}",
        "",
        "",
    ]

    for i, (input_args, expected) in enumerate(test_cases):
        args_repr = ", ".join(repr(a) for a in input_args)
        lines.append(f"def test_case_{i}():")
        lines.append(f"    result = {program_name}({args_repr})")
        lines.append(f"    assert result == {expected!r}")
        lines.append("")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _rewrite_graph_test(test_source: str, program_name: str) -> str | None:
    """Rewrite a graph-based pytest test to import from ``buggy`` module.

    The original test imports from ``python_programs.<name>`` / ``node``.
    We rewrite to import from ``buggy`` (which will contain both the Node
    class and the algorithm function).
    """
    lines: list[str] = []
    for raw_line in test_source.splitlines():
        stripped = raw_line.strip()

        # Skip the pytest.use_correct branching — always use buggy.
        if "pytest.use_correct" in stripped:
            continue
        if "from correct_python_programs" in stripped:
            continue

        # Rewrite imports from python_programs or node.
        if stripped.startswith("from python_programs."):
            lines.append(f"from buggy import {program_name}")
            continue
        if stripped.startswith("from node import") or stripped.startswith("import node"):
            lines.append("from buggy import Node")
            continue

        # Skip pytest import (only used for use_correct flag).
        if stripped == "import pytest":
            continue

        # Skip slow-test markers that depend on pytest.run_slow.
        if "pytest.run_slow" in stripped or "pytest.mark.skip" in stripped:
            continue

        lines.append(raw_line)

    body = "\n".join(lines).strip()
    if not body:
        return None

    return body + "\n"


# ---------------------------------------------------------------------------
# Helper: load JSON test cases
# ---------------------------------------------------------------------------


def _load_json_testcases(json_path: Path) -> list[tuple[list[Any], Any]]:
    """Load QuixBugs JSON test case file (one JSON array per line)."""
    cases: list[tuple[list[Any], Any]] = []
    for raw_line in json_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, list) and len(parsed) == 2:
            input_args, expected = parsed
            if not isinstance(input_args, list):
                input_args = [input_args]
            cases.append((input_args, expected))
    return cases


# ---------------------------------------------------------------------------
# Helper: prepare code with Node class for graph-based programs
# ---------------------------------------------------------------------------


def _prepend_node_class(code: str) -> str:
    """Prepend the Node class to graph-based program code."""
    return _NODE_CLASS + code


def _strip_docstring_tail(code: str) -> str:
    """Strip trailing docstring block from QuixBugs program files.

    QuixBugs files often end with a triple-quoted docstring describing the
    algorithm. We remove it so the buggy code stays clean.
    """

    def _is_triple_quoted_literal(segment: str) -> bool:
        stripped = segment.lstrip()
        prefix_len = 0
        while prefix_len < len(stripped) and stripped[prefix_len] in "rRuUbBfF":
            prefix_len += 1
        return stripped[prefix_len : prefix_len + 3] in {'"""', "'''"}

    try:
        module = ast.parse(code)
    except SyntaxError:
        return code

    if not module.body:
        return code

    last_stmt = module.body[-1]
    if not isinstance(last_stmt, ast.Expr):
        return code

    if not isinstance(last_stmt.value, ast.Constant) or not isinstance(last_stmt.value.value, str):
        return code

    if last_stmt.col_offset != 0:
        return code

    segment = ast.get_source_segment(code, last_stmt)
    if segment is None or not _is_triple_quoted_literal(segment):
        return code

    result = "\n".join(code.splitlines()[: last_stmt.lineno - 1]).rstrip()
    return result + "\n" if result else code


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------


def import_quixbugs(
    output_dir: Path,
    *,
    repo_path: Path,
    max_cases: int = 0,
    validate: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Import cases from a cloned QuixBugs repository.

    Returns list of imported case metadata dicts (or preview dicts if dry_run).
    """
    buggy_dir = repo_path / "python_programs"
    correct_dir = repo_path / "correct_python_programs"
    json_dir = repo_path / "json_testcases"
    test_dir = repo_path / "python_testcases"

    if not buggy_dir.exists():
        raise FileNotFoundError(
            f"QuixBugs buggy programs directory not found: {buggy_dir}. "
            f"Clone the repo: git clone https://github.com/jkoppel/QuixBugs"
        )

    imported: list[dict[str, Any]] = []
    skipped = 0

    # Discover all buggy programs (exclude test files, node.py, __init__.py).
    program_files = sorted(
        p
        for p in buggy_dir.glob("*.py")
        if not p.name.startswith("_") and not p.name.endswith("_test.py") and p.name != "node.py"
    )

    for program_path in program_files:
        if max_cases > 0 and len(imported) >= max_cases:
            break

        program_name = program_path.stem
        is_graph = program_name in GRAPH_BASED_PROGRAMS

        # Read buggy and fixed code.
        correct_path = correct_dir / program_path.name
        if not correct_path.exists():
            skipped += 1
            continue

        buggy_code = program_path.read_text(encoding="utf-8")
        fixed_code = correct_path.read_text(encoding="utf-8")

        if not buggy_code.strip() or not fixed_code.strip():
            skipped += 1
            continue

        # Strip trailing docstrings from QuixBugs files.
        buggy_code = _strip_docstring_tail(buggy_code)
        fixed_code = _strip_docstring_tail(fixed_code)

        # For graph-based programs, prepend the Node class.
        if is_graph:
            buggy_code = _prepend_node_class(buggy_code)
            fixed_code = _prepend_node_class(fixed_code)

        # Apply ruff formatting.
        buggy_code = ruff_format(buggy_code)
        fixed_code = ruff_format(fixed_code)

        # Filter: no external deps beyond stdlib.
        if has_external_deps(buggy_code) or has_external_deps(fixed_code):
            skipped += 1
            continue

        if len(buggy_code.splitlines()) > MAX_CODE_LINES:
            skipped += 1
            continue

        # Generate oracle patch.
        oracle_patch = generate_oracle_patch(buggy_code, fixed_code)
        if not oracle_patch.strip():
            skipped += 1
            continue

        fix_lines = count_patch_lines(oracle_patch)
        if fix_lines > MAX_FIX_LINES:
            skipped += 1
            continue

        # Generate test file.
        test_code: str | None = None
        if is_graph:
            test_file = test_dir / f"test_{program_name}.py"
            if test_file.exists():
                raw_test = test_file.read_text(encoding="utf-8")
                test_code = _rewrite_graph_test(raw_test, program_name)
        else:
            json_file = json_dir / f"{program_name}.json"
            if json_file.exists():
                test_cases = _load_json_testcases(json_file)
                if test_cases:
                    test_code = _generate_json_test(program_name, test_cases)

        if test_code is None:
            skipped += 1
            continue

        # Build metadata.
        category = map_quixbugs_type(program_name)
        snapshot_dep = infer_snapshot_dependency(category)
        case_id = f"qb_{program_name}_easy"

        case_dir = output_dir / case_id
        if not dry_run and case_dir.exists():
            skipped += 1
            continue

        metadata: dict[str, Any] = {
            "description": f"QuixBugs: {program_name} (1-line bug in classic algorithm)",
            "category": category,
            "difficulty": "easy",
            "snapshot_dependency": snapshot_dep,
            "expected_exception": "AssertionError",
            "tags": ["quixbugs"],
            "python_args": ["-m", "pytest", "-q"],
            "timeout_sec": 30,
            "bug_source": "quixbugs",
            "contamination_risk": "high",
            "expected_fix_lines": fix_lines,
            "source_id": f"quixbugs_{program_name}",
        }

        if is_graph:
            metadata["tags"].append("graph_algorithm")

        case_info: dict[str, Any] = {
            "case_id": case_id,
            "metadata": metadata,
            "fix_lines": fix_lines,
        }

        if dry_run:
            imported.append(case_info)
            continue

        write_case_files(
            case_dir,
            buggy_code=buggy_code,
            test_code=test_code,
            oracle_patch=oracle_patch,
            metadata=metadata,
        )

        if validate:
            ok, msg = validate_generated_case(case_dir, schema_only=True)
            if not ok:
                print(f"  [skip] {case_id}: validation failed: {msg}")
                skipped += 1
                import shutil

                shutil.rmtree(case_dir, ignore_errors=True)
                continue

        imported.append(case_info)

    return imported
