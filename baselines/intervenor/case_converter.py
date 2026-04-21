"""Convert llmdebug eval cases into INTERVENOR bridge inputs.

Unlike the strict LDB bridge, INTERVENOR can operate on a full copied
workspace because the adapter executes each case's native test command in a
temporary directory and feeds the resulting failure text back into the
teacher/learner loop.
"""

from __future__ import annotations

import ast
import base64
import textwrap
from pathlib import Path

from baselines.adapter_base import BaselineCaseContext
from baselines.case_schemas import IntervenorCaseInput

_SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def convert_case(
    case_dir: Path,
    *,
    case_context: BaselineCaseContext | None = None,
) -> IntervenorCaseInput:
    """Convert an llmdebug case directory into an INTERVENOR adapter input."""
    case_id = case_context.case_id if case_context else case_dir.name
    buggy_path = case_dir / "buggy.py"
    test_path = case_dir / "test_buggy.py"
    buggy_code = buggy_path.read_text(encoding="utf-8")
    test_code = test_path.read_text(encoding="utf-8")
    entry_point = _extract_entry_point(buggy_code, test_code)
    description = case_context.description if case_context and case_context.description else ""

    return IntervenorCaseInput.from_dict(
        {
            "task_id": case_id,
            "entry_point": entry_point,
            "prompt": _build_prompt(buggy_code, description, entry_point),
            "buggy_code": buggy_code,
            "test": test_code,
            "description": description,
            "python_args": list(case_context.python_args) if case_context else [],
            "timeout_sec": int(case_context.timeout_sec) if case_context else 0,
            "files": _package_case_files(case_dir),
        }
    )


def _extract_entry_point(buggy_code: str, test_code: str) -> str:
    """Best-effort symbolic name for diagnostics and prompt context."""
    try:
        tree = ast.parse(buggy_code)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name

    try:
        test_tree = ast.parse(test_code)
    except SyntaxError:
        return "solution"

    for stmt in test_tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "buggy":
            for alias in stmt.names:
                if alias.asname is None:
                    return alias.name
    return "solution"


def _build_prompt(buggy_code: str, description: str, entry_point: str) -> str:
    """Build compact task context appended to the learner prompt."""
    try:
        tree = ast.parse(buggy_code)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != entry_point:
            continue
        source = ast.get_source_segment(buggy_code, node)
        if not source:
            break
        lines = source.splitlines()
        prompt_lines = [f"Primary symbol: {entry_point}", "Definition:", lines[0]]
        docstring = ast.get_docstring(node, clean=False)
        if docstring:
            prompt_lines.append(textwrap.indent(f'"""{docstring}"""', "    "))
        if description:
            prompt_lines.append(f"Description: {description}")
        return "\n".join(prompt_lines)

    parts = [f"Primary symbol: {entry_point}"]
    if description:
        parts.append(f"Description: {description}")
    return "\n".join(parts)


def _package_case_files(case_dir: Path) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(case_dir)
        if any(part in _SKIP_DIR_NAMES for part in relative.parts):
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            payload.append(
                {
                    "path": relative.as_posix(),
                    "encoding": "base64",
                    "content": base64.b64encode(raw).decode("ascii"),
                }
            )
        else:
            payload.append(
                {
                    "path": relative.as_posix(),
                    "encoding": "utf-8",
                    "content": text,
                }
            )
    return payload
