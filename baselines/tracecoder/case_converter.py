"""Convert llmdebug eval cases into TraceCoder's patched llmdebug dataset rows."""

from __future__ import annotations

import ast
import base64
from pathlib import Path
from typing import cast

from baselines.adapter_base import BaselineCaseContext, BaselineCaseIncompatibleError
from baselines.case_information import (
    REFERENCE_TESTS_HEADER,
    TEST_SOURCE_FULL,
    case_information_record,
    render_reference_tests,
)
from baselines.case_schemas import TraceCoderCaseInput

_SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_MAX_TEST_NAME_ITEMS = 8
_MAX_EXPECTATION_ITEMS = 8


def convert_case(
    case_dir: Path,
    *,
    case_context: BaselineCaseContext | None = None,
) -> TraceCoderCaseInput:
    """Package a full llmdebug case for TraceCoder's patched CLI."""
    case_id = case_context.case_id if case_context else case_dir.name
    buggy_path = case_dir / "buggy.py"
    test_path = case_dir / "test_buggy.py"
    buggy_code = buggy_path.read_text(encoding="utf-8")
    test_code = test_path.read_text(encoding="utf-8")
    python_args = list(case_context.python_args) if case_context else ["-m", "pytest", "-q"]
    if python_args[:2] != ["-m", "pytest"]:
        raise BaselineCaseIncompatibleError(
            reason="non_pytest_command_unsupported",
            detail=f"{case_id}: TraceCoder llmdebug bridge expects python -m pytest commands",
        )

    entry_point = _extract_entry_point(buggy_code, test_code)
    timeout_sec = int(case_context.timeout_sec) if case_context and case_context.timeout_sec else 0

    return cast(
        TraceCoderCaseInput,
        TraceCoderCaseInput.from_dict(
            {
                "task_id": case_id,
                "entry_point": entry_point,
                "complete_prompt": _build_prompt(
                    buggy_code=buggy_code,
                    test_code=test_code,
                    entry_point=entry_point,
                ),
                "buggy_code": buggy_code,
                "test": test_code,
                "files": _package_case_files(case_dir),
                "python_args": python_args,
                "timeout_sec": timeout_sec,
                "repair_seeded": True,
                "execution_mode": "llmdebug_native",
                "case_information": case_information_record(
                    test_source_exposure=TEST_SOURCE_FULL,
                    prompt_files=("buggy.py", "test_buggy.py"),
                ),
            }
        ),
    )


def _extract_entry_point(buggy_code: str, test_code: str) -> str:
    """Best-effort symbolic name for prompt context and diagnostics."""
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


def _build_prompt(*, buggy_code: str, test_code: str, entry_point: str) -> str:
    """Build the task prompt passed into TraceCoder's planning/repair prompts."""
    parts = [
        "Repair the provided buggy.py implementation so all reference tests pass.",
        f"Primary symbol: {entry_point}",
    ]
    summary_lines = _summarize_reference_tests(test_code=test_code, entry_point=entry_point)
    if summary_lines:
        parts.append("Reference test summary:")
        parts.extend(summary_lines)
    parts.append(render_reference_tests(test_code).rstrip())

    try:
        tree = ast.parse(buggy_code)
    except SyntaxError:
        return "\n".join(parts)

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != entry_point:
            continue
        source = ast.get_source_segment(buggy_code, node)
        if not source:
            break
        signature = source.splitlines()[0]
        parts.insert(2, f"Current definition: {signature}")
        break
    return "\n".join(parts)


def compact_problem_prompt(prompt: str, *, max_chars: int) -> str:
    """Clamp the generated task description, keeping everything from the test section on.

    The budget used to apply to the whole prompt, which for longer cases cut off the end of
    ``test_buggy.py`` — while the record still claimed ``test_source_exposure="full"``
    (BAS-07) — and dropped the retry context the harness appends on attempt 2+ (BAS-02).
    Only the leading summary is compactable; the full reference tests and anything appended
    after them are preserved.
    """
    text = str(prompt or "").strip()
    limit = max(1, int(max_chars))
    index = text.find(REFERENCE_TESTS_HEADER)
    if index < 0:
        return _clamp_text(text, limit=limit)
    head = _clamp_text(text[:index].rstrip(), limit=limit)
    preserved = text[index:]
    return f"{head}\n{preserved}" if head else preserved


def _clamp_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _summarize_reference_tests(*, test_code: str, entry_point: str) -> list[str]:
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        line_count = len(test_code.splitlines())
        return [f"- test_buggy.py could not be parsed; inspect the {line_count}-line file locally."]

    imported_symbols = _imported_buggy_symbols(tree=tree)
    test_names = _test_function_names(tree=tree)
    expectations = _expectation_summaries(tree=tree)

    lines: list[str] = []
    if imported_symbols:
        lines.append(f"- Imports from buggy: {', '.join(imported_symbols)}")
    elif entry_point:
        lines.append(f"- Expected to preserve the public symbol `{entry_point}`.")
    if test_names:
        rendered_names = ", ".join(test_names[:_MAX_TEST_NAME_ITEMS])
        if len(test_names) > _MAX_TEST_NAME_ITEMS:
            rendered_names += f", ... (+{len(test_names) - _MAX_TEST_NAME_ITEMS} more)"
        lines.append(f"- Test functions: {rendered_names}")
    if expectations:
        lines.append("- Representative expectations:")
        lines.extend(f"  - {item}" for item in expectations[:_MAX_EXPECTATION_ITEMS])
        if len(expectations) > _MAX_EXPECTATION_ITEMS:
            lines.append(
                f"  - ... (+{len(expectations) - _MAX_EXPECTATION_ITEMS} more assertions/raises checks)"
            )
    if not lines:
        lines.append("- Inspect test_buggy.py locally for exact expectations.")
    return lines


def _imported_buggy_symbols(*, tree: ast.AST) -> list[str]:
    symbols: list[str] = []
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "buggy":
            for alias in stmt.names:
                name = alias.name.strip()
                if name and name not in symbols:
                    symbols.append(name)
    return symbols


def _test_function_names(*, tree: ast.AST) -> list[str]:
    return [
        stmt.name
        for stmt in getattr(tree, "body", [])
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        and stmt.name.startswith("test")
    ]


def _expectation_summaries(*, tree: ast.AST) -> list[str]:
    items: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            rendered = ast.unparse(node.test).strip()
            if rendered and rendered not in items:
                items.append(rendered)
            continue
        if isinstance(node, ast.With):
            for item in node.items:
                context = item.context_expr
                if (
                    isinstance(context, ast.Call)
                    and isinstance(context.func, ast.Attribute)
                    and context.func.attr == "raises"
                ):
                    rendered = ast.unparse(context).strip()
                    if rendered and rendered not in items:
                        items.append(rendered)
    return items


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
