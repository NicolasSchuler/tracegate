"""Convert llmdebug eval cases into lambda-RLM baseline inputs."""

from __future__ import annotations

import ast
import base64
from pathlib import Path

from baselines.adapter_base import BaselineCaseContext
from baselines.case_schemas import LambdaRLMCaseInput

_SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_PRIMARY_PROMPT_FILES = ("buggy.py", "test_buggy.py")


def convert_case(
    case_dir: Path,
    *,
    case_context: BaselineCaseContext | None = None,
) -> LambdaRLMCaseInput:
    """Convert an llmdebug case directory into lambda-RLM adapter input."""
    case_id = case_context.case_id if case_context else case_dir.name
    buggy_path = case_dir / "buggy.py"
    test_path = case_dir / "test_buggy.py"
    buggy_code = buggy_path.read_text(encoding="utf-8")
    test_code = test_path.read_text(encoding="utf-8")
    description = case_context.description if case_context and case_context.description else ""
    files = _package_case_files(case_dir)
    return LambdaRLMCaseInput.from_dict(
        {
            "task_id": case_id,
            "entry_point": _extract_entry_point(buggy_code, test_code),
            "buggy_code": buggy_code,
            "test": test_code,
            "description": description,
            "python_args": list(case_context.python_args) if case_context else [],
            "timeout_sec": int(case_context.timeout_sec)
            if case_context and case_context.timeout_sec
            else 0,
            "files": files,
            "prompt_context": _build_prompt_context(
                files=files,
                entry_point=_extract_entry_point(buggy_code, test_code),
                description=description,
            ),
            "category": case_context.category if case_context else None,
            "difficulty": case_context.difficulty if case_context else None,
            "snapshot_dependency": case_context.snapshot_dependency if case_context else None,
        }
    )


def _extract_entry_point(buggy_code: str, test_code: str) -> str:
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


def _build_prompt_context(
    *,
    files: list[dict[str, str]],
    entry_point: str,
    description: str,
) -> str:
    sections = [
        "You are repairing buggy.py so the provided tests pass.",
        f"Primary symbol: {entry_point}",
    ]
    if description:
        sections.append(f"Description: {description}")
    rendered_files: list[str] = []
    omitted_paths: list[str] = []
    ordered_paths = list(_PRIMARY_PROMPT_FILES) + sorted(
        {
            str(row.get("path") or "")
            for row in files
            if str(row.get("path") or "").endswith(".py")
            and str(row.get("path") or "") not in _PRIMARY_PROMPT_FILES
        }
    )
    file_map = {str(row.get("path") or ""): row for row in files}
    for path in ordered_paths:
        row = file_map.get(path)
        if row is None:
            continue
        encoding = str(row.get("encoding") or "utf-8")
        if encoding != "utf-8":
            omitted_paths.append(path)
            continue
        rendered_files.append(f"=== {path} ===\n{str(row.get('content') or '').rstrip()}")
    if rendered_files:
        sections.append("Workspace files:")
        sections.extend(rendered_files)
    if omitted_paths:
        sections.append("Omitted non-UTF8 files: " + ", ".join(omitted_paths))
    return "\n\n".join(section for section in sections if section.strip())


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
