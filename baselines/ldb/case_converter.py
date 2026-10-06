"""Convert llmdebug eval cases into the strict callable subset for LDB.

LDB's upstream pipeline expects HumanEval-like rows:
- ``task_id`` with a dataset-family prefix
- ``prompt`` containing a callable signature/docstring
- ``entry_point`` naming the callable under test
- ``test`` defining ``check(candidate)``
- ``given_tests`` as self-contained executable snippets
- ``seed`` as the buggy implementation

This bridge only accepts cases that can be lowered faithfully into that
contract. In particular, upstream LDB expects ``given_tests`` to be direct
top-level ``assert ...`` snippets, not wrapped pytest functions. Unsupported
task shapes raise ``BaselineCaseIncompatibleError`` so the eval harness can
record an explicit baseline failure instead of silently fabricating a
mismatched task.
"""

from __future__ import annotations

import ast
import copy
import textwrap
from dataclasses import dataclass
from pathlib import Path

from baselines.adapter_base import BaselineCaseIncompatibleError
from baselines.case_information import TEST_SOURCE_ASSERTS, case_information_record
from baselines.case_schemas import LDBCaseInput


@dataclass(frozen=True)
class _LoweredLDBCase:
    entry_point: str
    prompt: str
    given_tests: list[str]
    check_function: str


def convert_case(
    case_dir: Path,
    *,
    case_id: str | None = None,
) -> LDBCaseInput:
    """Convert an llmdebug case directory into a strict LDB input row."""
    resolved_case_id = str(case_id or case_dir.name)
    extra_python_files = sorted(
        path.name
        for path in case_dir.glob("*.py")
        if path.name not in {"buggy.py", "test_buggy.py"} and path.is_file()
    )
    if extra_python_files:
        raise BaselineCaseIncompatibleError(
            reason="multifile_case_unsupported",
            detail=(
                f"{resolved_case_id}: additional Python files require filesystem-aware execution "
                f"({', '.join(extra_python_files)})"
            ),
        )
    buggy_code = (case_dir / "buggy.py").read_text(encoding="utf-8")
    test_code = (case_dir / "test_buggy.py").read_text(encoding="utf-8")
    lowered = _lower_case(
        case_id=resolved_case_id,
        buggy_code=buggy_code,
        test_code=test_code,
    )

    return LDBCaseInput.from_dict(
        {
            "task_id": f"llmdebug/{resolved_case_id}",
            "prompt": lowered.prompt,
            "entry_point": lowered.entry_point,
            "test": lowered.check_function,
            "seed": buggy_code,
            "given_tests": lowered.given_tests,
            "dataset_type": "llmdebug",
            "case_information": case_information_record(
                test_source_exposure=TEST_SOURCE_ASSERTS,
                prompt_files=("buggy.py", "test_buggy.py"),
            ),
        }
    )


def _extract_entry_point(  # pyright: ignore[reportUnusedFunction]
    buggy_code: str, test_code: str
) -> str:
    """Best-effort helper retained for unit tests and diagnostics."""
    try:
        lowered = _lower_case(case_id="adhoc", buggy_code=buggy_code, test_code=test_code)
    except BaselineCaseIncompatibleError:
        pass
    else:
        return lowered.entry_point

    try:
        tree = ast.parse(buggy_code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
            if isinstance(node, ast.ClassDef):
                return node.name
    except SyntaxError:
        pass

    import_match = None
    if test_code.strip():
        try:
            test_tree = ast.parse(test_code)
        except SyntaxError:
            test_tree = ast.Module(body=[], type_ignores=[])
        import_match = next(
            (
                alias.name
                for node in test_tree.body
                if isinstance(node, ast.ImportFrom) and node.module == "buggy"
                for alias in node.names
                if alias.asname is None
            ),
            None,
        )
    return import_match or "solution"


def _build_prompt(buggy_code: str, entry_point: str) -> str:
    """Build the callable signature/docstring prompt LDB expects."""
    try:
        tree = ast.parse(buggy_code)
    except SyntaxError:
        return f"def {entry_point}():"

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != entry_point:
            continue
        source = ast.get_source_segment(buggy_code, node)
        if not source:
            break
        lines = source.splitlines()
        prompt_lines = [lines[0]]
        if ast.get_docstring(node, clean=False):
            body_index = 1
            while body_index < len(lines):
                prompt_lines.append(lines[body_index])
                if lines[body_index].strip().endswith(('"""', "'''")):
                    break
                body_index += 1
        return "\n".join(prompt_lines)

    return f"def {entry_point}():"


def _extract_assertions(  # pyright: ignore[reportUnusedFunction]
    test_code: str, buggy_code: str
) -> list[str]:
    """Lower supported pytest tests into direct LDB assert snippets."""
    return _lower_case(case_id="adhoc", buggy_code=buggy_code, test_code=test_code).given_tests


def _lower_case(
    *,
    case_id: str,
    buggy_code: str,
    test_code: str,
) -> _LoweredLDBCase:
    try:
        buggy_tree = ast.parse(buggy_code)
    except SyntaxError as exc:
        raise BaselineCaseIncompatibleError(
            reason="buggy_syntax_error",
            detail=f"{case_id}: {exc.msg}",
        ) from exc
    try:
        test_tree = ast.parse(test_code)
    except SyntaxError as exc:
        raise BaselineCaseIncompatibleError(
            reason="test_syntax_error",
            detail=f"{case_id}: {exc.msg}",
        ) from exc

    if _contains_subprocess(test_tree):
        raise BaselineCaseIncompatibleError(
            reason="subprocess_task_unsupported",
            detail=f"{case_id}: tests execute buggy.py via subprocess instead of calling an importable symbol",
        )
    if _contains_async(buggy_tree) or _contains_async(test_tree):
        raise BaselineCaseIncompatibleError(
            reason="async_task_unsupported",
            detail=f"{case_id}: LDB bridge only supports synchronous callable tasks",
        )

    import_stmt = _extract_buggy_import(test_tree, case_id=case_id)
    imported_alias = import_stmt.names[0]
    entry_point = imported_alias.name
    if imported_alias.asname is not None:
        raise BaselineCaseIncompatibleError(
            reason="aliased_buggy_import_unsupported",
            detail=f"{case_id}: from buggy import {entry_point} as {imported_alias.asname}",
        )

    top_level_defs = {
        node.name: node
        for node in buggy_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    entry_node = top_level_defs.get(entry_point)
    if entry_node is None:
        raise BaselineCaseIncompatibleError(
            reason="entry_point_not_found",
            detail=f"{case_id}: imported symbol {entry_point!r} is not a top-level definition in buggy.py",
        )
    if not isinstance(entry_node, ast.FunctionDef):
        kind = type(entry_node).__name__
        raise BaselineCaseIncompatibleError(
            reason="non_function_entry_point_unsupported",
            detail=f"{case_id}: imported symbol {entry_point!r} is a {kind}, not a synchronous function",
        )

    support_stmts: list[ast.stmt] = []
    test_functions: list[ast.FunctionDef] = []
    for stmt in test_tree.body:
        if _is_ignorable_test_support(stmt):
            continue
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "buggy":
            continue
        if isinstance(stmt, ast.AsyncFunctionDef) and stmt.name.startswith("test"):
            raise BaselineCaseIncompatibleError(
                reason="async_test_function_unsupported",
                detail=f"{case_id}: async pytest test {stmt.name!r}",
            )
        if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test"):
            if stmt.args.args or stmt.args.kwonlyargs or stmt.args.vararg or stmt.args.kwarg:
                raise BaselineCaseIncompatibleError(
                    reason="fixture_based_test_unsupported",
                    detail=f"{case_id}: pytest test {stmt.name!r} requires injected arguments",
                )
            test_functions.append(stmt)
            continue
        support_stmts.append(stmt)

    if not test_functions:
        raise BaselineCaseIncompatibleError(
            reason="no_pytest_test_functions",
            detail=f"{case_id}: expected top-level test_* functions in test_buggy.py",
        )

    if support_stmts:
        raise BaselineCaseIncompatibleError(
            reason="test_support_code_unsupported",
            detail=(
                f"{case_id}: LDB bridge requires direct assert-style tests without top-level "
                "helper code in test_buggy.py"
            ),
        )

    support_code = _render_module_statements(support_stmts)
    given_tests: list[str] = []
    for index, test_fn in enumerate(test_functions, start=1):
        snippet = _extract_direct_assert_snippet(test_fn, case_id=case_id).strip() + "\n"
        _validate_python(snippet, case_id=case_id, reason=f"given_test_{index}_invalid")
        given_tests.append(snippet)

    check_lines: list[str] = []
    if support_code:
        check_lines.append(support_code.rstrip())
        check_lines.append("")
    check_lines.append("def check(candidate):")
    replacer = _CandidateNameRewriter(source_name=entry_point)
    for index, test_fn in enumerate(test_functions, start=1):
        rewritten = replacer.visit(
            ast.fix_missing_locations(_clone_function(test_fn, f"_test_{index}"))
        )
        rewritten.decorator_list = []
        nested_source = textwrap.indent(ast.unparse(rewritten).strip(), "    ")
        check_lines.append(nested_source)
        check_lines.append(f"    {rewritten.name}()")
        check_lines.append("")
    check_function = "\n".join(check_lines).rstrip() + "\n"
    _validate_python(check_function, case_id=case_id, reason="check_function_invalid")

    return _LoweredLDBCase(
        entry_point=entry_point,
        prompt=_build_prompt(buggy_code, entry_point),
        given_tests=given_tests,
        check_function=check_function,
    )


def _extract_buggy_import(test_tree: ast.Module, *, case_id: str) -> ast.ImportFrom:
    buggy_imports = [
        stmt
        for stmt in test_tree.body
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "buggy"
    ]
    if len(buggy_imports) != 1 or len(buggy_imports[0].names) != 1:
        raise BaselineCaseIncompatibleError(
            reason="unsupported_test_import_shape",
            detail=f"{case_id}: expected exactly one 'from buggy import <symbol>' statement",
        )
    return buggy_imports[0]


def _render_module_statements(statements: list[ast.stmt]) -> str:
    rendered: list[str] = []
    for stmt in statements:
        source = ast.unparse(stmt).strip()
        if source:
            rendered.append(source)
    return "\n\n".join(rendered).strip()


def _extract_direct_assert_snippet(test_fn: ast.FunctionDef, *, case_id: str) -> str:
    body = [stmt for stmt in test_fn.body if not _is_docstring_stmt(stmt)]
    if len(body) != 1 or not isinstance(body[0], ast.Assert):
        raise BaselineCaseIncompatibleError(
            reason="non_assert_test_body_unsupported",
            detail=(
                f"{case_id}: pytest test {test_fn.name!r} must contain exactly one direct "
                "assert statement for LDB"
            ),
        )
    return ast.unparse(body[0]).strip()


def _is_ignorable_test_support(stmt: ast.stmt) -> bool:
    if _is_docstring_stmt(stmt):
        return True
    if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
        imported_names = {alias.name for alias in stmt.names}
        return imported_names == {"annotations"}
    return False


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _validate_python(source: str, *, case_id: str, reason: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise BaselineCaseIncompatibleError(
            reason=reason,
            detail=f"{case_id}: {exc.msg}",
        ) from exc


def _contains_subprocess(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "subprocess" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            return True
    return False


def _contains_async(tree: ast.AST) -> bool:
    return any(
        isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith))
        for node in ast.walk(tree)
    )


def _clone_function(test_fn: ast.FunctionDef, name: str) -> ast.FunctionDef:
    cloned = ast.FunctionDef(
        name=name,
        args=copy.deepcopy(test_fn.args),
        body=copy.deepcopy(test_fn.body),
        decorator_list=[],
        returns=copy.deepcopy(test_fn.returns),
        type_comment=test_fn.type_comment,
    )
    return ast.fix_missing_locations(cloned)


class _CandidateNameRewriter(ast.NodeTransformer):
    def __init__(self, *, source_name: str) -> None:
        self._source_name = source_name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self._source_name:
            return ast.copy_location(ast.Name(id="candidate", ctx=node.ctx), node)
        return node
