"""The sync and async OpenAI-compatible helpers must stay twins (X-04).

The async variants are used by SWE-Bench generation only, so drift went unnoticed:
``_http_get_json_async`` lacked the bad-URL handler and the async proposers lacked
``system_prompt_override``. After normalizing what legitimately differs (async/await,
``AsyncClient``, the async client pool and ``asyncio.sleep``), each pair must be
identical.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PATCHERS = Path(__file__).resolve().parents[1] / "evals" / "patchers"
_MODULES = ("openai_compat_patcher.py", "openai_transport.py", "openai_probes.py")
_SYNC_EQUIVALENTS = {
    "AsyncClient": "Client",
    "_get_async_http_client": "_get_thread_http_client",
    "asyncio": "time",
}


class _Normalize(ast.NodeTransformer):
    def _as(self, node: ast.AST, cls: type[ast.AST]) -> ast.AST:
        converted = ast.copy_location(cls(**{f: getattr(node, f) for f in node._fields}), node)
        self.generic_visit(converted)
        return converted

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = node.name.removesuffix("_async")
        return self._as(node, ast.FunctionDef)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = node.name.removesuffix("_async")
        self.generic_visit(node)
        return node

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        return self._as(node, ast.With)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        return self._as(node, ast.For)

    def visit_Await(self, node: ast.Await) -> ast.AST:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        name = node.id.removesuffix("_async")
        node.id = _SYNC_EQUIVALENTS.get(name, name)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        name = node.attr.removesuffix("_async")
        node.attr = _SYNC_EQUIVALENTS.get(name, name)
        return node


def _functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for module in _MODULES:
        tree = ast.parse((_PATCHERS / module).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[node.name] = node
    return found


def _twins() -> list[str]:
    functions = _functions()
    return sorted(
        name.removesuffix("_async")
        for name in functions
        if name.endswith("_async") and name.removesuffix("_async") in functions
    )


def _normalized(node: ast.AST) -> str:
    tree = ast.parse(ast.unparse(node))
    return ast.unparse(ast.fix_missing_locations(_Normalize().visit(tree)))


def test_the_twins_are_found() -> None:
    assert len(_twins()) >= 10


@pytest.mark.parametrize("name", _twins())
def test_async_twin_matches_its_sync_counterpart(name: str) -> None:
    functions = _functions()
    assert _normalized(functions[f"{name}_async"]) == _normalized(functions[name])
