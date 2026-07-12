"""Regression guards for the package's interface dependency boundary."""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "llmdebug"
_INTERFACE_MODULES = frozenset(
    {
        "llmdebug",
        "llmdebug._cli_display",
        "llmdebug.cli",
        "llmdebug.jupyter_plugin",
        "llmdebug.mcp_contract",
        "llmdebug.mcp_diff_tools",
        "llmdebug.mcp_git_context",
        "llmdebug.mcp_rca",
        "llmdebug.mcp_server",
        "llmdebug.mcp_snapshot_tools",
        "llmdebug.mcp_types",
        "llmdebug.middleware",
        "llmdebug.pytest_plugin",
    }
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(_PACKAGE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["llmdebug", *parts])


def _resolve_from_import(module_name: str, node: ast.ImportFrom) -> set[str]:
    if node.level == 0:
        return {node.module} if node.module is not None else set()

    package_parts = module_name.split(".")[:-1]
    levels_up = node.level - 1
    if levels_up:
        package_parts = package_parts[:-levels_up]
    if node.module is not None:
        return {".".join([*package_parts, *node.module.split(".")])}

    package_name = ".".join(package_parts)
    targets: set[str] = set()
    for alias in node.names:
        relative_module = _PACKAGE_ROOT.joinpath(*package_parts[1:], alias.name)
        if (
            relative_module.with_suffix(".py").is_file()
            or (relative_module / "__init__.py").is_file()
        ):
            targets.add(".".join([*package_parts, alias.name]))
        else:
            # ``from . import exported_name`` reads an attribute from the
            # package facade rather than importing a sibling module.
            targets.add(package_name)
    return targets


def _import_targets(path: Path) -> set[str]:
    module_name = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            targets.update(_resolve_from_import(module_name, node))
    return targets


def test_inner_package_modules_do_not_import_interface_adapters() -> None:
    violations: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        module_name = _module_name(path)
        if module_name == "llmdebug" or module_name in _INTERFACE_MODULES:
            continue
        violations.extend(
            f"{module_name} imports interface module {target}"
            for target in sorted(_import_targets(path))
            if target in _INTERFACE_MODULES
        )

    assert violations == [], "\n".join(violations)


def test_relative_facade_attribute_import_resolves_to_package_interface() -> None:
    node = ast.parse("from . import __version__").body[0]
    assert isinstance(node, ast.ImportFrom)

    assert _resolve_from_import("llmdebug.capture", node) == {"llmdebug"}


def test_relative_sibling_module_import_does_not_resolve_to_facade() -> None:
    node = ast.parse("from . import _json as json").body[0]
    assert isinstance(node, ast.ImportFrom)

    assert _resolve_from_import("llmdebug.output", node) == {"llmdebug._json"}
