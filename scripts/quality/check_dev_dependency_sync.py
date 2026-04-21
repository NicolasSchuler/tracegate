#!/usr/bin/env python3
"""Fail when the duplicated dev dependency lists in pyproject.toml diverge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _load_dev_dependencies(pyproject_path: Path) -> tuple[list[str], list[str]]:
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional_deps = payload.get("project", {}).get("optional-dependencies", {}).get("dev", [])
    dependency_group = payload.get("dependency-groups", {}).get("dev", [])
    if not isinstance(optional_deps, list) or not all(
        isinstance(item, str) for item in optional_deps
    ):
        raise RuntimeError("[project.optional-dependencies].dev must be a list of strings")
    if not isinstance(dependency_group, list) or not all(
        isinstance(item, str) for item in dependency_group
    ):
        raise RuntimeError("[dependency-groups].dev must be a list of strings")
    return list(optional_deps), list(dependency_group)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that pyproject.toml dev dependency lists stay in sync."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml (default: ./pyproject.toml).",
    )
    args = parser.parse_args()

    try:
        optional_deps, dependency_group = _load_dev_dependencies(args.pyproject)
    except (FileNotFoundError, RuntimeError, tomllib.TOMLDecodeError) as exc:
        print(f"dependency-sync: ERROR: {exc}")
        return 2

    if optional_deps == dependency_group:
        print("dependency-sync: PASS")
        return 0

    optional_only = [dep for dep in optional_deps if dep not in dependency_group]
    group_only = [dep for dep in dependency_group if dep not in optional_deps]
    print("dependency-sync: FAIL: dev dependency lists diverged.")
    if optional_only:
        print("only in [project.optional-dependencies].dev:")
        for dep in optional_only:
            print(f"  - {dep}")
    if group_only:
        print("only in [dependency-groups].dev:")
        for dep in group_only:
            print(f"  - {dep}")
    if not optional_only and not group_only:
        print("lists contain the same members but in a different order.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
