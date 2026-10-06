"""Architecture guard for ORC-01/ORC-02 (2026-10-04 review).

``evals.eval_main_support`` and ``evals.eval_attempt_support`` forward attribute *reads*
to ``evals.run_eval``. Assigning ``shim.<NAME> = ...`` binds on the shim and never reaches
``run_eval``'s readers, which silently disabled ``--deterministic --seed`` and the container
pool. Production code must use the ``run_eval.set_active_*`` setters instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SHIMS = {"eval_main_support", "eval_attempt_support"}
_EVALS = Path(__file__).resolve().parents[1] / "evals"
_DATA_DIRS = {"cases", "artifacts", "results", "swebench_runs", "swebench_pro_runs"}
_DATA_DIRS |= {"swebench_pro_assets"}


def _harness_modules() -> list[Path]:
    return sorted(
        path
        for path in _EVALS.rglob("*.py")
        if not _DATA_DIRS.intersection(path.relative_to(_EVALS).parts)
    )


def _shim_aliases(tree: ast.Module) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"evals", None}:
            for alias in node.names:
                if alias.name in _SHIMS:
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] in _SHIMS and alias.asname:
                    aliases.add(alias.asname)
    return aliases


def _shim_writes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _shim_aliases(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in aliases
        ):
            found.append(f"{path.name}:{node.lineno}: setattr({node.args[0].id}, ...)")
        found.extend(
            f"{path.name}:{node.lineno}: {target.value.id}.{target.attr} = ..."
            for target in targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in aliases
        )
    return found


def test_no_module_attribute_writes_through_forwarding_shims() -> None:
    offenders = [hit for path in _harness_modules() for hit in _shim_writes(path)]
    assert not offenders, "writes through forwarding shims:\n" + "\n".join(offenders)


def test_guard_detects_a_shim_write(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from evals import eval_main_support as legacy\nlegacy._ACTIVE_RUN_SETTINGS = None\n",
        encoding="utf-8",
    )
    assert _shim_writes(sample) == ["sample.py:2: legacy._ACTIVE_RUN_SETTINGS = ..."]
