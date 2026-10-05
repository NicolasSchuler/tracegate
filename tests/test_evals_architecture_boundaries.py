"""Dependency boundaries of the eval harness (X-02, ORC-01, ORC-02).

``evals.run_eval`` is the CLI entrypoint and compatibility facade. The modules it
runs must not import it back: that cycle needed function-local imports, and writes
through the forwarding shims never reached ``run_eval``'s readers, which silently
disabled ``--deterministic --seed`` and the container pool.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVALS = _REPO / "evals"
_DATA_DIRS = frozenset(
    {"cases", "artifacts", "results", "swebench_runs", "swebench_pro_runs", "swebench_pro_assets"}
)
# run_eval attributes that a monkeypatch can usefully replace: run_eval reads them when
# it builds EvalSeams or dispatches to the main flow. Anything else run_eval merely
# re-exports, so patching it there would silently do nothing.
_RUN_EVAL_SEAMS = frozenset(
    {
        "_REPO_ROOT",
        "load_cases",
        "make_patcher",
        "run_one_condition",
        "_build_run_settings_fingerprint",
        "_write_resolved_run_config_lockfile",
        "_load_existing_run_state",
        "_repair_trailing_results_tail",
        "check_testcontainers_runtime_available",
        "cleanup_stale_eval_containers",
        "_RunResultsLock",
        "run_main_flow",
    }
)


def _harness_modules() -> list[Path]:
    return sorted(
        path
        for path in _EVALS.rglob("*.py")
        if not _DATA_DIRS.intersection(path.relative_to(_EVALS).parts)
    )


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(path: Path) -> set[str]:
    """Every module an import statement in *path* names, at any nesting level."""
    package = (
        _module_name(path).rsplit(".", 1)[0] if path.name != "__init__.py" else (_module_name(path))
    )
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, *([base] if base else [])])
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def test_no_harness_module_imports_run_eval() -> None:
    offenders = [
        _module_name(path)
        for path in _harness_modules()
        if path.name != "run_eval.py" and "evals.run_eval" in _imported_modules(path)
    ]
    assert not offenders, f"modules importing evals.run_eval back: {offenders}"


def test_forwarding_shims_stay_deleted() -> None:
    for shim in ("eval_attempt_support.py", "eval_main_support.py"):
        assert not (_EVALS / shim).exists(), f"{shim} was a forwarding shim (X-02)"


def test_run_eval_keeps_no_module_state() -> None:
    tree = ast.parse((_EVALS / "run_eval.py").read_text(encoding="utf-8"))
    globals_used = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Global)]
    assert not globals_used, f"run_eval.py rebinds module globals at lines {globals_used}"


def _module_aliases(tree: ast.Module) -> set[str]:
    """Names bound to modules by ``import x as y`` or ``from evals import module``."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module in {"evals", None}:
            for alias in node.names:
                name = alias.name
                if (_EVALS / f"{name}.py").is_file() or (_EVALS / name).is_dir():
                    aliases.add(alias.asname or name)
    return aliases


def _module_attribute_writes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _module_aliases(tree)
    hits: list[str] = []
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
            hits.append(f"{path.name}:{node.lineno}: setattr({node.args[0].id}, ...)")
        hits.extend(
            f"{path.name}:{node.lineno}: {target.value.id}.{target.attr} = ..."
            for target in targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in aliases
        )
    return hits


def test_no_module_writes_another_modules_attributes() -> None:
    offenders = [hit for path in _harness_modules() for hit in _module_attribute_writes(path)]
    assert not offenders, "module attribute writes:\n" + "\n".join(offenders)


def test_attribute_write_guard_detects_a_write(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from evals import eval_config as cfg\ncfg._REPO_ROOT = None\nsetattr(cfg, 'x', 1)\n",
        encoding="utf-8",
    )
    assert len(_module_attribute_writes(sample)) == 2


def test_tests_patch_only_seams_that_run_eval_reads() -> None:
    pattern = re.compile(r'setattr\(\s*(?:run_eval|run_eval_mod),\s*"([A-Za-z_]+)"')
    string_pattern = re.compile(r'"evals\.run_eval\.([A-Za-z_]+)"')
    patched: dict[str, set[str]] = {}
    for path in sorted((_REPO / "tests").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in pattern.findall(text) + string_pattern.findall(text):
            patched.setdefault(name, set()).add(path.name)
    no_ops = {name: sorted(files) for name, files in patched.items() if name not in _RUN_EVAL_SEAMS}
    assert not no_ops, (
        "these run_eval attributes are re-exports; patch them where they are used instead: "
        f"{no_ops}"
    )


def test_seam_list_matches_what_run_eval_reads() -> None:
    source = (_EVALS / "run_eval.py").read_text(encoding="utf-8")
    seams_body = source[source.index("def _current_seams") :]
    seams_body = seams_body[: seams_body.index("\n\n\n")]
    read = set(re.findall(r"=\s*([A-Za-z_]+),\n", seams_body)) | {"run_main_flow"}
    assert read == _RUN_EVAL_SEAMS


# The harness drives the baselines, so it may import them; the reverse would close a cycle.
# compare_results is a CLI over eval results that nothing imports.
_BASELINE_MODULES_ALLOWED_TO_IMPORT_EVALS = frozenset({"baselines.compare_results"})


def _baseline_modules() -> list[Path]:
    root = _REPO / "baselines"
    return sorted(
        path
        for path in root.rglob("*.py")
        if not {"repos", "envs", "stubs"}.intersection(path.relative_to(root).parts)
    )


def test_baselines_do_not_import_the_harness() -> None:
    offenders = {
        _module_name(path): sorted(
            name for name in _imported_modules(path) if name.split(".")[0] == "evals"
        )
        for path in _baseline_modules()
        if _module_name(path) not in _BASELINE_MODULES_ALLOWED_TO_IMPORT_EVALS
    }
    offenders = {module: names for module, names in offenders.items() if names}
    assert not offenders, f"baselines importing evals: {offenders}"


def _is_analysis_module(name: str) -> bool:
    stem = name.rsplit(".", 1)[-1]
    return stem == "analyze_results" or stem.startswith("analysis_")


def test_only_analysis_api_imports_analyze_results_internals() -> None:
    offenders: list[str] = []
    for path in [*_harness_modules(), *_baseline_modules()]:
        if _is_analysis_module(_module_name(path)):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and _is_analysis_module(node.module or ""):
                private = [alias.name for alias in node.names if alias.name.startswith("_")]
                if private:
                    offenders.append(f"{_module_name(path)}: {private}")
    assert not offenders, "import these from evals.analysis_api instead:\n" + "\n".join(offenders)
