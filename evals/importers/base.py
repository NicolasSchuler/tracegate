"""Shared utilities for dataset importers.

Functions for generating oracle patches, writing case files, validating
generated cases, and shared filter helpers used across importers.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import _json as json
from ..eval_patches import _build_diff_from_changes, _diff_lines
from .determinism import is_deterministic as is_deterministic
from .determinism import nondeterminism_reasons as nondeterminism_reasons


def generate_oracle_patch(
    buggy: str,
    fixed: str,
    filename: str = "buggy.py",
) -> str:
    """Generate a unified diff (oracle.patch) from buggy to fixed code.

    Returns a git-apply-compatible unified diff string. A source without a trailing
    newline gets the ``\\ No newline at end of file`` marker; the old builder merged the
    last removed and added lines into one (DAT-10).
    """
    return _build_diff_from_changes([(filename, buggy, fixed)])


def write_case_files(
    case_dir: Path,
    *,
    buggy_code: str,
    test_code: str,
    oracle_patch: str,
    metadata: dict[str, Any],
) -> None:
    """Write all required case files to a directory.

    Creates: buggy.py, test_buggy.py, oracle.patch, case.json.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "buggy.py").write_text(buggy_code, encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(test_code, encoding="utf-8")
    (case_dir / "oracle.patch").write_text(oracle_patch, encoding="utf-8")
    (case_dir / "case.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_generated_case(
    case_dir: Path,
    *,
    schema_only: bool = True,
) -> tuple[bool, str]:
    """Validate a generated case directory against eval contract.

    When schema_only=True, checks file existence and case.json structure only.
    When schema_only=False, also runs baseline + oracle validation (requires pytest).
    """
    required_files = ("buggy.py", "test_buggy.py", "case.json", "oracle.patch")
    for fname in required_files:
        if not (case_dir / fname).exists():
            return False, f"missing required file: {fname}"

    # Validate case.json.
    try:
        meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, f"invalid case.json: {e}"

    if not isinstance(meta, dict):
        return False, "case.json is not a JSON object"

    from evals.run_eval import VALID_CATEGORIES, VALID_DIFFICULTIES, VALID_SNAPSHOT_DEPENDENCIES

    category = meta.get("category")
    if category not in VALID_CATEGORIES:
        return False, f"invalid category: {category}"

    difficulty = meta.get("difficulty")
    if difficulty not in VALID_DIFFICULTIES:
        return False, f"invalid difficulty: {difficulty}"

    snapshot_dependency = meta.get("snapshot_dependency")
    if snapshot_dependency not in VALID_SNAPSHOT_DEPENDENCIES:
        return False, f"invalid snapshot_dependency: {snapshot_dependency}"

    expected_exception = meta.get("expected_exception")
    if not isinstance(expected_exception, str) or not expected_exception:
        return False, "missing or empty expected_exception"

    # Check oracle.patch is non-empty.
    patch_text = (case_dir / "oracle.patch").read_text(encoding="utf-8")
    if not patch_text.strip():
        return False, "oracle.patch is empty"

    if not schema_only:
        # Full validation: baseline must fail, oracle must fix.
        from evals.validate_cases import validate_one_case

        python_args = meta.get("python_args", ["-m", "pytest", "-q"])
        timeout_sec = meta.get("timeout_sec", 120)
        return validate_one_case(
            case_dir, expected_exception, python_args, timeout_sec, "json_compact"
        )

    return True, "schema valid"


def count_patch_lines(patch: str) -> int:
    """Count the number of changed lines (additions + deletions) in a patch."""
    added = 0
    removed = 0
    for line in _diff_lines(patch):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added + removed


# ---------------------------------------------------------------------------
# Shared filter helpers (used by multiple importers)
# ---------------------------------------------------------------------------

# Conservative stdlib-only import allowlist.
# Importers can explicitly extend this for known-safe externals.
_STDLIB_MODULES = frozenset(sys.stdlib_module_names)
_FALLBACK_IMPORT_LINE_RE = re.compile(r"^\s*(?:import|from)\s+", re.MULTILINE)


def _extract_import_roots_from_ast(code: str) -> set[str]:
    tree = ast.parse(code)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root:
                    roots.add(root)
            continue
        if isinstance(node, ast.ImportFrom):
            # Relative imports are local to the snippet/package context.
            if node.level and node.level > 0:
                continue
            if node.module:
                root = node.module.split(".", 1)[0]
                if root:
                    roots.add(root)
    return roots


def _extract_import_roots_fallback(code: str) -> set[str]:
    roots: set[str] = set()
    for raw in code.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("import "):
            payload = line[len("import ") :]
            for chunk in payload.split(","):
                token = chunk.strip().split()
                if not token:
                    continue
                root = token[0].split(".", 1)[0]
                if root:
                    roots.add(root)
            continue
        if line.startswith("from "):
            payload = line[len("from ") :]
            module = payload.split(" import ", 1)[0].strip()
            if module and not module.startswith("."):
                root = module.split(".", 1)[0]
                if root:
                    roots.add(root)
    return roots


def has_external_deps(code: str, *, allowed_externals: set[str] | None = None) -> bool:
    """Check if code imports anything beyond stdlib.

    Uses AST parsing when possible, with a conservative line-based fallback for
    syntactically invalid code snippets.
    """
    allowed = {"__future__"} | _STDLIB_MODULES
    if allowed_externals:
        allowed |= {m.strip() for m in allowed_externals if m and m.strip()}

    try:
        roots = _extract_import_roots_from_ast(code)
    except SyntaxError:
        if not _FALLBACK_IMPORT_LINE_RE.search(code):
            return False
        roots = _extract_import_roots_fallback(code)

    return any(root not in allowed for root in roots)


def ruff_format(code: str) -> str:
    """Format Python code through ruff, returning original on failure."""
    try:
        result = subprocess.run(
            ["ruff", "format", "--stdin-filename=buggy.py"],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return code
