"""The files a case directory must contain: one contract for importers and validate_cases.

The importers' check required ``buggy.py`` and ``validate_cases`` required
``test_buggy.py``, but native-repro cases (``repro_mode = native_dataset_test``:
BugsInPy, PyBugHive, TrickyBugs) keep their sources and tests at repository paths, so
one validator or the other rejected every such case (DAT-06). Both now call
:func:`validate_case_contract`.
"""

from __future__ import annotations

from pathlib import Path

from . import _json as json
from .eval_patches import _extract_diff_paths

NATIVE_REPRO_MODE = "native_dataset_test"
REQUIRED_CASE_FILES = ("test_buggy.py", "case.json", "oracle.patch")


def validate_case_contract(case_path: Path) -> tuple[bool, str]:
    """Check that *case_path* contains the files its kind of case needs.

    Every case needs ``case.json`` and ``oracle.patch``. A native-repro case needs every
    file its oracle patch changes. Any other case needs ``test_buggy.py`` and at least one
    non-test Python implementation file. Returns ``(ok, message)``.
    """
    for required_name in ("case.json", "oracle.patch"):
        if not (case_path / required_name).exists():
            return False, f"missing required file: {required_name}"
    if _repro_mode(case_path) == NATIVE_REPRO_MODE:
        oracle_text = (case_path / "oracle.patch").read_text(encoding="utf-8")
        for rel_path in _extract_diff_paths(oracle_text):
            if not (case_path / rel_path).exists():
                return False, f"missing required file: {rel_path}"
        return True, "ok"
    if not (case_path / "test_buggy.py").exists():
        return False, "missing required file: test_buggy.py"
    implementation_files = [
        p
        for p in case_path.rglob("*.py")
        if p.name != "test_buggy.py" and not p.name.startswith("test_")
    ]
    if not implementation_files:
        return False, "missing implementation python file (non-test)"
    return True, "ok"


def _repro_mode(case_path: Path) -> str:
    try:
        meta = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    return str(meta.get("repro_mode") or "") if isinstance(meta, dict) else ""
