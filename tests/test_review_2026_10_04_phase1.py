"""Regression tests for Phase 1 (result integrity) of the 2026-10-04 review remediation.

Critical findings live in ``test_review_2026_10_04_criticals.py``; this file covers the
High/Medium Phase 1 findings that need a run through ``run_one_condition``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from evals import run_eval
from evals.eval_types import Condition
from evals.patchers.base import PatchResult

_FIX = (
    "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,2 @@\n"
    " def f() -> int:\n-    return 1\n+    return 2\n"
)


@dataclass
class _DiffPatcher:
    diffs: list[str | None]
    name: str = "scripted"
    prompts: list[str] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        diff = self.diffs[min(len(self.prompts), len(self.diffs)) - 1]
        return PatchResult(diff=diff, meta={})


def _case(tmp_path: Path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="phase1_case",
        path=case_dir,
        category="logic",
        difficulty="easy",
        snapshot_dependency="helpful",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=60,
    )


def _run(
    tmp_path: Path, patcher: _DiffPatcher, *, k: int = 1, **inputs: Any
) -> list[dict[str, Any]]:
    case = _case(tmp_path)
    workdir = tmp_path / "work"
    shutil.copytree(case.path, workdir)
    previous = run_eval.set_active_execution_config(run_eval.ExecutionConfig(executor="local"))
    try:
        return run_eval.run_one_condition(
            case=case,
            condition=Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=tmp_path / "ctx",
            patcher=patcher,
            k=k,
            output_format="json_compact",
            run_id="phase1",
            include_files_context=True,
            **inputs,
        )
    finally:
        run_eval.set_active_execution_config(previous)


def _attempts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if int(r.get("attempt", 0)) >= 1]


def test_att14_patch_verifier_runs_for_non_adaptive_conditions_by_default(
    tmp_path: Path,
) -> None:
    records = _run(tmp_path, _DiffPatcher([_FIX]), force_patch_attempt=False)
    record = _attempts(records)[0]
    assert record["patch_meta"]["patch_verifier_scope"] == "all"
    assert record["patch_meta"]["patch_verifier_enabled"] is True
    assert record["success"] is True


_UNAPPLIABLE = (
    "--- a/buggy.py\n+++ b/buggy.py\n@@ -7,2 +7,2 @@\n"
    " def g() -> int:\n-    return 5\n+    return 6\n"
)


def test_att07_forced_marker_recovery_stays_apply_failure_and_is_reverted(
    tmp_path: Path,
) -> None:
    records = _run(
        tmp_path,
        _DiffPatcher([_UNAPPLIABLE]),
        force_patch_attempt=True,
        patch_verifier_scope="disabled",
    )
    record = _attempts(records)[0]
    meta = record["patch_meta"]
    assert record["note"] == "patch_apply_failed"
    assert record["success"] is False
    assert meta["runner_forced_patch_recovery"] is True
    assert meta["runner_forced_patch_recovery_reverted"] is True
    assert meta["patch_file_paths"] == ["buggy.py"]
    assert "llmdebug-force-attempt" not in (tmp_path / "work" / "buggy.py").read_text(
        encoding="utf-8"
    )


def parsed_cli_defaults(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Parse ``run_eval`` CLI defaults without running an eval."""
    captured: dict[str, Any] = {}

    def _capture(*, parser: Any, args: Any, argv_tokens: Any) -> int:
        del parser, argv_tokens
        captured.update(vars(args))
        return 0

    monkeypatch.setattr("evals.eval_main_flow.run_main_flow", _capture)
    assert run_eval.main([]) == 0
    return captured


def test_att07_force_patch_attempt_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parsed_cli_defaults(monkeypatch)["force_patch_attempt"] is False


def test_llm13_patcher_forced_marker_diff_is_recognized_as_noop() -> None:
    from evals.eval_patch_constants import FORCED_PATCH_MARKER
    from evals.eval_patches import _is_forced_noop_marker_diff

    patcher_style = (
        "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,3 @@\n"
        f" def f() -> int:\n     return 1\n+{FORCED_PATCH_MARKER}\n"
    )
    runner_style = patcher_style.replace(FORCED_PATCH_MARKER, f"{FORCED_PATCH_MARKER}-2")
    assert _is_forced_noop_marker_diff(patcher_style)
    assert _is_forced_noop_marker_diff(runner_style)
    assert not _is_forced_noop_marker_diff(_FIX)
