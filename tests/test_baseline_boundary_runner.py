"""Runner-level seam tests for baseline patch application and bookkeeping."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from baselines.adapter_base import (
    BaselineAdapter,
    BaselineCaseContext,
    BaselineConfig,
    BaselineFileEdit,
    BaselineResult,
)
from evals import run_eval
from evals.patchers.baseline_patcher import BaselinePatcher
from llmdebug import _json as json


class _RunnerStubAdapter(BaselineAdapter):
    def __init__(self, repair_result: BaselineResult) -> None:
        self._repair_result = repair_result

    @property
    def name(self) -> str:
        return "intervenor"

    def setup(self, config: BaselineConfig) -> None:
        del config

    def convert_case(
        self,
        case_dir: Path,
        *,
        case_context: BaselineCaseContext | None = None,
    ) -> dict[str, Any]:
        return {
            "case_dir": str(case_dir),
            "case_id": case_context.case_id if case_context is not None else case_dir.name,
        }

    def run_repair(self, case_input: dict[str, Any], config: BaselineConfig) -> BaselineResult:
        del case_input, config
        return self._repair_result


def _make_case_spec(repo_root: Path, case_id: str) -> run_eval.CaseSpec:
    case_dir = repo_root / "synthetic_cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def _write_single_file_case(case_dir: Path) -> None:
    (case_dir / "buggy.py").write_text(
        "def compute() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import compute\n\n\ndef test_compute() -> None:\n    assert compute() == 2\n",
        encoding="utf-8",
    )


def _write_multifile_case(case_dir: Path) -> None:
    (case_dir / "buggy.py").write_text(
        "from helper import compute_base\n\n\n"
        "def compute() -> int:\n"
        "    return compute_base() + 1\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text(
        "def compute_base() -> int:\n    return 0\n",
        encoding="utf-8",
    )
    (case_dir / "obsolete.py").write_text("OBSOLETE = True\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from pathlib import Path\n\n"
        "from buggy import compute\n"
        "from extra import created_value\n\n\n"
        "def test_compute() -> None:\n"
        "    assert compute() == 2\n\n\n"
        "def test_extra_file() -> None:\n"
        "    assert created_value() == 2\n\n\n"
        "def test_obsolete_file_removed() -> None:\n"
        "    assert not Path(__file__).with_name('obsolete.py').exists()\n",
        encoding="utf-8",
    )


def _run_boundary_case(
    *,
    tmp_path: Path,
    monkeypatch,
    case_id: str,
    repair_result: BaselineResult,
    write_case,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = tmp_path / case_id
    repo_root.mkdir(parents=True, exist_ok=True)
    case = _make_case_spec(repo_root, case_id)
    write_case(case.path)

    patcher = BaselinePatcher(
        adapter=_RunnerStubAdapter(repair_result),
        config=BaselineConfig(
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            repo_dir=repo_root,
        ),
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "make_patcher", lambda _args: patcher)

    run_id = f"{case_id}-run"
    argv = [
        "--run-id",
        run_id,
        "--patcher",
        "baseline",
        "--baseline-name",
        "intervenor",
        "--baseline-repo-dir",
        str(repo_root),
        "--baseline-python",
        sys.executable,
        "--openai-base-url",
        "http://127.0.0.1:1234",
        "--openai-model",
        "test-model",
        "--k",
        "1",
        "--jobs",
        "1",
        "--max-cases",
        "1",
        "--executor",
        "local",
    ]
    if extra_args:
        argv.extend(extra_args)
    exit_code = run_eval.main(argv)

    assert exit_code == 0
    results_path = repo_root / "evals" / "results" / f"{run_id}.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    return rows[0]


def test_runner_records_success_for_single_file_baseline_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _run_boundary_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        case_id="baseline-single-file-success",
        repair_result=BaselineResult(
            repaired_code="def compute() -> int:\n    return 2\n",
            success=True,
        ),
        write_case=_write_single_file_case,
    )

    assert record["success"] is True
    assert record["note"] == "success"
    assert record["after_returncode"] == 0
    assert record["patch_meta"]["baseline_success"] is True


def test_runner_records_success_for_multifile_add_update_delete_baseline_diff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _run_boundary_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        case_id="baseline-multifile-success",
        repair_result=BaselineResult(
            file_edits=[
                BaselineFileEdit(
                    path="buggy.py",
                    content=(
                        "from extra import created_value\n"
                        "from helper import compute_base\n\n\n"
                        "def compute() -> int:\n"
                        "    return compute_base() + created_value()\n"
                    ),
                ),
                BaselineFileEdit(
                    path="extra.py",
                    content="def created_value() -> int:\n    return 2\n",
                ),
                BaselineFileEdit(path="obsolete.py", content=None),
            ],
            success=True,
        ),
        write_case=_write_multifile_case,
    )

    assert record["success"] is True
    assert record["note"] == "success"
    assert record["after_returncode"] == 0
    assert record["patch_meta"]["baseline_success"] is True


def test_runner_records_patch_apply_failed_but_preserves_baseline_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _run_boundary_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        case_id="baseline-patch-apply-failed",
        repair_result=BaselineResult(
            patch_diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1 @@\n"
                "-def compute() -> int:\n"
                "-    return 1+    return 2"
            ),
            success=True,
        ),
        write_case=_write_single_file_case,
        extra_args=["--no-force-patch-attempt"],
    )

    assert record["success"] is False
    assert record["note"] == "patch_apply_failed"
    assert record["patch_meta"]["baseline_success"] is True


def test_runner_records_syntax_preflight_failed_but_preserves_baseline_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _run_boundary_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        case_id="baseline-syntax-preflight-failed",
        repair_result=BaselineResult(
            repaired_code="def compute() -> int:\n    return )\n",
            success=True,
        ),
        write_case=_write_single_file_case,
        extra_args=["--no-syntax-preflight-fallthrough"],
    )

    assert record["success"] is False
    assert record["note"] == "syntax_preflight_failed"
    assert record["patch_meta"]["baseline_success"] is True
    assert int(record["patch_meta"]["syntax_preflight_failures"]) > 0
