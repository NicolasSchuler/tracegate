"""Regression tests for the Critical findings of the 2026-10-04 code review.

See ``docs/2026-10-04-code-review-findings.md``. Each test exercises the real seam
the finding is about and is marked ``xfail(strict=True)`` until its fix lands, so
removing the marker is part of the fix commit.
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# ATT-01: the model can edit the test oracle (conftest exit-status hack).
# ---------------------------------------------------------------------------

_CONFTEST_HACK_DIFF = (
    "diff --git a/conftest.py b/conftest.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/conftest.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def pytest_sessionfinish(session, exitstatus):\n"
    "+    session.exitstatus = 0\n"
)


@dataclass
class _FixedDiffPatcher:
    diff: str
    name: str = "fixed_diff"
    calls: list[str] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> Any:
        from evals.patchers.base import PatchResult

        del context_dir, cwd
        self.calls.append(prompt)
        return PatchResult(diff=self.diff, meta={})


def _make_failing_case(tmp_path: Path) -> Any:
    from evals import run_eval

    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True)
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="att01_case",
        path=case_dir,
        category="logic",
        difficulty="easy",
        snapshot_dependency="helpful",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=60,
    )


def test_att01_conftest_exit_status_hack_is_not_a_success(tmp_path: Path) -> None:
    from evals import run_eval
    from evals.eval_types import Condition

    case = _make_failing_case(tmp_path)
    workdir = tmp_path / "workdir"
    shutil.copytree(case.path, workdir)
    patcher = _FixedDiffPatcher(diff=_CONFTEST_HACK_DIFF)

    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )
    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=tmp_path / "ctx",
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="att01",
            include_files_context=True,
            force_patch_attempt=False,
            # Exercise the oracle itself; the verifier would reject this patch earlier.
            patch_verifier_scope="disabled",
        )
    finally:
        run_eval.set_active_execution_config(previous_execution)

    attempts = [r for r in records if int(r.get("attempt", 0)) >= 1]
    assert patcher.calls, "patcher was never asked for a patch"
    assert attempts
    assert not any(bool(r.get("success")) for r in attempts)
    first = attempts[0]
    assert first["note"] == "tests_still_failing"
    assert first["patch_meta"]["test_surface_modified"] is True
    assert first["patch_meta"]["test_surface_restored_paths"] == ["conftest.py"]
    assert first["patch_meta"]["pytest_verdict_source"] == "junit"
    assert first["patch_meta"]["pytest_verdict_counts"]["failures"] == 1
    assert not (workdir / "conftest.py").exists()


# ---------------------------------------------------------------------------
# ATT-02 (D1): case descriptions must not reach any model-facing baseline input.
# ---------------------------------------------------------------------------

_SENTINEL = "SENTINEL-ROOT-CAUSE-off-by-one-in-loop"


def test_att02_baseline_inputs_omit_case_description(tmp_path: Path) -> None:
    """Every baseline arm's model-facing input is free of dataset descriptions/labels."""
    import json

    from baselines.intervenor.case_converter import convert_case as intervenor_convert
    from baselines.lambda_rlm.case_converter import convert_case as lambda_convert
    from baselines.ldb.case_converter import convert_case as ldb_convert
    from baselines.rlm.case_converter import convert_case as rlm_convert
    from baselines.tracecoder.case_converter import convert_case as tracecoder_convert
    from evals.eval_attempt_flow import _write_baseline_case_context
    from evals.patchers.baseline_patcher import _load_baseline_case_context

    case = dataclasses.replace(
        _make_failing_case(tmp_path), description=_SENTINEL, category="SENTINEL-CATEGORY"
    )
    (case.path / "case.json").write_text(
        json.dumps({"description": _SENTINEL, "category": "SENTINEL-CATEGORY"}),
        encoding="utf-8",
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    _write_baseline_case_context(attempt_dir, case)
    context_file = attempt_dir / "baseline_case_context.json"
    assert _SENTINEL not in context_file.read_text(encoding="utf-8")

    # Even a context file that still carries the labels must not leak them.
    context_file.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "python_args": list(case.python_args),
                "description": _SENTINEL,
            }
        ),
        encoding="utf-8",
    )
    loaded = _load_baseline_case_context(context_file, default_case_id=case.case_id)
    assert loaded.case_context is not None
    assert any("hidden_case_fields_dropped" in issue for issue in loaded.issues)
    context = loaded.case_context

    workdir = tmp_path / "work"
    shutil.copytree(case.path, workdir)
    (workdir / "case.json").unlink()
    inputs = [
        ldb_convert(workdir, case_id=case.case_id),
        intervenor_convert(workdir, case_context=context),
        tracecoder_convert(workdir, case_context=context),
        lambda_convert(workdir, case_context=context),
        rlm_convert(workdir, case_context=context),
    ]
    for converted in inputs:
        rendered = repr(converted.to_dict())
        assert _SENTINEL not in rendered
        assert "SENTINEL-CATEGORY" not in rendered
        assert converted.case_information is not None
        assert converted.case_information["case_info_policy"] == "neutral_v1"


# ---------------------------------------------------------------------------
# BAS-01 (D3): LDB-incompatible cases are not scored as LDB failures.
# ---------------------------------------------------------------------------


def test_bas01_unsupported_baseline_rows_are_excluded_from_pairs() -> None:
    from evals.analyze_results import _derive_case_condition_rows
    from evals.stats import analyze_paired_results

    def _record(case_id: str, condition: str, success: bool, **patch_meta: Any) -> dict[str, Any]:
        return {
            "run_id": "r1",
            "case_id": case_id,
            "condition": condition,
            "attempt": 1,
            "success": success,
            "category": "logic",
            "patch_meta": dict(patch_meta),
        }

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for case_id in ("c1", "c2", "c3"):
        grouped[("r1", case_id, "base_repair")] = [_record(case_id, "base_repair", True)]
    grouped[("r1", "c1", "ldb")] = [_record("c1", "ldb", True)]
    for case_id in ("c2", "c3"):
        grouped[("r1", case_id, "ldb")] = [
            _record(case_id, "ldb", False, error="case_incompatible", baseline_supported=False)
        ]

    rows = _derive_case_condition_rows(grouped)
    unsupported = [r for r in rows if r["condition"] == "ldb" and r["case_id"] != "c1"]
    assert all(r.get("baseline_supported") is False for r in unsupported)
    result = analyze_paired_results(rows, condition_a="base_repair", condition_b="ldb", seed=0)
    assert result["n_pairs"] == 1


# ---------------------------------------------------------------------------
# POL-01 (D6): the bandit must be able to learn with default settings.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="POL-01")
def test_pol01_default_decay_is_one_and_decay_is_per_arm() -> None:
    import numpy as np

    from evals.adaptive_online_policy import (
        AdaptiveOnlinePolicyController,
        OnlinePolicyConfig,
    )
    from evals.eval_types import (
        DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
        DEFAULT_ADAPTIVE_ONLINE_DECAY,
    )

    assert DEFAULT_ADAPTIVE_ONLINE_DECAY == 1.0
    assert DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY == 1.0
    assert OnlinePolicyConfig.normalize(mode="on").decay == 1.0

    controller = AdaptiveOnlinePolicyController(OnlinePolicyConfig.normalize(mode="on", decay=0.5))
    features = {"bias": 1.0, "x": 0.5}
    controller._ensure_feature_state(cleaned_features=features)
    before = controller.state
    assert controller._update_model(action="frame_only", reward=1.0, feature_values=features)
    after = controller.state
    untouched = [a for a in before.A if a != "frame_only"]
    assert untouched
    for action in untouched:
        np.testing.assert_allclose(after.A[action], before.A[action])
        np.testing.assert_allclose(after.b[action], before.b[action])


# ---------------------------------------------------------------------------
# STA-01: missing token telemetry is not zero tokens.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="STA-01")
def test_sta01_missing_token_usage_is_none_not_zero() -> None:
    from evals.analyze_results import _token_usage_breakdown

    records = [
        {"attempt": 1, "attempt_metadata": {"token_usage_available": False}},
        {"attempt": 2, "attempt_metadata": {"token_usage_available": False}},
    ]
    assert _token_usage_breakdown(records) == (None, None, None)


# ---------------------------------------------------------------------------
# STA-02: repeated runs of the same case are not independent pairs.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="STA-02")
def test_sta02_repeated_runs_do_not_inflate_significance() -> None:
    from evals.stats import analyze_paired_results

    outcomes = [(False, True)] * 6 + [(True, False)] * 2 + [(True, True)] * 20
    rows: list[dict[str, Any]] = []
    for run_id in ("r1", "r2", "r3"):
        for idx, (a_ok, b_ok) in enumerate(outcomes):
            for condition, ok in (("base_repair", a_ok), ("layered_adaptive_llm", b_ok)):
                rows.append(
                    {
                        "run_id": run_id,
                        "case_id": f"c{idx}",
                        "condition": condition,
                        "category": "logic",
                        "success": ok,
                    }
                )
    single = analyze_paired_results(
        [r for r in rows if r["run_id"] == "r1"], seed=0, n_bootstrap=500
    )
    pooled = analyze_paired_results(rows, seed=0, n_bootstrap=500)
    assert pooled["n_pairs"] == single["n_pairs"] == len(outcomes)
    assert pooled["overall_mcnemar"]["p_value"] == pytest.approx(
        single["overall_mcnemar"]["p_value"]
    )


# ---------------------------------------------------------------------------
# DAT-03: the SWE-Bench llmdebug condition needs a snapshot directory.
# ---------------------------------------------------------------------------


def test_dat03_llmdebug_condition_without_snapshot_dir_fails_fast(tmp_path: Path) -> None:
    from evals.swebench.generate import generate_lite_predictions

    with pytest.raises(ValueError, match="snapshot"):
        generate_lite_predictions(
            run_id="dat03",
            conditions="baseline,llmdebug",
            openai_base_url="http://127.0.0.1:9",
            openai_model="stub",
            openai_api_key=None,
            root=tmp_path,
        )


@pytest.mark.parametrize("cli_module", ["evals.swebench.cli", "evals.swebench_pro.cli"])
def test_dat03_cli_rejects_llmdebug_without_snapshot_dir(
    cli_module: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib

    cli = importlib.import_module(cli_module)
    command = "generate-lite" if cli_module == "evals.swebench.cli" else "generate-pro"
    exit_code = cli.main(
        [
            command,
            "--run-id",
            "dat03-cli",
            "--conditions",
            "baseline,llmdebug",
            "--openai-base-url",
            "http://127.0.0.1:9",
            "--openai-model",
            "stub",
        ]
    )
    assert exit_code == 1
    assert "--snapshot-dir" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CAP-01: a local integer wider than 64 bits must not abort capture.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="CAP-01")
def test_cap01_big_int_local_does_not_abort_capture() -> None:
    from llmdebug import SnapshotConfig
    from llmdebug.capture import capture_exception

    def boom() -> None:
        big = 2**70
        assert big < 0, "boom"

    try:
        boom()
    except AssertionError as exc:
        cfg = SnapshotConfig(out_dir=tempfile.mkdtemp())
        path = capture_exception("cap01", exc, exc.__traceback__, cfg)
    assert path is not None
    assert str(2**70) in Path(path).read_text(encoding="utf-8")
