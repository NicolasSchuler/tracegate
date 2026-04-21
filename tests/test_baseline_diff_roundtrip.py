"""Regression tests for baseline diff extraction round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from baselines.adapter_base import BaselineResult
from baselines.intervenor.adapter import IntervenorAdapter
from baselines.intervenor.result_extractor import make_diff as intervenor_make_diff
from baselines.ldb.adapter import LDBAdapter
from baselines.ldb.result_extractor import make_diff as ldb_make_diff
from baselines.tracecoder.adapter import TraceCoderAdapter
from baselines.tracecoder.result_extractor import make_diff as tracecoder_make_diff
from evals.eval_patches import apply_diff


@pytest.mark.parametrize(
    "adapter",
    [
        pytest.param(IntervenorAdapter(), id="intervenor"),
        pytest.param(LDBAdapter(), id="ldb"),
        pytest.param(TraceCoderAdapter(), id="tracecoder"),
    ],
)
def test_baseline_extract_diff_round_trips_without_trailing_newline(
    tmp_path: Path,
    adapter: IntervenorAdapter | LDBAdapter | TraceCoderAdapter,
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    original = "fH) ZM9TB(rKr\nG+s\nmjQ8mT3OA\n94:HhblZa_QFPiyCE\n-5lk\n) nMLpQ-AK\nlXOELxyxeH8ilq"
    repaired = "-Y"
    (case_dir / "buggy.py").write_text(original, encoding="utf-8")

    diff = adapter.extract_diff(case_dir, BaselineResult(repaired_code=repaired, success=True))

    assert diff is not None
    assert "-lXOELxyxeH8ilq\n\\ No newline at end of file\n+-Y\n" in diff
    assert "\\ No newline at end of file\n" in diff

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "buggy.py").write_text(original, encoding="utf-8")
    ok, msg = apply_diff(workdir, diff)

    assert ok is True, msg
    assert (workdir / "buggy.py").read_text(encoding="utf-8") == repaired


@pytest.mark.parametrize(
    ("make_diff", "name"),
    [
        pytest.param(intervenor_make_diff, "intervenor", id="intervenor"),
        pytest.param(ldb_make_diff, "ldb", id="ldb"),
        pytest.param(tracecoder_make_diff, "tracecoder", id="tracecoder"),
    ],
)
def test_result_extractor_make_diff_preserves_missing_trailing_newline(
    tmp_path: Path,
    make_diff,
    name: str,
) -> None:
    del name
    original_path = tmp_path / "buggy.py"
    original_path.write_text("print('before')\n", encoding="utf-8")

    diff = make_diff(original_path, "print('after')")

    assert diff is not None
    assert "+print('after')\n\\ No newline at end of file\n" in diff

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "buggy.py").write_text("print('before')\n", encoding="utf-8")
    ok, msg = apply_diff(workdir, diff)

    assert ok is True, msg
    assert (workdir / "buggy.py").read_text(encoding="utf-8") == "print('after')"
