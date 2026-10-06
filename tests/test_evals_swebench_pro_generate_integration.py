"""End-to-end integration tests for the SWE-bench Pro generation pipeline.

These tests exercise the public ``generate_pro_predictions()`` entry-point
together with the I/O helpers (metadata, instance-IDs, handoff artefacts)
to verify the full pipeline without network, Docker, or real datasets.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from evals.patchers.base import PatchResult
from evals.swebench.prompt_policy import PromptInputs
from evals.swebench.types import (
    CONDITION_BASELINE,
    CONDITION_LLMDEBUG,
    VALID_CONDITIONS,
    PredictionRecord,
)
from evals.swebench_pro.generate import generate_pro_predictions
from evals.swebench_pro.io import (
    copy_instructions_path,
    ensure_run_layout,
    instance_ids_path,
    load_json,
    metadata_path,
    prediction_path,
    read_predictions,
    run_dir_for_id,
    run_index_path,
    validate_paired_predictions,
    write_copy_instructions,
    write_predictions,
    write_run_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INSTANCE_ID_RE = re.compile(r"Instance ID:\s*(\S+)")


def _make_pro_rows(n: int) -> list[dict[str, Any]]:
    """Build *n* fake SWE-bench Pro dataset rows with all expected fields."""
    return [
        {
            "instance_id": f"pro-e2e-{i}",
            "repo": f"demo/pro-repo-{i}",
            "base_commit": f"commit-{i:04x}",
            "problem_statement": f"Fix Pro bug number {i}.",
            "repo_language": "Python",
            "requirements": f"numpy>={i}.0",
            "interface": f"def solve_{i}(x): ...",
            "fail_to_pass": None,
            "pass_to_pass": None,
            "patch": f"diff --git a/gold b/gold\n+gold-{i}\n",
            "test_patch": "",
        }
        for i in range(1, n + 1)
    ]


def _install_fake_datasets(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    """Inject a fake ``datasets`` module so ``load_dataset`` returns *rows*."""
    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = lambda _name, split: rows  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)


def _make_mock_patcher_cls(
    *,
    diff_for_id: dict[str, str] | None = None,
    default_diff: str = "diff --git a/f b/f\n+mock-patch\n",
    error: Exception | None = None,
) -> type:
    """Return a mock patcher class compatible with ``OpenAICompatPatcher``.

    The class accepts any constructor kwargs and returns predictable patches.
    Per-instance control is available via *diff_for_id*; all other instances
    get *default_diff*.  If *error* is set, every call raises that exception.
    """
    _diff_for_id = diff_for_id or {}

    class _MockPatcher:
        def __init__(self, **kwargs: Any) -> None:
            pass  # Accept all OpenAICompatPatcher constructor kwargs

        def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            if error is not None:
                raise error
            # Extract instance_id from prompt for per-instance behaviour
            match = _INSTANCE_ID_RE.search(prompt)
            instance_id = match.group(1) if match else ""
            diff = _diff_for_id.get(instance_id, default_diff)
            return PatchResult(diff=diff, meta=None)

    return _MockPatcher


_COMMON_OPENAI_KWARGS: dict[str, Any] = {
    "openai_base_url": "http://fake:8080/v1",
    "openai_model": "test-model",
    "openai_api_key": "sk-fake",
    "openai_http2": False,
    "openai_async": False,
    "openai_timeout_sec": 10.0,
    "openai_max_tokens": 512,
    "openai_temperature": 0.0,
    "openai_response_mode": "diff",
}


# ---------------------------------------------------------------------------
# Test 1: Full end-to-end pipeline
# ---------------------------------------------------------------------------


def test_generate_pro_predictions_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full pipeline: 5 rows, max_instances=3, both conditions — verify all artefacts."""
    rows = _make_pro_rows(5)
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(
        "evals.swebench_pro.generate.OpenAICompatPatcher",
        _make_mock_patcher_cls(),
    )

    root = tmp_path / "repo-root"
    summary = generate_pro_predictions(
        run_id="e2e-run-1",
        conditions=["baseline", "llmdebug"],
        max_instances=3,
        root=root,
        **_COMMON_OPENAI_KWARGS,
        prompt_inputs=PromptInputs(snapshot_dir=tmp_path),
    )

    # -- Summary assertions --
    assert summary.run_id == "e2e-run-1"
    assert summary.instance_count == 3
    assert len(summary.condition_summaries) == 2

    run_dir = summary.run_dir

    # -- Both condition directories exist --
    for condition in VALID_CONDITIONS:
        assert (run_dir / condition).is_dir()

    # -- Each condition has a JSONL file with 3 predictions --
    for cs in summary.condition_summaries:
        assert cs.condition in {CONDITION_BASELINE, CONDITION_LLMDEBUG}
        assert cs.total_instances == 3
        assert cs.generated == 3
        assert cs.resumed == 0
        assert cs.empty_patch_count == 0
        assert cs.request_error_count == 0

        records = read_predictions(cs.predictions_path)
        assert len(records) == 3
        for record in records:
            assert record.instance_id
            assert record.model_patch
            assert "mock-patch" in record.model_patch

    # -- Metadata JSON written with correct schema version --
    meta_file = metadata_path(run_dir)
    assert meta_file.exists()
    meta = load_json(meta_file)
    assert meta["schema_version"] == "swebench_pro_run_v1"
    assert meta["run_id"] == "e2e-run-1"
    assert meta["instance_count"] == 3
    assert set(meta["conditions"]) == {"baseline", "llmdebug"}  # type: ignore[arg-type]
    assert isinstance(meta.get("condition_summaries"), list)
    assert len(meta["condition_summaries"]) == 2  # type: ignore[arg-type]

    # -- Instance IDs file written --
    ids_file = instance_ids_path(run_dir)
    assert ids_file.exists()
    written_ids = ids_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(written_ids) == 3
    # IDs should be sorted (no instance_ids_file provided)
    assert written_ids == sorted(written_ids)


# ---------------------------------------------------------------------------
# Test 2: Single condition raises ValueError (paired enforcement)
# ---------------------------------------------------------------------------


def test_generate_pro_predictions_single_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing a single condition must raise ValueError — Pro requires paired conditions."""
    rows = _make_pro_rows(3)
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(
        "evals.swebench_pro.generate.OpenAICompatPatcher",
        _make_mock_patcher_cls(),
    )

    root = tmp_path / "repo-root"
    with pytest.raises(ValueError, match="conditions must include exactly baseline,llmdebug"):
        generate_pro_predictions(
            run_id="single-cond",
            conditions="baseline",
            root=root,
            **_COMMON_OPENAI_KWARGS,
            prompt_inputs=PromptInputs(snapshot_dir=tmp_path),
        )


# ---------------------------------------------------------------------------
# Test 3: Resume skips already-generated predictions
# ---------------------------------------------------------------------------


def test_generate_pro_predictions_with_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-write 2 predictions, resume=True with 5 instances → 3 newly generated."""
    rows = _make_pro_rows(5)
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(
        "evals.swebench_pro.generate.OpenAICompatPatcher",
        _make_mock_patcher_cls(default_diff="diff --git a/f b/f\n+freshly-generated\n"),
    )

    root = tmp_path / "repo-root"

    # Pre-create the run layout and seed predictions for both conditions.
    run_dir = run_dir_for_id("resume-run", root=root)
    ensure_run_layout(run_dir, conditions=list(VALID_CONDITIONS))

    pre_existing = [
        PredictionRecord(
            instance_id="pro-e2e-1",
            model_patch="diff --git a/f b/f\n+pre-existing-1\n",
            model_name_or_path="old-model",
        ),
        PredictionRecord(
            instance_id="pro-e2e-2",
            model_patch="diff --git a/f b/f\n+pre-existing-2\n",
            model_name_or_path="old-model",
        ),
    ]
    for condition in VALID_CONDITIONS:
        write_predictions(prediction_path(run_dir, condition), pre_existing)

    summary = generate_pro_predictions(
        run_id="resume-run",
        conditions=["baseline", "llmdebug"],
        max_instances=0,  # 0 = all instances
        resume=True,
        root=root,
        **_COMMON_OPENAI_KWARGS,
        prompt_inputs=PromptInputs(snapshot_dir=tmp_path),
    )

    assert summary.instance_count == 5

    for cs in summary.condition_summaries:
        assert cs.resumed == 2
        assert cs.generated == 3
        assert cs.total_instances == 5

        records = read_predictions(cs.predictions_path)
        assert len(records) == 5

        by_id = {r.instance_id: r for r in records}
        # Pre-existing records are preserved with their original patches
        assert "pre-existing-1" in by_id["pro-e2e-1"].model_patch
        assert "pre-existing-2" in by_id["pro-e2e-2"].model_patch
        # Newly generated records carry the stub's patch content
        for i in range(3, 6):
            assert "freshly-generated" in by_id[f"pro-e2e-{i}"].model_patch


# ---------------------------------------------------------------------------
# Test 4: Empty patches counted but still produce records
# ---------------------------------------------------------------------------


def test_generate_pro_predictions_empty_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patcher returns empty patches for 2 of 5 instances — health passes, records present."""
    rows = _make_pro_rows(5)
    _install_fake_datasets(monkeypatch, rows)

    diff_map = {
        "pro-e2e-2": "",
        "pro-e2e-4": "",
    }
    monkeypatch.setattr(
        "evals.swebench_pro.generate.OpenAICompatPatcher",
        _make_mock_patcher_cls(
            diff_for_id=diff_map,
            default_diff="diff --git a/f b/f\n+valid-patch\n",
        ),
    )

    root = tmp_path / "repo-root"
    summary = generate_pro_predictions(
        run_id="empty-patch-run",
        conditions=["baseline", "llmdebug"],
        max_instances=5,
        root=root,
        **_COMMON_OPENAI_KWARGS,
        prompt_inputs=PromptInputs(snapshot_dir=tmp_path),
    )

    for cs in summary.condition_summaries:
        # 2 empty patches per condition
        assert cs.empty_patch_count == 2
        assert cs.request_error_count == 0
        assert cs.generated == 5

        records = read_predictions(cs.predictions_path)
        assert len(records) == 5

        by_id = {r.instance_id: r for r in records}
        # Empty-patch instances still have records with empty model_patch
        assert by_id["pro-e2e-2"].model_patch == ""
        assert by_id["pro-e2e-4"].model_patch == ""
        # Others carry valid patches
        assert "valid-patch" in by_id["pro-e2e-1"].model_patch
        assert "valid-patch" in by_id["pro-e2e-3"].model_patch
        assert "valid-patch" in by_id["pro-e2e-5"].model_patch


# ---------------------------------------------------------------------------
# Test 5: All-fail triggers health check RuntimeError
# ---------------------------------------------------------------------------


def test_generate_pro_predictions_all_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the patcher always raises, the all-fail health check aborts with RuntimeError."""
    rows = _make_pro_rows(3)
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(
        "evals.swebench_pro.generate.OpenAICompatPatcher",
        _make_mock_patcher_cls(error=ConnectionError("server down")),
    )

    root = tmp_path / "repo-root"
    with pytest.raises(RuntimeError, match="all patch requests failed"):
        generate_pro_predictions(
            run_id="all-fail-run",
            conditions=["baseline", "llmdebug"],
            max_instances=3,
            root=root,
            **_COMMON_OPENAI_KWARGS,
            prompt_inputs=PromptInputs(snapshot_dir=tmp_path),
        )


# ---------------------------------------------------------------------------
# Test 6: Handoff artefacts after generation
# ---------------------------------------------------------------------------


def test_handoff_after_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate predictions → write handoff files → validate paired predictions."""
    rows = _make_pro_rows(3)
    _install_fake_datasets(monkeypatch, rows)
    monkeypatch.setattr(
        "evals.swebench_pro.generate.OpenAICompatPatcher",
        _make_mock_patcher_cls(),
    )

    root = tmp_path / "repo-root"
    summary = generate_pro_predictions(
        run_id="handoff-run",
        conditions=["baseline", "llmdebug"],
        max_instances=3,
        root=root,
        **_COMMON_OPENAI_KWARGS,
        prompt_inputs=PromptInputs(snapshot_dir=tmp_path),
    )

    run_dir = summary.run_dir
    conditions = list(VALID_CONDITIONS)

    # -- Write handoff artefacts --
    index_file = write_run_index(run_dir, run_id="handoff-run", conditions=conditions)
    instructions_file = write_copy_instructions(run_dir, conditions=conditions)

    # -- Run index JSON exists and has expected schema --
    assert index_file == run_index_path(run_dir)
    assert index_file.exists()
    index = load_json(index_file)
    assert index["schema_version"] == "swebench_pro_handoff_index_v1"
    assert index["run_id"] == "handoff-run"
    assert "predictions" in index
    predictions_map = index["predictions"]
    assert isinstance(predictions_map, dict)
    assert set(predictions_map.keys()) == {"baseline", "llmdebug"}

    # -- Copy instructions file exists and contains expected content --
    assert instructions_file == copy_instructions_path(run_dir)
    assert instructions_file.exists()
    text = instructions_file.read_text(encoding="utf-8")
    assert "SWE-Bench Pro Handoff" in text
    assert "score_swebench_pro_official" in text
    assert str(run_dir / "docker_eval") in text

    assert index["official_eval_root"] == str(run_dir / "docker_eval")

    # -- validate_paired_predictions passes with no errors --
    ids_file = instance_ids_path(run_dir)
    expected_ids = ids_file.read_text(encoding="utf-8").strip().splitlines()
    validate_paired_predictions(
        run_dir,
        conditions=conditions,
        expected_ids=expected_ids,
    )
