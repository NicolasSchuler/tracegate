"""Resuming SWE-Bench generation must keep the stored prompt policy (DAT-01/02).

Resume reuses the existing predictions but rewrites ``run_metadata.json``. Without a
check, resuming without ``--include-test-ids`` / ``--include-hints`` relabelled
leaked-test predictions as official-task-only.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from evals import _json as json
from evals.patchers.base import PatchResult
from evals.swebench import generate as lite_generate
from evals.swebench.io import metadata_path, prediction_path, read_predictions
from evals.swebench.prompt_policy import PromptInputs, prompt_policy_fingerprint
from evals.swebench.types import CONDITION_BASELINE, VALID_CONDITIONS
from evals.swebench_pro import generate as pro_generate

_IDS = ["inst-1", "inst-2"]


class _StubPatcher:
    """Answers every proposal with the same usable diff."""

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(diff="diff --git a/f.py b/f.py\n+patched\n", meta={})


def _rows() -> dict[str, dict[str, Any]]:
    return {
        instance_id: {
            "instance_id": instance_id,
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": f"Fix {instance_id}.",
            "hints_text": "a maintainer hint",
            "FAIL_TO_PASS": ["tests/test_x.py::test_a"],
            "fail_to_pass": ["tests/test_x.py::test_a"],
        }
        for instance_id in _IDS
    }


_FLAVOURS = [
    (lite_generate, "generate_lite_predictions"),
    (pro_generate, "generate_pro_predictions"),
]


def _generate(
    *,
    module: ModuleType,
    entrypoint: str,
    root: Path,
    snapshot_dir: Path,
    include_test_ids: bool,
    resume: bool,
) -> Any:
    return getattr(module, entrypoint)(
        run_id="policy-run",
        conditions=",".join(VALID_CONDITIONS),
        openai_base_url="http://127.0.0.1:9",
        openai_model="stub-model",
        openai_api_key=None,
        prompt_inputs=PromptInputs(snapshot_dir=snapshot_dir, include_test_ids=include_test_ids),
        resume=resume,
        root=root,
    )


@pytest.fixture
def _stubbed(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    for module in (lite_generate, pro_generate):
        monkeypatch.setattr(module, "_load_dataset_rows", lambda **_kwargs: _rows())
        monkeypatch.setattr(module, "OpenAICompatPatcher", lambda **_kwargs: _StubPatcher())
    del request


@pytest.mark.parametrize(("module", "entrypoint"), _FLAVOURS, ids=["lite", "pro"])
@pytest.mark.usefixtures("_stubbed")
def test_resume_rejects_a_changed_prompt_policy(
    tmp_path: Path, module: ModuleType, entrypoint: str
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    summary = _generate(
        module=module,
        entrypoint=entrypoint,
        root=tmp_path,
        snapshot_dir=snapshot_dir,
        include_test_ids=True,
        resume=False,
    )
    run_dir = summary.run_dir
    before = metadata_path(run_dir).read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="different prompt policy"):
        _generate(
            module=module,
            entrypoint=entrypoint,
            root=tmp_path,
            snapshot_dir=snapshot_dir,
            include_test_ids=False,
            resume=True,
        )

    # The rejected resume left the recorded policy and the predictions untouched.
    assert metadata_path(run_dir).read_text(encoding="utf-8") == before
    stored = json.loads(before)
    assert stored["prompt_policy"]["include_test_ids"] is True
    assert stored["prompt_policy_fingerprint"] == prompt_policy_fingerprint(stored["prompt_policy"])
    assert [
        record.instance_id
        for record in read_predictions(prediction_path(run_dir, CONDITION_BASELINE))
    ] == _IDS


@pytest.mark.parametrize(("module", "entrypoint"), _FLAVOURS, ids=["lite", "pro"])
@pytest.mark.usefixtures("_stubbed")
def test_resume_accepts_the_stored_prompt_policy(
    tmp_path: Path, module: ModuleType, entrypoint: str
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    _generate(
        module=module,
        entrypoint=entrypoint,
        root=tmp_path,
        snapshot_dir=snapshot_dir,
        include_test_ids=True,
        resume=False,
    )
    summary = _generate(
        module=module,
        entrypoint=entrypoint,
        root=tmp_path,
        snapshot_dir=snapshot_dir,
        include_test_ids=True,
        resume=True,
    )
    assert [item.resumed for item in summary.condition_summaries] == [len(_IDS)] * len(
        VALID_CONDITIONS
    )
    assert [item.generated for item in summary.condition_summaries] == [0] * len(VALID_CONDITIONS)


@pytest.mark.parametrize(("module", "entrypoint"), _FLAVOURS, ids=["lite", "pro"])
@pytest.mark.usefixtures("_stubbed")
def test_resume_rejects_metadata_without_a_recorded_policy(
    tmp_path: Path, module: ModuleType, entrypoint: str
) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    summary = _generate(
        module=module,
        entrypoint=entrypoint,
        root=tmp_path,
        snapshot_dir=snapshot_dir,
        include_test_ids=True,
        resume=False,
    )
    metadata_file = metadata_path(summary.run_dir)
    legacy = json.loads(metadata_file.read_text(encoding="utf-8"))
    legacy.pop("prompt_policy")
    legacy.pop("prompt_policy_fingerprint")
    metadata_file.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError, match="predates prompt-policy bookkeeping"):
        _generate(
            module=module,
            entrypoint=entrypoint,
            root=tmp_path,
            snapshot_dir=snapshot_dir,
            include_test_ids=True,
            resume=True,
        )
