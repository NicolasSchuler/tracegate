"""DAT-18: SWE-Bench generation records ``finish_reason`` per prediction.

The official predictions file keeps its three keys; truncation and the patcher error go to
``<condition>/generation_meta.jsonl``.
"""

from __future__ import annotations

import socket
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from evals import _json as json
from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from evals.swebench import generate as lite_generate
from evals.swebench.generation_meta import generation_meta_from_patch_meta
from evals.swebench.io import ensure_run_layout, prediction_path, read_predictions
from evals.swebench.prompt_policy import PromptInputs
from evals.swebench.types import CONDITION_BASELINE, SWEBENCH_OPENAI_MAX_TOKENS_DEFAULT
from evals.swebench_pro import generate as pro_generate

_IDS = ["inst-1", "inst-2", "inst-3"]
_DIFF = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b\n"


def _rows() -> dict[str, dict[str, Any]]:
    return {
        iid: {
            "instance_id": iid,
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": f"Fix issue in {iid}.",
            "repo_language": "Python",
            "requirements": "",
            "interface": "",
        }
        for iid in _IDS
    }


class _Patcher:
    """inst-2's answer is truncated; the others finish normally."""

    def _result(self, prompt: str) -> PatchResult:
        if "inst-2" in prompt:
            return PatchResult(
                diff=None, meta={"error": "truncated_response", "finish_reason": "length"}
            )
        return PatchResult(diff=_DIFF, meta={"finish_reason": "stop"})

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del context_dir, cwd
        return self._result(prompt)

    async def apropose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del context_dir, cwd
        return self._result(prompt)


@pytest.mark.parametrize("module", [lite_generate, pro_generate], ids=["lite", "pro"])
@pytest.mark.parametrize(
    ("jobs", "use_async"), [(1, False), (2, False), (1, True)], ids=["serial", "threads", "async"]
)
def test_dat18_generation_meta_records_finish_reason_per_prediction(
    tmp_path: Path, module: ModuleType, jobs: int, use_async: bool
) -> None:
    run_dir = tmp_path / "run"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])

    summary = module._generate_for_condition(
        run_dir=run_dir,
        condition=CONDITION_BASELINE,
        selected_ids=_IDS,
        rows_by_id=_rows(),
        patcher=cast(OpenAICompatPatcher, _Patcher()),
        model_label="stub::baseline",
        resume=False,
        prompt_inputs=PromptInputs(snapshot_dir=None),
        jobs=jobs,
        openai_async=use_async,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
    )

    meta_path = summary.predictions_path.with_name("generation_meta.jsonl")
    lines = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
    by_id = {line["instance_id"]: line for line in lines}
    assert set(by_id) == set(_IDS)
    assert by_id["inst-2"]["finish_reason"] == "length"
    assert by_id["inst-2"]["truncated"] is True
    assert by_id["inst-2"]["patcher_error"] == "truncated_response"
    assert by_id["inst-1"] == {
        "instance_id": "inst-1",
        "finish_reason": "stop",
        "truncated": False,
        "patcher_error": "",
        "request_error": "",
    }
    # The official predictions file keeps its three keys.
    assert set(read_predictions(prediction_path(run_dir, CONDITION_BASELINE))[0].as_json()) == {
        "instance_id",
        "model_patch",
        "model_name_or_path",
    }


def test_dat18_generation_meta_uses_the_fallback_call_finish_reason() -> None:
    meta = {
        "error": "no_json_payload",
        "finish_reason": "stop",
        "fallback_meta": {"error": "truncated_response", "finish_reason": "length"},
    }
    summary = generation_meta_from_patch_meta(meta)
    assert summary["finish_reason"] == "length"
    assert summary["truncated"] is True


def test_dat18_default_completion_cap_is_raised() -> None:
    assert SWEBENCH_OPENAI_MAX_TOKENS_DEFAULT == 4096


def _dead_local_endpoint() -> str:
    """A bound-then-released localhost port, so every request fails to connect."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    return f"http://127.0.0.1:{port}"


@pytest.mark.parametrize("module", [lite_generate, pro_generate], ids=["lite", "pro"])
def test_dat08_returned_transport_failure_is_not_persisted_as_a_prediction(
    tmp_path: Path, module: ModuleType
) -> None:
    """DAT-08: the real patcher *returns* transport outages instead of raising.

    Classifying only raised exceptions persisted the outage as an empty prediction,
    which ``--resume`` then skipped forever.
    """
    run_dir = tmp_path / "run"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])
    patcher = OpenAICompatPatcher(
        base_url=_dead_local_endpoint(),
        model="stub-model",
        api_key="local",
        http2=False,
        response_mode="diff",
        timeout_sec=1.0,
        transport_max_retries=0,
        max_propose_sec=5.0,
    )

    with pytest.raises(RuntimeError, match="all patch requests failed"):
        module._generate_for_condition(
            run_dir=run_dir,
            condition=CONDITION_BASELINE,
            selected_ids=_IDS[:1],
            rows_by_id=_rows(),
            patcher=patcher,
            model_label="stub::baseline",
            resume=False,
            prompt_inputs=PromptInputs(snapshot_dir=None),
            jobs=1,
            openai_async=False,
            max_request_errors=25,
            max_request_error_rate=0.5,
            min_requests_for_error_rate_check=10,
        )

    predictions_path = prediction_path(run_dir, CONDITION_BASELINE)
    assert read_predictions(predictions_path) == []
    sidecar = predictions_path.with_name("generation_meta.jsonl")
    entries = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert [entry["instance_id"] for entry in entries] == _IDS[:1]
    assert "patcher_transport" in entries[0]["request_error"]
