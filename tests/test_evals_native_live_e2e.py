"""Live end-to-end integration tests for native eval conditions.

These tests run the real native repair path against a live OpenAI-compatible
server. The native patcher is exercised in both explicit thinking modes so the
OpenAI-compat request-shaping and reasoning-fallback path stay covered end to
end. For endpoints that support explicit thinking control, prefer running with
`--live-openai-compat-profile qwen_openai_raw` or `qwen_dashscope`.

Test matrix: 2 conditions x {thinking_off, thinking_on} x {solvable, unsolvable} = 8 tests.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from evals import run_eval
from evals.eval_types import Condition
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher

from . import _live_e2e_support as live_support

pytestmark = pytest.mark.integration

live_openai_base_url = live_support.live_openai_base_url
live_openai_model = live_support.live_openai_model
live_openai_compat_profile = live_support.live_openai_compat_profile
_skip_if_live_server_unreachable = live_support._skip_if_live_server_unreachable
solvable_case = live_support.solvable_case
unsolvable_case = live_support.unsolvable_case

_MODEL_TIMEOUT_SEC = 600.0


def _make_case_spec(case_id: str, case_dir: Path) -> run_eval.CaseSpec:
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-x", "-q", "test_buggy.py"],
        timeout_sec=30,
    )


def _make_live_patcher(
    *,
    base_url: str,
    model: str,
    compat_profile: str,
    thinking_enabled: bool,
) -> OpenAICompatPatcher:
    return OpenAICompatPatcher(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        timeout_sec=_MODEL_TIMEOUT_SEC,
        max_propose_sec=_MODEL_TIMEOUT_SEC,
        temperature=0.0,
        compat_profile=compat_profile,
        allow_reasoning_fallback=thinking_enabled,
        enable_thinking=thinking_enabled,
        response_mode="edits",
        force_patch_attempt=False,
    )


def _attempt_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if int(record.get("attempt", 0)) >= 1]


def _first_attempt_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = _attempt_records(records)
    if not attempts:
        pytest.fail("No attempt record found")
    metadata = attempts[0].get("attempt_metadata")
    if not isinstance(metadata, dict):
        pytest.fail("First attempt metadata missing")
    return metadata


def _first_patch_meta(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = _attempt_records(records)
    if not attempts:
        pytest.fail("No attempt record found")
    patch_meta = attempts[0].get("patch_meta")
    if not isinstance(patch_meta, dict):
        pytest.fail("First attempt patch metadata missing")
    return patch_meta


def _first_prompt_text(records: list[dict[str, Any]]) -> str:
    for record in _attempt_records(records):
        attempt_metadata = record.get("attempt_metadata")
        if not isinstance(attempt_metadata, dict):
            continue
        prompt_path = attempt_metadata.get("prompt_path")
        if isinstance(prompt_path, str) and prompt_path:
            return Path(prompt_path).read_text(encoding="utf-8")
    pytest.fail("No prompt artifact found in attempt metadata")


def _assert_condition_prompt(condition: Condition, prompt_text: str) -> None:
    cot_phrase = "Write out your reasoning"
    if condition is Condition.COT_REPAIR:
        assert "step by step" in prompt_text
        assert cot_phrase in prompt_text
        return
    assert cot_phrase not in prompt_text


def _run_live_condition(
    *,
    tmp_path: Path,
    case_dir: Path,
    condition: Condition,
    live_openai_base_url: str,
    live_openai_model: str,
    live_openai_compat_profile: str,
    thinking_enabled: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    case = _make_case_spec(f"{condition.value}_{case_dir.name}", case_dir)
    thinking_tag = "thinking_on" if thinking_enabled else "thinking_off"
    workdir = tmp_path / "workdir" / condition.value / thinking_tag
    context_dir = tmp_path / "contexts" / condition.value / thinking_tag
    workdir.parent.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(case.path, workdir)
    previous_execution = run_eval.set_active_execution_config(
        run_eval.ExecutionConfig(executor="local")
    )
    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=condition,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=_make_live_patcher(
                base_url=live_openai_base_url,
                model=live_openai_model,
                compat_profile=live_openai_compat_profile,
                thinking_enabled=thinking_enabled,
            ),
            k=2,
            output_format="json_compact",
            run_id=f"live-{condition.value}-{thinking_tag}-{case_dir.name}",
            include_files_context=True,
            force_patch_attempt=True,
            retry_prompt_mode="full",
            openai_response_mode_autoswitch="retry_to_diff",
            record_prompt="full",
            record_evidence="none",
        )
        return workdir, records
    finally:
        run_eval.set_active_execution_config(previous_execution)


@pytest.mark.parametrize(
    "condition",
    [Condition.BASE_REPAIR, Condition.COT_REPAIR],
    ids=["base_repair", "cot_repair"],
)
@pytest.mark.parametrize("thinking_enabled", [False, True], ids=["thinking_off", "thinking_on"])
def test_live_native_condition_solves_simplest_case(
    tmp_path: Path,
    solvable_case: Path,
    condition: Condition,
    thinking_enabled: bool,
    live_openai_base_url: str,
    live_openai_model: str,
    live_openai_compat_profile: str,
) -> None:
    workdir, records = _run_live_condition(
        tmp_path=tmp_path,
        case_dir=solvable_case,
        condition=condition,
        live_openai_base_url=live_openai_base_url,
        live_openai_model=live_openai_model,
        live_openai_compat_profile=live_openai_compat_profile,
        thinking_enabled=thinking_enabled,
    )

    attempts = _attempt_records(records)
    assert attempts, "Expected at least one attempt record"
    assert all(record.get("condition") == condition.value for record in attempts)
    assert any(bool(record.get("success")) for record in attempts), records

    first_patch_meta = _first_patch_meta(records)
    assert first_patch_meta.get("compat_profile_requested") == live_openai_compat_profile
    assert first_patch_meta.get("enable_thinking") is thinking_enabled
    assert first_patch_meta.get("allow_reasoning_fallback") is thinking_enabled

    prompt_text = _first_prompt_text(records)
    _assert_condition_prompt(condition, prompt_text)
    first_attempt_metadata = _first_attempt_metadata(records)
    if condition is Condition.COT_REPAIR:
        assert first_attempt_metadata.get("plan_patch_invoked") is True
        plan_prompt_path = first_attempt_metadata.get("plan_patch_prompt_path")
        plan_response_path = first_attempt_metadata.get("plan_patch_response_path")
        assert isinstance(plan_prompt_path, str) and Path(plan_prompt_path).exists()
        assert isinstance(plan_response_path, str) and Path(plan_response_path).exists()
        assert Path(plan_response_path).read_text(encoding="utf-8").strip()
    else:
        assert first_attempt_metadata.get("plan_patch_invoked") is not True

    rerun = live_support.run_case_pytest(workdir, timeout_sec=30)
    assert rerun.returncode == 0, rerun.stdout + "\n" + rerun.stderr


@pytest.mark.parametrize(
    "condition",
    [Condition.BASE_REPAIR, Condition.COT_REPAIR],
    ids=["base_repair", "cot_repair"],
)
@pytest.mark.parametrize("thinking_enabled", [False, True], ids=["thinking_off", "thinking_on"])
def test_live_native_condition_handles_unsolvable_case_coherently(
    tmp_path: Path,
    unsolvable_case: Path,
    condition: Condition,
    thinking_enabled: bool,
    live_openai_base_url: str,
    live_openai_model: str,
    live_openai_compat_profile: str,
) -> None:
    workdir, records = _run_live_condition(
        tmp_path=tmp_path,
        case_dir=unsolvable_case,
        condition=condition,
        live_openai_base_url=live_openai_base_url,
        live_openai_model=live_openai_model,
        live_openai_compat_profile=live_openai_compat_profile,
        thinking_enabled=thinking_enabled,
    )

    attempts = _attempt_records(records)
    assert attempts, "Expected at least one attempt record"
    assert all(record.get("condition") == condition.value for record in attempts)

    first_patch_meta = _first_patch_meta(records)
    assert first_patch_meta.get("compat_profile_requested") == live_openai_compat_profile
    assert first_patch_meta.get("enable_thinking") is thinking_enabled
    assert first_patch_meta.get("allow_reasoning_fallback") is thinking_enabled

    prompt_text = _first_prompt_text(records)
    _assert_condition_prompt(condition, prompt_text)

    live_support.assert_python_file_parses(workdir / "buggy.py")
