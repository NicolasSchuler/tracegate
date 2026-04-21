"""Live end-to-end integration tests for baseline adapters.

These tests run against a real LLM server and verify that each baseline
adapter can actually produce repair attempts.

- Default server: http://127.0.0.1:1234/v1
- Default model: qwen_qwen3.5-4b
- All baseline repos cloned in baselines/repos/
- All baseline envs provisioned in baselines/envs/

Test matrix:  5 baselines x {solvable, unsolvable} x {thinking_off, thinking_on} = 20 tests.

Run with:
    uv run pytest -m integration tests/test_baseline_live_e2e.py -v

Quick smoke (thinking off only):
    uv run pytest -m integration tests/test_baseline_live_e2e.py -v -k thinking_off

Override endpoint/model/profile dynamically:
    uv run pytest -m integration tests/test_baseline_live_e2e.py -v -k thinking_off \
      --live-openai-base-url http://127.0.0.1:1234/v1 \
      --live-openai-model mlx-community/Qwen3.5-4B-MLX-8bit \
      --live-openai-compat-profile auto
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from baselines.adapter_base import (
    BaselineCaseContext,
    BaselineConfig,
)

from . import _live_e2e_support as live_support

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_API_TIMEOUT_SEC = live_support.API_TIMEOUT_SEC
_PROCESS_TIMEOUT_SEC = live_support.PROCESS_TIMEOUT_SEC
_CASE_TIMEOUT_SEC = live_support.CASE_TIMEOUT_SEC
_SOLVABLE_BUGGY = live_support.SOLVABLE_BUGGY
_SOLVABLE_TEST = live_support.SOLVABLE_TEST
_UNSOLVABLE_BUGGY = live_support.UNSOLVABLE_BUGGY

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration
live_openai_base_url = live_support.live_openai_base_url
live_openai_model = live_support.live_openai_model
live_openai_compat_profile = live_support.live_openai_compat_profile
_skip_if_live_server_unreachable = live_support._skip_if_live_server_unreachable
solvable_case = live_support.solvable_case
unsolvable_case = live_support.unsolvable_case


def _env_python(baseline_name: str) -> Path:
    """Return the provisioned env Python for a baseline."""
    return _REPO_ROOT / "baselines" / "envs" / baseline_name / "bin" / "python"


def _repo_dir(baseline_name: str) -> Path:
    """Return the cloned repo directory for a baseline."""
    name_map = {
        "lambda_rlm": "lambda-RLM",
        "rlm": "rlm",
        "intervenor": "INTERVENOR",
        "ldb": "LLMDebugger",
        "tracecoder": "TraceCoder",
    }
    return _REPO_ROOT / "baselines" / "repos" / name_map[baseline_name]


def _skip_missing_env(baseline_name: str) -> None:
    """Skip test if the baseline's env or repo is not available."""
    env = _env_python(baseline_name)
    repo = _repo_dir(baseline_name)
    if not env.exists():
        pytest.skip(f"Env python not found: {env}")
    if not repo.exists():
        pytest.skip(f"Repo not found: {repo}")


# ---------------------------------------------------------------------------
# Shared config builder
# ---------------------------------------------------------------------------


def _shared_extra(*, compat_profile: str) -> dict[str, Any]:
    return {
        "baseline_openai_compat_profile": compat_profile,
        "baseline_openai_allow_reasoning_fallback": True,
        "baseline_openai_enable_thinking": False,
    }


def _make_config(
    baseline_name: str,
    *,
    model: str,
    api_base_url: str,
    compat_profile: str,
    extra: dict[str, Any] | None = None,
    max_attempts: int = 3,
    thinking_enabled: bool = False,
) -> BaselineConfig:
    merged = {**_shared_extra(compat_profile=compat_profile), **(extra or {})}
    merged["baseline_openai_enable_thinking"] = thinking_enabled
    return BaselineConfig(
        model=model,
        api_base_url=api_base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        max_attempts=max_attempts,
        api_timeout_sec=_API_TIMEOUT_SEC,
        process_timeout_sec=_PROCESS_TIMEOUT_SEC,
        timeout_sec=_CASE_TIMEOUT_SEC,
        repo_dir=_repo_dir(baseline_name),
        python_executable=_env_python(baseline_name),
        extra=merged,
    )


@pytest.fixture
def make_live_config(
    live_openai_base_url: str,
    live_openai_model: str,
    live_openai_compat_profile: str,
) -> Callable[..., BaselineConfig]:
    def _build(
        baseline_name: str,
        *,
        extra: dict[str, Any] | None = None,
        max_attempts: int = 3,
        thinking_enabled: bool = False,
    ) -> BaselineConfig:
        return _make_config(
            baseline_name,
            model=live_openai_model,
            api_base_url=live_openai_base_url,
            compat_profile=live_openai_compat_profile,
            extra=extra,
            max_attempts=max_attempts,
            thinking_enabled=thinking_enabled,
        )

    return _build


def _make_case_context(case_id: str) -> BaselineCaseContext:
    return BaselineCaseContext(
        case_id=case_id,
        python_args=("-m", "pytest", "-x", "-q", "test_buggy.py"),
        timeout_sec=30,
        description="E2E integration test case.",
    )


# ===========================================================================
# Intervenor E2E Tests
# ===========================================================================


@pytest.mark.parametrize("thinking_enabled", [False, True], ids=["thinking_off", "thinking_on"])
class TestIntervenorLiveE2E:
    """Live E2E tests for the Intervenor baseline."""

    @staticmethod
    def _make_adapter():
        from baselines.intervenor.adapter import IntervenorAdapter

        return IntervenorAdapter()

    def test_solvable_case(
        self,
        solvable_case: Path,
        tmp_path: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("intervenor")
        adapter = self._make_adapter()
        config = make_live_config(
            "intervenor",
            extra={
                "baseline_intervenor_teacher_max_tokens": 1024,
                "baseline_intervenor_learner_max_tokens": 2048,
            },
            thinking_enabled=thinking_enabled,
        )
        adapter.setup(config)
        ctx = _make_case_context("intervenor_solvable")
        case_input = adapter.convert_case(solvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_solved_result(
            result,
            baseline_name="intervenor",
            original_buggy=_SOLVABLE_BUGGY,
            test_code=_SOLVABLE_TEST,
            tmp_path=tmp_path,
        )

    def test_unsolvable_case(
        self,
        unsolvable_case: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("intervenor")
        adapter = self._make_adapter()
        config = make_live_config(
            "intervenor",
            extra={
                "baseline_intervenor_teacher_max_tokens": 1024,
                "baseline_intervenor_learner_max_tokens": 2048,
            },
            thinking_enabled=thinking_enabled,
        )
        adapter.setup(config)
        ctx = _make_case_context("intervenor_unsolvable")
        case_input = adapter.convert_case(unsolvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_unsolved_result(
            result,
            baseline_name="intervenor",
            original_buggy=_UNSOLVABLE_BUGGY,
        )


# ===========================================================================
# Lambda_RLM E2E Tests
# ===========================================================================


_LAMBDA_RLM_EXTRA: dict[str, Any] = {
    "baseline_lambda_context_window_chars": 50_000,
    "baseline_lambda_accuracy_target": 0.80,
    "baseline_lambda_max_tokens": 2048,
    # Small models often misclassify code repair as "summarization" (→ prose),
    # force QA so the repair question reaches the leaf nodes.
    "baseline_lambda_force_task_type": "qa",
}


@pytest.mark.parametrize("thinking_enabled", [False, True], ids=["thinking_off", "thinking_on"])
class TestLambdaRLMLiveE2E:
    """Live E2E tests for the Lambda_RLM baseline."""

    @staticmethod
    def _make_adapter():
        from baselines.lambda_rlm.adapter import LambdaRLMAdapter

        return LambdaRLMAdapter()

    def test_solvable_case(
        self,
        solvable_case: Path,
        tmp_path: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("lambda_rlm")
        adapter = self._make_adapter()
        config = make_live_config(
            "lambda_rlm",
            extra=_LAMBDA_RLM_EXTRA,
            thinking_enabled=thinking_enabled,
        )
        adapter.setup(config)
        ctx = _make_case_context("lambda_rlm_solvable")
        case_input = adapter.convert_case(solvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_solved_result(
            result,
            baseline_name="lambda_rlm",
            original_buggy=_SOLVABLE_BUGGY,
            test_code=_SOLVABLE_TEST,
            tmp_path=tmp_path,
        )

    def test_unsolvable_case(
        self,
        unsolvable_case: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("lambda_rlm")
        adapter = self._make_adapter()
        config = make_live_config(
            "lambda_rlm",
            extra=_LAMBDA_RLM_EXTRA,
            thinking_enabled=thinking_enabled,
        )
        adapter.setup(config)
        ctx = _make_case_context("lambda_rlm_unsolvable")
        case_input = adapter.convert_case(unsolvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_unsolved_result(
            result,
            baseline_name="lambda_rlm",
            original_buggy=_UNSOLVABLE_BUGGY,
        )


# ===========================================================================
# RLM E2E Tests
# ===========================================================================


@pytest.mark.parametrize("thinking_enabled", [False, True], ids=["thinking_off", "thinking_on"])
class TestRLMLiveE2E:
    """Live E2E tests for the RLM baseline.

    Note: RLM uses tool calling which small models may struggle with,
    especially when combined with thinking mode.
    """

    @staticmethod
    def _make_adapter():
        from baselines.rlm.adapter import RLMAdapter

        return RLMAdapter()

    def test_solvable_case(
        self,
        solvable_case: Path,
        tmp_path: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("rlm")
        adapter = self._make_adapter()
        config = make_live_config(
            "rlm",
            extra={
                "baseline_rlm_max_depth": 2,
                "baseline_rlm_max_iterations": 8,
                "baseline_rlm_max_tokens": 2048,
            },
            thinking_enabled=thinking_enabled,
        )
        adapter.setup(config)
        ctx = _make_case_context("rlm_solvable")
        case_input = adapter.convert_case(solvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_solved_result(
            result,
            baseline_name="rlm",
            original_buggy=_SOLVABLE_BUGGY,
            test_code=_SOLVABLE_TEST,
            tmp_path=tmp_path,
        )

    def test_unsolvable_case(
        self,
        unsolvable_case: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("rlm")
        adapter = self._make_adapter()
        config = make_live_config(
            "rlm",
            extra={
                "baseline_rlm_max_depth": 2,
                "baseline_rlm_max_iterations": 8,
                "baseline_rlm_max_tokens": 2048,
            },
            thinking_enabled=thinking_enabled,
        )
        adapter.setup(config)
        ctx = _make_case_context("rlm_unsolvable")
        case_input = adapter.convert_case(unsolvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_unsolved_result(
            result,
            baseline_name="rlm",
            original_buggy=_UNSOLVABLE_BUGGY,
        )


# ===========================================================================
# LDB E2E Tests
# ===========================================================================


@pytest.mark.parametrize("thinking_enabled", [False, True], ids=["thinking_off", "thinking_on"])
class TestLDBLiveE2E:
    """Live E2E tests for the LDB baseline."""

    @staticmethod
    def _make_adapter():
        from baselines.ldb.adapter import LDBAdapter

        return LDBAdapter()

    def test_solvable_case(
        self,
        solvable_case: Path,
        tmp_path: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("ldb")
        adapter = self._make_adapter()
        config = make_live_config("ldb", thinking_enabled=thinking_enabled)
        adapter.setup(config)
        ctx = _make_case_context("ldb_solvable")
        case_input = adapter.convert_case(solvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_solved_result(
            result,
            baseline_name="ldb",
            original_buggy=_SOLVABLE_BUGGY,
            test_code=_SOLVABLE_TEST,
            tmp_path=tmp_path,
        )

    def test_unsolvable_case(
        self,
        unsolvable_case: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("ldb")
        adapter = self._make_adapter()
        config = make_live_config("ldb", thinking_enabled=thinking_enabled)
        adapter.setup(config)
        ctx = _make_case_context("ldb_unsolvable")
        case_input = adapter.convert_case(unsolvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_unsolved_result(
            result,
            baseline_name="ldb",
            original_buggy=_UNSOLVABLE_BUGGY,
        )


# ===========================================================================
# TraceCoder E2E Tests
# ===========================================================================


@pytest.mark.parametrize("thinking_enabled", [False, True], ids=["thinking_off", "thinking_on"])
class TestTraceCoderLiveE2E:
    """Live E2E tests for the TraceCoder baseline."""

    @staticmethod
    def _make_adapter():
        from baselines.tracecoder.adapter import TraceCoderAdapter

        return TraceCoderAdapter()

    def test_solvable_case(
        self,
        solvable_case: Path,
        tmp_path: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("tracecoder")
        adapter = self._make_adapter()
        config = make_live_config("tracecoder", thinking_enabled=thinking_enabled)
        adapter.setup(config)
        ctx = _make_case_context("tracecoder_solvable")
        case_input = adapter.convert_case(solvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        if result.success:
            live_support.assert_baseline_solved_result(
                result,
                baseline_name="tracecoder",
                original_buggy=_SOLVABLE_BUGGY,
                test_code=_SOLVABLE_TEST,
                tmp_path=tmp_path,
            )
        else:
            live_support.assert_baseline_classified_failure(
                result,
                baseline_name="tracecoder",
                original_buggy=_SOLVABLE_BUGGY,
            )

    def test_unsolvable_case(
        self,
        unsolvable_case: Path,
        thinking_enabled: bool,
        make_live_config: Callable[..., BaselineConfig],
    ) -> None:
        _skip_missing_env("tracecoder")
        adapter = self._make_adapter()
        config = make_live_config("tracecoder", thinking_enabled=thinking_enabled)
        adapter.setup(config)
        ctx = _make_case_context("tracecoder_unsolvable")
        case_input = adapter.convert_case(unsolvable_case, case_context=ctx).to_dict()
        result = adapter.run_repair(case_input, config)
        live_support.assert_baseline_unsolved_result(
            result,
            baseline_name="tracecoder",
            original_buggy=_UNSOLVABLE_BUGGY,
        )
