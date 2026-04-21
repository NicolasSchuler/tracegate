"""Provisioned runtime checks against pinned external baseline repos and envs."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import pytest

from baselines.adapter_base import BaselineCaseContext, BaselineConfig
from baselines.intervenor.adapter import IntervenorAdapter
from baselines.ldb.adapter import LDBAdapter, _has_llmdebug_dataset_support
from baselines.pins import get_pinned_repo_commit
from baselines.rlm.adapter import RLMAdapter
from baselines.tracecoder.adapter import TraceCoderAdapter, _has_llmdebug_support
from baselines.tracecoder.case_converter import convert_case
from evals import run_eval
from evals.patchers.baseline_patcher import build_baseline_patcher
from tests._openai_stub_server import OpenAIStubServer

pytestmark = [pytest.mark.integration, pytest.mark.baseline_runtime]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LDB_REPO = _REPO_ROOT / "baselines" / "repos" / "LLMDebugger"
_INTERVENOR_REPO = _REPO_ROOT / "baselines" / "repos" / "INTERVENOR"
_TRACECODER_REPO = _REPO_ROOT / "baselines" / "repos" / "TraceCoder"
_LAMBDA_RLM_REPO = _REPO_ROOT / "baselines" / "repos" / "lambda-RLM"
_RLM_REPO = _REPO_ROOT / "baselines" / "repos" / "rlm"
_ENV_ROOT = _REPO_ROOT / "baselines" / "envs"
_ENV_MANIFEST = _ENV_ROOT / "runtime_manifest.json"
_LDB_ENV_PYTHON = _ENV_ROOT / "ldb" / "bin" / "python"
_INTERVENOR_ENV_PYTHON = _ENV_ROOT / "intervenor" / "bin" / "python"
_TRACECODER_ENV_PYTHON = _ENV_ROOT / "tracecoder" / "bin" / "python"
_LAMBDA_RLM_ENV_PYTHON = _ENV_ROOT / "lambda_rlm" / "bin" / "python"
_RLM_ENV_PYTHON = _ENV_ROOT / "rlm" / "bin" / "python"
_LDB_COMMIT = get_pinned_repo_commit("ldb")
_INTERVENOR_COMMIT = get_pinned_repo_commit("intervenor")
_TRACECODER_COMMIT = get_pinned_repo_commit("tracecoder")
_LAMBDA_RLM_COMMIT = get_pinned_repo_commit("lambda_rlm")
_RLM_COMMIT = get_pinned_repo_commit("rlm")


def _git_stdout(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_tracecoder_case(case_dir: Path, *, kind: str) -> tuple[str, str, str]:
    case_dir.mkdir()
    if kind == "subprocess":
        (case_dir / "buggy.py").write_text(
            "from helper import run_value\n\n\ndef compute():\n    return run_value() + 1\n",
            encoding="utf-8",
        )
        (case_dir / "helper.py").write_text(
            "import subprocess\n"
            "import sys\n"
            "from pathlib import Path\n\n\n"
            "def run_value():\n"
            "    worker = Path(__file__).with_name('worker.py')\n"
            "    completed = subprocess.run(\n"
            "        [sys.executable, str(worker)],\n"
            "        capture_output=True,\n"
            "        text=True,\n"
            "        check=True,\n"
            "    )\n"
            "    return int(completed.stdout.strip())\n",
            encoding="utf-8",
        )
        (case_dir / "worker.py").write_text("print(3)\n", encoding="utf-8")
        (case_dir / "test_buggy.py").write_text(
            "from buggy import compute\n\n\ndef test_compute():\n    assert compute() == 3\n",
            encoding="utf-8",
        )
        return (
            "from helper import run_value\n\n\ndef compute():\n    return run_value()\n",
            "from helper import run_value\n\n\n"
            "def compute():\n"
            "    print(f'trace={run_value()}')\n"
            "    return run_value() + 1\n",
            "Return the worker output without modification.",
        )

    if kind == "multifile":
        (case_dir / "buggy.py").write_text(
            "from helper import scale\n\n\n"
            "def area(width, height):\n"
            "    return scale(width * height) - 1\n",
            encoding="utf-8",
        )
        (case_dir / "helper.py").write_text(
            "def scale(value):\n    return value\n",
            encoding="utf-8",
        )
        (case_dir / "test_buggy.py").write_text(
            "from buggy import area\n\n\ndef test_area():\n    assert area(2, 3) == 6\n",
            encoding="utf-8",
        )
        return (
            "from helper import scale\n\n\n"
            "def area(width, height):\n"
            "    return scale(width * height)\n",
            "from helper import scale\n\n\n"
            "def area(width, height):\n"
            "    current = scale(width * height)\n"
            "    print(f'area={current}')\n"
            "    return current - 1\n",
            "Return the exact scaled area.",
        )

    raise AssertionError(f"unsupported TraceCoder test case kind: {kind}")


def _write_tracecoder_fake_responses(
    path: Path,
    *,
    instrumented_code: str,
    repaired_code: str,
    use_input_output_tokens: bool = False,
) -> None:
    prompt_key = "input_tokens" if use_input_output_tokens else "prompt_tokens"
    completion_key = "output_tokens" if use_input_output_tokens else "completion_tokens"
    payload = [
        {
            "text": f"```python\n{instrumented_code}```",
            prompt_key: 11,
            completion_key: 5,
        },
        {
            "text": (
                "REPAIR_PLAN_START\n"
                "```python\n"
                "1. Remove the incorrect -/+ 1 offset from the return path.\n"
                "```\n"
                "REPAIR_PLAN_END\n\n"
                "Instrumentation_Suggestions_START\n"
                "```python\n"
                "Print the computed intermediate value before returning.\n"
                "```\n"
                "Instrumentation_Suggestions_END"
            ),
            prompt_key: 13,
            completion_key: 7,
        },
        {
            "text": f"CODE:\n```python\n{repaired_code}```\nEND_CODE",
            prompt_key: 17,
            completion_key: 9,
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _chat_json_response(content: str) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _write_single_file_baseline_case(case_dir: Path) -> tuple[str, str]:
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text(
        "from helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text(
        "def offset(value):\n    return value * 2\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import compute\n\n\ndef test_compute():\n    assert compute(3) == 6\n",
        encoding="utf-8",
    )
    return (
        "from helper import offset\n\n\ndef compute(value):\n    return offset(value)\n",
        "Fix the arithmetic offset in buggy.py only.",
    )


def _write_ldb_compatible_case(case_dir: Path) -> tuple[str, str]:
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return (
        "def add(a, b):\n    return a + b\n",
        "Fix the add function in buggy.py only.",
    )


def _write_baseline_case_context(
    context_dir: Path,
    *,
    case_id: str,
    description: str,
) -> None:
    context_dir.mkdir()
    (context_dir / "baseline_case_context.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "python_args": ["-m", "pytest", "-q", "test_buggy.py"],
                "timeout_sec": 10,
                "description": description,
                "category": "logic_error",
                "difficulty": "easy",
                "snapshot_dependency": "none",
            }
        ),
        encoding="utf-8",
    )


def _make_baseline_case_spec(
    *,
    case_id: str,
    case_dir: Path,
    description: str,
) -> run_eval.CaseSpec:
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="logic_error",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q", "test_buggy.py"],
        timeout_sec=10,
        description=description,
    )


@pytest.mark.skipif(not _LAMBDA_RLM_REPO.is_dir(), reason="real lambda-RLM repo not provisioned")
def test_real_lambda_rlm_repo_is_pinned_and_layout_ready() -> None:
    assert _git_stdout(_LAMBDA_RLM_REPO, "rev-parse", "HEAD") == _LAMBDA_RLM_COMMIT
    assert (_LAMBDA_RLM_REPO / "pyproject.toml").is_file()
    assert (_LAMBDA_RLM_REPO / "rlm" / "__init__.py").is_file()


@pytest.mark.skipif(not _RLM_REPO.is_dir(), reason="real rlm repo not provisioned")
def test_real_rlm_repo_is_pinned_and_layout_ready() -> None:
    assert _git_stdout(_RLM_REPO, "rev-parse", "HEAD") == _RLM_COMMIT
    assert (_RLM_REPO / "pyproject.toml").is_file()
    assert (_RLM_REPO / "rlm" / "__init__.py").is_file()

    adapter = RLMAdapter()
    adapter.setup(BaselineConfig(repo_dir=_RLM_REPO))
    assert adapter.repo_commit == _RLM_COMMIT


@pytest.mark.skipif(not _LDB_REPO.is_dir(), reason="real LDB repo not provisioned")
def test_real_ldb_repo_is_pinned_patched_and_setup_ready() -> None:
    assert _git_stdout(_LDB_REPO, "rev-parse", "HEAD") == _LDB_COMMIT
    assert _has_llmdebug_dataset_support(_LDB_REPO) is True

    adapter = LDBAdapter()
    adapter.setup(BaselineConfig(repo_dir=_LDB_REPO))
    assert adapter.repo_commit == _LDB_COMMIT


@pytest.mark.skipif(not _INTERVENOR_REPO.is_dir(), reason="real INTERVENOR repo not provisioned")
def test_real_intervenor_repo_is_pinned_and_setup_ready() -> None:
    assert _git_stdout(_INTERVENOR_REPO, "rev-parse", "HEAD") == _INTERVENOR_COMMIT

    adapter = IntervenorAdapter()
    adapter.setup(BaselineConfig(repo_dir=_INTERVENOR_REPO))
    assert adapter.repo_commit == _INTERVENOR_COMMIT
    assert "%%%buggy_code%%%" in adapter._teacher_prompt_template
    assert "%%%repair_method%%%" in adapter._learner_prompt_template


@pytest.mark.skipif(
    not (
        _LDB_REPO.is_dir()
        and _INTERVENOR_REPO.is_dir()
        and _TRACECODER_REPO.is_dir()
        and _LAMBDA_RLM_REPO.is_dir()
        and _RLM_REPO.is_dir()
    ),
    reason="real baseline repos not provisioned",
)
def test_build_baseline_patchers_against_real_repos() -> None:
    ldb = build_baseline_patcher(
        baseline_name="ldb",
        repo_dir=_LDB_REPO,
        model="test-model",
        max_attempts=2,
    )
    intervenor = build_baseline_patcher(
        baseline_name="intervenor",
        repo_dir=_INTERVENOR_REPO,
        model="test-model",
        max_attempts=2,
    )
    tracecoder = build_baseline_patcher(
        baseline_name="tracecoder",
        repo_dir=_TRACECODER_REPO,
        model="test-model",
        max_attempts=2,
    )
    lambda_rlm = build_baseline_patcher(
        baseline_name="lambda_rlm",
        repo_dir=_LAMBDA_RLM_REPO,
        model="test-model",
        max_attempts=2,
    )
    rlm = build_baseline_patcher(
        baseline_name="rlm",
        repo_dir=_RLM_REPO,
        model="test-model",
        max_attempts=2,
    )

    assert ldb.name == "ldb"
    assert intervenor.name == "intervenor"
    assert tracecoder.name == "tracecoder"
    assert lambda_rlm.name == "lambda_rlm"
    assert rlm.name == "rlm"


@pytest.mark.skipif(not _TRACECODER_REPO.is_dir(), reason="real TraceCoder repo not provisioned")
def test_real_tracecoder_repo_is_pinned_patched_and_setup_ready() -> None:
    assert _git_stdout(_TRACECODER_REPO, "rev-parse", "HEAD") == _TRACECODER_COMMIT
    assert _has_llmdebug_support(_TRACECODER_REPO) is True

    adapter = TraceCoderAdapter()
    adapter.setup(BaselineConfig(repo_dir=_TRACECODER_REPO))
    assert adapter.repo_commit == _TRACECODER_COMMIT


@pytest.mark.skipif(
    not (_TRACECODER_REPO.is_dir() and _TRACECODER_ENV_PYTHON.is_file()),
    reason="real TraceCoder repo/env not provisioned",
)
@pytest.mark.parametrize("case_kind", ["subprocess", "multifile"])
def test_real_tracecoder_cli_runs_mocked_llmdebug_cases(
    tmp_path: Path,
    case_kind: str,
) -> None:
    case_dir = tmp_path / case_kind
    repaired_code, instrumented_code, description = _write_tracecoder_case(case_dir, kind=case_kind)
    dataset_path = tmp_path / f"{case_kind}.jsonl"
    output_csv = tmp_path / f"{case_kind}.csv"
    responses_path = tmp_path / f"{case_kind}_responses.json"
    _write_tracecoder_fake_responses(
        responses_path,
        instrumented_code=instrumented_code,
        repaired_code=repaired_code,
    )

    row = convert_case(
        case_dir,
        case_context=BaselineCaseContext(
            case_id=case_dir.name,
            python_args=("-m", "pytest", "-q", "test_buggy.py"),
            timeout_sec=10,
            description=description,
            category="logic_error",
            difficulty="easy",
            snapshot_dependency="none",
        ),
    )
    dataset_row = row.to_dict() if hasattr(row, "to_dict") else row
    dataset_path.write_text(json.dumps(dataset_row) + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_BASE_URL": "http://127.0.0.1:9999",
        "TRACECODER_FAKE_RESPONSES_PATH": str(responses_path),
        "PYTHONPATH": os.pathsep.join(
            [
                str(_REPO_ROOT),
                *([str(os.environ.get("PYTHONPATH", ""))] if os.environ.get("PYTHONPATH") else []),
            ]
        ),
    }
    result = subprocess.run(
        [
            str(_TRACECODER_ENV_PYTHON),
            str(_TRACECODER_REPO / "trace_learn_coder.py"),
            "-d",
            "llmdebug",
            "--dataset-path",
            str(dataset_path),
            "-m",
            "test-model",
            "--max-problems",
            "1",
            "--max-debug-attempts",
            "2",
            "--timeout",
            "10",
            "-o",
            str(output_csv),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_TRACECODER_REPO),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert output_csv.is_file()

    with output_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    output_row = rows[0]
    assert output_row["task_id"] == case_dir.name
    assert output_row["direct_gen_passed"] == "False"
    assert output_row["final_passed"] == "True"
    assert output_row["final_code"].strip() == repaired_code.strip()
    assert output_row["stopped_reason"] == "Debug cycle completed"

    direct_eval = json.loads(output_row["direct_gen_eval"])
    final_eval = json.loads(output_row["final_eval"])
    debug_session_log = json.loads(output_row["debug_session_log"])
    assert direct_eval["passed"] is False
    assert final_eval["passed"] is True
    assert len(debug_session_log) == 1
    assert int(output_row["total_prompt_tokens"]) == 41
    assert int(output_row["total_completion_tokens"]) == 21
    assert int(output_row["total_llm_calls"]) == 3
    assert float(output_row["total_llm_duration_sec"]) >= 0.0
    assert float(output_row["total_eval_duration_sec"]) >= 0.0
    assert float(output_row["processing_time_seconds"]) >= float(
        output_row["total_llm_duration_sec"]
    )


@pytest.mark.skipif(
    not (_TRACECODER_REPO.is_dir() and _TRACECODER_ENV_PYTHON.is_file()),
    reason="real TraceCoder repo/env not provisioned",
)
def test_tracecoder_patcher_propagates_mocked_cli_metadata(tmp_path: Path) -> None:
    case_dir = tmp_path / "subprocess_case"
    repaired_code, instrumented_code, description = _write_tracecoder_case(
        case_dir, kind="subprocess"
    )
    responses_path = tmp_path / "responses.json"
    _write_tracecoder_fake_responses(
        responses_path,
        instrumented_code=instrumented_code,
        repaired_code=repaired_code,
    )
    context_dir = tmp_path / "context"
    _write_baseline_case_context(
        context_dir,
        case_id=case_dir.name,
        description=description,
    )

    patcher = build_baseline_patcher(
        baseline_name="tracecoder",
        repo_dir=_TRACECODER_REPO,
        python_executable=_TRACECODER_ENV_PYTHON,
        model="test-model",
        max_attempts=2,
        timeout_sec=30,
        extra={"tracecoder_fake_responses_path": str(responses_path)},
    )
    result = patcher.propose_patch(prompt="", context_dir=context_dir, cwd=case_dir)

    assert patcher.name == "tracecoder"
    assert result.diff is not None
    assert "-    return run_value() + 1" in result.diff
    assert "+    return run_value()" in result.diff
    assert result.meta["baseline"] == "tracecoder"
    assert result.meta["baseline_success"] is True
    assert result.meta["iterations_used"] == 1
    assert result.meta["llm_call_count"] == 3

    usage = result.meta["usage"]
    assert usage == {"prompt_tokens": 41, "completion_tokens": 21, "total_tokens": 62}
    assert int(result.meta["prompt_chars_total"]) > 0
    assert int(result.meta["completion_chars_total"]) > 0
    assert int(result.meta["chars_total"]) == int(result.meta["prompt_chars_total"]) + int(
        result.meta["completion_chars_total"]
    )
    baseline_metrics = result.meta["baseline_metrics"]
    assert baseline_metrics["llm_call_count"] == 3
    assert baseline_metrics["llm_duration_sec"] >= 0.0
    assert baseline_metrics["eval_duration_sec"] >= 0.0
    assert baseline_metrics["internal_total_duration_sec"] >= baseline_metrics["llm_duration_sec"]
    assert baseline_metrics["orchestration_duration_sec"] >= 0.0
    assert baseline_metrics["usage_available"] is True
    assert baseline_metrics["usage_source"] == "provider_reported"
    assert int(baseline_metrics["prompt_chars_total"]) > 0
    assert int(baseline_metrics["completion_chars_total"]) > 0
    assert int(baseline_metrics["chars_total"]) == int(
        baseline_metrics["prompt_chars_total"]
    ) + int(baseline_metrics["completion_chars_total"])

    baseline_meta = result.meta["baseline_meta"]
    assert baseline_meta["stopped_reason"] == "Debug cycle completed"
    assert baseline_meta["method_instrumentation"] is True
    assert baseline_meta["method_two_step_repair"] is True
    assert baseline_meta["direct_gen_eval"]["passed"] is False
    assert baseline_meta["final_eval"]["passed"] is True
    assert len(baseline_meta["debug_session_log"]) == 1


@pytest.mark.skipif(
    not (_TRACECODER_REPO.is_dir() and _TRACECODER_ENV_PYTHON.is_file()),
    reason="real TraceCoder repo/env not provisioned",
)
def test_tracecoder_patcher_accepts_input_output_usage_fields(tmp_path: Path) -> None:
    case_dir = tmp_path / "subprocess_case_io_usage"
    repaired_code, instrumented_code, description = _write_tracecoder_case(
        case_dir, kind="subprocess"
    )
    responses_path = tmp_path / "responses_io_usage.json"
    _write_tracecoder_fake_responses(
        responses_path,
        instrumented_code=instrumented_code,
        repaired_code=repaired_code,
        use_input_output_tokens=True,
    )
    context_dir = tmp_path / "context"
    _write_baseline_case_context(
        context_dir,
        case_id=case_dir.name,
        description=description,
    )

    patcher = build_baseline_patcher(
        baseline_name="tracecoder",
        repo_dir=_TRACECODER_REPO,
        python_executable=_TRACECODER_ENV_PYTHON,
        model="test-model",
        max_attempts=2,
        timeout_sec=30,
        extra={"tracecoder_fake_responses_path": str(responses_path)},
    )
    result = patcher.propose_patch(prompt="", context_dir=context_dir, cwd=case_dir)

    assert result.meta["usage"] == {
        "prompt_tokens": 41,
        "completion_tokens": 21,
        "total_tokens": 62,
    }


@pytest.mark.skipif(
    not (_TRACECODER_REPO.is_dir() and _TRACECODER_ENV_PYTHON.is_file()),
    reason="real TraceCoder repo/env not provisioned",
)
def test_tracecoder_patcher_reaches_openai_compat_stub_server(tmp_path: Path) -> None:
    case_dir = tmp_path / "tracecoder_stub_case"
    repaired_code, instrumented_code, description = _write_tracecoder_case(
        case_dir, kind="subprocess"
    )
    context_dir = tmp_path / "context_tracecoder_stub"
    _write_baseline_case_context(
        context_dir,
        case_id=case_dir.name,
        description=description,
    )

    with OpenAIStubServer() as server:
        for content in (
            f"```python\n{instrumented_code}```",
            (
                "REPAIR_PLAN_START\n"
                "```python\n"
                "1. Remove the incorrect + 1 offset from the return path.\n"
                "```\n"
                "REPAIR_PLAN_END\n\n"
                "Instrumentation_Suggestions_START\n"
                "```python\n"
                "Print the worker output before returning.\n"
                "```\n"
                "Instrumentation_Suggestions_END"
            ),
            f"CODE:\n```python\n{repaired_code}```\nEND_CODE",
        ):
            server.enqueue_json(
                method="POST",
                path="/v1/chat/completions",
                body=_chat_json_response(content),
            )

        patcher = build_baseline_patcher(
            baseline_name="tracecoder",
            repo_dir=_TRACECODER_REPO,
            python_executable=_TRACECODER_ENV_PYTHON,
            model="test-model",
            api_base_url=server.base_url_v1,
            api_key="sk-test",
            max_attempts=2,
            timeout_sec=30,
        )
        result = patcher.propose_patch(prompt="", context_dir=context_dir, cwd=case_dir)

    assert result.diff is not None
    assert "-    return run_value() + 1" in result.diff
    assert "+    return run_value()" in result.diff
    assert result.meta["baseline_success"] is True
    assert result.meta["llm_call_count"] == 3
    requests = server.requests_for(method="POST", path="/v1/chat/completions")
    assert len(requests) == 3
    assert {req["headers"]["authorization"] for req in requests} == {"Bearer sk-test"}
    for req in requests:
        payload = req["json"]
        assert isinstance(payload, dict)
        assert payload["model"] == "test-model"


@pytest.mark.skipif(
    not (_LDB_REPO.is_dir() and _LDB_ENV_PYTHON.is_file()),
    reason="real LDB repo/env not provisioned",
)
def test_ldb_patcher_reaches_openai_compat_stub_server(tmp_path: Path) -> None:
    case_dir = tmp_path / "ldb_stub_case"
    repaired_code, description = _write_ldb_compatible_case(case_dir)
    context_dir = tmp_path / "context_ldb_stub"
    _write_baseline_case_context(
        context_dir,
        case_id=case_dir.name,
        description=description,
    )

    with OpenAIStubServer() as server:
        for _ in range(2):
            server.enqueue_json(
                method="POST",
                path="/v1/chat/completions",
                body=_chat_json_response(repaired_code),
            )

        patcher = build_baseline_patcher(
            baseline_name="ldb",
            repo_dir=_LDB_REPO,
            python_executable=_LDB_ENV_PYTHON,
            model="test-model",
            api_base_url=server.base_url_v1,
            api_key="sk-test",
            max_attempts=2,
            api_timeout_sec=15.0,
            process_timeout_sec=30.0,
            timeout_sec=30.0,
        )
        result = patcher.propose_patch(prompt="", context_dir=context_dir, cwd=case_dir)

    assert result.diff is not None
    assert "-    return a - b" in result.diff
    assert "+    return a + b" in result.diff
    assert result.meta["baseline_success"] is True
    assert result.meta["llm_call_count"] == 2
    baseline_metrics = result.meta["baseline_metrics"]
    assert baseline_metrics["llm_call_count"] == 2
    assert baseline_metrics["usage_available"] is True
    requests = server.requests_for(method="POST", path="/v1/chat/completions")
    assert len(requests) == 2
    assert {req["headers"]["authorization"] for req in requests} == {"Bearer sk-test"}
    for req in requests:
        payload = req["json"]
        assert isinstance(payload, dict)
        assert payload["model"] == "test-model"


@pytest.mark.skipif(
    not (_INTERVENOR_REPO.is_dir() and _INTERVENOR_ENV_PYTHON.is_file()),
    reason="real INTERVENOR repo/env not provisioned",
)
def test_intervenor_patcher_reaches_openai_compat_stub_server(tmp_path: Path) -> None:
    case_dir = tmp_path / "intervenor_stub_case"
    repaired_code, description = _write_single_file_baseline_case(case_dir)
    context_dir = tmp_path / "context_intervenor_stub"
    _write_baseline_case_context(
        context_dir,
        case_id=case_dir.name,
        description=description,
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response("Remove the extra + 1 offset in compute."),
        )
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(repaired_code),
        )

        patcher = build_baseline_patcher(
            baseline_name="intervenor",
            repo_dir=_INTERVENOR_REPO,
            python_executable=_INTERVENOR_ENV_PYTHON,
            model="test-model",
            api_base_url=server.base_url_v1,
            api_key="sk-test",
            max_attempts=1,
            api_timeout_sec=15.0,
            process_timeout_sec=30.0,
            timeout_sec=30.0,
        )
        result = patcher.propose_patch(prompt="", context_dir=context_dir, cwd=case_dir)

    assert result.diff is not None
    assert "-    return offset(value) + 1" in result.diff
    assert "+    return offset(value)" in result.diff
    assert result.meta["baseline_success"] is True
    assert result.meta["llm_call_count"] == 2
    requests = server.requests_for(method="POST", path="/v1/chat/completions")
    assert len(requests) == 2
    assert {req["headers"]["authorization"] for req in requests} == {"Bearer sk-test"}
    first_payload = requests[0]["json"]
    second_payload = requests[1]["json"]
    assert isinstance(first_payload, dict)
    assert isinstance(second_payload, dict)
    assert first_payload["model"] == "test-model"
    assert second_payload["model"] == "test-model"
    assert "experienced and insightful programming instructor" in str(
        first_payload["messages"][0]["content"]
    )


@pytest.mark.skipif(
    not (_LAMBDA_RLM_REPO.is_dir() and _LAMBDA_RLM_ENV_PYTHON.is_file()),
    reason="real lambda-RLM repo/env not provisioned",
)
def test_lambda_rlm_patcher_reaches_openai_compat_stub_server(tmp_path: Path) -> None:
    case_dir = tmp_path / "lambda_stub_case"
    repaired_code, description = _write_single_file_baseline_case(case_dir)
    context_dir = tmp_path / "context_lambda_stub"
    _write_baseline_case_context(
        context_dir,
        case_id=case_dir.name,
        description=description,
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response("6"),
        )
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(repaired_code),
        )

        patcher = build_baseline_patcher(
            baseline_name="lambda_rlm",
            repo_dir=_LAMBDA_RLM_REPO,
            python_executable=_LAMBDA_RLM_ENV_PYTHON,
            model="test-model",
            api_base_url=server.base_url_v1,
            api_key="sk-test",
            max_attempts=1,
            api_timeout_sec=15.0,
            process_timeout_sec=30.0,
            timeout_sec=30.0,
        )
        result = patcher.propose_patch(prompt="", context_dir=context_dir, cwd=case_dir)

    assert result.diff is not None
    assert "-    return offset(value) + 1" in result.diff
    assert "+    return offset(value)" in result.diff
    assert result.meta["baseline_success"] is True
    assert result.meta["llm_call_count"] == 2
    requests = server.requests_for(method="POST", path="/v1/chat/completions")
    assert len(requests) == 2
    first_payload = requests[0]["json"]
    second_payload = requests[1]["json"]
    assert isinstance(first_payload, dict)
    assert isinstance(second_payload, dict)
    assert requests[0]["headers"]["authorization"] == "Bearer sk-test"
    assert first_payload["model"] == "test-model"
    assert second_payload["model"] == "test-model"
    assert "Reply with ONLY a single digit" in str(first_payload["messages"][0]["content"])


@pytest.mark.skipif(
    not (_RLM_REPO.is_dir() and _RLM_ENV_PYTHON.is_file()),
    reason="real rlm repo/env not provisioned",
)
def test_rlm_patcher_reaches_openai_compat_stub_server(tmp_path: Path) -> None:
    case_dir = tmp_path / "rlm_stub_case"
    repaired_code, description = _write_single_file_baseline_case(case_dir)
    context_dir = tmp_path / "context_rlm_stub"
    _write_baseline_case_context(
        context_dir,
        case_id=case_dir.name,
        description=description,
    )
    response_text = (
        "```repl\n"
        f"fixed_code = {repaired_code!r}\n"
        "write_buggy_py(fixed_code)\n"
        "```\n"
        "FINAL_VAR(fixed_code)"
    )

    with OpenAIStubServer() as server:
        server.enqueue_json(
            method="POST",
            path="/v1/chat/completions",
            body=_chat_json_response(response_text),
        )

        patcher = build_baseline_patcher(
            baseline_name="rlm",
            repo_dir=_RLM_REPO,
            python_executable=_RLM_ENV_PYTHON,
            model="test-model",
            api_base_url=server.base_url_v1,
            api_key="sk-test",
            max_attempts=1,
            api_timeout_sec=15.0,
            process_timeout_sec=30.0,
            timeout_sec=30.0,
        )
        result = patcher.propose_patch(prompt="", context_dir=context_dir, cwd=case_dir)

    assert result.diff is not None
    assert "-    return offset(value) + 1" in result.diff
    assert "+    return offset(value)" in result.diff
    assert result.meta["baseline_success"] is True
    assert result.meta["llm_call_count"] == 1
    requests = server.requests_for(method="POST", path="/v1/chat/completions")
    assert len(requests) == 1
    payload = requests[0]["json"]
    assert isinstance(payload, dict)
    assert requests[0]["headers"]["authorization"] == "Bearer sk-test"
    assert payload["model"] == "test-model"
    assert "REPL environment" in str(payload["messages"][0]["content"])


@pytest.mark.parametrize(
    ("baseline_name", "expected_request_count"),
    [
        ("ldb", 2),
        ("intervenor", 2),
        ("tracecoder", 3),
        ("lambda_rlm", 2),
        ("rlm", 1),
    ],
)
def test_run_eval_baseline_shorthand_reaches_openai_compat_stub_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    baseline_name: str,
    expected_request_count: int,
) -> None:
    runtime_by_name = {
        "ldb": (_LDB_REPO, _LDB_ENV_PYTHON),
        "intervenor": (_INTERVENOR_REPO, _INTERVENOR_ENV_PYTHON),
        "tracecoder": (_TRACECODER_REPO, _TRACECODER_ENV_PYTHON),
        "lambda_rlm": (_LAMBDA_RLM_REPO, _LAMBDA_RLM_ENV_PYTHON),
        "rlm": (_RLM_REPO, _RLM_ENV_PYTHON),
    }
    repo_dir, python_executable = runtime_by_name[baseline_name]
    if not repo_dir.is_dir() or not python_executable.is_file():
        pytest.skip(f"real {baseline_name} repo/env not provisioned")

    isolated_root = tmp_path / "isolated_repo"
    isolated_root.mkdir(parents=True, exist_ok=True)
    (isolated_root / "synthetic_cases").mkdir(parents=True, exist_ok=True)
    case_dir = isolated_root / "synthetic_cases" / f"{baseline_name}_harness_stub_case"

    if baseline_name == "tracecoder":
        repaired_code, instrumented_code, description = _write_tracecoder_case(
            case_dir, kind="subprocess"
        )
    elif baseline_name == "ldb":
        repaired_code, description = _write_ldb_compatible_case(case_dir)
        instrumented_code = ""
    else:
        repaired_code, description = _write_single_file_baseline_case(case_dir)
        instrumented_code = ""

    case = _make_baseline_case_spec(
        case_id=case_dir.name,
        case_dir=case_dir,
        description=description,
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", isolated_root)
    monkeypatch.setattr(run_eval, "load_cases", lambda _cases_dir: [case])
    monkeypatch.setattr(run_eval, "_default_baseline_repo_dir", lambda _name: repo_dir)
    monkeypatch.setattr(
        run_eval,
        "_autodetect_baseline_python",
        lambda _name: str(python_executable),
    )

    run_id = f"{baseline_name}-baseline-harness-smoke"
    with OpenAIStubServer() as server:
        if baseline_name == "tracecoder":
            for content in (
                f"```python\n{instrumented_code}```",
                (
                    "REPAIR_PLAN_START\n"
                    "```python\n"
                    "1. Remove the incorrect + 1 offset from the return path.\n"
                    "```\n"
                    "REPAIR_PLAN_END\n\n"
                    "Instrumentation_Suggestions_START\n"
                    "```python\n"
                    "Print the worker output before returning.\n"
                    "```\n"
                    "Instrumentation_Suggestions_END"
                ),
                f"CODE:\n```python\n{repaired_code}```\nEND_CODE",
            ):
                server.enqueue_json(
                    method="POST",
                    path="/v1/chat/completions",
                    body=_chat_json_response(content),
                )
        elif baseline_name == "ldb":
            for _ in range(expected_request_count):
                server.enqueue_json(
                    method="POST",
                    path="/v1/chat/completions",
                    body=_chat_json_response(repaired_code),
                )
        elif baseline_name == "intervenor":
            server.enqueue_json(
                method="POST",
                path="/v1/chat/completions",
                body=_chat_json_response("Remove the extra + 1 offset in compute."),
            )
            server.enqueue_json(
                method="POST",
                path="/v1/chat/completions",
                body=_chat_json_response(repaired_code),
            )
        elif baseline_name == "lambda_rlm":
            server.enqueue_json(
                method="POST",
                path="/v1/chat/completions",
                body=_chat_json_response("6"),
            )
            server.enqueue_json(
                method="POST",
                path="/v1/chat/completions",
                body=_chat_json_response(repaired_code),
            )
        else:
            server.enqueue_json(
                method="POST",
                path="/v1/chat/completions",
                body=_chat_json_response(
                    "```repl\n"
                    f"fixed_code = {repaired_code!r}\n"
                    "write_buggy_py(fixed_code)\n"
                    "```\n"
                    "FINAL_VAR(fixed_code)"
                ),
            )

        exit_code = run_eval.main(
            [
                "--run-id",
                run_id,
                "--baseline",
                baseline_name,
                "--openai-base-url",
                server.base_url_v1,
                "--openai-api-key",
                "sk-test",
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
        )
        requests = server.requests_for(method="POST", path="/v1/chat/completions")

    assert exit_code == 0
    assert len(requests) == expected_request_count
    assert {request["headers"]["authorization"] for request in requests} == {"Bearer sk-test"}
    for request in requests:
        payload = request["json"]
        assert isinstance(payload, dict)
        assert payload["model"] == "test-model"

    results_path = isolated_root / "evals" / "results" / f"{run_id}.jsonl"
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["case_id"] == case.case_id
    assert record["condition"] == baseline_name
    assert record["success"] is True
    assert record["note"] == "success"
    patch_meta = record["patch_meta"]
    assert patch_meta["baseline"] == baseline_name
    assert patch_meta["baseline_success"] is True
    assert patch_meta["llm_call_count"] == expected_request_count


@pytest.mark.skipif(not _ENV_MANIFEST.is_file(), reason="baseline runtime envs not provisioned")
def test_runtime_env_manifest_matches_expected_paths() -> None:
    payload = json.loads(_ENV_MANIFEST.read_text(encoding="utf-8"))

    assert payload["mode"] == "smoke"
    assert payload["creation_backend"] == "python_venv_copies"
    assert payload["ldb"]["python_executable"] == str(_LDB_ENV_PYTHON)
    assert payload["ldb"]["python_repo_relative"] == "baselines/envs/ldb/bin/python"
    assert payload["intervenor"]["python_executable"] == str(_INTERVENOR_ENV_PYTHON)
    assert payload["intervenor"]["python_repo_relative"] == "baselines/envs/intervenor/bin/python"
    assert payload["tracecoder"]["python_executable"] == str(_TRACECODER_ENV_PYTHON)
    assert payload["tracecoder"]["python_repo_relative"] == "baselines/envs/tracecoder/bin/python"
    assert _LDB_ENV_PYTHON.is_file()
    assert _INTERVENOR_ENV_PYTHON.is_file()
    assert _TRACECODER_ENV_PYTHON.is_file()
    if _LAMBDA_RLM_ENV_PYTHON.is_file():
        assert payload["lambda_rlm"]["python_executable"] == str(_LAMBDA_RLM_ENV_PYTHON)
        assert (
            payload["lambda_rlm"]["python_repo_relative"] == "baselines/envs/lambda_rlm/bin/python"
        )
        assert payload["lambda_rlm"]["repo_dir"] == str(_LAMBDA_RLM_REPO)
        assert payload["lambda_rlm"]["repo_dir_repo_relative"] == "baselines/repos/lambda-RLM"
    if _RLM_ENV_PYTHON.is_file():
        assert payload["rlm"]["python_executable"] == str(_RLM_ENV_PYTHON)
        assert payload["rlm"]["python_repo_relative"] == "baselines/envs/rlm/bin/python"
        assert payload["rlm"]["repo_dir"] == str(_RLM_REPO)
        assert payload["rlm"]["repo_dir_repo_relative"] == "baselines/repos/rlm"


@pytest.mark.skipif(not _ENV_MANIFEST.is_file(), reason="baseline runtime envs not provisioned")
def test_runtime_envs_import_expected_smoke_dependencies() -> None:
    ldb_result = subprocess.run(
        [
            str(_LDB_ENV_PYTHON),
            "-c",
            (
                "import astroid, graphviz, transformers, vllm; "
                "from astroid.builder import AstroidBuilder; "
                "from openai import OpenAI; "
                "assert int(astroid.__version__.split('.')[0]) < 4; "
                "AstroidBuilder(); "
                "OpenAI(api_key='sk-test', base_url='http://127.0.0.1:1234/v1'); "
                "print('ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    intervenor_result = subprocess.run(
        [str(_INTERVENOR_ENV_PYTHON), "-c", "import openai, numpy, backoff; print('ok')"],
        check=False,
        capture_output=True,
        text=True,
    )
    tracecoder_result = subprocess.run(
        [str(_TRACECODER_ENV_PYTHON), "-c", "import openai, pandas, pytest, tqdm; print('ok')"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert ldb_result.returncode == 0, ldb_result.stderr
    assert intervenor_result.returncode == 0, intervenor_result.stderr
    assert tracecoder_result.returncode == 0, tracecoder_result.stderr
    if _LAMBDA_RLM_ENV_PYTHON.is_file():
        lambda_result = subprocess.run(
            [str(_LAMBDA_RLM_ENV_PYTHON), "-c", "import openai, rlm; print('ok')"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert lambda_result.returncode == 0, lambda_result.stderr
    if _RLM_ENV_PYTHON.is_file():
        rlm_result = subprocess.run(
            [str(_RLM_ENV_PYTHON), "-c", "import openai, rlm; print('ok')"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rlm_result.returncode == 0, rlm_result.stderr
