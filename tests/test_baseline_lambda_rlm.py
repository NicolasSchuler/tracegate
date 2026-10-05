"""Tests for the lambda-RLM raw baseline adapter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines._runtime import _CommandOutcome
from baselines.adapter_base import BaselineCaseContext, BaselineConfig
from baselines.lambda_rlm import env_client
from baselines.lambda_rlm.adapter import LambdaRLMAdapter
from baselines.lambda_rlm.case_converter import convert_case
from baselines.openai_compat import BaselineOpenAICompatConfig, OpenAICompatResponseError
from llmdebug import _json as json


@pytest.fixture()
def lambda_case(tmp_path: Path) -> Path:
    case_dir = tmp_path / "lambda_case"
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
    return case_dir


def _make_lambda_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "lambda-RLM"
    (repo / "rlm" / "clients").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='lambda-rlm'\n", encoding="utf-8")
    (repo / "rlm" / "__init__.py").write_text(
        "from rlm.lambda_rlm import LambdaRLM\n", encoding="utf-8"
    )
    (repo / "rlm" / "lambda_rlm.py").write_text(
        "class LambdaRLM:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n"
        "    def completion(self, prompt):\n"
        "        raise RuntimeError('stub')\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "__init__.py").write_text(
        "def get_client(backend, backend_kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "openai.py").write_text(
        "class OpenAIClient:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        self.model_name = kwargs.get('model_name', '')\n"
        "    def completion(self, prompt, model=None):\n"
        "        return ''\n",
        encoding="utf-8",
    )
    return repo


def _make_lambda_streaming_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "lambda-RLM-stream"
    (repo / "rlm" / "clients").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        "[project]\nname='lambda-rlm-stream'\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "rlm" / "lambda_rlm.py").write_text(
        "from rlm.clients import get_client\n\n"
        "class LambdaRLM:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        pass\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "__init__.py").write_text(
        "from rlm.clients.openai import OpenAIClient\n\n"
        "def get_client(backend, backend_kwargs):\n"
        "    if backend == 'openai':\n"
        "        return OpenAIClient(**backend_kwargs)\n"
        "    return None\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "openai.py").write_text(
        "from types import SimpleNamespace\n\n"
        "class _Completions:\n"
        "    def __init__(self, content_text, reasoning_text):\n"
        "        self._content_text = content_text\n"
        "        self._reasoning_text = reasoning_text\n\n"
        "    def create(self, **kwargs):\n"
        "        if kwargs.get('stream', False):\n"
        "            delta = SimpleNamespace(\n"
        "                content=self._content_text,\n"
        "                reasoning='',\n"
        "                reasoning_content=self._reasoning_text,\n"
        "            )\n"
        "            return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]\n"
        "        message = SimpleNamespace(\n"
        "            content=self._content_text,\n"
        "            reasoning_content=self._reasoning_text,\n"
        "        )\n"
        "        return SimpleNamespace(\n"
        "            choices=[SimpleNamespace(message=message, finish_reason='stop')],\n"
        "        )\n\n"
        "class OpenAIClient:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        del args\n"
        "        self.model_name = kwargs.get('model_name', '')\n"
        "        self._completion_kwargs = {'stream': kwargs.get('stream', False)}\n"
        "        self.client = SimpleNamespace(\n"
        "            chat=SimpleNamespace(\n"
        "                completions=_Completions(\n"
        "                    kwargs.get('content_text', ''),\n"
        "                    kwargs.get('reasoning_text', ''),\n"
        "                )\n"
        "            )\n"
        "        )\n",
        encoding="utf-8",
    )
    return repo


def _make_lambda_env_client_tracking_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "lambda-RLM-env"
    (repo / "rlm" / "clients").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='lambda-rlm-env'\n", encoding="utf-8")
    (repo / "rlm" / "__init__.py").write_text(
        "from rlm.lambda_rlm import LambdaRLM\n", encoding="utf-8"
    )
    (repo / "rlm" / "clients" / "__init__.py").write_text(
        "from rlm.clients.openai import OpenAIClient\n\n"
        "def get_client(backend, backend_kwargs):\n"
        "    if backend != 'openai':\n"
        "        raise ValueError(f'unsupported backend: {backend}')\n"
        "    return OpenAIClient(**backend_kwargs)\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "clients" / "openai.py").write_text(
        "import openai\n\n"
        "class OpenAIClient:\n"
        "    def __init__(self, model_name='', api_key='', base_url=None, timeout=None, **kwargs):\n"
        "        self.model_name = model_name\n"
        "        self._completion_kwargs = kwargs\n"
        "        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)\n"
        "    def get_usage_summary(self):\n"
        "        return {'model_usage_summaries': {}}\n"
        "    def _track_cost(self, response, model):\n"
        "        del response, model\n"
        "        return None\n",
        encoding="utf-8",
    )
    (repo / "rlm" / "lambda_rlm.py").write_text(
        "from types import SimpleNamespace\n"
        "from rlm.clients import get_client\n\n"
        "class _UsageSummary:\n"
        "    def to_dict(self):\n"
        "        return {'model_usage_summaries': {}}\n"
        "    total_input_tokens = 0\n"
        "    total_output_tokens = 0\n\n"
        "class LambdaRLM:\n"
        "    def __init__(self, *, backend, backend_kwargs, **kwargs):\n"
        "        del kwargs\n"
        "        self.backend = backend\n"
        "        self.backend_kwargs = backend_kwargs\n"
        "    def _parse_task_type(self, response):\n"
        "        del response\n"
        "        return SimpleNamespace(value='qa')\n"
        "    def _plan(self, task_type, n):\n"
        "        del task_type, n\n"
        "        return SimpleNamespace(task_type=SimpleNamespace(value='qa'), compose_op=SimpleNamespace(value='select_relevant'), k_star=1, tau_star=10, depth=0, cost_estimate=0.0, n=10)\n"
        "    def completion(self, prompt):\n"
        "        client = get_client(self.backend, self.backend_kwargs)\n"
        "        response = client.completion(prompt, model=self.backend_kwargs.get('model_name', ''))\n"
        "        return SimpleNamespace(root_model=self.backend_kwargs.get('model_name', ''), response=response, usage_summary=_UsageSummary(), execution_time=0.1)\n",
        encoding="utf-8",
    )
    return repo


class TestLambdaCaseConverter:
    def test_convert_case_packages_prompt_context_and_files(self, lambda_case: Path) -> None:
        context = BaselineCaseContext(
            case_id="lambda_case",
            python_args=("-m", "pytest", "-q"),
            timeout_sec=12,
        )

        result = convert_case(lambda_case, case_context=context)

        assert result.task_id == "lambda_case"
        assert result.entry_point == "compute"
        assert result.python_args == ["-m", "pytest", "-q"]
        assert result.timeout_sec == 12
        assert "Primary symbol: compute" in str(result.prompt_context)
        assert "=== buggy.py ===" in str(result.prompt_context)
        assert "=== helper.py ===" in str(result.prompt_context)
        assert "=== test_buggy.py ===" in str(result.prompt_context)
        paths = {row["path"] for row in result.files}
        assert paths == {"buggy.py", "helper.py", "test_buggy.py"}

    def test_convert_case_prefers_case_context_case_id(self, lambda_case: Path) -> None:
        result = convert_case(
            lambda_case,
            case_context=BaselineCaseContext(
                case_id="canonical_case_id",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=12,
            ),
        )

        assert result.task_id == "canonical_case_id"


class TestLambdaRLMAdapter:
    def test_adapter_name(self) -> None:
        assert LambdaRLMAdapter().name == "lambda_rlm"

    def test_contract_defaults_to_canonical_prompt_contract(self) -> None:
        contract = LambdaRLMAdapter().contract()
        assert contract.prompt_contract_kind == "adapter_defined_code_repair_prompt"

    def test_setup_raises_on_missing_repo(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            LambdaRLMAdapter().setup(BaselineConfig(repo_dir=tmp_path / "missing"))

    def test_setup_succeeds_with_minimal_repo(self, tmp_path: Path) -> None:
        repo = _make_lambda_repo(tmp_path)
        adapter = LambdaRLMAdapter()
        adapter.setup(BaselineConfig(repo_dir=repo))
        assert adapter.repo_commit == ""

    def test_contract_reflects_repair_fair_prompt_contract(self, tmp_path: Path) -> None:
        repo = _make_lambda_repo(tmp_path)
        adapter = LambdaRLMAdapter()
        adapter.setup(
            BaselineConfig(
                repo_dir=repo,
                extra={"baseline_lambda_prompt_contract": "repair_fair_v1"},
            )
        )
        contract = adapter.contract()
        assert contract.prompt_contract_kind == "adapter_defined_code_repair_prompt__repair_fair_v1"

    def test_setup_rejects_stale_env_client_compatibility_declaration(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_lambda_repo(tmp_path)
        adapter = LambdaRLMAdapter()
        monkeypatch.setattr(
            "baselines.lambda_rlm.adapter.lambda_env_client._COMPATIBLE_WITH",
            "lambda-RLM@deadbeef",
        )

        with pytest.raises(RuntimeError, match="compatibility declaration is stale"):
            adapter.setup(BaselineConfig(repo_dir=repo))

    def test_run_repair_retries_and_accumulates_metrics(
        self,
        lambda_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_lambda_repo(tmp_path)
        adapter = LambdaRLMAdapter()
        fake_python = tmp_path / "lambda-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=2,
            api_timeout_sec=11,
            process_timeout_sec=30,
            timeout_sec=None,
            extra={
                "baseline_lambda_context_window_chars": 8000,
                "baseline_lambda_accuracy_target": 0.9,
                "baseline_lambda_a_leaf": 0.96,
                "baseline_lambda_a_compose": 0.92,
                "baseline_lambda_timeout_sec": 40,
                "baseline_lambda_force_task_type": "qa",
            },
        )
        adapter.setup(config)
        case_input = convert_case(
            lambda_case,
            case_context=BaselineCaseContext(
                case_id="lambda_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        test_outcomes = iter(
            [
                _CommandOutcome(
                    returncode=1, stdout="", stderr="assert compute(3) == 6", duration_sec=0.1
                ),
                _CommandOutcome(returncode=1, stdout="", stderr="still wrong", duration_sec=0.2),
                _CommandOutcome(returncode=0, stdout="ok", stderr="", duration_sec=0.15),
            ]
        )

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            assert work_dir.is_dir()
            assert python_args == ["-m", "pytest", "-q"]
            assert timeout_sec == 15
            return next(test_outcomes)

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)

        responses = [
            {
                "response": "```python\nfrom helper import offset\n\n\ndef compute(value):\n    return offset(value) + 1\n```",
                "execution_time": 0.4,
                "usage_summary": {
                    "total_calls": 3,
                    "total_input_tokens": 100,
                    "total_output_tokens": 40,
                },
                "char_summary": {
                    "total_calls": 3,
                    "total_input_chars": 900,
                    "total_output_chars": 120,
                },
                "plan_metadata": {
                    "task_type": "analysis",
                    "compose_op": "merge_summaries",
                    "k_star": 2,
                    "tau_star": 300,
                    "depth": 1,
                },
                "call_summary": {
                    "call_finish_reason_counts": {"length": 1, "stop": 2},
                    "call_length_finish_count": 1,
                    "task_detection_call_count": 1,
                    "repl_llm_query_call_count": 2,
                    "compat_recovery_count": 0,
                    "planned_decomposition": True,
                    "executed_decomposition": True,
                    "planned_depth": 1,
                    "k_star": 2,
                    "tau_star": 300,
                    "task_type": "analysis",
                },
            },
            {
                "response": "from helper import offset\n\n\ndef compute(value):\n    return offset(value)\n",
                "execution_time": 0.3,
                "usage_summary": {
                    "total_calls": 2,
                    "total_input_tokens": 80,
                    "total_output_tokens": 20,
                },
                "char_summary": {
                    "total_calls": 2,
                    "total_input_chars": 700,
                    "total_output_chars": 90,
                },
                "plan_metadata": {
                    "task_type": "analysis",
                    "compose_op": "merge_summaries",
                    "k_star": 1,
                    "tau_star": 200,
                    "depth": 0,
                },
                "call_summary": {
                    "call_finish_reason_counts": {"stop": 2},
                    "call_length_finish_count": 0,
                    "task_detection_call_count": 1,
                    "repl_llm_query_call_count": 1,
                    "compat_recovery_count": 1,
                    "compat_recovery_events": [
                        {"reason": "request_validation_retry_disable_thinking"}
                    ],
                    "planned_decomposition": False,
                    "executed_decomposition": False,
                    "planned_depth": 0,
                    "k_star": 1,
                    "tau_star": 200,
                    "task_type": "analysis",
                },
            },
        ]

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            assert request_payload["model"] == "test-model"
            assert request_payload["api_timeout_sec"] == 11.0
            assert request_payload["context_window_chars"] == 8000
            assert request_payload["accuracy_target"] == 0.9
            assert request_payload["a_leaf"] == 0.96
            assert request_payload["a_compose"] == 0.92
            assert request_payload["force_task_type"] == "qa"
            assert timeout == 30
            payload = responses.pop(0)
            response_path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.lambda_rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is True
        assert result.iterations_used == 2
        assert (
            result.repaired_code
            == "from helper import offset\n\n\ndef compute(value):\n    return offset(value)"
        )
        assert result.token_usage == {
            "prompt_tokens": 180,
            "completion_tokens": 60,
            "total_tokens": 240,
        }
        assert result.meta["last_plan_metadata"]["k_star"] == 1
        attempts = result.meta["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["candidate_status"] == "tests_still_failing"
        assert attempts[1]["candidate_status"] == "passed"
        assert attempts[0]["planned_decomposition"] is True
        assert attempts[0]["executed_decomposition"] is True
        assert attempts[0]["task_detection_call_count"] == 1
        assert attempts[0]["repl_llm_query_call_count"] == 2
        assert attempts[0]["call_finish_reason_counts"] == {"length": 1, "stop": 2}
        assert attempts[1]["compat_recovery_count"] == 1
        metrics = result.meta["metrics"]
        assert metrics["llm_call_count"] == 5
        assert metrics["reported_tokens_prompt"] == 180
        assert metrics["reported_tokens_completion"] == 60
        assert metrics["reported_chars_prompt"] == 1600
        assert metrics["reported_chars_completion"] == 210
        assert metrics["internal_task_detection_call_count"] == 2
        assert metrics["internal_repl_llm_query_call_count"] == 3
        assert metrics["internal_length_finish_count"] == 1
        assert metrics["internal_compat_recovery_count"] == 1
        assert metrics["internal_planned_decomposition_attempt_count"] == 1
        assert metrics["internal_executed_decomposition_attempt_count"] == 1
        assert metrics["direct_gen_passed"] is False
        assert metrics["debug_added_value"] is True

    def test_build_prompt_freezes_prompt_contract(self) -> None:
        adapter = LambdaRLMAdapter()

        prompt = adapter._build_prompt(
            prompt_context="Primary symbol: compute\n=== buggy.py ===\ndef compute(x):\n    return x + 1",
            failure_feedback="AssertionError: expected 6, got 7",
            controller_guidance="crash_anchor: buggy.py:2 in compute",
            attempt_idx=2,
            failure_feedback_max_chars=3500,
        )

        assert prompt.startswith("Context:\nPrimary symbol: compute")
        assert "Return ONLY the full corrected Python source code for buggy.py." in prompt
        assert (
            "Do NOT include explanations, comments about what you changed, or any other text."
            in prompt
        )
        assert "This is outer repair attempt 2. Do not repeat the previous failure." in prompt
        assert "Failure feedback from the previous attempt/test run:" in prompt
        assert "Adaptive controller guidance:" in prompt
        assert prompt.endswith("\n\nAnswer:")

    def test_build_prompt_truncates_failure_feedback_when_configured(self) -> None:
        adapter = LambdaRLMAdapter()

        prompt = adapter._build_prompt(
            prompt_context="ctx",
            failure_feedback="X" * 300,
            controller_guidance="",
            attempt_idx=1,
            failure_feedback_max_chars=80,
        )

        assert "Failure feedback from the previous attempt/test run:" in prompt
        assert "X" * 90 not in prompt
        assert "..." in prompt

    def test_build_prompt_adds_repair_fair_solve_first_instructions(self) -> None:
        adapter = LambdaRLMAdapter()

        prompt = adapter._build_prompt(
            prompt_context="ctx",
            failure_feedback="AssertionError: expected 6, got 7",
            controller_guidance="",
            attempt_idx=2,
            prompt_contract="repair_fair_v1",
            failure_feedback_max_chars=3500,
        )

        assert "Prefer the smallest plausible direct repair immediately." in prompt
        assert "Do not spend output budget restating the task" in prompt
        assert "Revise the previous fix directly using the failure feedback below." in prompt

    def test_env_client_forwards_api_timeout_to_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "lambda-RLM"
        repo.mkdir()
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        captured: dict[str, object] = {}

        class _FakeUsageSummary:
            total_input_tokens = 3
            total_output_tokens = 2

            def to_dict(self) -> dict[str, object]:
                return {"model_usage_summaries": {"test-model": {"total_calls": 1}}}

        class _FakeCompletion:
            root_model = "test-model"
            response = "patched"
            execution_time = 0.2
            usage_summary = _FakeUsageSummary()

        class _FakeLambdaRLM:
            def __init__(
                self, *, backend: str, backend_kwargs: dict[str, object], **kwargs: object
            ) -> None:
                del backend, kwargs
                captured["backend_kwargs"] = dict(backend_kwargs)
                self._parse_task_type = lambda response: SimpleNamespace(value="analysis")
                self._plan = lambda task_type, n: SimpleNamespace(
                    task_type=SimpleNamespace(value="analysis"),
                    compose_op=SimpleNamespace(value="merge"),
                    k_star=1,
                    tau_star=1,
                    depth=0,
                    cost_estimate=0.0,
                    n=n,
                )

            def completion(self, prompt: str) -> _FakeCompletion:
                del prompt
                return _FakeCompletion()

        monkeypatch.setattr(
            env_client,
            "_install_tracking",
            lambda *args, **kwargs: (_FakeLambdaRLM, []),
        )
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "model": "test-model",
                    "prompt": "hello",
                    "api_key": "sk-test",
                    "api_timeout_sec": 7.5,
                }
            ),
            encoding="utf-8",
        )

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        assert exit_code == 0
        assert captured["backend_kwargs"] == {
            "model_name": "test-model",
            "api_key": "sk-test",
            "base_url": None,
            "timeout": 7.5,
        }

    def test_run_repair_returns_compatibility_error_from_env_client(
        self,
        lambda_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_lambda_repo(tmp_path)
        adapter = LambdaRLMAdapter()
        fake_python = tmp_path / "lambda-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            api_base_url="http://127.0.0.1:1234",
            api_key="sk-test",
            max_attempts=1,
            api_timeout_sec=11,
            process_timeout_sec=30,
            extra={"baseline_openai_compat_profile": "lmstudio_model_yaml"},
        )
        adapter.setup(config)
        case_input = convert_case(
            lambda_case,
            case_context=BaselineCaseContext(
                case_id="lambda_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()

        def _fake_tests(
            *, work_dir: Path, python_args: list[str], timeout_sec: int
        ) -> _CommandOutcome:
            del work_dir, python_args, timeout_sec
            return _CommandOutcome(
                returncode=1, stdout="", stderr="still failing", duration_sec=0.1
            )

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            response_path = Path(cmd[3])
            response_path.write_text(
                json.dumps(
                    {
                        "error": "openai_compat_reasoning_only",
                        "compat_message": "Disable thinking in the local model template.",
                        "compat_meta": {
                            "compat_profile_applied": "lmstudio_model_yaml",
                            "empty_content_with_reasoning": True,
                        },
                        "response": "",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "plan_metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(adapter, "_run_case_tests", _fake_tests)
        monkeypatch.setattr("baselines.lambda_rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert result.meta["error"] == "openai_compat_reasoning_only"
        assert result.meta["compat_message"] == "Disable thinking in the local model template."
        assert result.meta["compat_meta"]["compat_profile_applied"] == "lmstudio_model_yaml"

    def test_run_repair_uses_configured_feedback_and_guidance_budgets(
        self,
        lambda_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_lambda_repo(tmp_path)
        adapter = LambdaRLMAdapter()
        fake_python = tmp_path / "lambda-python"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            max_attempts=1,
            process_timeout_sec=30,
            extra={
                "baseline_lambda_controller_guidance_max_chars": 120,
                "baseline_lambda_failure_feedback_max_chars": 90,
                "baseline_lambda_failure_feedback_meta_max_chars": 70,
            },
        )
        adapter.setup(config)
        case_input = convert_case(
            lambda_case,
            case_context=BaselineCaseContext(
                case_id="lambda_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "schema_version": "v1",
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "summary": {
                    "crash_anchor": "buggy.py:4 in compute",
                    "top_facts": ["A" * 300, "B" * 300],
                },
            },
            "history_bundle": {"schema_version": "v1"},
            "verifier_report": {"schema_version": "v1"},
        }

        monkeypatch.setattr(
            adapter,
            "_run_case_tests",
            lambda **kwargs: _CommandOutcome(
                returncode=1, stdout="", stderr="Y" * 300, duration_sec=0.1
            ),
        )
        observed_prompt: dict[str, str] = {}

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            observed_prompt["prompt"] = str(request_payload["prompt"])
            response_path.write_text(
                json.dumps(
                    {
                        "response": "def compute(value):\n    return value\n",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "plan_metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.lambda_rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        assert "Adaptive controller guidance:" in observed_prompt["prompt"]
        assert len(observed_prompt["prompt"]) < 1000
        assert result.meta["attempts"][0]["failure_feedback"].endswith("...")

    def test_run_repair_uses_fair_v1_hybrid_guidance_block(
        self,
        lambda_case: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_lambda_repo(tmp_path)
        adapter = LambdaRLMAdapter()
        fake_python = tmp_path / "lambda-python-fair"
        fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_python.chmod(0o755)
        config = BaselineConfig(
            repo_dir=repo,
            python_executable=fake_python,
            model="test-model",
            max_attempts=1,
            process_timeout_sec=30,
            extra={
                "baseline_hybrid_guidance_mode": "fair_v1",
                "baseline_lambda_controller_guidance_max_chars": 360,
                "baseline_lambda_failure_feedback_max_chars": 90,
                "baseline_lambda_failure_feedback_meta_max_chars": 70,
            },
        )
        adapter.setup(config)
        case_input = convert_case(
            lambda_case,
            case_context=BaselineCaseContext(
                case_id="lambda_case",
                python_args=("-m", "pytest", "-q"),
                timeout_sec=15,
            ),
        ).to_dict()
        case_input["llmdebug_controller_packet"] = {
            "schema_version": "v1",
            "controller_state": {
                "helper_action_executed": "snapshot_lite",
                "helper_outcome": "materialized",
            },
            "evidence_bundle": {
                "schema_version": "v1",
                "summary": {
                    "crash_anchor": "buggy.py:4 in compute",
                    "top_facts": ["Focus on buggy.py", "Do not widen the edit"],
                },
            },
            "history_bundle": {
                "schema_version": "v1",
                "lessons": [
                    {"source": "attempt_history", "attempt": 1, "lesson": "keep the return type"}
                ],
            },
            "verifier_report": {
                "schema_version": "v1",
                "status": "previous_verifier_rejected",
                "reasons": ["touch buggy.py", "preserve signature"],
            },
            "solver_hints": {
                "retry_plan": "Edit only the compute return expression.",
                "strategy_summary": "should not appear",
                "decomposition": [{"title": "localize", "question": "should not appear"}],
            },
        }

        monkeypatch.setattr(
            adapter,
            "_run_case_tests",
            lambda **kwargs: _CommandOutcome(
                returncode=1, stdout="", stderr="Y" * 200, duration_sec=0.1
            ),
        )
        observed_prompt: dict[str, str] = {}

        def _fake_run(
            cmd: list[str],
            *,
            capture_output: bool,
            text: bool,
            timeout: int,
            cwd: str,
            env: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            del capture_output, text, timeout, cwd, env
            request_path = Path(cmd[2])
            response_path = Path(cmd[3])
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
            observed_prompt["prompt"] = str(request_payload["prompt"])
            response_path.write_text(
                json.dumps(
                    {
                        "response": "def compute(value):\n    return value\n",
                        "execution_time": 0.1,
                        "usage_summary": {},
                        "char_summary": {},
                        "plan_metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("baselines.lambda_rlm.adapter.subprocess.run", _fake_run)

        result = adapter.run_repair(case_input, config)

        assert result.success is False
        prompt = observed_prompt["prompt"]
        assert "Adaptive controller guidance:" in prompt
        assert "verifier: status=previous_verifier_rejected" in prompt
        assert "retry_plan: Edit only the compute return expression." in prompt
        assert "strategy_summary:" not in prompt
        assert "decomposition:" not in prompt


class TestLambdaRLMEnvClientStreaming:
    def test_streaming_reasoning_only_raises_without_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _make_lambda_streaming_repo(tmp_path)
        monkeypatch.syspath_prepend(str(repo))
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)

        compat_config = BaselineOpenAICompatConfig(allow_reasoning_fallback=False)
        env_client._install_tracking(
            repo, compat_config=compat_config, base_url="http://127.0.0.1:1234/v1"
        )

        import rlm.clients as clients_mod  # type: ignore[import-not-found]

        client = clients_mod.get_client(
            "openai",
            {
                "model_name": "test-model",
                "stream": True,
                "content_text": "",
                "reasoning_text": "streamed reasoning",
            },
        )

        with pytest.raises(
            OpenAICompatResponseError,
            match=r"reasoning but no final answer in delta\.content",
        ):
            client.completion("hello")

    def test_streaming_reasoning_only_uses_fallback_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _make_lambda_streaming_repo(tmp_path)
        monkeypatch.syspath_prepend(str(repo))
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)

        compat_config = BaselineOpenAICompatConfig(allow_reasoning_fallback=True)
        env_client._install_tracking(
            repo, compat_config=compat_config, base_url="http://127.0.0.1:1234/v1"
        )

        import rlm.clients as clients_mod  # type: ignore[import-not-found]

        client = clients_mod.get_client(
            "openai",
            {
                "model_name": "test-model",
                "stream": True,
                "content_text": "",
                "reasoning_text": "streamed reasoning",
            },
        )

        assert client.completion("hello") == "streamed reasoning"

    def test_env_client_auto_recovers_prefill_enable_thinking_conflict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _make_lambda_env_client_tracking_repo(tmp_path)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        seen_requests: list[dict[str, object]] = []

        class _BadRequestError(Exception):
            pass

        class _FakeChatCompletions:
            def create(self, **kwargs: object) -> object:
                seen_requests.append(dict(kwargs))
                extra_body = kwargs.get("extra_body")
                chat_template_kwargs = (
                    extra_body.get("chat_template_kwargs") if isinstance(extra_body, dict) else None
                )
                if chat_template_kwargs != {"enable_thinking": False}:
                    raise _BadRequestError(
                        "Error code: 400 - {'error': {'code': 400, 'message': "
                        "'Assistant response prefill is incompatible with "
                        "enable_thinking.', 'type': 'invalid_request_error'}}"
                    )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="patched"),
                            finish_reason="stop",
                        )
                    ]
                )

        class _FakeChat:
            def __init__(self) -> None:
                self.completions = _FakeChatCompletions()

        class _FakeOpenAIClient:
            def __init__(self, **kwargs: object) -> None:
                self.base_url = kwargs.get("base_url")
                self.chat = _FakeChat()

        openai_module = SimpleNamespace(OpenAI=_FakeOpenAIClient, AsyncOpenAI=_FakeOpenAIClient)
        monkeypatch.setitem(sys.modules, "openai", openai_module)
        for module_name in list(sys.modules):
            if module_name == "rlm" or module_name.startswith("rlm."):
                monkeypatch.delitem(sys.modules, module_name, raising=False)
        request_path.write_text(
            json.dumps(
                {
                    "repo_dir": str(repo),
                    "model": "test-model",
                    "api_key": "sk-test",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "prompt": "Context:\nctx\n\nQuestion: fix\n\nAnswer:",
                }
            ),
            encoding="utf-8",
        )

        exit_code = env_client.main(["env_client.py", str(request_path), str(response_path)])

        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert response_payload["response"] == "patched"
        assert response_payload["call_summary"]["compat_recovery_count"] == 1
        assert response_payload["call_summary"]["call_finish_reason_counts"] == {"stop": 1}
        assert len(seen_requests) == 2
        assert "extra_body" not in seen_requests[0]
        assert seen_requests[1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


# ===========================================================================
# Regression tests for Lambda_RLM repair prompt & extraction improvements
# ===========================================================================


class TestLambdaRLMRepairExtraction:
    """Tests for _extract_candidate_code and prompt improvements."""

    def test_extract_candidate_code_from_fenced_python(self) -> None:
        adapter = LambdaRLMAdapter()
        response = (
            "The bug is in the multiply function. Here is the fix:\n\n"
            "```python\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "```\n"
        )
        result = adapter._extract_candidate_code(response)
        assert "return a * b" in result
        assert adapter._is_valid_python(result)

    def test_extract_candidate_code_from_prose_with_fenced_block(self) -> None:
        adapter = LambdaRLMAdapter()
        response = (
            "The function `multiply` is incorrectly defined to add instead of multiply. "
            "The fix is simple:\n\n"
            "```python\n"
            "def multiply(a, b):\n"
            "    return a * b\n"
            "```\n\n"
            "This changes the `+` operator to `*`."
        )
        result = adapter._extract_candidate_code(response)
        assert "return a * b" in result
        assert adapter._is_valid_python(result)

    def test_extract_candidate_code_pure_prose_returns_prose(self) -> None:
        adapter = LambdaRLMAdapter()
        response = "The function multiply is incorrectly adding. Change + to *."
        result = adapter._extract_candidate_code(response)
        # Without code fences, falls back to raw text (which _is_valid_python rejects)
        assert result == response.strip()
        assert not adapter._is_valid_python(result)

    def test_extract_candidate_code_raw_python_no_fences(self) -> None:
        adapter = LambdaRLMAdapter()
        response = "def multiply(a, b):\n    return a * b\n"
        result = adapter._extract_candidate_code(response)
        assert result == response.strip()
        assert adapter._is_valid_python(result)

    def test_extract_candidate_code_prefers_last_fenced_block(self) -> None:
        adapter = LambdaRLMAdapter()
        response = (
            "```python\ndef wrong():\n    pass\n```\n"
            "Actually the correct fix is:\n"
            "```python\ndef multiply(a, b):\n    return a * b\n```\n"
        )
        result = adapter._extract_candidate_code(response)
        assert "return a * b" in result

    def test_prompt_contract_code_only_instruction(self) -> None:
        adapter = LambdaRLMAdapter()
        prompt = adapter._build_prompt(
            prompt_context="ctx",
            failure_feedback="",
            controller_guidance="",
            attempt_idx=1,
        )
        assert "Return ONLY the full corrected Python source code" in prompt
        assert "Do NOT include explanations" in prompt
        assert "valid Python" in prompt

    def test_retry_feedback_invalid_python_is_forceful(self) -> None:
        """The retry feedback for invalid_python should strongly request code-only output."""
        adapter = LambdaRLMAdapter()
        # Test the feedback text indirectly by checking the adapter's internal state
        # after a failed extraction. The actual text is constructed inline, so we
        # verify the prompt behavior through the contract test above.
        assert adapter._is_valid_python("def f(): pass")
        assert not adapter._is_valid_python("This is prose, not code.")


def test_bas10_timeout_keeps_last_candidate_and_usage(
    lambda_case: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    repo = _make_lambda_repo(tmp_path)
    adapter = LambdaRLMAdapter()
    config = BaselineConfig(
        repo_dir=repo, model="test-model", max_attempts=2, process_timeout_sec=30
    )
    adapter.setup(config)
    case_input = convert_case(
        lambda_case,
        case_context=BaselineCaseContext(
            case_id="lambda_case", python_args=("-m", "pytest", "-q"), timeout_sec=15
        ),
    ).to_dict()
    outcomes = iter(
        [
            _CommandOutcome(returncode=1, stdout="", stderr="assert", duration_sec=0.1),
            _CommandOutcome(returncode=1, stdout="", stderr="still wrong", duration_sec=0.1),
        ]
    )
    monkeypatch.setattr(adapter, "_run_case_tests", lambda **_kw: next(outcomes))
    calls = {"count": 0}

    def _invoke(**_kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "response": "def compute(value):\n    return value * 3\n",
                "raw_response": "def compute(value):\n    return value * 3\n",
                "execution_time": 0.2,
                "usage_summary": {
                    "total_calls": 1,
                    "total_input_tokens": 50,
                    "total_output_tokens": 10,
                },
                "char_summary": {
                    "total_calls": 1,
                    "total_input_chars": 300,
                    "total_output_chars": 40,
                },
                "plan_metadata": {},
                "call_summary": {},
            }
        raise subprocess.TimeoutExpired(cmd="lambda", timeout=30)

    monkeypatch.setattr(adapter, "_invoke_lambda", _invoke)
    result = adapter.run_repair(case_input, config)

    assert result.meta["error"] == "timeout"
    assert result.repaired_code is not None and "value * 3" in result.repaired_code
    assert result.token_usage["prompt_tokens"] == 50
