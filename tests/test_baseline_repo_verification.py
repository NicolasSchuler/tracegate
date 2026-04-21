"""Tests for baseline repo smoke verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from baselines.patch_manifest import PATCH_MANIFEST
from baselines.verify_repos import BaselineRepoVerificationError, verify_baseline_repos


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_valid_repo_root(root: Path) -> None:
    ldb = root / "LLMDebugger"
    _write(ldb / "programming" / "main.py", "def main():\n    return 0\n")
    _write(
        ldb / "programming" / "ldb.py",
        (
            'DATASET = "llmdebug"\n'
            "llm_duration_sec = 0.0\n"
            "usage_available = True\n"
            "is_passing = False\n"
            "test_feedback = []\n"
        ),
    )
    _write(ldb / "programming" / "utils.py", 'SUPPORTED = {"llmdebug": True}\n')
    _write(
        ldb / "programming" / "generators" / "model.py",
        (
            "def _record_metrics():\n    return None\n"
            "LLMDEBUG_BASELINE_OPENAI_REQUESTED_MODEL = 'test-model'\n"
            "super().__init__(model_name)\n"
        ),
    )
    _write(
        ldb / "programming" / "generators" / "py_generate.py",
        'dataset_type in ["HumanEval", "MBPP", "llmdebug"]\n',
    )

    intervenor = root / "INTERVENOR"
    _write(intervenor / "prompt" / "code_teacher.txt", "teacher prompt\n")
    _write(intervenor / "prompt" / "code_learner.txt", "learner prompt\n")
    _write(intervenor / "runner.py", "def main():\n    return 0\n")

    tracecoder = root / "TraceCoder"
    _write(
        tracecoder / "config.py",
        'DATASET_PATHS = {"llmdebug": {"data_path": "x"}}\n# --dataset-path\n',
    )
    _write(tracecoder / "datasets" / "llmdebug" / "__init__.py", "\n")
    _write(
        tracecoder / "problem_processor.py",
        (
            "repair_seeded = True\n"
            "execution_mode = 'llmdebug_native'\n"
            "prompt_chars_total = 0\n"
            "completion_chars_total = 0\n"
            "usage_missing_call_count = 0\n"
            "method_instrumentation = True\n"
            "method_two_step_repair = True\n"
            "direct_gen_passed = False\n"
            "final_passed = False\n"
            "stopped_reason = ''\n"
            "total_llm_duration_sec = 0.0\n"
            "total_eval_duration_sec = 0.0\n"
            "build_exact_test_context = object()\n"
            "exact_test_context_meta = {}\n"
        ),
    )
    _write(
        tracecoder / "reporting.py",
        (
            "def build_llmdebug_failure_feedback(*args, **kwargs):\n"
            "    return ''\n\n"
            "def save_metrics_sidecar():\n"
            "    return None\n"
        ),
    )
    _write(
        tracecoder / "trace_learn_coder.py",
        "def main():\n    llmdebug_metrics_path = None\n    return llmdebug_metrics_path\n",
    )
    _write(
        tracecoder / "src" / "generation.py",
        (
            "OPENAI_BASE_URL = 'http://127.0.0.1:8080/v1'\n"
            "TRACECODER_FAKE_RESPONSES_PATH = ''\n"
            "TRACECODER_ANALYSIS_MAX_TOKENS = '2048'\n"
            "TRACECODER_IMPLEMENTATION_MAX_TOKENS = '4096'\n"
            "input_tokens = 0\n"
            "output_tokens = 0\n"
        ),
    )
    _write(tracecoder / "src" / "dataset_loader.py", "def load_dataset():\n    return []\n")
    _write(
        tracecoder / "src" / "exact_test_context.py",
        (
            "TRACECODER_EXACT_TEST_CONTEXT_MAX_CHARS = '3000'\n\n"
            "def build_exact_test_context(*args, **kwargs):\n"
            "    return '', {}\n"
        ),
    )
    _write(
        tracecoder / "src" / "progress.py", "def emit_progress(*args, **kwargs):\n    return None\n"
    )
    _write(tracecoder / "src" / "traceRunner.py", "def main():\n    return 0\n")
    _write(
        tracecoder / "datasets" / "llmdebug" / "evaluation.py",
        "def check_correctness(*args, **kwargs):\n    return {}\n",
    )

    lambda_rlm = root / "lambda-RLM"
    _write(
        lambda_rlm / "pyproject.toml",
        "[project]\nname = 'lambda-rlm'\nversion = '0.1.0'\n",
    )
    _write(lambda_rlm / "rlm" / "__init__.py", "__all__ = ['main']\n")

    rlm = root / "rlm"
    _write(rlm / "pyproject.toml", "[project]\nname = 'rlms'\nversion = '0.1.0'\n")
    _write(rlm / "rlm" / "__init__.py", "__all__ = ['RLM']\n")
    _write(rlm / "rlm" / "core" / "rlm.py", "class RLM:\n    pass\n")
    _write(rlm / "rlm" / "environments" / "local_repl.py", "class LocalREPL:\n    pass\n")
    _write(
        rlm / "rlm" / "clients" / "openai.py",
        "class OpenAIClient:\n    def __init__(self, *args, **kwargs):\n        pass\n",
    )


def test_verify_baseline_repos_accepts_minimal_valid_layout(tmp_path: Path) -> None:
    _build_valid_repo_root(tmp_path)

    result = verify_baseline_repos(tmp_path)

    assert set(result) == {"LLMDebugger", "INTERVENOR", "TraceCoder", "lambda-RLM", "rlm"}
    assert result["TraceCoder"] >= 1


def test_verify_baseline_repos_rejects_tracecoder_syntax_errors(tmp_path: Path) -> None:
    _build_valid_repo_root(tmp_path)
    _write(
        tmp_path / "TraceCoder" / "trace_learn_coder.py",
        'llmdebug_metrics_path = None\nEXPERIMENT_ID = f"oops" OUTPUT_CSV_FILENAME = "broken.csv"\n',
    )

    with pytest.raises(
        BaselineRepoVerificationError,
        match=r"TraceCoder: syntax verification failed for trace_learn_coder\.py",
    ):
        verify_baseline_repos(tmp_path)


def test_setup_repos_script_invokes_repo_verifier() -> None:
    script = (Path(__file__).resolve().parents[1] / "baselines" / "setup_repos.sh").read_text(
        encoding="utf-8"
    )

    assert "verify_repos.py" in script
    assert "from baselines.pins import PINNED_REPOS" in script


def test_patch_manifest_lists_expected_governed_artifacts() -> None:
    relative_paths = {entry.relative_path for entry in PATCH_MANIFEST}
    assert "baselines/ldb/patches/0001-add-llmdebug-dataset-support.patch" in relative_paths
    assert "baselines/tracecoder/patches/0001-add-llmdebug-repair-mode.patch" in relative_paths
    assert "baselines/lambda_rlm/env_client.py" in relative_paths
    assert "baselines/rlm/env_client.py" in relative_paths
