"""Tests for isolated baseline runtime env wiring."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.adapter_base import BaselineConfig
from baselines.intervenor.adapter import IntervenorAdapter
from baselines.ldb.adapter import LDBAdapter
from evals import baseline_runtime, run_eval


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_fake_python(path: Path, *, version: str, log_path: Path | None = None) -> Path:
    log_literal = repr(str(log_path)) if log_path is not None else "None"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f"#!{sys.executable}\n"
            "import ast\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            "def _write_python(path_str: str) -> None:\n"
            "    path = pathlib.Path(path_str)\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text('#!/bin/sh\\nexit 0\\n', encoding='utf-8')\n"
            "    path.chmod(0o755)\n"
            "\n"
            "def _project_dependencies(repo_path: pathlib.Path) -> list[str]:\n"
            "    for raw_line in (repo_path / 'pyproject.toml').read_text(encoding='utf-8').splitlines():\n"
            "        if raw_line.startswith('dependencies = '):\n"
            "            parsed = ast.literal_eval(raw_line.split('=', 1)[1].strip())\n"
            "            if isinstance(parsed, list):\n"
            "                return [str(item).strip() for item in parsed if str(item).strip()]\n"
            "            return []\n"
            "    return []\n"
            "\n"
            "def _append_log(entry: str) -> None:\n"
            f"    log_target = {log_literal}\n"
            "    if not log_target:\n"
            "        return\n"
            "    log_path = pathlib.Path(log_target)\n"
            "    log_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    with log_path.open('a', encoding='utf-8') as handle:\n"
            "        handle.write(entry + '\\n')\n"
            "\n"
            "def main(argv: list[str]) -> int:\n"
            "    if len(argv) >= 2 and argv[1] == '-c':\n"
            "        script = argv[2]\n"
            "        if 'sys.version_info' in script:\n"
            f"            print({version!r})\n"
            "            return 0\n"
            "        if 'sys.executable' in script and 'parent.parent / \"lib\"' in script:\n"
            "            print(pathlib.Path(argv[0]).resolve().parent.parent / 'lib')\n"
            "            return 0\n"
            "        raise SystemExit(f'unsupported fake python -c invocation: {argv!r}')\n"
            "    if len(argv) >= 3 and argv[1] == '-':\n"
            "        repo_path = pathlib.Path(argv[2])\n"
            "        for dependency in _project_dependencies(repo_path):\n"
            "            print(dependency)\n"
            "        return 0\n"
            "    if len(argv) >= 7 and argv[1:6] == ['-m', 'venv', '--clear', '--copies', '--without-pip']:\n"
            "        _append_log(' '.join(argv[1:]))\n"
            "        _write_python(str(pathlib.Path(argv[6]) / 'bin' / 'python'))\n"
            "        return 0\n"
            "    raise SystemExit(f'unsupported fake python invocation: {argv!r}')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main(sys.argv))\n"
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_fake_python_with_runtime_lib_dependency(
    path: Path,
    *,
    version: str,
    log_path: Path | None = None,
) -> Path:
    log_literal = repr(str(log_path)) if log_path is not None else "None"
    version_tag = ".".join(version.split(".")[:2])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f"#!{sys.executable}\n"
            "import ast\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            "def _write_python(path_str: str) -> None:\n"
            "    path = pathlib.Path(path_str)\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text(\n"
            "        '#!/bin/sh\\n'\n"
            '        \'lib_dir=$(CDPATH= cd -- "$(dirname "$0")/../lib" && pwd)\\n\'\n'
            f"        'if [ ! -e \"$lib_dir/libpython{version_tag}.so.1.0\" ]; then\\n'\n"
            f"        '  echo \"$0: error while loading shared libraries: libpython{version_tag}.so.1.0: cannot open shared object file: No such file or directory\" >&2\\n'\n"
            "        '  exit 127\\n'\n"
            "        'fi\\n'\n"
            "        'exit 0\\n',\n"
            "        encoding='utf-8',\n"
            "    )\n"
            "    path.chmod(0o755)\n"
            "\n"
            "def _project_dependencies(repo_path: pathlib.Path) -> list[str]:\n"
            "    for raw_line in (repo_path / 'pyproject.toml').read_text(encoding='utf-8').splitlines():\n"
            "        if raw_line.startswith('dependencies = '):\n"
            "            parsed = ast.literal_eval(raw_line.split('=', 1)[1].strip())\n"
            "            if isinstance(parsed, list):\n"
            "                return [str(item).strip() for item in parsed if str(item).strip()]\n"
            "            return []\n"
            "    return []\n"
            "\n"
            "def _append_log(entry: str) -> None:\n"
            f"    log_target = {log_literal}\n"
            "    if not log_target:\n"
            "        return\n"
            "    log_path = pathlib.Path(log_target)\n"
            "    log_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    with log_path.open('a', encoding='utf-8') as handle:\n"
            "        handle.write(entry + '\\n')\n"
            "\n"
            "def main(argv: list[str]) -> int:\n"
            "    if len(argv) >= 2 and argv[1] == '-c':\n"
            "        script = argv[2]\n"
            "        if 'sys.version_info' in script:\n"
            f"            print({version!r})\n"
            "            return 0\n"
            "        if 'sys.executable' in script and 'parent.parent / \"lib\"' in script:\n"
            "            print(pathlib.Path(argv[0]).resolve().parent.parent / 'lib')\n"
            "            return 0\n"
            "        raise SystemExit(f'unsupported fake python -c invocation: {argv!r}')\n"
            "    if len(argv) >= 3 and argv[1] == '-':\n"
            "        repo_path = pathlib.Path(argv[2])\n"
            "        for dependency in _project_dependencies(repo_path):\n"
            "            print(dependency)\n"
            "        return 0\n"
            "    if len(argv) >= 7 and argv[1:6] == ['-m', 'venv', '--clear', '--copies', '--without-pip']:\n"
            "        _append_log(' '.join(argv[1:]))\n"
            "        _write_python(str(pathlib.Path(argv[6]) / 'bin' / 'python'))\n"
            "        return 0\n"
            "    raise SystemExit(f'unsupported fake python invocation: {argv!r}')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main(sys.argv))\n"
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_fake_uv(
    path: Path,
    *,
    found_pythons: dict[str, Path] | None = None,
    log_path: Path | None = None,
) -> Path:
    found_mapping = {key: str(value) for key, value in (found_pythons or {}).items()}
    log_literal = repr(str(log_path)) if log_path is not None else "None"
    path.write_text(
        (
            f"#!{sys.executable}\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            f"FOUND = {found_mapping!r}\n"
            f"LOG_PATH = {log_literal}\n"
            "\n"
            "def _write_python(path_str: str) -> None:\n"
            "    path = pathlib.Path(path_str)\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    path.write_text('#!/bin/sh\\nexit 0\\n', encoding='utf-8')\n"
            "    path.chmod(0o755)\n"
            "\n"
            "def _append_log(entry: str) -> None:\n"
            "    if not LOG_PATH:\n"
            "        return\n"
            "    log_path = pathlib.Path(LOG_PATH)\n"
            "    log_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    with log_path.open('a', encoding='utf-8') as handle:\n"
            "        handle.write(entry + '\\n')\n"
            "\n"
            "def main(argv: list[str]) -> int:\n"
            "    if len(argv) >= 4 and argv[1:3] == ['python', 'find']:\n"
            "        resolved = FOUND.get(argv[3], '')\n"
            "        if not resolved:\n"
            "            return 1\n"
            "        print(resolved)\n"
            "        return 0\n"
            "    if len(argv) >= 5 and argv[1] == 'pip' and argv[2] == 'install':\n"
            "        _append_log(' '.join(argv[1:]))\n"
            "        python_index = argv.index('--python') + 1\n"
            "        _write_python(argv[python_index])\n"
            "        return 0\n"
            "    raise SystemExit(f'unsupported uv invocation: {argv!r}')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main(sys.argv))\n"
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _make_baseline_repo(repo_root: Path, repo_name: str) -> Path:
    repo_dir = repo_root / "baselines" / "repos" / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    return repo_dir


def _write_minimal_project_pyproject(
    repo_dir: Path,
    *,
    name: str,
    version: str = "0.1.0",
    dependencies: list[str] | None = None,
) -> None:
    dependency_block = ", ".join(repr(item) for item in (dependencies or []))
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "pyproject.toml").write_text(
        (
            "[project]\n"
            f"name = {name!r}\n"
            f"version = {version!r}\n"
            f"dependencies = [{dependency_block}]\n"
        ),
        encoding="utf-8",
    )


def _script_test_path(*candidate_dirs: Path) -> str:
    tool_dir = candidate_dirs[0].parent / "_script_test_tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    for tool_name in ("cat", "cp", "dirname", "mkdir"):
        source = shutil.which(tool_name)
        if source is None:
            raise AssertionError(f"required shell utility not found on PATH: {tool_name}")
        target = tool_dir / tool_name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

    ordered_entries: list[str] = []
    for entry in [*(str(path) for path in candidate_dirs), str(tool_dir)]:
        if entry not in ordered_entries:
            ordered_entries.append(entry)
    return ":".join(ordered_entries)


def _script_test_env(*candidate_dirs: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = _script_test_path(*candidate_dirs)
    for key in ("BASELINE_ENV_PYTHON", "BASELINE_REPO_ROOT", "BASELINE_ENV_MODE"):
        env.pop(key, None)
    return env


def test_make_patcher_auto_detects_baseline_env_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _make_baseline_repo(repo_root, "LLMDebugger")
    env_python = _write_executable(repo_root / "baselines" / "envs" / "ldb" / "bin" / "python")

    captured: dict[str, object] = {}

    def _fake_build_baseline_patcher(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="baseline",
        baseline_name="ldb",
        baseline_repo_dir=str(repo_root / "baselines" / "repos" / "LLMDebugger"),
        baseline_python="",
        baseline_process_timeout_sec=None,
        openai_model="test-model",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    run_eval.make_patcher(args)

    assert captured["python_executable"] == str(env_python)


def test_make_patcher_lambda_rlm_forwards_baseline_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _make_baseline_repo(repo_root, "lambda-RLM")
    env_python = _write_executable(
        repo_root / "baselines" / "envs" / "lambda_rlm" / "bin" / "python"
    )

    captured: dict[str, object] = {}

    def _fake_build_baseline_patcher(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="baseline",
        baseline="",
        baseline_name="lambda_rlm",
        baseline_repo_dir=str(repo_root / "baselines" / "repos" / "lambda-RLM"),
        baseline_python="",
        baseline_process_timeout_sec=None,
        baseline_lambda_context_window_chars=12345,
        baseline_lambda_accuracy_target=0.91,
        baseline_lambda_a_leaf=0.96,
        baseline_lambda_a_compose=0.93,
        baseline_lambda_verbose=True,
        baseline_lambda_temperature=0.2,
        baseline_lambda_top_p=0.8,
        baseline_lambda_max_tokens=512,
        baseline_lambda_stream=False,
        baseline_lambda_timeout_sec=77,
        baseline_lambda_force_task_type="qa",
        baseline_openai_compat_profile="qwen_openai_raw",
        baseline_openai_allow_reasoning_fallback=True,
        baseline_openai_enable_thinking=False,
        openai_model="test-model",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    run_eval.make_patcher(args)

    assert captured["baseline_name"] == "lambda_rlm"
    assert captured["python_executable"] == str(env_python)
    assert captured["extra"] == {
        "baseline_hybrid_guidance_mode": "tuned_v1",
        "baseline_openai_compat_profile": "qwen_openai_raw",
        "baseline_openai_allow_reasoning_fallback": True,
        "baseline_openai_enable_thinking": False,
        "baseline_lambda_context_window_chars": 12345,
        "baseline_lambda_accuracy_target": 0.91,
        "baseline_lambda_a_leaf": 0.96,
        "baseline_lambda_a_compose": 0.93,
        "baseline_lambda_verbose": True,
        "baseline_lambda_stream": False,
        "baseline_lambda_timeout_sec": 77,
        "baseline_lambda_force_task_type": "qa",
        "baseline_lambda_temperature": 0.2,
        "baseline_lambda_top_p": 0.8,
        "baseline_lambda_max_tokens": 512,
        "baseline_lambda_controller_guidance_max_chars": 2500,
        "baseline_lambda_failure_feedback_max_chars": 3500,
        "baseline_lambda_failure_feedback_meta_max_chars": 2000,
    }


def test_make_patcher_rlm_forwards_baseline_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _make_baseline_repo(repo_root, "rlm")
    env_python = _write_executable(repo_root / "baselines" / "envs" / "rlm" / "bin" / "python")

    captured: dict[str, object] = {}

    def _fake_build_baseline_patcher(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="baseline",
        baseline="",
        baseline_name="rlm",
        baseline_repo_dir=str(repo_root / "baselines" / "repos" / "rlm"),
        baseline_python="",
        baseline_process_timeout_sec=None,
        baseline_rlm_max_depth=3,
        baseline_rlm_max_iterations=10,
        baseline_rlm_timeout_sec=77,
        baseline_rlm_max_errors=6,
        baseline_rlm_max_tokens=512,
        baseline_rlm_max_total_tokens=4000,
        baseline_rlm_compaction=True,
        baseline_rlm_compaction_threshold_pct=0.7,
        baseline_rlm_temperature=0.2,
        baseline_rlm_top_p=0.8,
        baseline_rlm_verbose=True,
        baseline_openai_compat_profile="qwen_dashscope",
        baseline_openai_allow_reasoning_fallback=False,
        baseline_openai_enable_thinking=False,
        openai_model="test-model",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    run_eval.make_patcher(args)

    assert captured["baseline_name"] == "rlm"
    assert captured["python_executable"] == str(env_python)
    assert captured["extra"] == {
        "baseline_hybrid_guidance_mode": "tuned_v1",
        "baseline_openai_compat_profile": "qwen_dashscope",
        "baseline_openai_allow_reasoning_fallback": False,
        "baseline_openai_enable_thinking": False,
        "baseline_rlm_max_depth": 3,
        "baseline_rlm_max_iterations": 10,
        "baseline_rlm_timeout_sec": 77,
        "baseline_rlm_max_errors": 6,
        "baseline_rlm_compaction": True,
        "baseline_rlm_compaction_threshold_pct": 0.7,
        "baseline_rlm_verbose": True,
        "baseline_rlm_temperature": 0.2,
        "baseline_rlm_top_p": 0.8,
        "baseline_rlm_max_tokens": 512,
        "baseline_rlm_max_total_tokens": 4000,
        "baseline_rlm_controller_guidance_max_chars": 2500,
        "baseline_rlm_failure_feedback_max_chars": 3500,
        "baseline_rlm_failure_feedback_meta_max_chars": 2000,
    }


def test_make_patcher_tracecoder_forwards_shared_openai_compat_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _make_baseline_repo(repo_root, "TraceCoder")
    _write_executable(repo_root / "baselines" / "envs" / "tracecoder" / "bin" / "python")

    captured: dict[str, object] = {}

    def _fake_build_baseline_patcher(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="baseline",
        baseline="",
        baseline_name="tracecoder",
        baseline_repo_dir=str(repo_root / "baselines" / "repos" / "TraceCoder"),
        baseline_python="",
        baseline_process_timeout_sec=None,
        baseline_openai_compat_profile="lmstudio_model_yaml",
        baseline_openai_allow_reasoning_fallback=False,
        baseline_openai_enable_thinking=False,
        openai_model="test-model",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    run_eval.make_patcher(args)

    assert captured["baseline_name"] == "tracecoder"
    assert captured["extra"] == {
        "baseline_hybrid_guidance_mode": "tuned_v1",
        "baseline_openai_compat_profile": "lmstudio_model_yaml",
        "baseline_openai_allow_reasoning_fallback": False,
        "baseline_openai_enable_thinking": False,
        "baseline_tracecoder_analysis_max_tokens": 4096,
        "baseline_tracecoder_controller_guidance_max_chars": 2000,
        "baseline_tracecoder_exact_test_context_max_chars": 3000,
        "baseline_tracecoder_failure_feedback_max_chars": 6000,
        "baseline_tracecoder_failure_feedback_tail_max_chars": 2000,
        "baseline_tracecoder_implementation_max_tokens": 4096,
        "baseline_tracecoder_instrumentation_output_max_chars": 12000,
        "baseline_tracecoder_instrumentation_output_max_lines": 80,
        "baseline_tracecoder_max_no_improvement_streak": 2,
        "baseline_tracecoder_problem_summary_max_chars": 4000,
    }


def test_make_patcher_baseline_shorthand_auto_fills_repo_dir_and_env_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _make_baseline_repo(repo_root, "LLMDebugger")
    env_python = _write_executable(repo_root / "baselines" / "envs" / "ldb" / "bin" / "python")

    captured: dict[str, object] = {}

    def _fake_build_baseline_patcher(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="command",
        baseline="ldb",
        baseline_name="",
        baseline_repo_dir="",
        baseline_python="",
        baseline_process_timeout_sec=None,
        openai_model="test-model",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    run_eval.make_patcher(args)

    assert captured["baseline_name"] == "ldb"
    assert captured["repo_dir"] == str(repo_root / "baselines" / "repos" / "LLMDebugger")
    assert captured["python_executable"] == str(env_python)


def test_make_patcher_forwards_split_baseline_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _make_baseline_repo(repo_root, "INTERVENOR")
    _write_executable(repo_root / "baselines" / "envs" / "intervenor" / "bin" / "python")
    captured: dict[str, object] = {}

    def _fake_build_baseline_patcher(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="baseline",
        baseline="",
        baseline_name="intervenor",
        baseline_repo_dir=str(repo_root / "baselines" / "repos" / "INTERVENOR"),
        baseline_python="",
        baseline_process_timeout_sec=45.0,
        openai_model="",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    run_eval.make_patcher(args)

    assert captured["api_timeout_sec"] == 30.0
    assert captured["process_timeout_sec"] == 45.0
    # D4: the deprecated timeout alias became the shared generation contract.
    generation = captured["generation"]
    assert generation.temperature == 0.0
    assert generation.max_completion_tokens == 2048


def test_validate_external_baseline_runtime_accepts_auto_detected_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_dir = _make_baseline_repo(repo_root, "LLMDebugger")
    env_python = _write_executable(repo_root / "baselines" / "envs" / "ldb" / "bin" / "python")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)

    resolved = run_eval._validate_external_baseline_runtime_requirements(
        baseline_name="ldb",
        baseline_repo_dir=str(repo_dir),
        baseline_python="",
    )

    assert resolved == str(env_python)


def test_validate_external_baseline_runtime_accepts_explicit_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_dir = _make_baseline_repo(repo_root, "TraceCoder")
    explicit_python = _write_executable(tmp_path / "custom-env" / "bin" / "python")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)

    resolved = run_eval._validate_external_baseline_runtime_requirements(
        baseline_name="tracecoder",
        baseline_repo_dir=str(repo_dir),
        baseline_python=str(explicit_python),
    )

    assert resolved == str(explicit_python)


def test_validate_external_baseline_runtime_rejects_missing_repo_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    env_python = _write_executable(repo_root / "baselines" / "envs" / "ldb" / "bin" / "python")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match=r"setup_repos\.sh"):
        run_eval._validate_external_baseline_runtime_requirements(
            baseline_name="ldb",
            baseline_repo_dir=str(repo_root / "baselines" / "repos" / "LLMDebugger"),
            baseline_python="",
        )

    assert env_python.is_file()


def test_validate_external_baseline_runtime_rejects_missing_default_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_dir = _make_baseline_repo(repo_root, "INTERVENOR")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match=r"setup_envs\.sh"):
        run_eval._validate_external_baseline_runtime_requirements(
            baseline_name="intervenor",
            baseline_repo_dir=str(repo_dir),
            baseline_python="",
        )


def test_validate_external_baseline_runtime_rejects_missing_explicit_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_dir = _make_baseline_repo(repo_root, "lambda-RLM")
    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match=r"setup_envs\.sh"):
        run_eval._validate_external_baseline_runtime_requirements(
            baseline_name="lambda_rlm",
            baseline_repo_dir=str(repo_dir),
            baseline_python=str(tmp_path / "missing-python"),
        )


def test_make_patcher_rejects_missing_baseline_env_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_dir = _make_baseline_repo(repo_root, "TraceCoder")
    build_calls = 0

    def _fake_build_baseline_patcher(**_kwargs):
        nonlocal build_calls
        build_calls += 1
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="baseline",
        baseline="",
        baseline_name="tracecoder",
        baseline_repo_dir=str(repo_dir),
        baseline_python="",
        baseline_process_timeout_sec=None,
        openai_model="test-model",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    with pytest.raises(SystemExit, match=r"setup_envs\.sh"):
        run_eval.make_patcher(args)

    assert build_calls == 0


def test_make_patcher_rewrites_stale_repo_local_baseline_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_dir = _make_baseline_repo(repo_root, "TraceCoder")
    env_python = _write_executable(
        repo_root / "baselines" / "envs" / "tracecoder" / "bin" / "python"
    )
    stale_root = tmp_path / "stale-checkout"

    captured: dict[str, object] = {}

    def _fake_build_baseline_patcher(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(
        "evals.patchers.baseline_patcher.build_baseline_patcher",
        _fake_build_baseline_patcher,
    )

    args = argparse.Namespace(
        patcher="baseline",
        baseline="",
        baseline_name="tracecoder",
        baseline_repo_dir=str(stale_root / "baselines" / "repos" / "TraceCoder"),
        baseline_python=str(stale_root / "baselines" / "envs" / "tracecoder" / "bin" / "python"),
        baseline_process_timeout_sec=None,
        baseline_openai_compat_profile="lmstudio_model_yaml",
        baseline_openai_allow_reasoning_fallback=False,
        baseline_openai_enable_thinking=False,
        openai_model="test-model",
        openai_base_url="http://127.0.0.1:1234",
        openai_api_key="",
        k=2,
        openai_timeout_sec=30.0,
    )

    run_eval.make_patcher(args)

    assert args.baseline_repo_dir == str(repo_dir)
    assert args.baseline_python == str(env_python)
    assert captured["repo_dir"] == str(repo_dir)
    assert captured["python_executable"] == str(env_python)


def test_resolve_external_runtime_uses_manifest_repo_dir_when_default_repo_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_repo = tmp_path / "external-repos" / "lambda-RLM"
    external_repo.mkdir(parents=True)
    env_python = _write_executable(
        repo_root / "baselines" / "envs" / "lambda_rlm" / "bin" / "python"
    )
    manifest_path = repo_root / "baselines" / "envs" / "runtime_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "lambda_rlm": {
                    "python_executable": str(env_python),
                    "python_repo_relative": "baselines/envs/lambda_rlm/bin/python",
                    "repo_dir": str(external_repo),
                    "repo_dir_repo_relative": "baselines/repos/lambda-RLM",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_eval, "_REPO_ROOT", repo_root)

    monkeypatch.setattr(baseline_runtime, "_REPO_ROOT", repo_root)

    resolved_repo, resolved_python = run_eval._resolve_external_baseline_runtime_requirements(
        baseline_name="lambda_rlm",
        baseline_repo_dir=str(repo_root / "baselines" / "repos" / "lambda-RLM"),
        baseline_python="",
    )

    assert resolved_repo == str(external_repo)
    assert resolved_python == str(env_python)


def test_setup_envs_script_parses() -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"

    result = subprocess.run(
        ["bash", "-n", str(script)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_setup_envs_script_skips_missing_optional_lambda_rlm_repo(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"
    venv_log = tmp_path / "venv-log.txt"
    _write_fake_python(tmp_path / "python3.12", version="3.12.8", log_path=venv_log)
    _write_fake_uv(tmp_path / "uv")

    env_root = tmp_path / "envs"
    repo_root = tmp_path / "repos"
    repo_root.mkdir()
    env = _script_test_env(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(script), str(env_root), str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert (env_root / "ldb" / "bin" / "python").is_file()
    assert (env_root / "intervenor" / "bin" / "python").is_file()
    assert (env_root / "tracecoder" / "bin" / "python").is_file()
    assert not (env_root / "lambda_rlm" / "bin" / "python").exists()
    assert not (env_root / "rlm" / "bin" / "python").exists()
    assert "Skipping optional controller backend env" in result.stdout
    assert "Skipping optional standalone baseline env" in result.stdout

    manifest = json.loads((env_root / "runtime_manifest.json").read_text(encoding="utf-8"))
    venv_invocations = venv_log.read_text(encoding="utf-8").splitlines()
    assert venv_invocations == [
        "-m venv --clear --copies --without-pip " + str(env_root / "ldb"),
        "-m venv --clear --copies --without-pip " + str(env_root / "intervenor"),
        "-m venv --clear --copies --without-pip " + str(env_root / "tracecoder"),
    ]
    assert (env_root / "ldb" / "lib" / "python3.12" / "site-packages" / "vllm.py").read_text(
        encoding="utf-8"
    ) == (
        (
            Path(__file__).resolve().parents[1] / "baselines" / "stubs" / "vllm_stub" / "vllm.py"
        ).read_text(encoding="utf-8")
    )
    assert manifest["creation_backend"] == "python_venv_copies"
    assert manifest["ldb"]["python_executable"] == str(env_root / "ldb" / "bin" / "python")
    assert manifest["ldb"]["python_repo_relative"] == "baselines/envs/ldb/bin/python"
    assert manifest["intervenor"]["python_executable"] == str(
        env_root / "intervenor" / "bin" / "python"
    )
    assert manifest["intervenor"]["python_repo_relative"] == "baselines/envs/intervenor/bin/python"
    assert manifest["tracecoder"]["python_executable"] == str(
        env_root / "tracecoder" / "bin" / "python"
    )
    assert manifest["tracecoder"]["python_repo_relative"] == "baselines/envs/tracecoder/bin/python"
    assert "lambda_rlm" not in manifest
    assert "rlm" not in manifest


def test_setup_envs_script_repairs_copied_python_runtime_libs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"
    venv_log = tmp_path / "venv-log.txt"
    _write_fake_python_with_runtime_lib_dependency(
        tmp_path / "bin" / "python3.12",
        version="3.12.8",
        log_path=venv_log,
    )
    _write_fake_uv(tmp_path / "bin" / "uv")

    runtime_lib = tmp_path / "lib" / "libpython3.12.so.1.0"
    runtime_lib.parent.mkdir(parents=True, exist_ok=True)
    runtime_lib.write_text("stub runtime lib\n", encoding="utf-8")

    env_root = tmp_path / "envs"
    repo_root = tmp_path / "repos"
    repo_root.mkdir()
    env = _script_test_env(tmp_path / "bin")

    result = subprocess.run(
        ["/bin/bash", str(script), str(env_root), str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert (env_root / "ldb" / "lib" / "libpython3.12.so.1.0").read_text(encoding="utf-8") == (
        "stub runtime lib\n"
    )
    assert (env_root / "intervenor" / "lib" / "libpython3.12.so.1.0").read_text(
        encoding="utf-8"
    ) == "stub runtime lib\n"
    assert (env_root / "tracecoder" / "lib" / "libpython3.12.so.1.0").read_text(
        encoding="utf-8"
    ) == "stub runtime lib\n"
    assert venv_log.read_text(encoding="utf-8").splitlines() == [
        "-m venv --clear --copies --without-pip " + str(env_root / "ldb"),
        "-m venv --clear --copies --without-pip " + str(env_root / "intervenor"),
        "-m venv --clear --copies --without-pip " + str(env_root / "tracecoder"),
    ]


def test_setup_envs_script_rejects_python39_for_smoke_envs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"
    python39 = _write_fake_python(tmp_path / "python3", version="3.9.18")
    _write_fake_uv(tmp_path / "uv")

    env_root = tmp_path / "envs"
    repo_root = tmp_path / "repos"
    repo_root.mkdir()
    env = _script_test_env(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(script), str(env_root), str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "requires Python >=3.10" in result.stderr
    assert str(python39) in result.stderr


def test_setup_envs_script_prefers_python311_for_lambda_rlm(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"
    _write_fake_python(tmp_path / "python3", version="3.9.18")
    python311 = _write_fake_python(tmp_path / "python3.11", version="3.11.9")
    _write_fake_uv(tmp_path / "uv")

    env_root = tmp_path / "envs"
    repo_root = tmp_path / "repos"
    _write_minimal_project_pyproject(repo_root / "lambda-RLM", name="lambda-rlm")
    env = _script_test_env(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(script), str(env_root), str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"Venv python: {python311}" in result.stdout
    assert (env_root / "lambda_rlm" / "bin" / "python").is_file()


def test_setup_envs_script_uses_uv_python311_for_lambda_rlm(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"
    _write_fake_python(tmp_path / "python3", version="3.10.14")
    uv_python311 = _write_fake_python(tmp_path / "uv-python3.11", version="3.11.11")
    _write_fake_uv(tmp_path / "uv", found_pythons={"3.11": uv_python311})

    env_root = tmp_path / "envs"
    repo_root = tmp_path / "repos"
    _write_minimal_project_pyproject(repo_root / "lambda-RLM", name="lambda-rlm")
    env = _script_test_env(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(script), str(env_root), str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"Venv python: {uv_python311}" in result.stdout
    assert (env_root / "lambda_rlm" / "bin" / "python").is_file()


def test_setup_envs_script_rejects_python39_for_lambda_rlm(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"
    python39 = _write_fake_python(tmp_path / "python3", version="3.9.18")
    _write_fake_uv(tmp_path / "uv")

    env_root = tmp_path / "envs"
    repo_root = tmp_path / "repos"
    (repo_root / "lambda-RLM").mkdir(parents=True)
    env = _script_test_env(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(script), str(env_root), str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "requires Python >=3.11" in result.stderr
    assert str(python39) in result.stderr


def test_setup_envs_script_installs_lambda_rlm_and_rlm_runtime_dependencies(
    tmp_path: Path,
) -> None:
    script = Path(__file__).resolve().parents[1] / "baselines" / "setup_envs.sh"
    uv_log = tmp_path / "uv-log.txt"
    _write_fake_python(tmp_path / "python3.11", version="3.11.9")
    _write_fake_uv(tmp_path / "uv", log_path=uv_log)

    env_root = tmp_path / "envs"
    repo_root = tmp_path / "repos"
    lambda_repo = repo_root / "lambda-RLM"
    rlm_repo = repo_root / "rlm"
    _write_minimal_project_pyproject(
        lambda_repo,
        name="lambda-rlm",
        dependencies=["requests>=2.0", "rich>=13.0"],
    )
    _write_minimal_project_pyproject(
        rlm_repo,
        name="rlms",
        dependencies=["openai>=2.0", "python-dotenv>=1.0"],
    )
    env = _script_test_env(tmp_path)

    result = subprocess.run(
        ["/bin/bash", str(script), str(env_root), str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    uv_invocations = uv_log.read_text(encoding="utf-8").splitlines()
    assert (
        f"pip install --python {env_root / 'lambda_rlm' / 'bin' / 'python'} requests>=2.0 rich>=13.0"
        in uv_invocations
    )
    assert (
        f"pip install --python {env_root / 'rlm' / 'bin' / 'python'} openai>=2.0 python-dotenv>=1.0"
        in uv_invocations
    )
    assert all("--no-build-isolation" not in invocation for invocation in uv_invocations)
    assert all(str(lambda_repo) not in invocation for invocation in uv_invocations)
    assert all(str(rlm_repo) not in invocation for invocation in uv_invocations)


def test_intervenor_env_client_path_uses_configured_python(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf \'%s\' \'{"text":"patched","usage":{"prompt_tokens":3,"completion_tokens":4}}\' > "$3"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    adapter = IntervenorAdapter()
    config = BaselineConfig(
        repo_dir=tmp_path,
        python_executable=fake_python,
        timeout_sec=10,
        api_base_url="http://127.0.0.1:9999",
        api_key="sk-test",
    )

    text, usage, duration_sec, meta = adapter._generate_text(
        client=None,
        model="test-model",
        prompt="hello",
        max_tokens=32,
        config=config,
    )

    assert text == "patched"
    assert usage == {"prompt_tokens": 3, "completion_tokens": 4}
    assert duration_sec >= 0.0
    assert meta == {}


def test_intervenor_env_client_path_uses_api_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    adapter = IntervenorAdapter()
    config = BaselineConfig(
        repo_dir=tmp_path,
        python_executable=fake_python,
        api_timeout_sec=7,
        process_timeout_sec=21,
        timeout_sec=None,
        api_base_url="http://127.0.0.1:9999",
        api_key="sk-test",
    )

    observed: dict[str, object] = {}

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        cwd: str,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        observed["timeout"] = timeout
        request_path = Path(cmd[2])
        response_path = Path(cmd[3])
        observed["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
        response_path.write_text(
            json.dumps({"text": "patched", "usage": {"prompt_tokens": 3, "completion_tokens": 4}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("baselines.intervenor.adapter.subprocess.run", _fake_run)

    text, usage, _, meta = adapter._generate_text(
        client=None,
        model="test-model",
        prompt="hello",
        max_tokens=32,
        config=config,
    )

    assert observed["timeout"] == 7
    assert isinstance(observed["payload"], dict)
    assert observed["payload"]["timeout_sec"] == 7
    assert text == "patched"
    assert usage == {"prompt_tokens": 3, "completion_tokens": 4}
    assert meta == {}


def test_ldb_adapter_uses_configured_python_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "fake_ldb"
    programming = repo / "programming"
    programming.mkdir(parents=True)
    (programming / "main.py").write_text("# stub\n", encoding="utf-8")
    (programming / "ldb.py").write_text('DATASET = "llmdebug"\n', encoding="utf-8")
    (programming / "utils.py").write_text('DATASET = "llmdebug"\n', encoding="utf-8")

    adapter = LDBAdapter()
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    config = BaselineConfig(
        repo_dir=repo,
        python_executable=fake_python,
        model="test-model",
        timeout_sec=5,
    )
    adapter._config = config
    observed: dict[str, object] = {}

    import subprocess

    def _fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        cwd: str,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        observed["cmd"] = cmd
        run_name = cmd[cmd.index("--run_name") + 1]
        root_dir = Path(cmd[cmd.index("--root_dir") + 1])
        log_dir = root_dir / run_name
        log_dir.mkdir(parents=True)
        (log_dir / "results.jsonl").write_text(
            json.dumps({"solution": "def add(a, b):\n    return a + b\n", "is_solved": True})
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    from evals.subprocess_monitor import MonitoredSubprocessResult

    def _fake_monitored_run(
        *,
        cmd: list[str],
        cwd: Path | str | None,
        env: dict[str, str] | None,
        timeout_sec: int,
        **_kwargs,
    ) -> MonitoredSubprocessResult:
        completed = _fake_run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout_sec),
            cwd=str(cwd),
            env=dict(env or {}),
        )
        return MonitoredSubprocessResult(
            returncode=completed.returncode,
            duration_sec=0.01,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    monkeypatch.setattr("baselines.ldb.adapter.run_monitored_subprocess", _fake_monitored_run)

    adapter.run_repair(
        {
            "task_id": "llmdebug/my_case",
            "prompt": "def add(a, b):",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "seed": "def add(a, b):\n    return a - b\n",
            "given_tests": ["assert add(1, 2) == 3\n"],
        },
        config,
    )

    cmd = observed["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == str(fake_python)
