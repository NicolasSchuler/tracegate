from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _arg_value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def test_dockerfile_llama_defaults_to_repro_preset() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.llama").read_text(encoding="utf-8")

    assert "ENV LLAMA_PRESET=qwen35-coding-repro" in dockerfile
    assert "LLAMA_NO_MMAP=1" in dockerfile
    assert "LLAMA_UBATCH_SIZE=2048" in dockerfile
    assert "COPY --chmod=755" not in dockerfile
    assert "RUN chmod 755 /usr/local/bin/llama-entrypoint.sh" in dockerfile


def test_dockerfile_llama_targets_ada_by_default() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.llama").read_text(encoding="utf-8")

    assert "ARG GGML_CUDA_FA_ALL_QUANTS=OFF" in dockerfile
    assert '-DCMAKE_CUDA_ARCHITECTURES="89-real"' in dockerfile


def test_dockerfile_llama_blackwell_targets_blackwell_defaults() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.llama.blackwell").read_text(encoding="utf-8")

    assert "ENV LLAMA_PRESET=qwen35-coding-repro" in dockerfile
    assert "LLAMA_NO_MMAP=1" in dockerfile
    assert "LLAMA_UBATCH_SIZE=2048" in dockerfile
    assert "ARG GGML_CUDA_FA_ALL_QUANTS=ON" in dockerfile
    assert '-DCMAKE_CUDA_ARCHITECTURES="120a-real"' in dockerfile
    assert "COPY --chmod=755" not in dockerfile
    assert "RUN chmod 755 /usr/local/bin/llama-entrypoint.sh" in dockerfile


def test_llama_entrypoint_repro_preset_applies_repeatable_qwen_defaults(tmp_path: Path) -> None:
    fake_server = tmp_path / "llama-server"
    fake_server.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)
    fake_nproc = tmp_path / "nproc"
    fake_nproc.write_text("#!/usr/bin/env bash\nprintf '32\\n'\n", encoding="utf-8")
    fake_nproc.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["LLAMA_PRESET"] = "qwen35-coding-repro"
    env["LLAMA_NO_MMAP"] = "1"
    env["LLAMA_UBATCH_SIZE"] = "2048"

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "llama-entrypoint.sh"), "-m", "model.gguf"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    argv = result.stdout.splitlines()

    assert _arg_value(argv, "--flash-attn") == "on"
    assert _arg_value(argv, "--parallel") == "4"
    assert _arg_value(argv, "--ctx-size") == "1048576"
    assert _arg_value(argv, "--seed") == "42"
    assert _arg_value(argv, "--reasoning") == "on"
    assert _arg_value(argv, "--reasoning-format") == "deepseek"
    assert _arg_value(argv, "--temp") == "0.0"
    assert _arg_value(argv, "--top-p") == "1.0"
    assert _arg_value(argv, "--top-k") == "1"
    assert _arg_value(argv, "-t") == "8"
    assert _arg_value(argv, "--threads-batch") == "32"
    assert _arg_value(argv, "-ub") == "2048"
    assert "--swa-full" in argv
    assert "--kv-unified" not in argv
    assert "--no-mmap" in argv


def test_llama_entrypoint_explicit_thread_env_overrides_auto_defaults(tmp_path: Path) -> None:
    fake_server = tmp_path / "llama-server"
    fake_server.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)
    fake_nproc = tmp_path / "nproc"
    fake_nproc.write_text("#!/usr/bin/env bash\nprintf '32\\n'\n", encoding="utf-8")
    fake_nproc.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["LLAMA_PRESET"] = "qwen35-coding-repro"
    env["LLAMA_THREADS"] = "12"
    env["LLAMA_THREADS_BATCH"] = "24"
    env["LLAMA_THREADS_HTTP"] = "6"
    env["LLAMA_NO_MMAP"] = "1"
    env["LLAMA_UBATCH_SIZE"] = "2048"

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "llama-entrypoint.sh"), "-m", "model.gguf"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    argv = result.stdout.splitlines()

    assert _arg_value(argv, "-t") == "12"
    assert _arg_value(argv, "--threads-batch") == "24"
    assert _arg_value(argv, "--threads-http") == "6"
    assert _arg_value(argv, "-ub") == "2048"
    assert "--kv-unified" not in argv
    assert "--no-mmap" in argv


def test_llama_entrypoint_can_disable_swa_full(tmp_path: Path) -> None:
    fake_server = tmp_path / "llama-server"
    fake_server.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)
    fake_nproc = tmp_path / "nproc"
    fake_nproc.write_text("#!/usr/bin/env bash\nprintf '32\\n'\n", encoding="utf-8")
    fake_nproc.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["LLAMA_PRESET"] = "qwen35-coding-repro"
    env["LLAMA_SWA_FULL"] = "0"
    env["LLAMA_NO_MMAP"] = "1"
    env["LLAMA_UBATCH_SIZE"] = "2048"

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "llama-entrypoint.sh"), "-m", "model.gguf"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    argv = result.stdout.splitlines()

    assert "--swa-full" not in argv
