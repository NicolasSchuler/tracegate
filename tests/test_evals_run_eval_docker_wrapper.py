"""Tests for the Docker-based eval runner shell wrapper script."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest

from evals.docker_image_fingerprint import compute_eval_image_build_fingerprint
from llmdebug import _json as json

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "evals" / "docker" / "run_eval_docker.sh"
CURRENT_BUILD_FINGERPRINT = compute_eval_image_build_fingerprint(
    REPO_ROOT,
    include_baselines=False,
)
CURRENT_BASELINE_BUILD_FINGERPRINT = compute_eval_image_build_fingerprint(
    REPO_ROOT,
    include_baselines=True,
)


def _install_docker_stub(bin_dir: Path) -> None:
    stub = bin_dir / "docker"
    stub.write_text(
        """#!/usr/bin/env python3
from llmdebug import _json as json
import os
import signal
import sys
import time

log_path = os.environ.get("DOCKER_STUB_LOG", "")
if log_path:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")

args = sys.argv[1:]
if len(args) >= 2 and args[0] == "image" and args[1] == "inspect":
    rc = int(os.environ.get("DOCKER_STUB_IMAGE_INSPECT_RC", "0"))
    if rc != 0:
        raise SystemExit(rc)
    if "--format" in args:
        image_name = args[-1] if args else ""
        if image_name == os.environ.get(
            "DOCKER_STUB_BASELINE_IMAGE_NAME",
            "llmdebug-eval-baselines:latest",
        ):
            sys.stdout.write(os.environ.get("DOCKER_STUB_BASELINE_IMAGE_INSPECT_OUTPUT", ""))
        else:
            sys.stdout.write(os.environ.get("DOCKER_STUB_IMAGE_INSPECT_OUTPUT", ""))
    raise SystemExit(0)
if args and args[0] == "build":
    raise SystemExit(int(os.environ.get("DOCKER_STUB_BUILD_RC", "0")))
if args and args[0] == "create":
    is_probe = any("/repo/evals/results/.perm_probe" in item for item in args)
    code_var = "DOCKER_STUB_PROBE_RC" if is_probe else "DOCKER_STUB_CREATE_RC"
    sys.stdout.write(os.environ.get("DOCKER_STUB_CONTAINER_ID", "fake-container-id") + "\\n")
    raise SystemExit(int(os.environ.get(code_var, "0")))
if args and args[0] == "start":
    signal_name = os.environ.get("DOCKER_STUB_START_SIGNAL", "")
    if signal_name:
        os.kill(os.getppid(), getattr(signal, signal_name))
        time.sleep(float(os.environ.get("DOCKER_STUB_START_SIGNAL_DELAY_SEC", "0.2")))
    raise SystemExit(int(os.environ.get("DOCKER_STUB_START_RC", "0")))
if args and args[0] == "rm":
    raise SystemExit(0)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)


def _install_uname_stub(bin_dir: Path) -> None:
    stub = bin_dir / "uname"
    stub.write_text(
        """#!/usr/bin/env bash
printf "%s\\n" "${DOCKER_STUB_UNAME:-Linux}"
""",
        encoding="utf-8",
    )
    stub.chmod(0o755)


def _arg_value(args: list[str], key: str) -> str | None:
    for index, arg in enumerate(args):
        if arg == key:
            if index + 1 < len(args):
                return args[index + 1]
            return None
        if arg.startswith(f"{key}="):
            return arg.split("=", 1)[1]
    return None


def _arg_values(args: list[str], key: str) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == key:
            if index + 1 < len(args):
                out.append(args[index + 1])
            index += 2
            continue
        if arg.startswith(f"{key}="):
            out.append(arg.split("=", 1)[1])
        index += 1
    return out


def _run_wrapper(
    *,
    tmp_path: Path,
    args: list[str],
    uname_value: str,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "docker_calls.jsonl"
    _install_docker_stub(bin_dir)
    _install_uname_stub(bin_dir)

    env = os.environ.copy()
    for key in (
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLMDEBUG_EVAL_OPENAI_API_KEY",
        "Z_API",
        "DOCKER_EXTRA_ARGS",
    ):
        env.pop(key, None)

    env.update(
        {
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            "DOCKER_STUB_LOG": str(log_path),
            "DOCKER_STUB_UNAME": uname_value,
            "DOCKER_STUB_IMAGE_INSPECT_RC": "0",
            "DOCKER_STUB_IMAGE_INSPECT_OUTPUT": CURRENT_BUILD_FINGERPRINT,
            "DOCKER_STUB_BASELINE_IMAGE_INSPECT_OUTPUT": CURRENT_BASELINE_BUILD_FINGERPRINT,
            "DOCKER_STUB_BASELINE_IMAGE_NAME": "llmdebug-eval-baselines:latest",
            "DOCKER_STUB_RUN_RC": "0",
            "LLMDEBUG_EVAL_DOCKER_USER": "1000:1000",
        }
    )
    if env_overrides:
        env.update(env_overrides)

    proc = subprocess.run(
        ["bash", str(RUN_SCRIPT), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    calls: list[list[str]] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            calls.append(list(payload["argv"]))
    return proc, calls


def _main_run_call(calls: list[list[str]]) -> list[str]:
    create_calls = [call for call in calls if call and call[0] == "create"]
    if create_calls:
        return create_calls[-1]
    run_calls = [call for call in calls if call and call[0] == "run"]
    assert run_calls
    return run_calls[-1]


def _main_build_call(calls: list[list[str]]) -> list[str]:
    build_calls = [call for call in calls if call and call[0] == "build"]
    assert build_calls
    return build_calls[-1]


def _build_calls(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if call and call[0] == "build"]


def _split_docker_and_eval_args(
    docker_run_call: list[str], *, image: str | None = None
) -> tuple[list[str], list[str]]:
    assert docker_run_call and docker_run_call[0] in {"create", "run"}
    image_index = docker_run_call.index("-m") - 1 if image is None else docker_run_call.index(image)
    docker_args = docker_run_call[1:image_index]
    eval_args = docker_run_call[image_index + 1 :]
    return docker_args, eval_args


def test_run_eval_docker_forwards_openai_autoswitch_flags_and_forces_local_executor(
    tmp_path: Path,
) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "openai",
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://127.0.0.1:1234/v1",
            "--openai-response-mode",
            "edits",
            "--openai-response-mode-autoswitch",
            "retry_to_diff",
            "--openai-auto-context-window",
            "--openai-force-patch-attempt",
            "--executor",
            "testcontainers",
            "--container-network",
            "none",
            "--k",
            "3",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, eval_args = _split_docker_and_eval_args(_main_run_call(calls))
    assert _arg_value(docker_args, "--network") == "host"
    assert "--openai-auto-context-window" in eval_args
    assert _arg_value(eval_args, "--openai-response-mode-autoswitch") == "retry_to_diff"
    assert _arg_value(eval_args, "--openai-base-url") == "http://127.0.0.1:1234/v1"
    assert "--container-network" not in eval_args
    assert "testcontainers" not in eval_args
    assert _arg_value(eval_args, "--executor") == "local"
    assert _arg_value(eval_args, "--container-image") == "llmdebug-eval:latest"


def test_run_eval_docker_forwards_no_openai_http2_flag(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "openai",
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://127.0.0.1:1234/v1",
            "--no-openai-http2",
            "--k",
            "1",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    _, eval_args = _split_docker_and_eval_args(_main_run_call(calls))
    assert "--no-openai-http2" in eval_args


def test_run_eval_docker_allocates_tty_when_requested(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "null",
            "--k",
            "1",
        ],
        env_overrides={"LLMDEBUG_EVAL_DOCKER_TTY": "1"},
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, _ = _split_docker_and_eval_args(_main_run_call(calls))
    assert "--tty" in docker_args


def test_run_eval_docker_allocates_tty_by_default(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "null",
            "--k",
            "1",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, _ = _split_docker_and_eval_args(_main_run_call(calls))
    assert "--tty" in docker_args


def test_run_eval_docker_can_disable_tty_explicitly(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "null",
            "--k",
            "1",
        ],
        env_overrides={"LLMDEBUG_EVAL_DOCKER_TTY": "0"},
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, _ = _split_docker_and_eval_args(_main_run_call(calls))
    assert "--tty" not in docker_args


def test_run_eval_docker_translates_localhost_url_on_darwin(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Darwin",
        args=[
            "--patcher",
            "openai",
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://localhost:1234/v1",
            "--k",
            "1",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, eval_args = _split_docker_and_eval_args(_main_run_call(calls))
    assert not _arg_values(docker_args, "--network")
    assert _arg_value(eval_args, "--openai-base-url") == "http://host.docker.internal:1234/v1"
    assert "Translating localhost URL for Docker access" in proc.stderr


def test_run_eval_docker_uses_openai_env_defaults_with_key_precedence(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "openai",
            "--k",
            "1",
        ],
        env_overrides={
            "OPENAI_MODEL": "env-model",
            "OPENAI_BASE_URL": "http://127.0.0.1:9999/v1",
            "OPENAI_API_KEY": "openai-key",
            "LLMDEBUG_EVAL_OPENAI_API_KEY": "llmdebug-key",
            "Z_API": "z-key",
        },
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, eval_args = _split_docker_and_eval_args(_main_run_call(calls))
    assert _arg_value(eval_args, "--openai-model") == "env-model"
    assert _arg_value(eval_args, "--openai-base-url") == "http://127.0.0.1:9999/v1"
    assert _arg_value(eval_args, "--openai-api-key") == "z-key"
    assert _arg_value(docker_args, "--network") == "host"


def test_run_eval_docker_uses_repo_openai_base_url_default_when_unset(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "openai",
            "--openai-model",
            "env-model",
            "--k",
            "1",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, eval_args = _split_docker_and_eval_args(_main_run_call(calls))
    assert _arg_value(eval_args, "--openai-base-url") == "http://127.0.0.1:12000/v1"
    assert _arg_value(docker_args, "--network") == "host"


def test_run_eval_docker_keeps_explicit_network_from_docker_extra_args(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "openai",
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://localhost:1234/v1",
            "--k",
            "1",
        ],
        env_overrides={"DOCKER_EXTRA_ARGS": "--network bridge"},
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, _ = _split_docker_and_eval_args(_main_run_call(calls))
    assert _arg_values(docker_args, "--network") == ["bridge"]


def test_run_eval_docker_forwards_case_list_and_transport_retry_flags(tmp_path: Path) -> None:
    case_list_path = tmp_path / "failed_cases.txt"
    case_list_path.write_text("case_a\ncase_b\n", encoding="utf-8")

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--patcher",
            "openai",
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://127.0.0.1:1234/v1",
            "--case-list-file",
            str(case_list_path),
            "--openai-transport-max-retries",
            "9",
            "--openai-transport-retry-backoff-base-sec",
            "1.5",
            "--openai-transport-retry-backoff-max-sec",
            "12.0",
            "--k",
            "1",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    _, eval_args = _split_docker_and_eval_args(_main_run_call(calls))
    rewritten_case_list = _arg_value(eval_args, "--case-list-file")
    assert rewritten_case_list is not None
    assert rewritten_case_list.startswith("/repo/evals/artifacts/case_list_mounts/")
    assert rewritten_case_list.endswith("/001_failed_cases.txt")
    assert _arg_value(eval_args, "--openai-transport-max-retries") == "9"
    assert _arg_value(eval_args, "--openai-transport-retry-backoff-base-sec") == "1.5"
    assert _arg_value(eval_args, "--openai-transport-retry-backoff-max-sec") == "12.0"


def test_run_eval_docker_rewrites_config_path_for_container(tmp_path: Path) -> None:
    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "null"
conditions = "base_repair"
k = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--config",
            str(config_path),
            "--run-id",
            "docker-config-path-rewrite",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    _, eval_args = _split_docker_and_eval_args(_main_run_call(calls))
    rewritten_path = _arg_value(eval_args, "--config")
    assert rewritten_path is not None
    assert rewritten_path.startswith("/repo/evals/artifacts/config_mounts/")
    assert str(config_path) not in eval_args


def test_run_eval_docker_cleans_up_container_and_staged_config_mounts(
    tmp_path: Path,
) -> None:
    config_root = REPO_ROOT / "evals" / "artifacts" / "config_mounts"
    before = set(config_root.iterdir()) if config_root.exists() else set()

    config_path = tmp_path / "cleanup.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "null"
k = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--config",
            str(config_path),
            "--run-id",
            "docker-cleanup",
        ],
    )

    assert proc.returncode == 0, proc.stderr
    create_calls = [call for call in calls if call and call[0] == "create"]
    start_calls = [call for call in calls if call and call[0] == "start"]
    assert create_calls, calls
    assert start_calls, calls
    rm_calls = [call for call in calls if call and call[0] == "rm"]
    assert rm_calls, calls
    assert "-f" in rm_calls[-1], rm_calls[-1]
    assert "-v" in rm_calls[-1], rm_calls[-1]

    after = set(config_root.iterdir()) if config_root.exists() else set()
    assert after == before


@pytest.mark.parametrize(
    ("signal_name", "expected_returncode"),
    [
        ("SIGINT", 130),
        ("SIGTERM", 143),
        ("SIGHUP", 129),
        ("SIGQUIT", 131),
    ],
)
def test_run_eval_docker_stops_container_on_interrupt(
    tmp_path: Path,
    signal_name: str,
    expected_returncode: int,
) -> None:
    config_path = tmp_path / "interrupt.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "null"
k = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--config",
            str(config_path),
            "--run-id",
            "docker-interrupt",
        ],
        env_overrides={
            "DOCKER_STUB_START_SIGNAL": signal_name,
            "DOCKER_STUB_START_SIGNAL_DELAY_SEC": "0.1",
        },
    )

    expected_negative_returncode = -getattr(signal, signal_name).value
    assert proc.returncode in {expected_returncode, expected_negative_returncode}, proc.stderr
    assert any(call and call[0] == "start" for call in calls), calls
    assert any(call and call[0] == "rm" for call in calls), calls


@pytest.mark.parametrize(
    ("inspect_output", "expected_note"),
    [
        ("", "missing a build fingerprint label"),
        ("stale-fingerprint", "does not match current repo contents"),
    ],
)
def test_run_eval_docker_rebuilds_unlabeled_or_stale_image(
    tmp_path: Path,
    inspect_output: str,
    expected_note: str,
) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--patcher", "null", "--k", "1"],
        env_overrides={"DOCKER_STUB_IMAGE_INSPECT_OUTPUT": inspect_output},
    )

    assert proc.returncode == 0, proc.stderr
    build_call = _main_build_call(calls)
    assert f"LLMDEBUG_EVAL_BUILD_FINGERPRINT={CURRENT_BUILD_FINGERPRINT}" in _arg_values(
        build_call, "--build-arg"
    )
    assert expected_note in proc.stderr


def test_run_eval_docker_no_cache_env_forces_no_cache_build_flag(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--patcher", "null", "--k", "1"],
        env_overrides={
            "DOCKER_STUB_IMAGE_INSPECT_RC": "1",
            "LLMDEBUG_EVAL_BUILD_NO_CACHE": "1",
        },
    )

    assert proc.returncode == 0, proc.stderr
    build_call = _main_build_call(calls)
    assert "--no-cache" in build_call


def test_run_eval_docker_no_cache_env_rebuilds_fresh_image(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--patcher", "null", "--k", "1"],
        env_overrides={"LLMDEBUG_EVAL_BUILD_NO_CACHE": "1"},
    )

    assert proc.returncode == 0, proc.stderr
    build_call = _main_build_call(calls)
    assert "--no-cache" in build_call
    assert "no-cache rebuild requested" in proc.stdout


def test_run_eval_docker_reuses_fresh_image_without_rebuild(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--patcher", "null", "--k", "1"],
    )

    assert proc.returncode == 0, proc.stderr
    assert not _build_calls(calls)


def test_run_eval_docker_baseline_shorthand_selects_baseline_image_and_build_arg(
    tmp_path: Path,
) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--baseline",
            "tracecoder",
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://127.0.0.1:1234/v1",
            "--k",
            "1",
        ],
        env_overrides={"DOCKER_STUB_IMAGE_INSPECT_RC": "1"},
    )

    assert proc.returncode == 0, proc.stderr
    build_call = _main_build_call(calls)
    assert "--build-arg" in build_call
    assert "LLMDEBUG_EVAL_INCLUDE_BASELINES=1" in build_call
    assert "llmdebug-eval-baselines:latest" in build_call

    docker_run_call = _main_run_call(calls)
    docker_args, eval_args = _split_docker_and_eval_args(
        docker_run_call,
        image="llmdebug-eval-baselines:latest",
    )
    assert _arg_value(docker_args, "--network") == "host"
    assert _arg_value(eval_args, "--baseline") == "tracecoder"
    assert _arg_value(eval_args, "--executor") == "local"


def test_run_eval_docker_baseline_uses_openai_env_defaults_with_key_precedence(
    tmp_path: Path,
) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--baseline",
            "ldb",
            "--k",
            "1",
        ],
        env_overrides={
            "OPENAI_MODEL": "env-model",
            "OPENAI_BASE_URL": "http://127.0.0.1:9999/v1",
            "OPENAI_API_KEY": "openai-key",
            "LLMDEBUG_EVAL_OPENAI_API_KEY": "llmdebug-key",
            "Z_API": "z-key",
        },
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, eval_args = _split_docker_and_eval_args(
        _main_run_call(calls),
        image="llmdebug-eval-baselines:latest",
    )
    assert _arg_value(eval_args, "--openai-model") == "env-model"
    assert _arg_value(eval_args, "--openai-base-url") == "http://127.0.0.1:9999/v1"
    assert _arg_value(eval_args, "--openai-api-key") == "z-key"
    assert _arg_value(docker_args, "--network") == "host"


def test_run_eval_docker_baseline_config_selects_baseline_image(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(
        """
[run_eval]
baseline = "intervenor"
k = 1

[run_eval.openai]
base_url = "http://127.0.0.1:1234/v1"
model = "cfg-model"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--config",
            str(config_path),
        ],
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, eval_args = _split_docker_and_eval_args(
        _main_run_call(calls),
        image="llmdebug-eval-baselines:latest",
    )
    assert _arg_value(docker_args, "--network") == "host"
    assert _arg_value(eval_args, "--openai-model") == "cfg-model"
    assert _arg_value(eval_args, "--openai-base-url") == "http://127.0.0.1:1234/v1"


def test_run_eval_docker_config_normalizes_dashed_openai_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline-dashed.toml"
    config_path.write_text(
        """
[run_eval]
baseline = "ldb"
k = 1

[run_eval.openai]
base-url = "http://remote:9999/v1"
model = "cfg-model"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--config", str(config_path)],
    )

    assert proc.returncode == 0, proc.stderr
    docker_args, eval_args = _split_docker_and_eval_args(
        _main_run_call(calls),
        image="llmdebug-eval-baselines:latest",
    )
    assert not _arg_values(docker_args, "--network")
    assert _arg_value(eval_args, "--openai-model") == "cfg-model"
    assert _arg_value(eval_args, "--openai-base-url") == "http://remote:9999/v1"


def test_run_eval_docker_uses_configured_openai_api_key_env(tmp_path: Path) -> None:
    config_path = tmp_path / "baseline-key-env.toml"
    config_path.write_text(
        """
[run_eval]
baseline = "intervenor"
openai-api-key-env = "MY_KEY"
k = 1

[run_eval.openai]
base-url = "http://remote:9999/v1"
model = "cfg-model"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--config", str(config_path)],
        env_overrides={"MY_KEY": "custom-secret"},
    )

    assert proc.returncode == 0, proc.stderr
    _docker_args, eval_args = _split_docker_and_eval_args(
        _main_run_call(calls),
        image="llmdebug-eval-baselines:latest",
    )
    assert _arg_value(eval_args, "--openai-api-key") == "custom-secret"


def test_run_eval_docker_config_openai_api_key_env_uses_standard_fallback_chain(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "openai-key-env-fallback.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "openai"
openai_api_key_env = "OPENAI_API_KEY"
k = 1

[run_eval.openai]
base_url = "http://remote:9999/v1"
model = "cfg-model"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--config", str(config_path)],
        env_overrides={"Z_API": "z-secret-token"},
    )

    assert proc.returncode == 0, proc.stderr
    _docker_args, eval_args = _split_docker_and_eval_args(
        _main_run_call(calls),
        image="llmdebug-eval:latest",
    )
    assert _arg_value(eval_args, "--openai-api-key") == "z-secret-token"


def test_run_eval_docker_config_openai_api_key_env_defaults_to_local_when_missing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "openai-key-env-default.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "openai"
openai_api_key_env = "OPENAI_API_KEY"
k = 1

[run_eval.openai]
base_url = "http://remote:9999/v1"
model = "cfg-model"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--config", str(config_path)],
    )

    assert proc.returncode == 0, proc.stderr
    _docker_args, eval_args = _split_docker_and_eval_args(
        _main_run_call(calls),
        image="llmdebug-eval:latest",
    )
    assert _arg_value(eval_args, "--openai-api-key") == "local"


def test_run_eval_docker_cli_patcher_override_disables_config_baseline_mode(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "baseline-override.toml"
    config_path.write_text(
        """
[run_eval]
patcher = "baseline"
baseline_name = "ldb"
baseline_repo_dir = "/tmp/custom-baseline"
baseline_python = "/tmp/custom-python"
k = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=["--config", str(config_path), "--patcher", "null"],
    )

    assert proc.returncode == 0, proc.stderr
    _docker_args, eval_args = _split_docker_and_eval_args(
        _main_run_call(calls),
        image="llmdebug-eval:latest",
    )
    assert _arg_value(eval_args, "--patcher") == "null"


def test_run_eval_docker_baseline_respects_explicit_image_override(tmp_path: Path) -> None:
    proc, calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--baseline",
            "intervenor",
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://127.0.0.1:1234/v1",
            "--k",
            "1",
        ],
        env_overrides={
            "DOCKER_STUB_IMAGE_INSPECT_RC": "1",
            "LLMDEBUG_EVAL_IMAGE": "custom-eval:baseline",
        },
    )

    assert proc.returncode == 0, proc.stderr
    build_call = _main_build_call(calls)
    assert "custom-eval:baseline" in build_call
    assert "LLMDEBUG_EVAL_INCLUDE_BASELINES=1" in build_call
    assert "custom-eval:baseline" in _main_run_call(calls)


def test_run_eval_docker_rejects_custom_baseline_repo_dir(tmp_path: Path) -> None:
    proc, _calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--baseline",
            "ldb",
            "--baseline-repo-dir",
            str(tmp_path / "custom_repo"),
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://127.0.0.1:1234/v1",
        ],
    )

    assert proc.returncode == 2
    assert "custom --baseline-repo-dir" in proc.stderr


def test_run_eval_docker_rejects_custom_baseline_python(tmp_path: Path) -> None:
    proc, _calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--baseline",
            "ldb",
            "--baseline-python",
            str(tmp_path / "python"),
            "--openai-model",
            "qwen-demo",
            "--openai-base-url",
            "http://127.0.0.1:1234/v1",
        ],
    )

    assert proc.returncode == 2
    assert "custom --baseline-python" in proc.stderr


def test_run_eval_docker_reports_missing_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"
    proc, _calls = _run_wrapper(
        tmp_path=tmp_path,
        uname_value="Linux",
        args=[
            "--config",
            str(missing_path),
        ],
    )

    assert proc.returncode == 2
    assert "Config file not found" in proc.stderr
