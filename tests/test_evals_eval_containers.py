"""Tests for evals.eval_containers - container management helpers.

Covers: _coerce_subprocess_output, check_testcontainers_runtime_available,
_parse_docker_created_timestamp_utc, _container_created_at_utc,
cleanup_stale_eval_containers edge-cases, _EvalContainer, and
_EvalContainerPool.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals import eval_containers
from evals.eval_containers import (
    _coerce_subprocess_output,
    _container_created_at_utc,
    _EvalContainer,
    _EvalContainerPool,
    _parse_docker_created_timestamp_utc,
    check_testcontainers_runtime_available,
    cleanup_stale_eval_containers,
)
from evals.eval_types import (
    EVAL_CONTAINER_LABEL_KEY,
    EVAL_CONTAINER_LABEL_VALUE,
    PYTEST_TIMEOUT_RETURNCODE,
    ExecutionConfig,
    RunResult,
)

# ---------------------------------------------------------------------------
# _coerce_subprocess_output
# ---------------------------------------------------------------------------


class TestCoerceSubprocessOutput:
    """Tests for _coerce_subprocess_output helper."""

    def test_none_returns_empty_string(self) -> None:
        """None input returns empty string."""
        assert _coerce_subprocess_output(None) == ""

    def test_bytes_decoded_utf8(self) -> None:
        """Bytes input is decoded as UTF-8."""
        assert _coerce_subprocess_output(b"hello") == "hello"

    def test_bytes_with_invalid_utf8_replaced(self) -> None:
        """Invalid UTF-8 bytes are replaced rather than raising."""
        result = _coerce_subprocess_output(b"bad\xff byte")
        assert "bad" in result
        assert "byte" in result

    def test_str_passthrough(self) -> None:
        """String input is returned unchanged."""
        assert _coerce_subprocess_output("already a string") == "already a string"


# ---------------------------------------------------------------------------
# _parse_docker_created_timestamp_utc
# ---------------------------------------------------------------------------


class TestParsDockerCreatedTimestamp:
    """Tests for _parse_docker_created_timestamp_utc."""

    def test_empty_string_returns_none(self) -> None:
        """Empty/whitespace input returns None."""
        assert _parse_docker_created_timestamp_utc("") is None
        assert _parse_docker_created_timestamp_utc("   ") is None

    def test_z_suffix_parsed(self) -> None:
        """Timestamps ending in Z are parsed correctly."""
        result = _parse_docker_created_timestamp_utc("2025-01-15T10:30:00Z")
        assert result is not None
        assert result.tzinfo is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.hour == 10

    def test_fractional_seconds_truncated_to_six(self) -> None:
        """Docker nanosecond timestamps (9 digits) are truncated to 6."""
        result = _parse_docker_created_timestamp_utc("2025-01-15T10:30:00.123456789Z")
        assert result is not None
        assert result.microsecond == 123456

    def test_fractional_seconds_with_tz_offset(self) -> None:
        """Fractional seconds with explicit timezone offset are handled."""
        result = _parse_docker_created_timestamp_utc("2025-01-15T10:30:00.123456789+05:00")
        assert result is not None
        # Converted to UTC: 10:30 - 5:00 = 05:30
        assert result.hour == 5
        assert result.minute == 30

    def test_empty_frac_after_dot(self) -> None:
        """A dot followed immediately by tz offset (empty frac) is handled."""
        result = _parse_docker_created_timestamp_utc("2025-01-15T10:30:00.+00:00")
        assert result is not None
        assert result.hour == 10

    def test_trailing_dot_empty_frac(self) -> None:
        """A trailing dot with no fractional digits hits the else branch."""
        result = _parse_docker_created_timestamp_utc("2025-01-15T10:30:00.")
        assert result is not None
        assert result.hour == 10
        assert result.tzinfo == timezone.utc

    def test_naive_datetime_gets_utc(self) -> None:
        """A naive datetime (no tz info) gets UTC assumed."""
        result = _parse_docker_created_timestamp_utc("2025-01-15T10:30:00")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 10

    def test_invalid_format_returns_none(self) -> None:
        """Completely invalid date strings return None."""
        assert _parse_docker_created_timestamp_utc("not-a-date") is None

    def test_fractional_without_tz_and_no_z(self) -> None:
        """Fractional seconds without timezone and without Z suffix."""
        result = _parse_docker_created_timestamp_utc("2025-06-01T12:00:00.500000")
        assert result is not None
        assert result.microsecond == 500000
        assert result.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# _container_created_at_utc
# ---------------------------------------------------------------------------


class TestContainerCreatedAtUtc:
    """Tests for _container_created_at_utc."""

    def test_no_attrs_returns_none(self) -> None:
        """Container without attrs attribute returns None."""
        obj = object()
        assert _container_created_at_utc(obj) is None

    def test_attrs_not_dict_returns_none(self) -> None:
        """Container with non-dict attrs returns None."""
        obj = types.SimpleNamespace(attrs="not-a-dict")
        assert _container_created_at_utc(obj) is None

    def test_valid_attrs(self) -> None:
        """Container with valid Created timestamp is parsed."""
        obj = types.SimpleNamespace(attrs={"Created": "2025-03-10T08:00:00Z"})
        result = _container_created_at_utc(obj)
        assert result is not None
        assert result.year == 2025
        assert result.month == 3

    def test_missing_created_key(self) -> None:
        """Container attrs dict without Created returns None."""
        obj = types.SimpleNamespace(attrs={})
        result = _container_created_at_utc(obj)
        assert result is None


# ---------------------------------------------------------------------------
# check_testcontainers_runtime_available
# ---------------------------------------------------------------------------


class TestCheckTestcontainersRuntimeAvailable:
    """Tests for check_testcontainers_runtime_available."""

    def test_import_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When docker or testcontainers cannot be imported, returns False."""
        import builtins

        real_import = builtins.__import__

        def _fail_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
            if name == "docker" or name.startswith("testcontainers"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail_import)
        ok, msg = check_testcontainers_runtime_available()
        assert ok is False
        assert "testcontainers runtime is not available" in msg

    def test_docker_ping_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When docker ping fails, returns False with daemon message."""
        import builtins

        fake_docker = types.ModuleType("docker")
        fake_client = MagicMock()
        fake_client.ping.side_effect = ConnectionError("refused")
        fake_docker.from_env = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]

        fake_tc_container = types.ModuleType("testcontainers.core.container")
        fake_tc_container.DockerContainer = object  # type: ignore[attr-defined]

        real_import = builtins.__import__

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
            if name == "docker":
                return fake_docker
            if name == "testcontainers.core.container":
                return fake_tc_container
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        ok, msg = check_testcontainers_runtime_available()
        assert ok is False
        assert "docker daemon is not reachable" in msg

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When everything works, returns (True, 'ok')."""
        import builtins

        fake_docker = types.ModuleType("docker")
        fake_client = MagicMock()
        fake_client.ping.return_value = True
        fake_docker.from_env = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]

        fake_tc_container = types.ModuleType("testcontainers.core.container")
        fake_tc_container.DockerContainer = object  # type: ignore[attr-defined]

        real_import = builtins.__import__

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
            if name == "docker":
                return fake_docker
            if name == "testcontainers.core.container":
                return fake_tc_container
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        ok, msg = check_testcontainers_runtime_available()
        assert ok is True
        assert msg == "ok"


# ---------------------------------------------------------------------------
# cleanup_stale_eval_containers - extra edge-cases
# ---------------------------------------------------------------------------


class _FakeContainer:
    """Minimal fake Docker container for cleanup tests."""

    def __init__(
        self,
        *,
        status: str,
        created: str | None,
        remove_error: Exception | None = None,
    ) -> None:
        self.status = status
        self.attrs: dict[str, Any] = {"Created": created} if created is not None else {}
        self._remove_error = remove_error

    def remove(self, *, force: bool, v: bool) -> None:
        if self._remove_error is not None:
            raise self._remove_error


class _FakeDockerClient:
    """Minimal fake Docker client."""

    def __init__(
        self,
        containers: list[_FakeContainer] | None = None,
        *,
        list_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._containers = containers or []
        self._list_error = list_error
        self._close_error = close_error
        self.containers = self  # type: ignore[assignment]
        self.closed = False

    def list(self, *, all: bool, filters: dict[str, Any]) -> list[_FakeContainer]:
        if self._list_error is not None:
            raise self._list_error
        return list(self._containers)

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class TestCleanupStaleEvalContainersEdge:
    """Edge-case tests for cleanup_stale_eval_containers."""

    def test_list_raises_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When client.containers.list raises, returns error tuple."""
        client = _FakeDockerClient(list_error=RuntimeError("api error"))
        monkeypatch.setattr(eval_containers, "_docker_client_from_env", lambda: client)
        removed, failed, error = cleanup_stale_eval_containers(grace_sec=300)
        assert removed == 0
        assert failed == 0
        assert error is not None
        assert "unable to inspect containers" in error

    def test_close_error_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When client.close() raises, the error is swallowed."""
        stale = _FakeContainer(status="exited", created="2025-01-01T00:00:00Z")
        client = _FakeDockerClient([stale], close_error=OSError("socket error"))
        monkeypatch.setattr(eval_containers, "_docker_client_from_env", lambda: client)
        now = datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)
        removed, _failed, error = cleanup_stale_eval_containers(grace_sec=60, now_utc=now)
        assert removed == 1
        assert error is None

    def test_grace_zero_removes_all_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With grace_sec=0, all stopped containers are removed."""
        c1 = _FakeContainer(status="exited", created="2025-06-01T00:00:00Z")
        c2 = _FakeContainer(status="exited", created="2025-06-01T23:59:59Z")
        client = _FakeDockerClient([c1, c2])
        monkeypatch.setattr(eval_containers, "_docker_client_from_env", lambda: client)
        now = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        removed, _failed, error = cleanup_stale_eval_containers(grace_sec=0, now_utc=now)
        # grace==0 means the `if created_at is not None and grace > 0` branch
        # is False, so containers are removed regardless of age.
        assert removed == 2
        assert error is None


# ---------------------------------------------------------------------------
# _EvalContainer
# ---------------------------------------------------------------------------


class TestEvalContainer:
    """Tests for the _EvalContainer wrapper."""

    @staticmethod
    def _make_container(
        exec_return: tuple[int, Any] = (0, (b"ok", b"")),
    ) -> tuple[MagicMock, _EvalContainer]:
        """Create a fake docker container and _EvalContainer wrapper."""
        raw = MagicMock()
        raw.exec_run.return_value = exec_return
        workdirs = Path("/host/workdirs")
        wrapper = _EvalContainer(raw, workdirs)
        return raw, wrapper

    def test_exec_pytest_normal_run(self) -> None:
        """Normal execution returns RunResult with correct fields."""
        raw, wrapper = self._make_container(exec_return=(0, (b"PASSED", b"")))
        result = wrapper.exec_pytest(
            host_cwd=Path("/host/workdirs/case1"),
            command=["pytest", "test_a.py"],
            env={"PYTHONPATH": "/repo_src"},
            timeout_sec=60,
        )
        assert isinstance(result, RunResult)
        assert result.returncode == 0
        assert result.stdout == "PASSED"
        assert result.timed_out is False
        assert result.timeout_sec is None
        # Verify the command passed to exec_run
        call_kwargs = raw.exec_run.call_args
        assert call_kwargs[1]["cmd"] == ["timeout", "-k", "5", "60", "pytest", "test_a.py"]
        assert call_kwargs[1]["workdir"] == "/workdirs/case1"

    def test_exec_pytest_timeout(self) -> None:
        """When exit code is 124, result is marked as timed out."""
        _raw, wrapper = self._make_container(
            exec_return=(PYTEST_TIMEOUT_RETURNCODE, (b"", b"some err"))
        )
        result = wrapper.exec_pytest(
            host_cwd=Path("/host/workdirs/case2"),
            command=["pytest"],
            env={},
            timeout_sec=30,
        )
        assert result.timed_out is True
        assert result.timeout_sec == 30
        assert "pytest timed out after 30s" in result.stderr

    def test_exec_pytest_timeout_no_prior_stderr(self) -> None:
        """When timed out with empty stderr, only timeout message appears."""
        _raw, wrapper = self._make_container(exec_return=(PYTEST_TIMEOUT_RETURNCODE, (b"", b"")))
        result = wrapper.exec_pytest(
            host_cwd=Path("/host/workdirs/case3"),
            command=["pytest"],
            env={},
            timeout_sec=10,
        )
        assert result.timed_out is True
        assert result.stderr == "pytest timed out after 10s"

    def test_exec_pytest_non_tuple_output(self) -> None:
        """When exec_run returns non-tuple output, it becomes stdout."""
        _raw, wrapper = self._make_container(exec_return=(1, b"raw output"))
        result = wrapper.exec_pytest(
            host_cwd=Path("/host/workdirs/case4"),
            command=["pytest"],
            env={},
            timeout_sec=60,
        )
        assert result.stdout == "raw output"
        assert result.stderr == ""

    def test_exec_pytest_host_cwd_not_under_workdirs(self) -> None:
        """Raises RuntimeError if host_cwd is not under workdirs_host."""
        _raw, wrapper = self._make_container()
        with pytest.raises(RuntimeError, match=r"is not under workdirs_host"):
            wrapper.exec_pytest(
                host_cwd=Path("/somewhere/else"),
                command=["pytest"],
                env={},
                timeout_sec=60,
            )

    def test_exec_pytest_nested_workdir(self) -> None:
        """Nested host_cwd is correctly mapped to guest path."""
        raw, wrapper = self._make_container(exec_return=(0, (b"", b"")))
        result = wrapper.exec_pytest(
            host_cwd=Path("/host/workdirs/a/b/c"),
            command=["pytest"],
            env={},
            timeout_sec=60,
        )
        call_kwargs = raw.exec_run.call_args
        assert call_kwargs[1]["workdir"] == "/workdirs/a/b/c"
        assert result.returncode == 0

    def test_stop_calls_stop_and_remove(self) -> None:
        """stop() calls container.stop() and container.remove()."""
        raw, wrapper = self._make_container()
        wrapper.stop()
        raw.stop.assert_called_once_with(timeout=5)
        raw.remove.assert_called_once_with(force=True, v=True)

    def test_stop_swallows_stop_exception(self) -> None:
        """stop() swallows exception from container.stop()."""
        raw, wrapper = self._make_container()
        raw.stop.side_effect = RuntimeError("already stopped")
        wrapper.stop()  # should not raise
        raw.remove.assert_called_once()

    def test_stop_swallows_remove_exception(self) -> None:
        """stop() swallows exception from container.remove()."""
        raw, wrapper = self._make_container()
        raw.remove.side_effect = RuntimeError("already removed")
        wrapper.stop()  # should not raise
        raw.stop.assert_called_once()


# ---------------------------------------------------------------------------
# _EvalContainerPool
# ---------------------------------------------------------------------------


def _make_fake_docker_module(
    *,
    health_exit_code: int = 0,
    run_side_effect: Exception | None = None,
) -> tuple[types.ModuleType, MagicMock]:
    """Build a fake ``docker`` module and client for pool tests."""
    fake_docker = types.ModuleType("docker")
    fake_client = MagicMock()

    fake_raw_container = MagicMock()
    fake_raw_container.exec_run.return_value = (health_exit_code, b"ok")

    if run_side_effect is not None:
        fake_client.containers.run.side_effect = run_side_effect
    else:
        fake_client.containers.run.return_value = fake_raw_container

    fake_docker.from_env = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
    return fake_docker, fake_client


class TestEvalContainerPool:
    """Tests for _EvalContainerPool."""

    def test_init_creates_containers_and_health_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool init creates N containers and runs health check."""
        fake_docker, fake_client = _make_fake_docker_module()
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=False)
        pool = _EvalContainerPool(size=2, execution=execution, workdirs_host=tmp_path / "w")
        assert len(pool._containers) == 2
        assert pool._queue.qsize() == 2
        assert fake_client.containers.run.call_count == 2
        fake_client.close.assert_called_once()
        pool.shutdown()

    def test_init_with_mount_repo_src(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool init includes repo src volume when configured."""
        fake_docker, fake_client = _make_fake_docker_module()
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=True)
        pool = _EvalContainerPool(size=1, execution=execution, workdirs_host=tmp_path / "w")
        # Check volumes passed to run()
        call_kwargs = fake_client.containers.run.call_args
        volumes = call_kwargs[1]["volumes"]
        # Should have both workdirs and repo_src binds
        assert len(volumes) == 2
        pool.shutdown()

    def test_init_health_check_failure_shuts_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool init raises and shuts down on health check failure."""
        fake_docker, _fake_client = _make_fake_docker_module(health_exit_code=127)
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=False)
        with pytest.raises(RuntimeError, match=r"does not have GNU"):
            _EvalContainerPool(
                size=1,
                execution=execution,
                workdirs_host=tmp_path / "w",
            )

    def test_init_run_failure_shuts_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool init shuts down on container run failure."""
        fake_docker, _fake_client = _make_fake_docker_module(
            run_side_effect=RuntimeError("image not found")
        )
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=False)
        with pytest.raises(RuntimeError, match="image not found"):
            _EvalContainerPool(
                size=1,
                execution=execution,
                workdirs_host=tmp_path / "w",
            )

    def test_acquire_returns_container(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """acquire() returns a container from the pool."""
        fake_docker, _fake_client = _make_fake_docker_module()
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=False)
        pool = _EvalContainerPool(size=1, execution=execution, workdirs_host=tmp_path / "w")
        c = pool.acquire(timeout=1.0)
        assert isinstance(c, _EvalContainer)
        assert pool._queue.qsize() == 0
        pool.release(c)
        assert pool._queue.qsize() == 1
        pool.shutdown()

    def test_acquire_timeout_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """acquire() raises RuntimeError after timeout."""
        fake_docker, _fake_client = _make_fake_docker_module()
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=False)
        pool = _EvalContainerPool(size=1, execution=execution, workdirs_host=tmp_path / "w")
        # Drain the pool
        pool.acquire(timeout=0.1)
        with pytest.raises(RuntimeError, match=r"Timed out waiting for a container"):
            pool.acquire(timeout=0.1)
        pool.shutdown()

    def test_release_puts_container_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """release() returns the container to the pool queue."""
        fake_docker, _fake_client = _make_fake_docker_module()
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=False)
        pool = _EvalContainerPool(size=1, execution=execution, workdirs_host=tmp_path / "w")
        c = pool.acquire(timeout=1.0)
        pool.release(c)
        # Can acquire again
        c2 = pool.acquire(timeout=1.0)
        assert c2 is c
        pool.shutdown()

    def test_shutdown_stops_and_clears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """shutdown() stops all containers and clears the list."""
        fake_docker, _fake_client = _make_fake_docker_module()
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        execution = ExecutionConfig(container_mount_repo_src=False)
        pool = _EvalContainerPool(size=2, execution=execution, workdirs_host=tmp_path / "w")
        pool.shutdown()
        assert len(pool._containers) == 0

    def test_workdirs_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pool init creates the workdirs_host directory."""
        fake_docker, _fake_client = _make_fake_docker_module()
        monkeypatch.setitem(sys.modules, "docker", fake_docker)

        workdirs = tmp_path / "deep" / "nested" / "workdirs"
        assert not workdirs.exists()
        execution = ExecutionConfig(container_mount_repo_src=False)
        pool = _EvalContainerPool(size=1, execution=execution, workdirs_host=workdirs)
        assert workdirs.exists()
        pool.shutdown()
