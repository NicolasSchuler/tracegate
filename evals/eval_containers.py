"""Container management for the eval harness (Docker/testcontainers).

Extracted from ``evals.run_eval`` to keep the main orchestrator lean.
Re-exported by ``evals.run_eval`` for backward compatibility.
"""

from __future__ import annotations

import queue
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .eval_types import (
    _CONTAINER_ACTIVE_STATES,
    DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    EVAL_CONTAINER_LABEL_KEY,
    EVAL_CONTAINER_LABEL_VALUE,
    PYTEST_TIMEOUT_RETURNCODE,
    ExecutionConfig,
    RunResult,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _coerce_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def check_testcontainers_runtime_available() -> tuple[bool, str]:
    """Probe whether Docker and testcontainers are usable.

    Returns:
        Tuple of (available, message) where *message* is ``"ok"`` on success.
    """
    try:
        import docker  # type: ignore[import-not-found]
        from testcontainers.core.container import DockerContainer  # noqa: F401
    except Exception as exc:
        return (
            False,
            "testcontainers runtime is not available "
            f"({type(exc).__name__}: {exc}). Install llmdebug[evals].",
        )
    try:
        client = docker.from_env()
        client.ping()
        client.close()
    except Exception as exc:
        return False, f"docker daemon is not reachable ({type(exc).__name__}: {exc})."
    return True, "ok"


def _docker_client_from_env() -> Any:
    import docker  # type: ignore[import-not-found]

    return docker.from_env()


def _parse_docker_created_timestamp_utc(raw: Any) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        frac = tail
        suffix = ""
        tz_pos = max(frac.find("+"), frac.find("-"))
        if tz_pos > 0:
            suffix = frac[tz_pos:]
            frac = frac[:tz_pos]
        if frac:
            frac = frac[:6]
            text = f"{head}.{frac}{suffix}"
        else:
            text = f"{head}{suffix}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _container_created_at_utc(container: Any) -> datetime | None:
    attrs = getattr(container, "attrs", None)
    if not isinstance(attrs, dict):
        return None
    return _parse_docker_created_timestamp_utc(attrs.get("Created"))


def cleanup_stale_eval_containers(
    *,
    grace_sec: int = DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    now_utc: datetime | None = None,
) -> tuple[int, int, str | None]:
    """Remove stopped eval containers older than *grace_sec* seconds.

    Returns:
        Tuple of (removed_count, failed_count, error_message_or_None).
    """
    grace = max(0, int(grace_sec))
    try:
        client = _docker_client_from_env()
    except Exception as exc:
        return 0, 0, f"docker client unavailable ({type(exc).__name__}: {exc})"

    removed = 0
    failed = 0
    now_value = (
        now_utc.astimezone(timezone.utc) if now_utc is not None else datetime.now(timezone.utc)
    )
    label_filter = f"{EVAL_CONTAINER_LABEL_KEY}={EVAL_CONTAINER_LABEL_VALUE}"
    try:
        containers = client.containers.list(
            all=True,
            filters={"label": [label_filter]},
        )
        for container in containers:
            status = str(getattr(container, "status", "")).strip().lower()
            if status in _CONTAINER_ACTIVE_STATES:
                continue
            created_at = _container_created_at_utc(container)
            if created_at is not None and grace > 0:
                age_sec = (now_value - created_at).total_seconds()
                if age_sec < float(grace):
                    continue
            try:
                container.remove(force=True, v=True)
                removed += 1
            except Exception:
                failed += 1
    except Exception as exc:
        return 0, 0, f"unable to inspect containers ({type(exc).__name__}: {exc})"
    finally:
        try:
            client.close()
        except Exception:
            pass
    return removed, failed, None


class _EvalContainer:
    """Wraps a running docker-py container for reuse via ``exec_run``."""

    def __init__(self, container: Any, workdirs_host: Path) -> None:
        self._container = container
        self._workdirs_host = workdirs_host

    def exec_pytest(
        self,
        *,
        host_cwd: Path,
        command: list[str],
        env: dict[str, str],
        timeout_sec: int,
    ) -> RunResult:
        """Run *command* inside the container via ``exec_run``."""
        try:
            guest_rel = host_cwd.relative_to(self._workdirs_host)
        except ValueError as exc:
            raise RuntimeError(
                f"host_cwd {host_cwd} is not under workdirs_host {self._workdirs_host}"
            ) from exc
        guest_workdir = str(PurePosixPath("/workdirs") / guest_rel.as_posix())

        full_cmd = ["timeout", "-k", "5", str(timeout_sec), *command]

        start = time.perf_counter()
        exit_code, output = self._container.exec_run(
            cmd=full_cmd,
            environment=env,
            workdir=guest_workdir,
            demux=True,
        )
        end = time.perf_counter()

        if isinstance(output, tuple):
            stdout_raw, stderr_raw = output
        else:
            stdout_raw, stderr_raw = output, b""
        stdout = _coerce_subprocess_output(stdout_raw)
        stderr = _coerce_subprocess_output(stderr_raw)

        timed_out = exit_code == PYTEST_TIMEOUT_RETURNCODE
        if timed_out:
            timeout_message = f"pytest timed out after {timeout_sec}s"
            stderr = timeout_message if not stderr else f"{stderr}\n{timeout_message}"

        return RunResult(
            returncode=int(exit_code),
            duration_sec=float(end - start),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            timeout_sec=timeout_sec if timed_out else None,
        )

    def stop(self) -> None:
        """Stop and remove the container."""
        try:
            self._container.stop(timeout=5)
        except Exception:
            pass
        try:
            self._container.remove(force=True, v=True)
        except Exception:
            pass


class _EvalContainerPool:
    """Thread-safe pool of reusable Docker containers for eval pytest runs."""

    def __init__(
        self,
        *,
        size: int,
        execution: ExecutionConfig,
        workdirs_host: Path,
    ) -> None:
        import docker  # type: ignore[import-not-found]

        self._workdirs_host = workdirs_host
        workdirs_host.mkdir(parents=True, exist_ok=True)

        client: Any = docker.from_env()
        tmpfs_size = max(1, int(execution.container_tmpfs_size_mb))

        volumes: dict[str, dict[str, str]] = {
            str(workdirs_host): {"bind": "/workdirs", "mode": "rw"},
        }
        if execution.container_mount_repo_src:
            volumes[str(_SRC_ROOT)] = {"bind": "/repo_src", "mode": "ro"}

        self._containers: list[_EvalContainer] = []
        self._queue: queue.Queue[_EvalContainer] = queue.Queue()

        try:
            for _ in range(size):
                raw = client.containers.run(
                    execution.container_image,
                    command=["sleep", "infinity"],
                    detach=True,
                    read_only=True,
                    network_mode=execution.container_network,
                    tmpfs={"/tmp": f"rw,noexec,nosuid,nodev,size={tmpfs_size}m"},
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges:true"],
                    pids_limit=int(execution.container_pids_limit),
                    mem_limit=execution.container_memory,
                    nano_cpus=int(float(execution.container_cpus) * 1_000_000_000),
                    user=execution.container_user,
                    labels={EVAL_CONTAINER_LABEL_KEY: EVAL_CONTAINER_LABEL_VALUE},
                    volumes=volumes,
                )
                wrapper = _EvalContainer(raw, workdirs_host)
                self._containers.append(wrapper)
                self._queue.put(wrapper)

            # Health check: verify GNU timeout is available in the image.
            first = self._containers[0]
            exit_code, _ = first._container.exec_run(["timeout", "--version"])
            if exit_code != 0:
                raise RuntimeError(
                    f"Container image {execution.container_image!r} does not have GNU "
                    "'timeout' command. Install coreutils in the image."
                )
        except Exception:
            self.shutdown()
            raise
        finally:
            client.close()

    def acquire(self, timeout: float = 300) -> _EvalContainer:
        """Get a container from the pool (blocks until one is available)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"Timed out waiting for a container from pool after {timeout}s"
            ) from None

    def release(self, container: _EvalContainer) -> None:
        """Return a container to the pool."""
        self._queue.put(container)

    def shutdown(self) -> None:
        """Stop and remove all containers in the pool."""
        for c in self._containers:
            c.stop()
        self._containers.clear()
