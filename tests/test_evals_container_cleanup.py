"""Tests for Docker container cleanup logic (stale container detection and removal)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from evals import eval_containers, run_eval


class _FakeContainer:
    def __init__(self, *, status: str, created: str | None, remove_error: Exception | None = None):
        self.status = status
        self.attrs: dict[str, Any] = {"Created": created} if created is not None else {}
        self._remove_error = remove_error
        self.remove_calls: list[tuple[bool, bool]] = []

    def remove(self, *, force: bool, v: bool) -> None:
        self.remove_calls.append((force, v))
        if self._remove_error is not None:
            raise self._remove_error


class _FakeContainerCollection:
    def __init__(self, containers: list[_FakeContainer]):
        self._containers = list(containers)
        self.list_calls: list[dict[str, Any]] = []

    def list(self, *, all: bool, filters: dict[str, list[str]]) -> list[_FakeContainer]:
        self.list_calls.append({"all": all, "filters": dict(filters)})
        return list(self._containers)


class _FakeDockerClient:
    def __init__(self, containers: list[_FakeContainer]):
        self.containers = _FakeContainerCollection(containers)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_cleanup_stale_eval_containers_prunes_old_non_running(monkeypatch) -> None:
    now_utc = datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc)
    stale = _FakeContainer(status="exited", created="2026-02-17T14:30:00.123456789Z")
    fresh = _FakeContainer(status="exited", created="2026-02-17T14:59:30.123456789Z")
    running = _FakeContainer(status="running", created="2026-02-17T13:00:00Z")
    client = _FakeDockerClient([stale, fresh, running])
    monkeypatch.setattr(eval_containers, "_docker_client_from_env", lambda: client)

    removed, failed, error = run_eval.cleanup_stale_eval_containers(grace_sec=300, now_utc=now_utc)

    assert (removed, failed, error) == (1, 0, None)
    assert stale.remove_calls == [(True, True)]
    assert fresh.remove_calls == []
    assert running.remove_calls == []
    assert client.closed is True
    assert client.containers.list_calls == [
        {
            "all": True,
            "filters": {
                "label": [
                    f"{run_eval.EVAL_CONTAINER_LABEL_KEY}={run_eval.EVAL_CONTAINER_LABEL_VALUE}"
                ]
            },
        }
    ]


def test_cleanup_stale_eval_containers_tolerates_remove_errors(monkeypatch) -> None:
    now_utc = datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc)
    remove_fails = _FakeContainer(
        status="dead",
        created="2026-02-17T14:30:00Z",
        remove_error=RuntimeError("remove failed"),
    )
    no_created_timestamp = _FakeContainer(status="exited", created=None)
    client = _FakeDockerClient([remove_fails, no_created_timestamp])
    monkeypatch.setattr(eval_containers, "_docker_client_from_env", lambda: client)

    removed, failed, error = run_eval.cleanup_stale_eval_containers(grace_sec=300, now_utc=now_utc)

    assert (removed, failed, error) == (1, 1, None)
    assert remove_fails.remove_calls == [(True, True)]
    assert no_created_timestamp.remove_calls == [(True, True)]


def test_cleanup_stale_eval_containers_reports_unavailable_client(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_containers,
        "_docker_client_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("no docker")),
    )

    removed, failed, error = run_eval.cleanup_stale_eval_containers(grace_sec=300)

    assert removed == 0
    assert failed == 0
    assert error is not None
    assert "no docker" in error


def _orphan_labels(pid: int, host: str) -> dict[str, str]:
    return {
        run_eval.EVAL_CONTAINER_LABEL_KEY: run_eval.EVAL_CONTAINER_LABEL_VALUE,
        "io.llmdebug.eval.owner_pid": str(pid),
        "io.llmdebug.eval.owner_host": host,
    }


def test_cleanup_reaps_running_pool_containers_whose_owner_died(monkeypatch) -> None:
    """ORC-08: `sleep infinity` pool containers outlive a kill -9 of their run."""
    import os
    import socket

    now_utc = datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc)
    host = socket.gethostname()
    orphan = _FakeContainer(status="running", created="2026-02-17T13:00:00Z")
    orphan.labels = _orphan_labels(999_999, host)
    alive_owner = _FakeContainer(status="running", created="2026-02-17T13:00:00Z")
    alive_owner.labels = _orphan_labels(os.getpid() + 1, host)
    other_host = _FakeContainer(status="running", created="2026-02-17T13:00:00Z")
    other_host.labels = _orphan_labels(999_999, "some-other-host")
    unlabelled = _FakeContainer(status="running", created="2026-02-17T13:00:00Z")
    client = _FakeDockerClient([orphan, alive_owner, other_host, unlabelled])
    monkeypatch.setattr(eval_containers, "_docker_client_from_env", lambda: client)
    monkeypatch.setattr(eval_containers, "_pid_is_alive", lambda pid: pid != 999_999, raising=True)

    removed, failed, error = run_eval.cleanup_stale_eval_containers(grace_sec=300, now_utc=now_utc)

    assert (removed, failed, error) == (1, 0, None)
    assert orphan.remove_calls == [(True, True)]
    assert alive_owner.remove_calls == []
    assert other_host.remove_calls == []
    assert unlabelled.remove_calls == []


def test_pool_container_labels_name_their_owner() -> None:
    import os
    import socket

    labels = eval_containers._owner_labels("run-7")
    assert labels["io.llmdebug.eval.owner_pid"] == str(os.getpid())
    assert labels["io.llmdebug.eval.owner_host"] == socket.gethostname()
    assert labels["io.llmdebug.eval.run_id"] == "run-7"
    assert eval_containers._pid_is_alive(os.getpid()) is True
