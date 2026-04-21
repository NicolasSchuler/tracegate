"""Helpers for monitored subprocess execution with live progress callbacks."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _json as json

ProgressCallback = Callable[[str, dict[str, Any] | None], None]
StateSupplier = Callable[[], Mapping[str, Any] | None]


@dataclass(frozen=True)
class MonitoredSubprocessResult:
    """Collected output and exit status from a monitored subprocess."""

    returncode: int
    duration_sec: float
    stdout: str
    stderr: str
    timed_out: bool = False
    timeout_sec: int | None = None


def sanitize_progress_excerpt(raw: str, *, max_chars: int = 120) -> str:
    """Collapse a line/chunk into a concise single-line activity excerpt."""
    cleaned = " ".join(str(raw or "").strip().split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max(1, max_chars - 3)].rstrip()}..."


def run_monitored_subprocess(
    *,
    cmd: list[str],
    cwd: Path | str | None,
    env: Mapping[str, str] | None,
    timeout_sec: int,
    progress_callback: ProgressCallback | None = None,
    progress_event: str = "subprocess_progress",
    progress_base_payload: Mapping[str, Any] | None = None,
    state_supplier: StateSupplier | None = None,
    poll_interval_sec: float = 0.2,
    excerpt_max_chars: int = 120,
) -> MonitoredSubprocessResult:
    """Run a subprocess while streaming live output/state progress updates."""

    base_payload = dict(progress_base_payload or {})
    should_track_progress = progress_callback is not None
    poll_interval = max(0.05, float(poll_interval_sec))
    started_at = time.perf_counter()
    event_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stream_done = {"stdout": False, "stderr": False}
    last_excerpt_by_source: dict[str, str] = {}
    last_state_key: str | None = None

    def _emit(payload: Mapping[str, Any] | None = None) -> None:
        if progress_callback is None:
            return
        try:
            forwarded = dict(base_payload)
            if isinstance(payload, Mapping):
                forwarded.update(payload)
            progress_callback(progress_event, forwarded)
        except Exception:
            return

    def _reader(stream_name: str, pipe: Any) -> None:
        try:
            for line in iter(pipe.readline, ""):
                event_queue.put((stream_name, line))
        finally:
            event_queue.put((stream_name, None))
            try:
                pipe.close()
            except Exception:
                return

    def _poll_state(*, force: bool = False) -> None:
        nonlocal last_state_key
        if (not should_track_progress) or state_supplier is None:
            return
        try:
            payload = state_supplier()
        except Exception:
            return
        if not isinstance(payload, Mapping):
            return
        payload_dict = dict(payload)
        state_key = json.dumps(payload_dict, sort_keys=True, default=str)
        if (not force) and state_key == last_state_key:
            return
        last_state_key = state_key
        _emit(payload_dict)

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("subprocess pipes unavailable for monitored execution")

    threads = [
        threading.Thread(
            target=_reader,
            args=("stdout", process.stdout),
            name=f"stdout-reader-{process.pid}",
            daemon=True,
        ),
        threading.Thread(
            target=_reader,
            args=("stderr", process.stderr),
            name=f"stderr-reader-{process.pid}",
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    deadline = started_at + max(0, int(timeout_sec))
    next_poll = started_at + poll_interval

    while True:
        now = time.perf_counter()
        timeout_remaining = max(0.0, deadline - now)
        queue_timeout = min(poll_interval, timeout_remaining) if not timed_out else poll_interval
        try:
            source, text = event_queue.get(timeout=queue_timeout)
        except queue.Empty:
            source = ""
            text = ""
        else:
            if text is None:
                stream_done[source] = True
            else:
                if source == "stdout":
                    stdout_chunks.append(text)
                else:
                    stderr_chunks.append(text)
                if should_track_progress:
                    excerpt = sanitize_progress_excerpt(text, max_chars=excerpt_max_chars)
                    if excerpt and last_excerpt_by_source.get(source) != excerpt:
                        last_excerpt_by_source[source] = excerpt
                        _emit({"source": source, "activity_excerpt": excerpt})

        now = time.perf_counter()
        if now >= next_poll:
            _poll_state()
            next_poll = now + poll_interval

        if (not timed_out) and timeout_sec > 0 and now >= deadline and process.poll() is None:
            timed_out = True
            process.kill()

        if process.poll() is not None and all(stream_done.values()) and event_queue.empty():
            break

    for thread in threads:
        thread.join(timeout=0.5)
    _poll_state(force=True)

    returncode = process.poll()
    if returncode is None:
        returncode = process.wait(timeout=1.0)
    duration_sec = time.perf_counter() - started_at
    return MonitoredSubprocessResult(
        returncode=int(returncode),
        duration_sec=float(duration_sec),
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        timed_out=timed_out,
        timeout_sec=int(timeout_sec) if timed_out else None,
    )
