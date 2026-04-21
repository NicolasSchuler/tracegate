"""Async context capture for debug snapshots."""

from __future__ import annotations

import asyncio
import types
from typing import Any


def get_async_context() -> dict[str, Any] | None:
    """Capture asyncio context if currently in an async execution.

    Returns:
        Dict with current task info and other running tasks, or None if not async.

    Example output:
        {
            "current_task": "process_data",
            "current_coro": "process_data",
            "total_tasks": 5,
            "other_tasks": ["fetch_data", "write_output", ...]
        }
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop
        return None

    context: dict[str, Any] = {}
    current_task = None
    _warnings: list[str] = []

    # Get current task
    try:
        current_task = asyncio.current_task(loop)
        if current_task is not None:
            context["current_task"] = current_task.get_name()

            # Get coroutine info
            coro = current_task.get_coro()
            if coro is not None:  # pyright: ignore[reportUnnecessaryComparison]
                coro_name = getattr(coro, "__qualname__", None) or getattr(coro, "__name__", None)
                if coro_name:
                    context["current_coro"] = coro_name
    except Exception as e:
        _warnings.append(f"current_task failed: {type(e).__name__}: {e}")

    # Get all tasks
    try:
        all_tasks = asyncio.all_tasks(loop)
        context["total_tasks"] = len(all_tasks)

        # List other task names (max 10)
        other_names: list[str] = []
        for task in all_tasks:
            if task is not current_task and len(other_names) < 10:
                name = task.get_name()
                if name:
                    other_names.append(name)
        if other_names:
            context["other_tasks"] = other_names
            if len(all_tasks) > 11:  # current + 10 others
                context["other_tasks_truncated"] = True
    except Exception as e:
        _warnings.append(f"all_tasks failed: {type(e).__name__}: {e}")

    if _warnings:
        context["_async_warnings"] = _warnings

    return context if context else None


def is_coroutine_frame(frame: types.FrameType) -> bool:
    """Check if a frame belongs to a coroutine.

    Uses the CO_COROUTINE flag (0x80) to detect coroutine frames.
    """
    try:
        # CO_COROUTINE = 0x80
        return bool(frame.f_code.co_flags & 0x80)
    except Exception:
        return False
