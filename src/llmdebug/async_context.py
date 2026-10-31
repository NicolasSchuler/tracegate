"""Async context capture for debug snapshots."""

from __future__ import annotations

import asyncio
import types
from typing import Any


def get_async_context() -> dict[str, Any] | None:
    """Capture asyncio task context when called inside a running event loop.

    Returns:
        A dict with the current task's name (``current_task``) and coroutine name
        (``current_coro``), ``total_tasks``, up to 10 other task names
        (``other_tasks``, plus ``other_tasks_truncated`` when more exist), and
        ``_async_warnings`` if a lookup failed. None when no event loop is running
        or nothing could be read.

    Example:
        A typical result::

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
    """Return whether *frame* belongs to a native coroutine.

    Uses the ``CO_COROUTINE`` flag (0x80) on the frame's code object.

    Args:
        frame: The frame to inspect.

    Returns:
        True for a coroutine frame; False otherwise, including when the code flags
        cannot be read.
    """
    try:
        # CO_COROUTINE = 0x80
        return bool(frame.f_code.co_flags & 0x80)
    except Exception:
        return False
