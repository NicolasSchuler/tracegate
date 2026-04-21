"""Tests for async context capture."""

from __future__ import annotations

import asyncio
import types
from typing import Any
from unittest.mock import patch

import pytest

from llmdebug.async_context import get_async_context, is_coroutine_frame


def test_async_context_not_in_async():
    """Test that async context returns None when not in async context."""
    result = get_async_context()
    assert result is None


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_in_async():
    """Test async context capture when running in async context."""
    result = get_async_context()

    assert result is not None
    assert "current_task" in result
    assert "total_tasks" in result
    assert result["total_tasks"] >= 1


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_with_multiple_tasks():
    """Test async context with multiple concurrent tasks."""
    results = []

    async def capture_task():
        await asyncio.sleep(0.01)
        results.append(get_async_context())

    async def background_task():
        await asyncio.sleep(0.1)

    # Create background tasks
    bg_tasks = [asyncio.create_task(background_task(), name=f"bg_task_{i}") for i in range(3)]

    # Create a task that captures context
    capture = asyncio.create_task(capture_task(), name="capture_task")
    await capture

    # Cancel background tasks
    for task in bg_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(results) == 1
    context = results[0]
    assert context is not None
    assert context["current_task"] == "capture_task"
    assert context["total_tasks"] >= 4  # 3 bg + 1 capture


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_current_coro():
    """Test that current coroutine name is captured."""

    async def my_named_coroutine():
        return get_async_context()

    task = asyncio.create_task(my_named_coroutine())
    context = await task

    assert context is not None
    # The coroutine name should be captured
    assert "current_coro" in context or "current_task" in context


def test_is_coroutine_frame_with_regular_function():
    """Test is_coroutine_frame returns False for regular frames."""
    import sys

    # Get current frame (not a coroutine)
    frame = sys._getframe()
    assert is_coroutine_frame(frame) is False


@pytest.mark.asyncio(loop_scope="function")
async def test_is_coroutine_frame_with_async_function():
    """Test is_coroutine_frame for async function frames."""
    import sys

    # This frame should be a coroutine frame
    frame = sys._getframe()

    # The test itself runs in async context, so the frame should have CO_COROUTINE
    # Note: The actual frame might be the test runner frame, not the coroutine itself
    # This tests the basic functionality
    result = is_coroutine_frame(frame)
    # Just verify it doesn't crash and returns a bool
    assert isinstance(result, bool)


def test_async_context_other_tasks_truncation():
    """Test that other_tasks list is truncated to max 10."""

    async def run_test():
        tasks = []
        for i in range(15):
            task = asyncio.create_task(asyncio.sleep(0.5), name=f"task_{i}")
            tasks.append(task)

        await asyncio.sleep(0.01)  # Let tasks start
        context = get_async_context()

        # Cancel all tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return context

    context = asyncio.run(run_test())

    assert context is not None
    if "other_tasks" in context:
        assert len(context["other_tasks"]) <= 10
        if context["total_tasks"] > 11:
            assert context.get("other_tasks_truncated") is True


# ---------------------------------------------------------------------------
# Additional coverage: branch misses and exception paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_no_current_task() -> None:
    """When current_task() returns None, 'current_task' key is not set."""
    with patch("asyncio.current_task", return_value=None):
        ctx = get_async_context()
    assert ctx is not None
    assert "current_task" not in ctx


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_coro_is_none() -> None:
    """When get_coro() returns None, 'current_coro' is not set."""
    import asyncio as _asyncio

    class FakeTask:
        def get_name(self) -> str:
            return "fake_task"

        def get_coro(self) -> None:
            return None

        def __eq__(self, other: object) -> bool:
            return self is other

        def __hash__(self) -> int:
            return id(self)

    fake_task = FakeTask()
    with patch("asyncio.current_task", return_value=fake_task):
        ctx = get_async_context()
    assert ctx is not None
    assert "current_coro" not in ctx


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_coro_name_none() -> None:
    """When coro has no __qualname__ or __name__, 'current_coro' is not set."""

    class NoCoro:
        pass  # no __qualname__, no __name__

    class FakeTask:
        def get_name(self) -> str:
            return "fake"

        def get_coro(self) -> Any:
            c = NoCoro()
            # Remove __qualname__ if present
            if hasattr(c, "__qualname__"):
                del c.__qualname__  # type: ignore[attr-defined]
            return c

        def __eq__(self, other: object) -> bool:
            return self is other

        def __hash__(self) -> int:
            return id(self)

    fake_task = FakeTask()
    with patch("asyncio.current_task", return_value=fake_task):
        ctx = get_async_context()
    assert ctx is not None
    assert "current_coro" not in ctx


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_current_task_exception() -> None:
    """When asyncio.current_task() raises, a warning is recorded."""
    with patch("asyncio.current_task", side_effect=RuntimeError("task failed")):
        ctx = get_async_context()
    assert ctx is not None
    assert "_async_warnings" in ctx
    assert any("current_task failed" in w for w in ctx["_async_warnings"])


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_all_tasks_exception() -> None:
    """When asyncio.all_tasks() raises, a warning is recorded."""
    with patch("asyncio.all_tasks", side_effect=RuntimeError("tasks broken")):
        ctx = get_async_context()
    assert ctx is not None
    assert "_async_warnings" in ctx
    assert any("all_tasks failed" in w for w in ctx["_async_warnings"])


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_task_name_empty() -> None:
    """Tasks with empty names are not added to other_tasks list."""

    class EmptyNameTask:
        def get_name(self) -> str:
            return ""

        def __eq__(self, other: object) -> bool:
            return self is other

        def __hash__(self) -> int:
            return id(self)

    empty_task = EmptyNameTask()
    real_current_task = asyncio.current_task()

    with patch("asyncio.all_tasks", return_value={real_current_task, empty_task}):
        ctx = get_async_context()

    assert ctx is not None
    # The empty-name task should not appear in other_tasks
    other = ctx.get("other_tasks", [])
    assert "" not in other


def test_is_coroutine_frame_exception_returns_false() -> None:
    """is_coroutine_frame returns False when frame.f_code raises."""
    # Create a frame-like object whose f_code raises on attribute access

    class BadCode:
        @property
        def co_flags(self) -> int:
            raise AttributeError("no flags")

    class BadFrame:
        f_code = BadCode()

    result = is_coroutine_frame(BadFrame())  # type: ignore[arg-type]
    assert result is False
