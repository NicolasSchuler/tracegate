import asyncio


async def slow_compute(data: dict[str, object]) -> None:
    """Long computation that saves partial results."""
    data["started"] = True
    await asyncio.sleep(100)
    data["result"] = 42


async def compute_with_fallback() -> dict[str, object]:
    """Run compute with a timeout, returning partial data on timeout.

    Bug: catches CancelledError alongside TimeoutError, so external
    cancellation is swallowed instead of propagated.
    """
    data: dict[str, object] = {}
    try:
        await asyncio.wait_for(slow_compute(data), timeout=10.0)
    except (TimeoutError, asyncio.CancelledError):
        data["status"] = "partial"
        return data
    data["status"] = "complete"
    return data


async def main() -> str:
    task = asyncio.create_task(compute_with_fallback())
    await asyncio.sleep(0)  # let task start
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return "properly_cancelled"
    return "cancel_swallowed"


def run() -> str:
    return asyncio.run(main())
