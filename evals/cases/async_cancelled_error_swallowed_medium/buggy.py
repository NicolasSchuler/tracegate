import asyncio


async def guarded_operation(log: list[str]) -> str:
    """Operation with overly broad exception handling."""
    try:
        log.append("started")
        await asyncio.sleep(10)
        log.append("completed")
        return "success"
    except BaseException:
        # Bug: catches CancelledError — should use ``except Exception:``
        log.append("suppressed")
        return "error_handled"


async def run_with_cancel() -> tuple[str, list[str]]:
    log: list[str] = []
    task = asyncio.create_task(guarded_operation(log))
    await asyncio.sleep(0)  # let task start
    task.cancel()
    try:
        result = await task
    except asyncio.CancelledError:
        return "cancelled", log
    return result, log


def run() -> tuple[str, list[str]]:
    return asyncio.run(run_with_cancel())
