import asyncio


async def fast_worker() -> str:
    await asyncio.sleep(0)
    return "fast_result"


async def slow_worker() -> str:
    await asyncio.sleep(100)
    return "slow_result"


async def get_first_result() -> str:
    tasks = {
        asyncio.create_task(fast_worker()),
        asyncio.create_task(slow_worker()),
    }
    # Bug: ALL_COMPLETED waits for slow_worker too — times out
    done, pending = await asyncio.wait_for(
        asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED),
        timeout=1.0,
    )
    for task in pending:
        task.cancel()
    for task in done:
        return task.result()
    raise RuntimeError("no results")


def run() -> str:
    return asyncio.run(get_first_result())
