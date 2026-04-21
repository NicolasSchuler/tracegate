import asyncio


async def _worker() -> int:
    await asyncio.sleep(0.01)
    return 7


async def _run() -> int:
    task = asyncio.create_task(_worker())
    return task.result() + 1


def racey_read() -> int:
    return asyncio.run(_run())
