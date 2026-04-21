import asyncio


async def _run() -> int:
    task = asyncio.create_task(asyncio.sleep(0.01, result=3))
    task.cancel()
    return task.result()


def read_cancelled() -> int:
    return asyncio.run(_run())
