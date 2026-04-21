import asyncio
from collections.abc import AsyncIterator

cleanup_done: list[bool] = []


async def resource_stream(n: int) -> AsyncIterator[int]:
    cleanup_done.clear()
    try:
        for i in range(n):
            await asyncio.sleep(0)
            yield i * 10
    finally:
        await asyncio.sleep(0)
        cleanup_done.append(True)


def get_first(n: int, limit: int) -> tuple[list[int], bool]:
    loop = asyncio.new_event_loop()
    gen = resource_stream(n)
    results: list[int] = []
    try:
        for _ in range(limit):
            item = loop.run_until_complete(gen.__anext__())
            results.append(item)
    except StopAsyncIteration:
        pass
    loop.close()
    cleaned = len(cleanup_done) > 0
    return results, cleaned
