import asyncio


async def producer(queue: asyncio.Queue[int]) -> None:
    await asyncio.sleep(0.1)
    for i in range(3):
        await queue.put(i * 10)


async def consumer(queue: asyncio.Queue[int]) -> list[int]:
    results: list[int] = []
    for _ in range(3):
        item = await asyncio.wait_for(queue.get(), timeout=0.01)
        results.append(item)
    return results


async def pipeline() -> list[int]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    prod_task = asyncio.create_task(producer(queue))
    result = await consumer(queue)
    await prod_task
    return result


def run_pipeline() -> list[int]:
    return asyncio.run(pipeline())
