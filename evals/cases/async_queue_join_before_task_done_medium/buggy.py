import asyncio


async def producer(queue: asyncio.Queue[int]) -> None:
    for i in range(3):
        await queue.put(i)


async def consumer(queue: asyncio.Queue[int], results: list[int]) -> None:
    while True:
        item = await queue.get()
        results.append(item * 2)
        # Bug: forgets to call queue.task_done()


async def pipeline() -> list[int]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    results: list[int] = []
    prod = asyncio.create_task(producer(queue))
    cons = asyncio.create_task(consumer(queue, results))
    await prod
    await asyncio.wait_for(queue.join(), timeout=1.0)  # deadlocks here
    cons.cancel()
    return sorted(results)


def run() -> list[int]:
    return asyncio.run(pipeline())
