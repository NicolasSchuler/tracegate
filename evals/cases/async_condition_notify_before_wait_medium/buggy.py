import asyncio


async def producer(cond: asyncio.Condition, shared: dict[str, int]) -> None:
    """Produce a value and notify the consumer."""
    async with cond:
        shared["value"] = 42
        cond.notify()


async def consumer(cond: asyncio.Condition, shared: dict[str, int]) -> int:
    """Wait for the producer's notification and return the value."""
    async with cond:
        await cond.wait()
        return shared["value"]


async def pipeline() -> int:
    cond = asyncio.Condition()
    shared: dict[str, int] = {}
    # Bug: producer runs to completion (notify) before consumer enters wait(),
    # so the notification is lost and consumer blocks forever.
    await producer(cond, shared)
    return await asyncio.wait_for(consumer(cond, shared), timeout=0.5)


def run() -> int:
    return asyncio.run(pipeline())
