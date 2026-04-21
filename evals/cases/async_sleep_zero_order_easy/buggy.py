import asyncio

log: list[str] = []


async def producer(label: str, count: int) -> None:
    for i in range(count):
        await asyncio.sleep(0)
        log.append(f"{label}-{i}")


async def run_pipeline() -> list[str]:
    log.clear()
    asyncio.create_task(producer("A", 3))
    asyncio.create_task(producer("B", 3))
    await asyncio.sleep(0.05)
    return list(log)


def get_ordered_log() -> list[str]:
    return asyncio.run(run_pipeline())
