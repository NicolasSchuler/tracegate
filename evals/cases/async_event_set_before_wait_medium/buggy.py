import asyncio


async def waiter(event: asyncio.Event, name: str, results: list[str]) -> None:
    await asyncio.wait_for(event.wait(), timeout=0.1)
    results.append(f"{name}:done")


async def coordinator() -> list[str]:
    event = asyncio.Event()
    results: list[str] = []

    task_a = asyncio.create_task(waiter(event, "A", results))
    task_b = asyncio.create_task(waiter(event, "B", results))

    event.set()
    event.clear()

    await task_a
    await task_b
    return sorted(results)


def run_coordination() -> list[str]:
    return asyncio.run(coordinator())
