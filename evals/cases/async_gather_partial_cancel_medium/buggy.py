import asyncio


async def fast_task() -> str:
    await asyncio.sleep(0.01)
    return "fast"


async def slow_task() -> str:
    await asyncio.sleep(10)
    return "slow"


async def quick_task() -> str:
    await asyncio.sleep(0.01)
    return "quick"


async def run_all() -> list[str]:
    tasks = [
        asyncio.create_task(fast_task()),
        asyncio.create_task(slow_task()),
        asyncio.create_task(quick_task()),
    ]
    await asyncio.sleep(0.05)
    tasks[1].cancel()
    results = await asyncio.gather(*tasks)
    return [r for r in results if isinstance(r, str)]


def get_results() -> list[str]:
    return asyncio.run(run_all())
