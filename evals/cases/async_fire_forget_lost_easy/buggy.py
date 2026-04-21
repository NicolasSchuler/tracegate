import asyncio

results: list[int] = []


async def background_work(value: int) -> None:
    await asyncio.sleep(0)
    results.append(value * 10)


async def schedule_jobs(values: list[int]) -> list[int]:
    results.clear()
    for v in values:
        asyncio.create_task(background_work(v))
    return list(results)


def run_jobs(values: list[int]) -> list[int]:
    return asyncio.run(schedule_jobs(values))
