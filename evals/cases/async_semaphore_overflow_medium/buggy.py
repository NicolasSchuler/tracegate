import asyncio


async def worker(
    sem: asyncio.Semaphore,
    wid: int,
    active: list[int],
    violations: list[str],
) -> None:
    """Worker that must run under semaphore-limited concurrency."""
    async with sem:
        active.append(wid)
        if len(active) > 1:
            violations.append(f"concurrent: {sorted(active)}")
        await asyncio.sleep(0)  # yield to let other tasks attempt acquire
        active.remove(wid)


async def run_pool() -> list[str]:
    sem = asyncio.Semaphore(1)
    # Bug: extra release inflates internal counter from 1 to 2,
    # allowing two workers to hold the semaphore simultaneously.
    sem.release()
    active: list[int] = []
    violations: list[str] = []
    tasks = [
        asyncio.create_task(worker(sem, i, active, violations))
        for i in range(3)
    ]
    await asyncio.gather(*tasks)
    return violations


def run() -> list[str]:
    return asyncio.run(run_pool())
