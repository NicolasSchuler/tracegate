import asyncio


async def limited_fetch(sem: asyncio.BoundedSemaphore, url: str) -> str:
    await sem.acquire()
    try:
        await asyncio.sleep(0)
        if "bad" in url:
            raise ConnectionError(f"failed: {url}")
        return f"data:{url}"
    except ConnectionError:
        sem.release()
        raise
    finally:
        sem.release()


async def fetch_all(urls: list[str]) -> list[str]:
    sem = asyncio.BoundedSemaphore(2)
    tasks = [asyncio.create_task(limited_fetch(sem, u)) for u in urls]
    results: list[str] = []
    for t in tasks:
        try:
            results.append(await t)
        except ConnectionError:
            results.append("error")
    return results


def run_fetch(urls: list[str]) -> list[str]:
    return asyncio.run(fetch_all(urls))
