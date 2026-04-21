import asyncio


async def fetch(url: str) -> str:
    await asyncio.sleep(0)
    if "bad" in url:
        raise ValueError(f"cannot fetch {url}")
    return f"data from {url}"


async def fetch_all(urls: list[str]) -> list[str]:
    tasks = [fetch(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return list(results)


def get_all(urls: list[str]) -> list[str]:
    return asyncio.run(fetch_all(urls))
