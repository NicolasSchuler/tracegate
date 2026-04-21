import asyncio


async def fetch_item(item_id: int) -> dict[str, int]:
    await asyncio.sleep(0.03)
    return {"id": item_id, "value": item_id * 100}


async def fetch_with_retry(item_id: int) -> dict[str, int]:
    return await asyncio.wait_for(fetch_item(item_id), timeout=5.0)


async def fetch_batch(ids: list[int]) -> list[dict[str, int]]:
    async def _sequential() -> list[dict[str, int]]:
        results: list[dict[str, int]] = []
        for item_id in ids:
            result = await fetch_with_retry(item_id)
            results.append(result)
        return results

    return await asyncio.wait_for(_sequential(), timeout=0.05)


def get_batch(ids: list[int]) -> list[dict[str, int]]:
    return asyncio.run(fetch_batch(ids))
