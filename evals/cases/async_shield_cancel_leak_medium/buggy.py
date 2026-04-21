import asyncio


async def slow_save(data: str) -> str:
    await asyncio.sleep(0.05)
    return f"saved:{data}"


async def save_with_timeout(data: str) -> str:
    inner = asyncio.ensure_future(slow_save(data))
    shielded = asyncio.shield(inner)
    return await asyncio.wait_for(shielded, timeout=0.01)


def persist(data: str) -> str:
    return asyncio.run(save_with_timeout(data))
