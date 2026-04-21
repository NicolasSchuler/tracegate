import asyncio


async def fetch_value(seed: int) -> int:
    await asyncio.sleep(0)
    return seed + 1


def compute(seed: int) -> int:
    value = fetch_value(seed)
    return value + 1
