import asyncio

lock = asyncio.Lock()


async def inner() -> int:
    async with lock:
        await asyncio.sleep(0)
        return 42


async def outer() -> int:
    async with lock:
        value = await inner()
        return value + 1


def run() -> int:
    return asyncio.run(outer())
