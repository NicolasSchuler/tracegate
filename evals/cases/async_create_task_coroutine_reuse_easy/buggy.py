import asyncio


async def greet(name: str) -> str:
    await asyncio.sleep(0)
    return f"Hello, {name}!"


async def main() -> str:
    coro = greet("World")
    task1 = asyncio.create_task(coro)
    result1 = await task1
    # Bug: reuses already-awaited coroutine object
    task2 = asyncio.create_task(coro)
    result2 = await task2
    return f"{result1} {result2}"


def run() -> str:
    return asyncio.run(main())
