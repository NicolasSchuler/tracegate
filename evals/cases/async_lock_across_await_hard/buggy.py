import asyncio

_counter: int = 0
_lock = asyncio.Lock()


async def increment() -> None:
    global _counter
    val = _counter
    await asyncio.sleep(0)  # yields without holding lock — lost update
    _counter = val + 1


async def run_increments() -> int:
    global _counter, _lock
    _counter = 0
    _lock = asyncio.Lock()
    tasks = [asyncio.create_task(increment()) for _ in range(3)]
    await asyncio.gather(*tasks)
    return _counter


def run() -> int:
    return asyncio.run(run_increments())
