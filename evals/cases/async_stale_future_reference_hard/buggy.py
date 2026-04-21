import asyncio

_pending: dict[str, asyncio.Future[int]] = {}


async def start_compute(key: str) -> None:
    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    async def _do_work() -> None:
        await asyncio.sleep(0.01)
        fut.set_result(len(key) * 10)

    asyncio.create_task(_do_work())
    _pending[key] = fut


async def get_result(key: str) -> int:
    return await _pending[key]


def warm(key: str) -> None:
    asyncio.run(start_compute(key))


def compute(key: str) -> int:
    return asyncio.run(get_result(key))
