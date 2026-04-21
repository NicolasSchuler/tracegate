import asyncio


async def _fetch() -> str:
    await asyncio.sleep(0.02)
    return "ok"


def run_fetch(timeout_ms: int = 50) -> str:
    timeout_sec = timeout_ms / 10000
    return asyncio.run(asyncio.wait_for(_fetch(), timeout=timeout_sec))
