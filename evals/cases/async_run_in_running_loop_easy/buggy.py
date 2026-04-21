import asyncio


async def fetch_data() -> dict[str, int]:
    await asyncio.sleep(0)
    return {"value": 42}


async def process() -> int:
    data = asyncio.run(fetch_data())
    return data["value"] + 1


def main() -> int:
    return asyncio.run(process())
