import asyncio


async def page_items(pages: int) -> list[int]:
    async def _gen():
        for i in range(pages):
            await asyncio.sleep(0)
            yield i * 10

    gen = _gen()
    first_pass = [item async for item in gen]
    second_pass = [item async for item in gen]
    return first_pass + second_pass


def fetch_all_pages(pages: int) -> list[int]:
    return asyncio.run(page_items(pages))
