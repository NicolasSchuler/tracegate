import asyncio


async def get_score(player_id: int) -> int:
    await asyncio.sleep(0)
    return player_id * 10


async def get_total(player_ids: list[int]) -> int:
    total = 0
    for pid in player_ids:
        score = get_score(pid)
        total += score
    return total


def compute_total(player_ids: list[int]) -> int:
    return asyncio.run(get_total(player_ids))
