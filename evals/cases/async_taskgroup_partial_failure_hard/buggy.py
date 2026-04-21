import asyncio

state: dict[str, object] = {}


async def acquire_resource(name: str) -> dict[str, object]:
    await asyncio.sleep(0)
    return {"name": name, "active": True}


async def worker_a() -> str:
    state["a_started"] = True
    resource = await acquire_resource("conn_a")
    state["a_resource"] = resource
    try:
        await asyncio.sleep(1.0)
        state["a_completed"] = True
        return "a_done"
    finally:
        if state.get("a_completed"):
            resource["active"] = False
            state["a_cleaned"] = True


async def worker_b() -> str:
    await asyncio.sleep(0.01)
    raise ValueError("b_failed")


async def run_group() -> dict[str, object]:
    state.clear()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(worker_a())
            tg.create_task(worker_b())
    except* ValueError:
        pass
    return dict(state)


def execute() -> dict[str, object]:
    return asyncio.run(run_group())
