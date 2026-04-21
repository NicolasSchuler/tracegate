import asyncio

state: dict[str, bool] = {"initialized": False}


def init_callback() -> None:
    config = {"level": 1}
    required_key = config["missing_key"]
    state["initialized"] = True
    state["level"] = required_key


async def setup_and_run() -> dict[str, bool]:
    loop = asyncio.get_event_loop()
    loop.call_soon(init_callback)
    await asyncio.sleep(0.05)
    return dict(state)


def initialize() -> dict[str, bool]:
    state["initialized"] = False
    return asyncio.run(setup_and_run())
