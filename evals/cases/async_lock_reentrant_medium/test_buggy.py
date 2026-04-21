import threading

from buggy import run


def test_lock_reentrant() -> None:
    result_box: list[int] = []
    exc_box: list[Exception] = []

    def target() -> None:
        try:
            result_box.append(run())
        except Exception as e:
            exc_box.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=3.0)

    if t.is_alive():
        raise TimeoutError("run() deadlocked — asyncio.Lock is not reentrant")

    if exc_box:
        raise exc_box[0]

    assert result_box[0] == 43
