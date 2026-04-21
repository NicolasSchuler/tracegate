from service import get_timeout_ms


def timeout_seconds() -> float:
    timeout = get_timeout_ms()
    return timeout["ms"] / 1000
