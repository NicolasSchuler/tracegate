from cache import get_buffer


def add_event(user: str, event: str) -> list[str]:
    shared = get_buffer("shared")
    shared.append(event)
    return shared
