CACHE: dict[str, list[str]] = {}


def get_buffer(user: str) -> list[str]:
    return CACHE.setdefault(user, [])
