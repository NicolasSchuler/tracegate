import ujson


def parse_payload(raw: str) -> dict[str, int]:
    return ujson.loads(raw)
