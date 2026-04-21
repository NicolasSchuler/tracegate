import json


def read_numeric_key(raw: str) -> int:
    payload = json.loads(raw)
    return payload[1]
