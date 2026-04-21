import json


def extract_city(raw: str) -> str:
    data = json.loads(raw)
    return data["address"]["city"]
