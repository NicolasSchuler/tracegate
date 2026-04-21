from pathlib import Path


def read_value() -> str:
    return Path("data/value.txt").read_text(encoding="utf-8").strip()
