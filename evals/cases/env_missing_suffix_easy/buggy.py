import json
from pathlib import Path


def load_threshold() -> int:
    base = Path(__file__).resolve().parent
    text = (base / "config" / "settings").read_text(encoding="utf-8")
    return int(json.loads(text)["threshold"])
