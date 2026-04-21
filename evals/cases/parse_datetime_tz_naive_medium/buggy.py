from datetime import datetime


def parse_timestamp(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M")
