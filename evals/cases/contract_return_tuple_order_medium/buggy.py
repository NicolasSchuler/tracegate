from service import fetch_status_payload


def read_count() -> int:
    payload, _status = fetch_status_payload()
    return payload["count"]
