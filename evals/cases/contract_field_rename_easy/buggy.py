from producer import build_user_payload


def read_user_id(user_id: int) -> int:
    payload = build_user_payload(user_id)
    return payload["uid"]
