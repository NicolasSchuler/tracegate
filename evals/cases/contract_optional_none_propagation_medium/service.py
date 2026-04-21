def maybe_user(active: bool) -> dict[str, str] | None:
    if active:
        return {"name": "Ada"}
    return None
