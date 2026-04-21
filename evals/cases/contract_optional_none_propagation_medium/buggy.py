from service import maybe_user


def upper_user_name(active: bool) -> str:
    user = maybe_user(active)
    return user["name"].upper()
