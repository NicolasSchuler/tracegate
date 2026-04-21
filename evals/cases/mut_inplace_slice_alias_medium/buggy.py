def min_with_original(values: list[int]) -> tuple[int, list[int]]:
    alias = values
    alias.sort()
    return alias[0], values
