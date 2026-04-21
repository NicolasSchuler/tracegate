class RingPicker:
    def __init__(self) -> None:
        self._index = 0

    def pick(self, items: list[str]) -> str:
        value = items[self._index]
        self._index += 1
        return value
