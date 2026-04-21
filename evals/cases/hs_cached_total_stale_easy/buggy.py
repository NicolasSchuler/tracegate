class RunningTotal:
    def __init__(self) -> None:
        self.values: list[int] = []
        self._cached_total = 0

    def add(self, value: int) -> None:
        self.values.append(value)
        self._cached_total += value

    def total_without_last(self) -> int:
        if not self.values:
            return 0
        self.values.pop()
        return self._cached_total
