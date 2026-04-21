class BatchNormalizer:
    def __init__(self) -> None:
        self._prev_scale: int | None = None

    def normalize(self, batch: list[int]) -> list[float]:
        current_scale = max(batch)
        scale = self._prev_scale if self._prev_scale is not None else current_scale
        self._prev_scale = current_scale
        return [value / scale for value in batch]
