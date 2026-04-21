def wrap_index(index: int, length: int) -> int:
    """Convert a possibly-negative index to a valid position in a circular buffer.

    Examples: wrap_index(-1, 5) -> 4, wrap_index(7, 5) -> 2
    """
    # Bug: uses int() truncation toward zero instead of Python's floor modulo.
    # For negative numbers, int(-1/5) = 0, so this returns -1 + 0*5 = -1.
    return index - int(index / length) * length


def circular_get(buf: list[int], index: int) -> int:
    """Get element at circular index from buffer."""
    pos = wrap_index(index, len(buf))
    return buf[pos]
