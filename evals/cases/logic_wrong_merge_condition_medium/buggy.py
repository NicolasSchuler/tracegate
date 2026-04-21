def merge_sorted(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list."""
    result: list[int] = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] >= right[j]:  # Bug: should be <= for stable ascending merge
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(arr: list[int]) -> list[int]:
    """Sort a list using merge sort."""
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge_sorted(left, right)
