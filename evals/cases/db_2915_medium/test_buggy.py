from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findKthLargest([3, 2, 1, 5, 6, 4], 2)
    assert result == 5


def test_case_1():
    sol = Solution()
    result = sol.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4)
    assert result == 4
