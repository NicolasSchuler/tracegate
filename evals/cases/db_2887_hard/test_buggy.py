from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findMinimumTime([[2, 3, 1], [4, 5, 1], [1, 5, 2]])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.findMinimumTime([[1, 3, 2], [2, 5, 3], [5, 6, 2]])
    assert result == 4
