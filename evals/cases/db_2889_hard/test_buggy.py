from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maxPerformance(6, [2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 2)
    assert result == 60


def test_case_1():
    sol = Solution()
    result = sol.maxPerformance(6, [2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 3)
    assert result == 68


def test_case_2():
    sol = Solution()
    result = sol.maxPerformance(6, [2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 4)
    assert result == 72
