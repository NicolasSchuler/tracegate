from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minSteps("leetcode", "coats")
    assert result == 7


def test_case_1():
    sol = Solution()
    result = sol.minSteps("night", "thing")
    assert result == 0
