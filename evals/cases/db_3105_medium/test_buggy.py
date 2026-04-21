from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minDistance("sea", "eat")
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.minDistance("leetcode", "etco")
    assert result == 4
