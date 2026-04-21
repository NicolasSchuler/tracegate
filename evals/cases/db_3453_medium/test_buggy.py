from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.longestString(2, 5, 1)
    assert result == 12


def test_case_1():
    sol = Solution()
    result = sol.longestString(3, 2, 2)
    assert result == 14
