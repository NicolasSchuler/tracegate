from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.longestWPI([9,9,6,0,6,6,9])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.longestWPI([6,6,6])
    assert result == 0
