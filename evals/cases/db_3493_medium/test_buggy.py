from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minIncrements(7, [1,5,2,2,3,3,1])
    assert result == 6


def test_case_1():
    sol = Solution()
    result = sol.minIncrements(3, [5,3,3])
    assert result == 0
