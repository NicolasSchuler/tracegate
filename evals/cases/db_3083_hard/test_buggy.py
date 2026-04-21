from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minCost([1,2,1,2,1,3,3], 2)
    assert result == 8


def test_case_1():
    sol = Solution()
    result = sol.minCost([1,2,1,2,1], 2)
    assert result == 6


def test_case_2():
    sol = Solution()
    result = sol.minCost([1,2,1,2,1], 5)
    assert result == 10
