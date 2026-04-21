from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maxProfit([1,2,3,0,2])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.maxProfit([1])
    assert result == 0
