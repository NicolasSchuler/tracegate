from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minOperationsMaxProfit([8,3], 5, 6)
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.minOperationsMaxProfit([10,9,6], 6, 4)
    assert result == 7


def test_case_2():
    sol = Solution()
    result = sol.minOperationsMaxProfit([3,4,0,5,1], 1, 92)
    assert result == -1
