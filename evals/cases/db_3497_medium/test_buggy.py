from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maxProfit([7,1,5,3,6,4])
    assert result == 7


def test_case_1():
    sol = Solution()
    result = sol.maxProfit([1,2,3,4,5])
    assert result == 4


def test_case_2():
    sol = Solution()
    result = sol.maxProfit([7,6,4,3,1])
    assert result == 0
