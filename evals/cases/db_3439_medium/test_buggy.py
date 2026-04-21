from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minimumFuelCost([[0,1],[0,2],[0,3]], 5)
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.minimumFuelCost([[3,1],[3,2],[1,0],[0,4],[0,5],[4,6]], 2)
    assert result == 7


def test_case_2():
    sol = Solution()
    result = sol.minimumFuelCost([], 1)
    assert result == 0
