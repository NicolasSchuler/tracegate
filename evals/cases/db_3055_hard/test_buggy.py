from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minCost([[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.minCost([[1,1,3],[3,2,2],[1,1,4]])
    assert result == 0


def test_case_2():
    sol = Solution()
    result = sol.minCost([[1,2],[4,3]])
    assert result == 1
