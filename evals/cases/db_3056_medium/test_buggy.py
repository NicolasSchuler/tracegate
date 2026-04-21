from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.gridGame([[2,5,4],[1,5,1]])
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.gridGame([[3,3,1],[8,5,2]])
    assert result == 4


def test_case_2():
    sol = Solution()
    result = sol.gridGame([[1,3,1,15],[1,3,3,1]])
    assert result == 7
