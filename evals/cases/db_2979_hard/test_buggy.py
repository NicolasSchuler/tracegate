from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.componentValue([6,2,2,2,6], [[0,1],[1,2],[1,3],[3,4]])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.componentValue([2], [])
    assert result == 0
