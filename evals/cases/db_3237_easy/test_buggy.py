from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.averageValue([1,3,6,10,12,15])
    assert result == 9


def test_case_1():
    sol = Solution()
    result = sol.averageValue([1,2,4,7,10])
    assert result == 0
