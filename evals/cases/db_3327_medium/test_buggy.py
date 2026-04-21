from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.restoreArray([[2,1],[3,4],[3,2]])
    assert result == [1,2,3,4]


def test_case_1():
    sol = Solution()
    result = sol.restoreArray([[4,-2],[1,4],[-3,1]])
    assert result == [-2,4,1,-3]


def test_case_2():
    sol = Solution()
    result = sol.restoreArray([[100000,-100000]])
    assert result == [100000,-100000]
