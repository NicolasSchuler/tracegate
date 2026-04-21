from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.countPairs([3,1,2,2,2,1,3], 2)
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.countPairs([1,2,3,4], 1)
    assert result == 0
