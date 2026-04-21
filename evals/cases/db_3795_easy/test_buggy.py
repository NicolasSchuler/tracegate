from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findLHS([1,3,2,2,5,2,3,7])
    assert result == 5


def test_case_1():
    sol = Solution()
    result = sol.findLHS([1,2,3,4])
    assert result == 2


def test_case_2():
    sol = Solution()
    result = sol.findLHS([1,1,1,1])
    assert result == 0
