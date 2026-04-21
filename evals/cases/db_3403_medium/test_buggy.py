from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.getLastMoment(4, [4,3], [0,1])
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.getLastMoment(7, [], [0,1,2,3,4,5,6,7])
    assert result == 7


def test_case_2():
    sol = Solution()
    result = sol.getLastMoment(7, [0,1,2,3,4,5,6,7], [])
    assert result == 7
