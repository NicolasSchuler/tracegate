from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.flipgame([1,2,4,4,7], [1,3,4,1,3])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.flipgame([1], [1])
    assert result == 0
