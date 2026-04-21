from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maxPower([1,2,4,5,0], 1, 2)
    assert result == 5


def test_case_1():
    sol = Solution()
    result = sol.maxPower([4,4,4,4], 0, 3)
    assert result == 4
