from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maxStrength([3,-1,-5,2,5,-9])
    assert result == 1350


def test_case_1():
    sol = Solution()
    result = sol.maxStrength([-4,-5,-4])
    assert result == 20
