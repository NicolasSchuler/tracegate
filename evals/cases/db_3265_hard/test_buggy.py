from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.numberOfArithmeticSlices([2,4,6,8,10])
    assert result == 7


def test_case_1():
    sol = Solution()
    result = sol.numberOfArithmeticSlices([7,7,7,7,7])
    assert result == 16
