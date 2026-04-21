from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minimumDifference([90], 1)
    assert result == 0


def test_case_1():
    sol = Solution()
    result = sol.minimumDifference([9,4,1,7], 2)
    assert result == 2
