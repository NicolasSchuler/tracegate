from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.fourSumCount([1,2], [-2,-1], [-1,2], [0,2])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.fourSumCount([0], [0], [0], [0])
    assert result == 1
