from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maximumGap([3,6,9,1])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.maximumGap([10])
    assert result == 0
