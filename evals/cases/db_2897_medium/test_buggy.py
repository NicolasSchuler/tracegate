from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.sumSubarrayMins([3, 1, 2, 4])
    assert result == 17


def test_case_1():
    sol = Solution()
    result = sol.sumSubarrayMins([11, 81, 94, 43, 3])
    assert result == 444
