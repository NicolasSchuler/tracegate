from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.tribonacci(4)
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.tribonacci(25)
    assert result == 1389537
