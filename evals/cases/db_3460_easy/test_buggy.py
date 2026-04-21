from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.alternateDigitSum(521)
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.alternateDigitSum(111)
    assert result == 1


def test_case_2():
    sol = Solution()
    result = sol.alternateDigitSum(886996)
    assert result == 0
