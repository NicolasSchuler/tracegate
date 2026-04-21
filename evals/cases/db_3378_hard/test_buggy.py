from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.getMaxRepetitions("acb", 4, "ab", 2)
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.getMaxRepetitions("acb", 1, "acb", 1)
    assert result == 1
