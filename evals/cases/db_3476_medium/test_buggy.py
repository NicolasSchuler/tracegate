from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minFlips(2, 6, 5)
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.minFlips(4, 2, 7)
    assert result == 1


def test_case_2():
    sol = Solution()
    result = sol.minFlips(1, 2, 3)
    assert result == 0
