from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.kthFactor(12, 3)
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.kthFactor(7, 2)
    assert result == 7


def test_case_2():
    sol = Solution()
    result = sol.kthFactor(4, 4)
    assert result == -1
