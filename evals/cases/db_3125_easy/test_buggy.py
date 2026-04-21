from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.totalMoney(4)
    assert result == 10


def test_case_1():
    sol = Solution()
    result = sol.totalMoney(10)
    assert result == 37


def test_case_2():
    sol = Solution()
    result = sol.totalMoney(20)
    assert result == 96
