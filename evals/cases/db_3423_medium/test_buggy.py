from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.countGoodNumbers(1)
    assert result == 5


def test_case_1():
    sol = Solution()
    result = sol.countGoodNumbers(4)
    assert result == 400


def test_case_2():
    sol = Solution()
    result = sol.countGoodNumbers(50)
    assert result == 564908303
