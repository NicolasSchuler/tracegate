from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.fib(2)
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.fib(3)
    assert result == 2


def test_case_2():
    sol = Solution()
    result = sol.fib(4)
    assert result == 3
