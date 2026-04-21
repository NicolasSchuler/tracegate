from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.numWays("10101")
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.numWays("1001")
    assert result == 0


def test_case_2():
    sol = Solution()
    result = sol.numWays("0000")
    assert result == 3
