from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minBitFlips(10, 7)
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.minBitFlips(3, 4)
    assert result == 3
