from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maxNiceDivisors(5)
    assert result == 6


def test_case_1():
    sol = Solution()
    result = sol.maxNiceDivisors(8)
    assert result == 18
