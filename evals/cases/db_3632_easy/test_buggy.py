from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findComplement(5)
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.findComplement(1)
    assert result == 0
