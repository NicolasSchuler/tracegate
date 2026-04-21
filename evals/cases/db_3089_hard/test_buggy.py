from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.numPermsDISequence("DID")
    assert result == 5


def test_case_1():
    sol = Solution()
    result = sol.numPermsDISequence("D")
    assert result == 1
