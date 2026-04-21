from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.deleteGreatestValue([[1,2,4],[3,3,1]])
    assert result == 8


def test_case_1():
    sol = Solution()
    result = sol.deleteGreatestValue([[10]])
    assert result == 10
