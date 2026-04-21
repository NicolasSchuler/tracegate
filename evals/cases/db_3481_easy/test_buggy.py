from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.largestInteger(1234)
    assert result == 3412


def test_case_1():
    sol = Solution()
    result = sol.largestInteger(65875)
    assert result == 87655
