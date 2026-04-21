from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minSubsequence([4,3,10,9,8])
    assert result == [10,9]


def test_case_1():
    sol = Solution()
    result = sol.minSubsequence([4,4,7,6,7])
    assert result == [7,7,6]
