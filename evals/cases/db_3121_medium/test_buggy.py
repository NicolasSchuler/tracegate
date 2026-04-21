from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.longestSubstring("aaabb", 3)
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.longestSubstring("ababbc", 2)
    assert result == 5
