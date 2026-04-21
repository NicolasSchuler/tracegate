from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.longestValidSubstring("cbaaaabc", ["aaa","cb"])
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.longestValidSubstring("leetcode", ["de","le","e"])
    assert result == 4
