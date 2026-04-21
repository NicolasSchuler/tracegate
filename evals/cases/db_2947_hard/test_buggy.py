from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.longestPrefix("level")
    assert result == "l"


def test_case_1():
    sol = Solution()
    result = sol.longestPrefix("ababab")
    assert result == "abab"
