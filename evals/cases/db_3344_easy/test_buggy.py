from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.makeSmallestPalindrome("egcfe")
    assert result == "efcfe"


def test_case_1():
    sol = Solution()
    result = sol.makeSmallestPalindrome("abcd")
    assert result == "abba"


def test_case_2():
    sol = Solution()
    result = sol.makeSmallestPalindrome("seven")
    assert result == "neven"
