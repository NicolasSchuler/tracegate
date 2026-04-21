from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.arrangeWords("Leetcode is cool")
    assert result == "Is cool leetcode"


def test_case_1():
    sol = Solution()
    result = sol.arrangeWords("Keep calm and code on")
    assert result == "On and keep calm code"


def test_case_2():
    sol = Solution()
    result = sol.arrangeWords("To be or not to be")
    assert result == "To be or to be not"
