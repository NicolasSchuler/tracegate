from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.firstPalindrome(["abc", "car", "ada", "racecar", "cool"])
    assert result == "ada"


def test_case_1():
    sol = Solution()
    result = sol.firstPalindrome(["notapalindrome", "racecar"])
    assert result == "racecar"


def test_case_2():
    sol = Solution()
    result = sol.firstPalindrome(["def", "ghi"])
    assert result == ""
