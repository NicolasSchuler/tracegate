from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.sortString("aaaabbbbcccc")
    assert result == "abccbaabccba"


def test_case_1():
    sol = Solution()
    result = sol.sortString("rat")
    assert result == "art"
