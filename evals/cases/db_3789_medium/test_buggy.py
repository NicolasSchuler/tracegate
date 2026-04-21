from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.smallestString("cbabc")
    assert result == "baabc"


def test_case_1():
    sol = Solution()
    result = sol.smallestString("acbbc")
    assert result == "abaab"


def test_case_2():
    sol = Solution()
    result = sol.smallestString("leetcode")
    assert result == "kddsbncd"
