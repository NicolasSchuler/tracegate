from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.capitalizeTitle("capiTalIze tHe titLe")
    assert result == "Capitalize The Title"


def test_case_1():
    sol = Solution()
    result = sol.capitalizeTitle("First leTTeR of EACH Word")
    assert result == "First Letter of Each Word"


def test_case_2():
    sol = Solution()
    result = sol.capitalizeTitle("i lOve leetcode")
    assert result == "i Love Leetcode"
