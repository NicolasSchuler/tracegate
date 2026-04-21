from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.convertToTitle(1)
    assert result == "A"


def test_case_1():
    sol = Solution()
    result = sol.convertToTitle(28)
    assert result == "AB"


def test_case_2():
    sol = Solution()
    result = sol.convertToTitle(701)
    assert result == "ZY"
