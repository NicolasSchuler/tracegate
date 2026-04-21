from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.calPoints(["5","2","C","D","+"])
    assert result == 30


def test_case_1():
    sol = Solution()
    result = sol.calPoints(["5","-2","4","C","D","9","+","+"])
    assert result == 27


def test_case_2():
    sol = Solution()
    result = sol.calPoints(["1","C"])
    assert result == 0
