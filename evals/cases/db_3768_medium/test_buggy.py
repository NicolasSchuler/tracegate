from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.powerfulIntegers(2, 3, 10)
    assert result == [2,3,4,5,7,9,10]


def test_case_1():
    sol = Solution()
    result = sol.powerfulIntegers(3, 5, 15)
    assert result == [2,4,6,8,10,14]
