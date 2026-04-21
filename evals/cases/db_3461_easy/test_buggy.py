from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.selfDividingNumbers(1, 22)
    assert result == [1,2,3,4,5,6,7,8,9,11,12,15,22]


def test_case_1():
    sol = Solution()
    result = sol.selfDividingNumbers(47, 85)
    assert result == [48,55,66,77]
