from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.pivotArray([9,12,5,10,14,3,10], 10)
    assert result == [9,5,3,10,10,12,14]


def test_case_1():
    sol = Solution()
    result = sol.pivotArray([-3,4,3,2], 2)
    assert result == [-3,2,4,3]
