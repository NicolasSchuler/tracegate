from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.getSumAbsoluteDifferences([2,3,5])
    assert result == [4,3,5]


def test_case_1():
    sol = Solution()
    result = sol.getSumAbsoluteDifferences([1,4,6,8,10])
    assert result == [24,15,13,15,21]
