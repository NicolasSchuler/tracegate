from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.waysToSplitArray([10,4,-8,7])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.waysToSplitArray([2,3,1,0])
    assert result == 2
