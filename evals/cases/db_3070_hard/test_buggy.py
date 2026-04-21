from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minimumTotalCost([1,2,3,4,5], [1,2,3,4,5])
    assert result == 10


def test_case_1():
    sol = Solution()
    result = sol.minimumTotalCost([2,2,2,1,3], [1,2,2,3,3])
    assert result == 10


def test_case_2():
    sol = Solution()
    result = sol.minimumTotalCost([1,2,2], [1,2,2])
    assert result == -1
