from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minimumOperations([1,5,0,3,5])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.minimumOperations([0])
    assert result == 0
