from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minOperations([2,3,2,4,3], [9,6,9,3,15])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.minOperations([4,3,6], [8,2,6,10])
    assert result == -1
