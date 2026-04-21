from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minPairSum([3,5,2,3])
    assert result == 7


def test_case_1():
    sol = Solution()
    result = sol.minPairSum([3,5,4,2,4,6])
    assert result == 8
