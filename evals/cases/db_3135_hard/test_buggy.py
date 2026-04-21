from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.secondGreaterElement([2,4,0,9,6])
    assert result == [9,6,6,-1,-1]


def test_case_1():
    sol = Solution()
    result = sol.secondGreaterElement([3,3])
    assert result == [-1,-1]
