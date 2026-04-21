from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findMiddleIndex([2,3,-1,8,4])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.findMiddleIndex([1,-1,4])
    assert result == 2


def test_case_2():
    sol = Solution()
    result = sol.findMiddleIndex([2,5])
    assert result == -1
