from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.longestCycle([3,3,4,2,3])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.longestCycle([2,-1,3,1])
    assert result == -1
