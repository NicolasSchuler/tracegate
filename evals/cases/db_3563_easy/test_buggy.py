from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findMaxK([-1,2,-3,3])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.findMaxK([-1,10,6,7,-7,1])
    assert result == 7


def test_case_2():
    sol = Solution()
    result = sol.findMaxK([-10,8,6,7,-2,-3])
    assert result == -1
