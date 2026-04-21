from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.jump([2,3,1,1,4])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.jump([2,3,0,1,4])
    assert result == 2
