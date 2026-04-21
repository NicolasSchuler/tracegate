from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.getMaximumConsecutive([1,3])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.getMaximumConsecutive([1,1,1,4])
    assert result == 8


def test_case_2():
    sol = Solution()
    result = sol.getMaximumConsecutive([1,4,10,3,1])
    assert result == 20
