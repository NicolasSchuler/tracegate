from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findKthPositive([2,3,4,7,11], 5)
    assert result == 9


def test_case_1():
    sol = Solution()
    result = sol.findKthPositive([1,2,3,4], 2)
    assert result == 6
