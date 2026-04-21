from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findPeakElement([1,2,3,1])
    assert result == 2


def test_case_1():
    sol = Solution()
    result = sol.findPeakElement([1,2,1,3,5,6,4])
    assert result == 5
