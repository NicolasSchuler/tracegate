from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findMedianSortedArrays([1,3], [2])
    assert result == 2.00000


def test_case_1():
    sol = Solution()
    result = sol.findMedianSortedArrays([1,2], [3,4])
    assert result == 2.50000
