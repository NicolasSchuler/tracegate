from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minimumDeletions([2, 10, 7, 5, 4, 1, 8, 6])
    assert result == 5


def test_case_1():
    sol = Solution()
    result = sol.minimumDeletions([0, -4, 19, 1, 8, -2, -3, 5])
    assert result == 3


def test_case_2():
    sol = Solution()
    result = sol.minimumDeletions([101])
    assert result == 1
