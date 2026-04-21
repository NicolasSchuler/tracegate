from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.distributeCandies([1,1,2,2,3,3])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.distributeCandies([1,1,2,3])
    assert result == 2


def test_case_2():
    sol = Solution()
    result = sol.distributeCandies([6,6,6,6])
    assert result == 1
