from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.thirdMax([3,2,1])
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.thirdMax([1,2])
    assert result == 2


def test_case_2():
    sol = Solution()
    result = sol.thirdMax([2,2,3,1])
    assert result == 1
