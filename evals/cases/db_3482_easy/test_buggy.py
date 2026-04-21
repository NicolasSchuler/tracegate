from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.getMinDistance([1,2,3,4,5], 5, 3)
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.getMinDistance([1], 1, 0)
    assert result == 0


def test_case_2():
    sol = Solution()
    result = sol.getMinDistance([1,1,1,1,1,1,1,1,1,1], 1, 0)
    assert result == 0
