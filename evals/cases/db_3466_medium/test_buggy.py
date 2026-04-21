from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.stoneGameVI([1,3], [2,1])
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.stoneGameVI([1,2], [3,1])
    assert result == 0


def test_case_2():
    sol = Solution()
    result = sol.stoneGameVI([2,4,3], [1,6,7])
    assert result == -1
