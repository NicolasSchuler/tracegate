from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maxPoints([[1,1],[2,2],[3,3]])
    assert result == 3


def test_case_1():
    sol = Solution()
    result = sol.maxPoints([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]])
    assert result == 4
