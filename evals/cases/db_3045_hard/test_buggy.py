from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.slidingPuzzle([[1,2,3],[4,0,5]])
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.slidingPuzzle([[1,2,3],[5,4,0]])
    assert result == -1


def test_case_2():
    sol = Solution()
    result = sol.slidingPuzzle([[4,1,2],[5,0,3]])
    assert result == 5
