from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.maximumSumQueries([4,3,1,2], [2,4,9,5], [[4,1],[1,3],[2,5]])
    assert result == [6,10,7]


def test_case_1():
    sol = Solution()
    result = sol.maximumSumQueries([3,2,5], [2,3,4], [[4,4],[3,2],[1,1]])
    assert result == [9,9,9]


def test_case_2():
    sol = Solution()
    result = sol.maximumSumQueries([2,1], [2,3], [[3,3]])
    assert result == [-1]
