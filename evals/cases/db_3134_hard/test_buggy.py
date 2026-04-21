from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.countSubgraphsForEachDiameter(4, [[1,2],[2,3],[2,4]])
    assert result == [3,4,0]


def test_case_1():
    sol = Solution()
    result = sol.countSubgraphsForEachDiameter(2, [[1,2]])
    assert result == [1]


def test_case_2():
    sol = Solution()
    result = sol.countSubgraphsForEachDiameter(3, [[1,2],[2,3]])
    assert result == [2,1]
