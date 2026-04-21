from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.reachableNodes(7, [[0,1],[1,2],[3,1],[4,0],[0,5],[5,6]], [4,5])
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.reachableNodes(7, [[0,1],[0,2],[0,5],[0,4],[3,2],[6,5]], [4,2,1])
    assert result == 3
