from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findWinners([[1,3],[2,3],[3,6],[5,6],[5,7],[4,5],[4,8],[4,9],[10,4],[10,9]])
    assert result == [[1,2,10],[4,5,7,8]]


def test_case_1():
    sol = Solution()
    result = sol.findWinners([[2,3],[1,3],[5,4],[6,4]])
    assert result == [[1,2,5,6],[]]
