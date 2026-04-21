from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.answerQueries([4,5,2,1], [3,10,21])
    assert result == [2,3,4]


def test_case_1():
    sol = Solution()
    result = sol.answerQueries([2,3,4,5], [1])
    assert result == [0]
