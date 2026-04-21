from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.sortArray([5,2,3,1])
    assert result == [1,2,3,5]


def test_case_1():
    sol = Solution()
    result = sol.sortArray([5,1,1,2,0,0])
    assert result == [0,0,1,1,2,5]
