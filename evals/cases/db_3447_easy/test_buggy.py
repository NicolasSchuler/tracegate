from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.arrayRankTransform([40,10,20,30])
    assert result == [4,1,2,3]


def test_case_1():
    sol = Solution()
    result = sol.arrayRankTransform([100,100,100])
    assert result == [1,1,1]


def test_case_2():
    sol = Solution()
    result = sol.arrayRankTransform([37,12,28,9,100,56,80,5,12])
    assert result == [5,3,4,2,8,6,7,1,3]
