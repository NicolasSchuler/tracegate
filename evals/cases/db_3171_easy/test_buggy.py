from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.smallerNumbersThanCurrent([8,1,2,2,3])
    assert result == [4,0,1,1,3]


def test_case_1():
    sol = Solution()
    result = sol.smallerNumbersThanCurrent([6,5,4,8])
    assert result == [2,1,0,3]


def test_case_2():
    sol = Solution()
    result = sol.smallerNumbersThanCurrent([7,7,7,7])
    assert result == [0,0,0,0]
