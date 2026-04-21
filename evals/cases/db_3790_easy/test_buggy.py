from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.plusOne([1,2,3])
    assert result == [1,2,4]


def test_case_1():
    sol = Solution()
    result = sol.plusOne([4,3,2,1])
    assert result == [4,3,2,2]


def test_case_2():
    sol = Solution()
    result = sol.plusOne([9])
    assert result == [1,0]
