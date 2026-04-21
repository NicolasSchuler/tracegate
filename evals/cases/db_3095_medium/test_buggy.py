from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findLonely([10,6,5,8])
    assert result == [10,8]


def test_case_1():
    sol = Solution()
    result = sol.findLonely([1,3,5,3])
    assert result == [1,5]
