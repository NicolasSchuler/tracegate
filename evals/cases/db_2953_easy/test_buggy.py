from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.nextGreaterElement([4,1,2], [1,3,4,2])
    assert result == [-1,3,-1]


def test_case_1():
    sol = Solution()
    result = sol.nextGreaterElement([2,4], [1,2,3,4])
    assert result == [3,-1]
