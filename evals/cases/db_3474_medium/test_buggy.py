from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.arrayChange([1,2,4,6], [[1,3],[4,7],[6,1]])
    assert result == [3,2,7,1]


def test_case_1():
    sol = Solution()
    result = sol.arrayChange([1,2], [[1,3],[2,1],[3,2]])
    assert result == [2,1]
