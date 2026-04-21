from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.generate(5)
    assert result == [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]


def test_case_1():
    sol = Solution()
    result = sol.generate(1)
    assert result == [[1]]
