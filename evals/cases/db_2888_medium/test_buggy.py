from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.lexicalOrder(13)
    assert result == [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]


def test_case_1():
    sol = Solution()
    result = sol.lexicalOrder(2)
    assert result == [1, 2]
