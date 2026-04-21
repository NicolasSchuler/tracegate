from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minimumTotalPrice(
        4, [[0, 1], [1, 2], [1, 3]], [2, 2, 10, 6], [[0, 3], [2, 1], [2, 3]]
    )
    assert result == 23


def test_case_1():
    sol = Solution()
    result = sol.minimumTotalPrice(2, [[0, 1]], [2, 2], [[0, 0]])
    assert result == 1
