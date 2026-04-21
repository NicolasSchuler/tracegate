from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.palindromePairs(["abcd", "dcba", "lls", "s", "sssll"])
    assert result == [[0, 1], [1, 0], [3, 2], [2, 4]]


def test_case_1():
    sol = Solution()
    result = sol.palindromePairs(["bat", "tab", "cat"])
    assert result == [[0, 1], [1, 0]]


def test_case_2():
    sol = Solution()
    result = sol.palindromePairs(["a", ""])
    assert result == [[0, 1], [1, 0]]
