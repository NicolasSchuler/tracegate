from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.mostCommonWord("Bob hit a ball, the hit BALL flew far after it was hit.", ["hit"])
    assert result == "ball"


def test_case_1():
    sol = Solution()
    result = sol.mostCommonWord("a.", [])
    assert result == "a"
