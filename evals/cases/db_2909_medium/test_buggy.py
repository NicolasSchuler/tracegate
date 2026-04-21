from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.numOfPairs(["777","7","77","77"], "7777")
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.numOfPairs(["123","4","12","34"], "1234")
    assert result == 2


def test_case_2():
    sol = Solution()
    result = sol.numOfPairs(["1","1","1"], "11")
    assert result == 6
