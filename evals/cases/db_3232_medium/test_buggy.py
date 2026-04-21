from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.minimumPartition("165462", 60)
    assert result == 4


def test_case_1():
    sol = Solution()
    result = sol.minimumPartition("238182", 5)
    assert result == -1
