from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findValueOfPartition([1, 3, 2, 4])
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.findValueOfPartition([100, 1, 10])
    assert result == 9
