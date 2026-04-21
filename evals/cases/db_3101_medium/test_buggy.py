from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.sequentialDigits(100, 300)
    assert result == [123,234]


def test_case_1():
    sol = Solution()
    result = sol.sequentialDigits(1000, 13000)
    assert result == [1234,2345,3456,4567,5678,6789,12345]
