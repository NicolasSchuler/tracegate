from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findKthBit(3, 1)
    assert result == "0"


def test_case_1():
    sol = Solution()
    result = sol.findKthBit(4, 11)
    assert result == "1"
