from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.countGoodSubstrings("xyzzaz")
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.countGoodSubstrings("aababcabc")
    assert result == 4
