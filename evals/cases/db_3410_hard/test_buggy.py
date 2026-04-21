from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.countAnagrams("too hot")
    assert result == 18


def test_case_1():
    sol = Solution()
    result = sol.countAnagrams("aa")
    assert result == 1
