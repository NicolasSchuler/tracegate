from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.kthLargestNumber(["3","6","7","10"], 4)
    assert result == "3"


def test_case_1():
    sol = Solution()
    result = sol.kthLargestNumber(["2","21","12","1"], 3)
    assert result == "2"


def test_case_2():
    sol = Solution()
    result = sol.kthLargestNumber(["0","0"], 2)
    assert result == "0"
