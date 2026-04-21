from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.findTheArrayConcVal([7,52,2,4])
    assert result == 596


def test_case_1():
    sol = Solution()
    result = sol.findTheArrayConcVal([5,14,13,8,12])
    assert result == 673
