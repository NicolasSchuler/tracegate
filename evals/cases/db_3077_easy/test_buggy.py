from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.countPoints("B0B6G0R6R0R6G9")
    assert result == 1


def test_case_1():
    sol = Solution()
    result = sol.countPoints("B0R0G0R9R0B0G0")
    assert result == 1


def test_case_2():
    sol = Solution()
    result = sol.countPoints("G4")
    assert result == 0
