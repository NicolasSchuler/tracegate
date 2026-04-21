from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.decodeCiphertext("ch   ie   pr", 3)
    assert result == "cipher"


def test_case_1():
    sol = Solution()
    result = sol.decodeCiphertext("iveo    eed   l te   olc", 4)
    assert result == "i love leetcode"


def test_case_2():
    sol = Solution()
    result = sol.decodeCiphertext("coding", 1)
    assert result == "coding"
