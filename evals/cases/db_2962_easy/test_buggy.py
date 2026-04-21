from buggy import Solution


def test_case_0():
    sol = Solution()
    result = sol.fizzBuzz(3)
    assert result == ["1","2","Fizz"]


def test_case_1():
    sol = Solution()
    result = sol.fizzBuzz(5)
    assert result == ["1","2","Fizz","4","Buzz"]


def test_case_2():
    sol = Solution()
    result = sol.fizzBuzz(15)
    assert result == ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
