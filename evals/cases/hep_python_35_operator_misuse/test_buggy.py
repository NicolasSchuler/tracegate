from buggy import *







def check(max_element):
    assert max_element([1, 2, 3]) == 3
    assert max_element([5, 3, -5, 2, -3, 3, 9, 0, 124, 1, -10]) == 124

check(max_element)


def test_main():
    check(max_element)
