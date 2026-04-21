from buggy import min_with_original


def test_function_does_not_mutate_input_order() -> None:
    original = [3, 1, 2]
    smallest, seen = min_with_original(original)
    assert smallest == 1
    assert seen == [3, 1, 2]
