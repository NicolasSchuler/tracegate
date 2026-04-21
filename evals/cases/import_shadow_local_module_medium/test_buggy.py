from buggy import median_value


def test_shadow_local_module() -> None:
    assert median_value([1, 3, 2]) == 2.0
