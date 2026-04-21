from buggy import invoke


def test_none_callable_control_case() -> None:
    assert invoke() == "default"
