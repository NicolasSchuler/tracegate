from buggy import square_root


def test_import_typo_control_case() -> None:
    assert square_root(9.0) == 3.0
