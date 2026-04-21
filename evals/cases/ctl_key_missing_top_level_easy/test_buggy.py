from buggy import read_port


def test_key_control_case() -> None:
    assert read_port({"port": 8080}) == 8080
