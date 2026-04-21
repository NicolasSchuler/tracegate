import os

from buggy import read_value


def test_relative_cwd_shift(tmp_path) -> None:
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert read_value() == "hello"
    finally:
        os.chdir(previous)
