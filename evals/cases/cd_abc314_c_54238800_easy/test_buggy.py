import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 3\napzbqrcs\n1 2 3 1 2 2 1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'cszapqbr\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1\naa\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aa\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\na\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'a\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
