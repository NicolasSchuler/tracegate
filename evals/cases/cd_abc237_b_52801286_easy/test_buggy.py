import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 3\n1 2 3\n4 5 6\n7 8 9\n10 11 12\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 4 7 10\n2 5 8 11\n3 6 9 12\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n1000000000 1000000000\n1000000000 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1000000000 1000000000\n1000000000 1000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\n12345\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '12345\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
