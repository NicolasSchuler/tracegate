import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n3 1 4 5 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n2 4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='20\n9 7 19 7 10 4 13 9 4 8 10 15 16 3 18 19 12 13 2 12\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n1 2 5 6 8 11 14 17 18 20\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
