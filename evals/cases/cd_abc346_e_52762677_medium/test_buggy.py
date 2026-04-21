import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4 4\n1 2 5\n2 4 0\n1 3 3\n1 3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n0 5\n2 4\n5 3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 5\n1 1 1\n1 1 10\n2 1 100\n1 1 1000\n2 1 10000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n10000 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 5 10\n1 1 1\n1 2 2\n1 3 3\n1 4 4\n1 5 5\n2 1 6\n2 2 7\n2 3 8\n2 4 9\n2 5 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n6 5\n7 5\n8 5\n9 5\n10 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
