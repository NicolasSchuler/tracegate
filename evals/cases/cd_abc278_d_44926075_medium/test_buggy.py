import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n3 1 4 1 5\n6\n3 2\n2 3 4\n3 3\n1 1\n2 3 4\n3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n8\n5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1000000000\n8\n2 1 1000000000\n2 1 1000000000\n2 1 1000000000\n2 1 1000000000\n2 1 1000000000\n2 1 1000000000\n2 1 1000000000\n3 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n1 8 4 15 7 5 7 5 8 0\n20\n2 7 0\n3 7\n3 8\n1 7\n3 3\n2 4 4\n2 4 9\n2 10 5\n1 10\n2 4 2\n1 10\n2 3 1\n2 8 11\n2 3 14\n2 1 9\n3 8\n3 8\n3 1\n2 6 5\n3 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7\n5\n7\n21\n21\n19\n10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
