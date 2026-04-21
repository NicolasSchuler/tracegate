import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4 3\n3 1 1 2\n1 1 2 2\n3 4 1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2 2\n1 1 2 2\n2 1 1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 10 10\n4 1 2 7\n5 1 3 2\n2 9 4 4\n5 4 2 9\n2 9 1 9\n4 8 3 10\n5 7 1 5\n3 5 1 2\n3 8 2 10\n2 9 4 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '12\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
