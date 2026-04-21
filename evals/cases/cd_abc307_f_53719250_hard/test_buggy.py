import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 4\n1 2 2\n2 3 1\n2 4 3\n3 4 2\n1\n1\n2\n3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n1\n1\n2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 7\n1 2 2\n2 3 3\n3 4 1\n4 5 1\n5 6 3\n3 7 1\n4 7 1\n2\n1 6\n2\n2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n1\n2\n-1\n2\n0\n-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 1\n1 2 5\n2\n1 3\n3\n3 7 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n2\n0\n-1\n-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
