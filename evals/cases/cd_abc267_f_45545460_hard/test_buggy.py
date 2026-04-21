import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n1 2\n2 3\n3 4\n3 5\n3\n2 2\n5 3\n3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n1\n-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n1 2\n2 3\n3 5\n2 8\n3 4\n4 6\n4 9\n5 7\n9 10\n5\n1 1\n2 2\n3 3\n4 4\n5 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n4\n10\n-1\n-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n1 2\n4\n1 1\n1 2\n2 1\n2 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n-1\n1\n-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
