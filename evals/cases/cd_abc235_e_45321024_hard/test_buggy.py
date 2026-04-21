import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 6 3\n1 2 2\n2 3 3\n1 3 6\n2 4 5\n4 5 9\n3 5 8\n1 3 1\n3 4 7\n3 5 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nNo\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3 2\n1 2 100\n1 2 1000000000\n1 1 1\n1 2 2\n1 1 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
