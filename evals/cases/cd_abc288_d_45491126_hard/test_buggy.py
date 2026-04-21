import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 3\n3 -1 1 -2 2 0 5\n2\n1 6\n2 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='20 4\n-19 -66 -99 16 18 33 32 28 26 11 12 0 -16 4 21 21 37 17 55 -19\n5\n13 16\n4 11\n3 12\n13 18\n4 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nYes\nNo\nYes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
