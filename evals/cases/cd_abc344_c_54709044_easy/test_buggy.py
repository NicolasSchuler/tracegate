import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n0\n1\n0\n1\n0\n1\n0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n1 2 3\n2\n2 4\n6\n1 2 4 8 16 32\n4\n1 5 10 50\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nYes\nYes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
