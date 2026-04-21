import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 8\n1 1 2 3 1 2\n1 1\n1 2\n1 3\n1 4\n2 1\n2 2\n2 3\n4 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n5\n-1\n3\n6\n-1\n-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\n0 1000000000 999999999\n1000000000 1\n123456789 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
