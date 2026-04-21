import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n0 1 1 0\n1 0 0 1\n1 0 0 0\n0 1 0 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 3\n1 4\n1\n2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n0 1 0 1 1\n1 0 0 1 0\n0 0 0 0 1\n1 1 0 0 1\n1 0 1 1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 4 5\n1 4\n5\n1 2 5\n1 3 4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
