import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 5 6 5\n1 1 6\n1 3 3\n2 2 4\n2 4 2\n1 1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 8 3 3 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1000000000 1000000000 3 5\n1 1 2\n1 2 2\n1 3 2\n1 4 2\n1 5 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 5000000000 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
