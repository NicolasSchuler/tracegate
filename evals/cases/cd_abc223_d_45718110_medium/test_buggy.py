import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 3\n2 1\n3 4\n2 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 1 3 4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\n1 2\n1 2\n2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
