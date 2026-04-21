import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 4 7\n3 1 4 9 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4 4 4 7 7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 10 10\n11 10 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10 10 10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
