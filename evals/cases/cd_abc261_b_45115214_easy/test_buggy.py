import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n-WWW\nL-DD\nLD-W\nLDW-\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'incorrect\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n-D\nD-\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'correct\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
