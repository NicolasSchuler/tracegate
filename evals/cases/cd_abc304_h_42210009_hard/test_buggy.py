import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 4\n1 5\n2 1\n2 5\n4 3\n1 5\n1 3\n3 4\n1 3\n4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n3 1 4 2 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2\n1 2\n2 1\n1 2\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
