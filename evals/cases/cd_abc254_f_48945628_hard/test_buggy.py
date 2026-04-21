import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 5\n3 5 2\n8 1 3\n1 2 2 3\n1 3 1 3\n1 1 1 1\n2 2 2 2\n3 3 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n1\n11\n6\n10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\n9\n100\n1 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '109\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
