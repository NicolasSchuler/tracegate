import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n5\n2 1 4 5 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n1 2 3 4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n6\n2 5 3 4 3 4\n5\n10000 1000 100 1 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n1 3 4 5 6\n2\n4 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
