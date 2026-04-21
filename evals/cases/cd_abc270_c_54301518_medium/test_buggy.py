import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 2 5\n1 2\n1 3\n3 4\n3 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 1 3 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 1 2\n3 1\n2 5\n1 2\n4 1\n2 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2 1\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
