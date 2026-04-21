import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n1 0 1 0 0\n1 2 3 4 5\n0 2 2 0 0\n1 1 1 1 5\n0 1 2 0 0\n1 1 1 5 3\n1 1 1 0 0\n1 1 1 1 1\n0 0 0 0 1\n1 1 1 1 1\n100000000 100000000 100000000 0 0\n100000000 100000000 100000000 100000000 100000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n2\n3\n2\n0\n15000000000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n0 0 1 0 0\n1 1 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n1 0 0 0 0\n1 1 1 1 1\n0 1 0 0 0\n1 1 1 1 1\n0 0 1 0 0\n1 1 1 1 1\n0 0 0 1 0\n1 1 1 1 1\n0 0 0 0 1\n1 1 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n0\n0\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
