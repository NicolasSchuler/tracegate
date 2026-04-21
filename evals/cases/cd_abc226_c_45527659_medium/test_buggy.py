import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n3 0\n5 1 1\n7 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n1000000000 0\n1000000000 0\n1000000000 0\n1000000000 0\n1000000000 4 1 2 3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
