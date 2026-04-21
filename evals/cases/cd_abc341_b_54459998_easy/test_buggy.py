import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n5 7 0 3\n2 2\n4 3\n5 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n32 6 46 9 37 8 33 14 31 5\n5 5\n3 1\n4 3\n2 2\n3 2\n3 2\n4 4\n3 3\n3 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '45\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n1000000000 1000000000\n1000000000 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1000000001\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
