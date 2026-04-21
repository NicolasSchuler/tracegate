import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='13 3\n9 5 2 7 1 8 8 2 1 5 2 3 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '26\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 1\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10000000009\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='30 8\n8 55 26 97 48 37 47 35 55 5 17 62 2 60 23 99 73 34 75 7 46 82 84 29 41 32 31 52 32 60\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '189\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
