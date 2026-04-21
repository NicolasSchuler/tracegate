import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n3\n7 19 20\n4\n4 19 24 0\n2\n26 10\n3\n19 31 24\n19\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n1 4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n1\n1\n1\n2\n1\n3\n0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n37\n21 36 33 9 28 1 0 34 3 25 13 31 20 2 22 4 11 15 7 18 6 32 5 14 26 35 12 10 24 17 23 29 19 30 8 27 16\n36\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n0\n0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n36\n36\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
