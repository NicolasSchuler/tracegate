import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n4 1 -1 5 3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3 5 4 1 2 6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n-1 1 2 3 4 5 6 7 8 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5 6 7 8 9 10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='30\n3 25 20 6 18 12 26 1 29 -1 21 17 23 9 8 30 10 15 22 27 4 13 5 11 16 24 28 2 19 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10 17 12 6 4 21 11 24 26 7 30 16 25 2 28 27 20 3 1 8 15 18 5 23 13 22 19 29 9 14\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n-1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n-1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
