import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 5\n1 3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 3 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 15\n1 2 3 4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5 4 3 2 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 37\n9 2 1 3 8 7 10 4 5 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9 2 1 6 5 4 10 7 8 3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='36 1\n8 22 27 13 20 23 31 15 10 5 3 14 11 1 2 32 35 4 25 17 33 16 19 9 12 7 34 18 24 28 29 21 30 36 26 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 11 14 3 5 10 15 31 23 20 13 27 22 8 2 32 35 4 25 17 33 16 19 9 12 7 34 18 24 28 29 21 30 36 26 6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='54 1485\n49 16 50 12 21 35 44 40 25 17 3 15 13 22 7 34 54 8 52 20 32 37 29 1 53 24 39 5 10 51 33 6 47 2 23 43 36 11 31 4 27 30 42 45 14 38 18 41 48 28 46 9 19 26\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '54 34 7 22 13 15 3 17 25 40 44 35 21 12 50 16 49 8 52 20 32 37 29 1 53 24 39 5 10 51 33 6 47 2 23 43 36 11 31 4 27 30 42 45 14 38 18 41 48 28 46 9 19 26\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
