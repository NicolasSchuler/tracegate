import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 3\n1 3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 1 5 4 3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 0\n\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 6\n1 2 3 7 8 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4 3 2 1 5 6 10 9 8 7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='85 25\n2 3 4 6 12 13 21 24 28 31 32 42 44 48 50 51 58 65 66 69 71 76 77 78 81\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 5 4 3 2 7 6 8 9 10 11 14 13 12 15 16 17 18 19 20 22 21 23 25 24 26 27 29 28 30 33 32 31 34 35 36 37 38 39 40 41 43 42 45 44 46 47 49 48 52 51 50 53 54 55 56 57 59 58 60 61 62 63 64 67 66 65 68 70 69 72 71 73 74 75 79 78 77 76 80 82 81 83 84 85\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='67 1\n11\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5 6 7 8 9 10 12 11 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
