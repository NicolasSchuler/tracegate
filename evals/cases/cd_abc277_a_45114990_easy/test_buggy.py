import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 3\n2 3 1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 2\n3 5 1 4 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 6\n1 2 3 4 5 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='89 11\n57 5 9 4 82 48 50 10 20 11 80 40 87 81 16 15 68 38 58 14 35 69 49 77 52 18 7 37 32 53 17 25 43 6 70 85 79 3 66 47 8 24 26 59 65 42 19 33 45 2 22 74 28 51 64 83 34 63 31 89 39 29 54 73 60 71 21 23 75 67 36 12 88 56 72 86 55 41 61 62 30 78 44 13 27 46 84 76 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='69 3\n17 68 9 26 1 47 40 11 55 43 10 63 22 37 58 35 2 51 7 30 36 45 12 32 15 34 44 46 8 13 24 48 41 62 5 27 29 54 23 50 33 4 53 19 67 69 64 52 16 56 3 39 38 59 49 20 31 14 18 60 65 21 25 57 6 66 42 61 28\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '51\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
