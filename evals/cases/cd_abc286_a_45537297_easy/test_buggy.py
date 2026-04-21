import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 1 3 5 7\n1 2 3 4 5 6 7 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5 6 7 4 1 2 3 8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 2 3 4 5\n2 2 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 1 1 2 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1 1 2 2\n50 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '100 50\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 2 4 7 9\n22 75 26 45 72 81 47 29 97 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '22 47 29 97 72 81 75 26 45 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 1 50 51 100\n72 20 60 48 42 28 38 71 92 80 54 58 6 87 84 90 14 32 37 51 22 57 26 2 55 34 74 68 17 75 27 36 62 77 5 83 29 65 89 24 64 43 69 78 93 61 82 53 66 100 46 16 41 10 40 99 98 18 30 23 63 85 94 79 44 35 15 45 73 76 88 9 56 12 31 1 4 70 95 47 11 97 19 96 52 59 8 13 39 86 7 33 21 81 49 67 50 3 91 25\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '46 16 41 10 40 99 98 18 30 23 63 85 94 79 44 35 15 45 73 76 88 9 56 12 31 1 4 70 95 47 11 97 19 96 52 59 8 13 39 86 7 33 21 81 49 67 50 3 91 25 72 20 60 48 42 28 38 71 92 80 54 58 6 87 84 90 14 32 37 51 22 57 26 2 55 34 74 68 17 75 27 36 62 77 5 83 29 65 89 24 64 43 69 78 93 61 82 53 66 100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
