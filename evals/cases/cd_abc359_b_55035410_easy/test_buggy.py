import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n1 2 1 3 2 3\n',
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
        input='2\n1 1 2 2\n',
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
        input='4\n4 3 2 3 2 1 4 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n93 61 78 65 74 89 57 6 93 74 68 70 99 55 7 40 56 50 60 32 30 51 21 8 17 66 2 31 84 24 72 33 50 22 64 98 55 36 89 58 75 87 84 88 5 53 86 87 22 30 85 57 77 32 4 79 31 83 82 76 59 73 16 54 60 90 23 49 47 49 92 19 29 40 95 72 99 66 96 28 27 26 77 3 21 85 83 92 39 44 38 7 62 63 19 76 25 100 79 100 14 27 91 91 10 9 98 14 17 47 78 6 25 96 43 18 20 46 64 80 39 1 68 62 34 61 18 20 13 4 94 12 71 94 81 33 34 42 8 48 37 69 81 29 44 56 41 48 67 38 15 53 3 70 69 63 42 35 24 73 43 1 46 2 86 5 75 52 26 58 35 15 36 9 16 28 95 13 52 23 67 97 88 11 59 71 82 12 41 65 97 45 54 51 80 11 37 45 10 90\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n84 45 73 14 67 1 48 88 51 84 86 58 69 76 8 18 11 17 91 16 75 25 66 87 44 86 56 50 11 80 20 22 46 58 22 33 60 25 55 45 62 15 18 2 81 47 61 59 28 3 35 42 34 2 40 31 52 100 79 50 64 52 96 49 33 54 12 4 64 37 55 57 19 68 43 4 30 65 71 29 6 19 69 95 91 43 66 31 79 28 77 26 83 94 78 97 68 56 92 35 95 67 74 51 21 39 63 23 57 34 96 89 87 80 89 70 83 5 77 71 23 40 13 93 59 81 36 13 72 1 42 32 39 32 72 41 47 82 12 49 10 46 62 20 41 9 65 82 38 36 70 85 53 14 73 37 27 48 76 15 44 8 21 7 99 6 3 98 38 97 85 74 26 60 7 90 54 30 5 78 61 16 24 27 17 99 100 92 10 53 98 29 94 88 9 90 63 24 75 93\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
