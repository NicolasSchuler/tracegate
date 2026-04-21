import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n3 1 2\n',
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
        input='10\n9 8 6 5 10 3 1 2 4 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9 8 6 5 10 2 7 4 3 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n17 59 72 100 13 44 32 48 46 15 84 22 75 38 92 4 63 31 25 43 19 62 14 20 85 71 87 65 2 27 24 80 37 6 94 77 66 8 81 56 21 39 33 74 82 83 58 99 96 89 16 40 42 10 76 60 70 73 9 3 61 95 53 45 1 5 7 11 12 18 23 26 28 29 30 34 35 36 41 47 49 50 51 52 54 55 57 64 67 68 69 78 79 86 88 90 91 93 97 98\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '17 59 72 100 13 44 32 48 46 15 84 22 75 38 92 4 63 31 25 43 19 62 14 20 85 71 87 65 2 27 24 80 37 6 94 77 66 8 81 56 21 39 33 74 82 83 58 99 96 89 16 40 42 10 76 60 70 73 9 3 61 95 53 41 98 97 93 91 90 88 86 79 78 69 68 67 64 57 55 54 52 51 50 49 47 45 36 35 34 30 29 28 26 23 18 12 11 7 5 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n46 91 93 45 13 55 1 2 3 4 5 6 7 8 9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 47 48 49 50 51 52 53 54 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 92 94 95 96 97 98 99 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '46 91 93 45 13 54 100 99 98 97 96 95 94 92 90 89 88 87 86 85 84 83 82 81 80 79 78 77 76 75 74 73 72 71 70 69 68 67 66 65 64 63 62 61 60 59 58 57 56 55 53 52 51 50 49 48 47 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 12 11 10 9 8 7 6 5 4 3 2 1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n42 40 69 27 57 92 98 62 13 24 82 83 37 56 16 59 60 30 99 76 55 45 28 79 78 34 50 39 97 19 35 68 6 89 80 85 66 54 36 5 17 22 88 20 1 95 2 71 33 51 4 70 90 73 67 84 25 63 61 15 23 44 7 18 93 87 48 12 52 91 29 94 100 3 14 41 81 21 96 31 32 38 64 47 11 72 74 75 10 53 9 43 49 46 77 86 58 26 8 65\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '42 40 69 27 57 92 98 62 13 24 82 83 37 56 16 59 60 30 99 76 55 45 28 79 78 34 50 39 97 19 35 68 6 89 80 85 66 54 36 5 17 22 88 20 1 95 2 71 33 51 4 70 90 73 67 84 25 63 61 15 23 44 7 18 93 87 48 12 52 91 29 94 100 3 14 41 81 21 96 31 32 38 64 47 11 72 74 75 10 53 9 43 49 46 77 86 58 8 65 26\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
