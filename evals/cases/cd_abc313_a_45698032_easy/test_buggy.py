import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n5 15 2 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '11\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n15 5 2 10\n',
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
        input='3\n100 100 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='91\n75 26 45 72 81 47 29 97 2 75 25 82 84 17 56 32 2 28 37 57 39 18 11 79 6 40 68 68 16 40 63 93 49 91 10 55 68 31 80 57 18 34 28 76 55 21 80 22 45 11 67 67 74 91 4 35 34 65 80 21 95 1 52 25 31 2 53 96 22 89 99 7 66 32 2 68 33 75 92 84 10 94 28 54 12 9 80 43 21 51 92\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '25\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n97 7 25 67 17 38 100 86 4 83 20 6 81 58 59 53 2 54 62 25 35 79 64 27 49 32 95 100 20 58 39 92 30 67 89 58 81 100 66 73 29 75 81 70 55 18 28 7 35 98 52 30 11 69 48 84 54 13 14 15 86 34 82 92 26 8 53 62 57 50 31 61 85 88 5 80 64 90 52 47 43 40 93 69 70 16 43 7 25 99 12 63 99 71 76 55 17 90 43 27\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
