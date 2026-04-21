import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='12\n31 29 31 30 31 30 31 31 30 31 30 31\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '13\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n10 1 2 3 4 5 6 7 8 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='30\n73 8 55 26 97 48 37 47 35 55 5 17 62 2 60 23 99 73 34 75 7 46 82 84 29 41 32 31 52 32\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '15\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n60 51 95 37 1 68 48 35 18 1 68 26 27 11 47 78 73 54 27 91 48 2 63 60 77 75 80 41 79 60 2 81 51 59 33 96 92 67 76 89 5 84 49 76 68 80 63 33 96 22 72 19 56 51 89 29 25 61 9 19 73 92 83 27 55 81 15 67 2 95 95 6 55 54 98 95 59 54 70 51 69 38 93 75 20 86 73 70 91 31 49 79 46 69 31 29 97 41 80 88\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '26\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n19 78 76 24 48 65 82 21 56 21 73 51 12 85 94 5 18 11 49 49 9 17 67 5 65 80 1 11 67 90 87 82 59 92 15 24 20 27 64 43 36 25 66 6 97 87 39 67 18 43 66 37 49 81 38 36 97 25 4 52 66 82 60 87 17 74 8 68 60 100 95 1 60 19 52 3 75 44 94 83 66 95 44 35 34 97 40 58 19 32 82 91 39 25 38 66 50 61 12 56\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '25\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
