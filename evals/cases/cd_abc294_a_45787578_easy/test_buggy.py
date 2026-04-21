import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n1 2 3 5 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n2 2 2 3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 2 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n22 3 17 8 30 15 12 14 11 17\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '22 8 30 12 14\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n17 62 2 60 23\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '62 2 60\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='99\n73 34 75 7 46 82 84 29 41 32 31 52 32 60 51 95 37 1 68 48 35 18 1 68 26 27 11 47 78 73 54 27 91 48 2 63 60 77 75 80 41 79 60 2 81 51 59 33 96 92 67 76 89 5 84 49 76 68 80 63 33 96 22 72 19 56 51 89 29 25 61 9 19 73 92 83 27 55 81 15 67 2 95 95 6 55 54 98 95 59 54 70 51 69 38 93 75 20 86\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '34 46 82 84 32 52 32 60 68 48 18 68 26 78 54 48 2 60 80 60 2 96 92 76 84 76 68 80 96 22 72 56 92 2 6 54 98 54 70 38 20 86\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
