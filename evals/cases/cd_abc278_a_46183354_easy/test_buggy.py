import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\n2 7 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4\n9 9 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9 5\n1 2 3 4 5 6 7 8 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6 7 8 9 0 0 0 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='73 38\n30 85 27 45 65 71 52 86 52 48 60 40 98 37 38 5 92 34 30 93 70 39 83 81 67 21 7 21 85 39 51 91 58 81 34 54 38 40 24 57 5 40 70 30 53 28 94 25 70 87 20 77 76 50 33 60 41 84 69 14 6 81 49 1 67 55 27 38 95 80 6 22 22\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '24 57 5 40 70 30 53 28 94 25 70 87 20 77 76 50 33 60 41 84 69 14 6 81 49 1 67 55 27 38 95 80 6 22 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='42 72\n1 36 36 71 44 94 97 65 1 3 94 83 89 17 86 79 10 92 63 9 47 62 19 38 51 96 85 96 59 31 58 67 33 43 60 72 20 94 83 44 93 73\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
