import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n2 7 2\n',
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
        input='1\n3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='34\n50 71 98 82 28 19 56 57 58 75 96 33 42 52 52 24 85 90 13 31 62 53 23 55 71 15 22 80 87 2 25 41 59 99\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1806\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='59\n17 47 77 1 61 62 11 15 84 23 13 14 15 38 33 48 16 95 40 57 4 74 12 1 20 95 36 48 71 38 39 76 47 13 8 17 40 3 94 70 3 62 66 78 61 70 69 58 5 44 55 72 21 15 35 82 41 80 22\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2512\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='58\n85 63 44 82 26 84 17 84 89 20 96 41 14 66 21 14 70 73 47 54 1 94 59 24 89 96 37 2 20 52 52 47 88 38 3 5 19 16 23 95 89 43 77 37 47 93 90 94 74 63 96 96 11 44 90 33 2 49\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3078\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
