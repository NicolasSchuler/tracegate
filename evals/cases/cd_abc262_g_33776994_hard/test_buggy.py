import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n1 2 3 4 1 2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10\n1 1 1 1 1 1 1 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='49\n25 26 45 22 31 47 29 47 2 25 25 32 34 17 6 32 2 28 37 7 39 18 11 29 6 40 18 18 16 40 13 43 49 41 10 5 18 31 30 7 18 34 28 26 5 21 30 22 45\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '23\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='48\n17 17 24 41 4 35 34 15 30 21 45 1 2 25 31 2 3 46 22 39 49 7 16 32 2 18 33 25 42 34 10 44 28 4 12 9 30 43 21 1 42 20 47 7 25 17 17 38\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '19\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='48\n36 4 33 20 6 31 8 9 3 2 4 12 25 35 29 14 27 49 32 45 50 20 8 39 42 30 17 39 8 31 50 16 23 29 25 31 20 5 18 28 7 35 48 2 30 11 19 48\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '20\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
