import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 -2 -1\n',
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
        input='3\n0 0\n',
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
        input='6\n10 20 30 40 50\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-150\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='73\n34 71 -64 18 -43 -71 68 62 -40 64 70 13 -9 -97 -65 98 88 19 29 27 74 -29 -61 -67 42 16 -69 -24 17 17 79 -44 -20 -46 26 -80 -26 -17 14 22 -20 59 95 -7 2 34 -80 -93 68 -46 8 -11 9 -93 23 -77 4 62 -58 39 -17 52 -65 98 -92 81 96 -4 -99 55 57 93\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-269\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='92\n2 -97 -26 -8 -35 73 64 3 72 -56 13 -24 13 -87 -68 -67 9 -27 -49 88 86 -69 99 95 -29 -35 -64 85 26 -58 -32 -24 -2 -78 -14 -12 -26 -14 -20 76 -71 -97 -86 -48 1 58 99 78 -37 -75 -58 -96 -5 25 -65 80 14 63 -95 33 86 -92 73 32 22 38 -100 -81 -34 34 -58 89 89 -16 16 25 -21 -60 77 -15 27 69 -48 -23 66 44 82 -9 25 -45 -94\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '301\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
