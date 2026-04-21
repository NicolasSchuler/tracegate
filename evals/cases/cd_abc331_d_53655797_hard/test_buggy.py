import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\nWWB\nBBW\nWBW\n1 2 3 4\n0 3 4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 5\nBBBWWWBBBW\nWWWWWBBBWB\nBBBWBBWBBB\nBBBWWBWWWW\nWWWWBWBWBW\nWBBWBWBBBB\nWWBBBWWBWB\nWBWBWWBBBB\nWBWBWBBWWW\nWWWBWWBWWB\n5 21 21 93\n35 35 70 43\n55 72 61 84\n36 33 46 95\n0 0 999999999 999999999\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '621\n167\n44\n344\n500000000000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\nB\n0 0 999999999 999999999\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1000000000000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\nW\n0 0 999999999 999999999\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
