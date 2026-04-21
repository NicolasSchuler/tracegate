import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 7\n2 4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 60\n10 20 30 40 50\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '37634180\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 1000000000000000000\n1 2 4 8 16 32 64 128 256 512\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '75326268\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 2\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
