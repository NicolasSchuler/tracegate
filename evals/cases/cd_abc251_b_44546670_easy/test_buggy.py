import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 10\n1 3\n',
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
        input='2 1\n2 3\n',
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
        input='4 12\n3 3 3 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 251\n202 20 5 1 4 2 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '48\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='20 100\n36 49 73 38 30 85 27 45 65 71 52 86 52 48 60 40 98 37 38 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '60\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
