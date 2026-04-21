import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\r\n1 2 4 3\r\n4 3 1 2\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '())(\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\r\n1 3 2 4\r\n4 3 2 1\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\r\n2 1 5 3 4 6\r\n6 5 3 4 2 1\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\r\n2 1\r\n2 1\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = ')(\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
