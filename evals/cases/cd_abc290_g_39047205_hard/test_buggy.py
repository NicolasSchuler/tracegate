import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='11\n2 2 1\n2 2 2\n2 2 3\n2 2 4\n2 2 5\n2 2 6\n2 2 7\n1 999999999999999999 1\n1 999999999999999999 2\n1 999999999999999999 999999999999999999\n1 999999999999999999 1000000000000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n1\n1\n2\n1\n0\n1\n999999999999999998\n1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1 2 1\n',
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
        input='3\n1 2 1\n1 2 2\n1 2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n2 2 1\n2 2 2\n2 2 3\n2 2 4\n2 2 5\n2 2 6\n2 2 7\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n1\n1\n2\n1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='15\n3 2 1\n3 2 2\n3 2 3\n3 2 4\n3 2 5\n3 2 6\n3 2 7\n3 2 8\n3 2 9\n3 2 10\n3 2 11\n3 2 12\n3 2 13\n3 2 14\n3 2 15\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n1\n2\n2\n2\n1\n1\n2\n3\n2\n1\n2\n1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
