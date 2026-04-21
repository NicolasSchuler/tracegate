import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 7\n1 2\n1 4\n1 5\n2 4\n2 3\n3 5\n3 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n3\n2\n1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 7\n7 8\n3 4\n5 6\n5 7\n5 8\n6 7\n6 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n2\n2\n1\n1\n1\n1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
