import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n2 3\n1 1\n2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 5\n1 2\n2 3\n3 4\n3 5\n1 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='13 16\n7 9\n7 11\n3 8\n1 13\n11 11\n6 11\n8 13\n2 11\n3 3\n8 12\n9 11\n1 11\n5 13\n3 12\n6 9\n1 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 100\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
