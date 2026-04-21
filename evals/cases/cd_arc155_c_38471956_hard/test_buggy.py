import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\r\n1 2 3 4 5\r\n3 1 2 4 5\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\r\n1 2 4 6 5\r\n5 1 4 2 6\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\r\n2 10 4 3 6 2 6 8 5\r\n2 4 10 3 8 6 6 2 5\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\r\n1 2 3\r\n2 1 3\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\r\n1 3 5\r\n3 1 5\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
