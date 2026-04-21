import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\r\n1 2 3 4 5\r\n',
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
        input='5\r\n1 6 1 6 1\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\r\n1 6 6 6 1\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\r\n1\r\n',
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
        input='1\r\n1000000000\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
