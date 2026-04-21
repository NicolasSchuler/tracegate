import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 1 1 1 1\r\n',
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
        input='5 1 2 1 0\r\n',
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
        input='5 0 4 0 0\r\n',
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
        input='1 0 0 0 0\r\n',
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
        input='2 1 0 0 0\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
