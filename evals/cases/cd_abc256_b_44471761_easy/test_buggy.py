import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 1 3 2\n',
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
        input='3\n1 1 1\n',
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
        input='10\n2 2 4 1 1 1 4 2 2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n3 1 1 3 1 4 2 4 3 4 4 4 2 1 4 2 1 4 4 4 4 2 2 1 1 2 3 3 1 3 1 3 1 3 4 1 4 3 4 3 3 4 2 1 2 3 2 2 4 4 1 1 4 2 2 3 1 4 1 2 1 2 3 4 1 3 4 2 2 2 1 4 1 4 1 2 2 1 1 1 1 3 3 2 3 1 2 3 3 4 2 2 2 2 3 2 4 4 1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100\n4 3 4 2 4 2 1 1 1 3 2 4 3 1 1 2 3 2 1 2 4 3 1 3 2 3 2 1 3 4 1 4 3 3 2 3 1 3 4 1 4 4 2 4 4 3 4 1 1 3 2 1 4 3 4 4 4 2 1 2 1 1 2 3 4 1 3 1 2 2 2 4 4 2 1 4 2 2 3 3 3 3 2 2 4 3 2 1 1 4 2 4 4 4 2 1 4 3 2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '99\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
