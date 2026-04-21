import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n2\n1 2\n2\n1 3\n1\n2\n',
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
        input='4 2\n2\n1 2\n2\n1 3\n',
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
        input='6 6\n3\n2 3 6\n3\n2 4 6\n2\n3 6\n3\n1 5 6\n3\n1 3 6\n2\n1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '18\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 6\n5\n2 3 4 5 8\n5\n1 3 5 7 8\n5\n1 2 4 5 8\n5\n1 3 4 6 8\n3\n1 7 8\n5\n1 2 3 7 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '23\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9 4\n5\n2 3 5 7 8\n4\n3 5 6 8\n8\n1 2 3 4 6 7 8 9\n3\n5 6 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
