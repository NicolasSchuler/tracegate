import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 2\n2 1\n2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2 4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 9\n6 5\n1 2\n3 4\n5 3\n6 2\n1 5\n3 2\n6 4\n1 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 3 6 7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 1\n3 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2 3 4 5 6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 6\n4 3\n4 2\n4 7\n1 2\n3 5\n6 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 3 4 5 6 7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 7\n6 1\n5 2\n4 1\n4 2\n6 2\n6 5\n3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3 4 5 6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
