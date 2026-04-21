import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0\n0 0 1\n0 1 0\n1 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0\n0 0 1\n0 0 2\n0 1 0\n0 1 1\n0 2 0\n1 0 0\n1 0 1\n1 1 0\n2 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0\n0 0 1\n0 0 2\n0 0 3\n0 0 4\n0 0 5\n0 1 0\n0 1 1\n0 1 2\n0 1 3\n0 1 4\n0 2 0\n0 2 1\n0 2 2\n0 2 3\n0 3 0\n0 3 1\n0 3 2\n0 4 0\n0 4 1\n0 5 0\n1 0 0\n1 0 1\n1 0 2\n1 0 3\n1 0 4\n1 1 0\n1 1 1\n1 1 2\n1 1 3\n1 2 0\n1 2 1\n1 2 2\n1 3 0\n1 3 1\n1 4 0\n2 0 0\n2 0 1\n2 0 2\n2 0 3\n2 1 0\n2 1 1\n2 1 2\n2 2 0\n2 2 1\n2 3 0\n3 0 0\n3 0 1\n3 0 2\n3 1 0\n3 1 1\n3 2 0\n4 0 0\n4 0 1\n4 1 0\n5 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0 0\n0 0 1\n0 0 2\n0 0 3\n0 0 4\n0 0 5\n0 0 6\n0 1 0\n0 1 1\n0 1 2\n0 1 3\n0 1 4\n0 1 5\n0 2 0\n0 2 1\n0 2 2\n0 2 3\n0 2 4\n0 3 0\n0 3 1\n0 3 2\n0 3 3\n0 4 0\n0 4 1\n0 4 2\n0 5 0\n0 5 1\n0 6 0\n1 0 0\n1 0 1\n1 0 2\n1 0 3\n1 0 4\n1 0 5\n1 1 0\n1 1 1\n1 1 2\n1 1 3\n1 1 4\n1 2 0\n1 2 1\n1 2 2\n1 2 3\n1 3 0\n1 3 1\n1 3 2\n1 4 0\n1 4 1\n1 5 0\n2 0 0\n2 0 1\n2 0 2\n2 0 3\n2 0 4\n2 1 0\n2 1 1\n2 1 2\n2 1 3\n2 2 0\n2 2 1\n2 2 2\n2 3 0\n2 3 1\n2 4 0\n3 0 0\n3 0 1\n3 0 2\n3 0 3\n3 1 0\n3 1 1\n3 1 2\n3 2 0\n3 2 1\n3 3 0\n4 0 0\n4 0 1\n4 0 2\n4 1 0\n4 1 1\n4 2 0\n5 0 0\n5 0 1\n5 1 0\n6 0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
