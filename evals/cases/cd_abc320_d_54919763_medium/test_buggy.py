import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\n1 2 2 1\n1 3 -1 -2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0\n2 1\n-1 -2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\n2 1 -2 -1\n2 3 -3 -3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0\n2 1\n-1 -2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 7\n1 2 0 0\n1 2 0 0\n2 3 0 0\n3 1 0 0\n2 1 0 0\n3 2 0 0\n4 5 0 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0 0\n0 0\n0 0\nundecidable\nundecidable\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
