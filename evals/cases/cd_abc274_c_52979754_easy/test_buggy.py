import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n1\n1\n2\n2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 3 5 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n1\n1\n2\n2\n3\n3\n2\n2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
