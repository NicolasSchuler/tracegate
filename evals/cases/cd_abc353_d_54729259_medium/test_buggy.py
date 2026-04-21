import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n3 14 15\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2044\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n1001 5 1000000 1000000000 100000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '625549048\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n1000000000 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '174754156\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
