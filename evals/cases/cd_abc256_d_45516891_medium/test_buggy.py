import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n10 20\n20 30\n40 50\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10 30\n40 50\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n10 40\n30 60\n20 50\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10 60\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
