import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 7\nabcdefgh\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'abgfedch\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 7\nreviver\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'reviver\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 13\nmerrychristmas\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'meramtsirhcyrs\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\nz\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'z\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
