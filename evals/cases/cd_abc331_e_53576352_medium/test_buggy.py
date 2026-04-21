import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3 3\n2 1\n10 30 20\n1 2\n2 1\n2 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '31\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1 0\n1000000000 1\n1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 10 10\n47718 21994 74148 76721 98917 73766 29598 59035 69293 29127\n7017 46004 16086 62644 74928 57404 32168 45794 19493 71590\n1 3\n2 6\n4 5\n5 4\n5 5\n5 6\n5 7\n5 8\n5 10\n7 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '149076\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
