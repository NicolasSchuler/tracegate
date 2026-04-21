import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 9\n1 1 2\n3 1 2\n1 2 1\n3 1 2\n1 2 3\n1 3 2\n3 1 3\n2 1 2\n3 1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nYes\nNo\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 8\n1 1 2\n1 2 1\n3 1 2\n1 1 2\n1 1 2\n1 1 2\n2 1 2\n3 1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 30\n3 1 6\n3 5 4\n1 6 1\n3 1 7\n3 8 4\n1 1 6\n2 4 3\n1 6 5\n1 5 6\n1 1 8\n1 8 1\n2 3 10\n1 7 6\n3 5 6\n1 6 7\n3 6 7\n1 9 5\n3 8 6\n3 3 8\n2 6 9\n1 7 1\n3 10 8\n2 9 2\n1 10 9\n2 6 10\n2 6 8\n3 1 6\n3 1 8\n2 8 5\n1 9 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nNo\nNo\nNo\nYes\nYes\nNo\nNo\nNo\nYes\nYes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 200\n3 2 8\n1 5 7\n1 7 5\n1 9 10\n1 10 9\n1 7 1\n3 9 10\n3 5 7\n1 1 7\n1 3 7\n3 9 10\n1 7 3\n1 4 10\n1 10 4\n3 2 8\n3 1 7\n2 7 3\n1 7 3\n1 4 5\n1 5 4\n3 4 3\n2 3 7\n1 3 7\n1 6 7\n3 7 10\n1 7 6\n2 8 3\n2 5 8\n1 1 4\n1 3 1\n1 3 5\n3 3 2\n1 4 1\n3 4 3\n1 5 3\n3 6 7\n1 1 3\n1 9 8\n2 8 5\n3 9 10\n3 6 3\n2 7 1\n3 5 7\n3 10 7\n1 7 1\n2 8 6\n3 6 4\n1 8 9\n3 1 3\n1 3 2\n1 2 3\n1 6 5\n2 9 8\n1 7 10\n2 8 2\n3 8 9\n1 10 7\n1 5 6\n2 3 1\n3 7 4\n2 7 4\n3 6 8\n3 5 7\n3 4 5\n1 3 1\n2 8 10\n1 3 10\n2 9 3\n2 5 4\n1 10 3\n1 9 8\n1 10 3\n3 1 8\n3 7 9\n1 5 4\n3 1 3\n3 1 7\n1 5 4\n3 1 3\n1 4 7\n1 7 4\n1 6 7\n1 1 2\n3 10 8\n1 2 1\n2 10 5\n1 6 1\n1 1 6\n3 7 10\n2 4 2\n1 10 1\n3 2 7\n3 6 10\n1 1 10\n2 6 5\n1 6 5\n3 3 6\n1 1 3\n1 6 7\n2 4 7\n1 4 7\n1 3 8\n1 8 3\n1 2 6\n3 3 5\n1 9 10\n1 6 2\n1 10 8\n3 4 5\n1 8 10\n3 4 10\n1 1 10\n1 2 9\n1 4 7\n3 7 9\n2 6 9\n1 9 2\n1 1 3\n3 5 6\n1 6 9\n1 9 6\n1 10 6\n1 6 10\n1 4 5\n3 1 2\n2 1 6\n3 2 4\n1 1 6\n2 7 2\n3 3 10\n1 5 6\n2 1 7\n3 2 10\n2 1 5\n3 2 8\n3 4 8\n2 3 10\n1 3 10\n1 1 7\n2 4 8\n1 6 10\n3 3 9\n2 3 10\n3 4 5\n1 3 10\n1 9 2\n3 1 2\n1 3 5\n3 2 7\n2 7 10\n2 1 6\n2 9 8\n2 1 8\n1 7 10\n1 9 8\n1 1 6\n1 10 8\n2 4 1\n2 1 3\n3 6 9\n1 2 7\n2 3 5\n1 1 3\n3 3 8\n2 3 8\n1 7 2\n2 4 3\n1 7 10\n1 3 8\n3 6 9\n1 5 9\n3 10 5\n1 9 5\n3 5 1\n1 4 1\n3 4 5\n1 3 5\n1 2 5\n1 9 4\n3 1 2\n1 4 9\n1 2 10\n1 5 2\n1 10 2\n3 8 4\n1 3 4\n1 4 3\n3 10 5\n1 10 1\n2 2 4\n3 1 10\n3 6 9\n3 4 5\n1 8 6\n3 1 3\n2 1 9\n3 2 4\n3 5 8\n3 7 8\n1 6 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\nYes\nYes\nYes\nNo\nYes\nNo\nNo\nNo\nNo\nYes\nYes\nNo\nYes\nNo\nNo\nYes\nNo\nNo\nNo\nYes\nYes\nNo\nNo\nYes\nYes\nYes\nNo\nYes\nNo\nNo\nNo\nYes\nYes\nYes\nNo\nYes\nYes\nNo\nYes\nNo\nNo\nNo\nNo\nYes\nYes\nNo\nYes\nYes\nYes\nNo\nNo\nYes\nYes\nNo\nNo\nYes\nYes\nYes\nYes\nNo\nNo\nNo\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
