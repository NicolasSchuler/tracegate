import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 4\n1 2\n1 3\n1 4\n3 4\n1 3\n1 4\n2 3\n3 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 6\n1 2\n1 3\n1 4\n3 4\n3 5\n4 5\n1 2\n1 3\n1 4\n1 5\n3 5\n4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 17\n3 6\n2 6\n2 3\n2 4\n3 7\n2 7\n3 5\n2 5\n4 5\n6 8\n4 8\n1 6\n1 3\n1 4\n1 7\n1 5\n1 8\n4 8\n4 7\n7 8\n1 7\n6 8\n6 7\n3 8\n3 7\n1 3\n4 5\n1 5\n2 4\n2 8\n1 2\n2 6\n2 3\n2 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 6\n2 3\n2 4\n3 4\n1 2\n1 3\n1 4\n3 4\n1 3\n1 4\n2 3\n2 4\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
