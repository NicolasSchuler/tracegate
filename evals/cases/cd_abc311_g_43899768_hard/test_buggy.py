import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n5 4 3\n4 3 2\n3 2 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '48\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 5\n3 1 4 1 5\n9 2 6 5 3\n5 8 9 7 9\n3 2 3 8 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '231\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6 6\n1 300 300 300 300 300\n300 1 300 300 300 300\n300 300 1 300 300 300\n300 300 300 1 300 300\n300 300 300 300 1 300\n300 300 300 300 300 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '810000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 2\n1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
