import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2\n1 2\n3 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '499122178\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='17 5\n1 1 1 1 1\n1 1 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '17\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 10\n2 4 7 1 7 6 1 4 8 1\n5 1 5 2 5 8 4 6 1 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '608849831\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='16 1\n15\n10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '15\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='16 8\n12 4 1 15 11 7 13 6\n16 7 9 6 12 4 2 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '497933799\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
