import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n1 2\n4 7\n5 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n1 2\n4 7\n4 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 2\n1 100\n1 1000000000\n101 1000\n9982 44353\n1000000000 1000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='25 2\n4 17\n5 9\n6 13\n1 7\n9 20\n15 20\n8 16\n2 5\n6 16\n5 16\n2 6\n7 19\n3 17\n7 7\n7 16\n19 20\n2 17\n6 8\n5 8\n6 11\n6 17\n13 20\n7 12\n10 18\n1 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='23 1\n6 14\n9 17\n1 17\n3 6\n6 16\n8 10\n1 4\n4 7\n12 16\n1 1\n1 3\n17 19\n10 15\n9 15\n6 19\n3 19\n1 11\n8 9\n12 20\n12 19\n9 14\n11 16\n5 9\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
