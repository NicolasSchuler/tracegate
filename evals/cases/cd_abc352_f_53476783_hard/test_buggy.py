import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5 2\n2 3 3\n5 4 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3 -1 -1 -1 -1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-1 -1 -1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8 5\n6 7 3\n8 1 7\n4 5 1\n7 2 1\n6 2 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1 -1 -1 -1 -1 -1 -1 8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='16 120\n5 9 1\n7 9 2\n14 9 3\n6 9 4\n12 9 5\n10 9 6\n2 9 7\n15 9 8\n16 9 9\n4 9 10\n11 9 11\n13 9 12\n1 9 13\n3 9 14\n8 9 15\n7 5 1\n14 5 2\n6 5 3\n12 5 4\n10 5 5\n2 5 6\n15 5 7\n16 5 8\n4 5 9\n11 5 10\n13 5 11\n1 5 12\n3 5 13\n8 5 14\n14 7 1\n6 7 2\n12 7 3\n10 7 4\n2 7 5\n15 7 6\n16 7 7\n4 7 8\n11 7 9\n13 7 10\n1 7 11\n3 7 12\n8 7 13\n6 14 1\n12 14 2\n10 14 3\n2 14 4\n15 14 5\n16 14 6\n4 14 7\n11 14 8\n13 14 9\n1 14 10\n3 14 11\n8 14 12\n12 6 1\n10 6 2\n2 6 3\n15 6 4\n16 6 5\n4 6 6\n11 6 7\n13 6 8\n1 6 9\n3 6 10\n8 6 11\n10 12 1\n2 12 2\n15 12 3\n16 12 4\n4 12 5\n11 12 6\n13 12 7\n1 12 8\n3 12 9\n8 12 10\n2 10 1\n15 10 2\n16 10 3\n4 10 4\n11 10 5\n13 10 6\n1 10 7\n3 10 8\n8 10 9\n15 2 1\n16 2 2\n4 2 3\n11 2 4\n13 2 5\n1 2 6\n3 2 7\n8 2 8\n16 15 1\n4 15 2\n11 15 3\n13 15 4\n1 15 5\n3 15 6\n8 15 7\n4 16 1\n11 16 2\n13 16 3\n1 16 4\n3 16 5\n8 16 6\n11 4 1\n13 4 2\n1 4 3\n3 4 4\n8 4 5\n13 11 1\n1 11 2\n3 11 3\n8 11 4\n1 13 1\n3 13 2\n8 13 3\n3 1 1\n8 1 2\n8 3 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '14 8 15 11 2 5 3 16 1 7 12 6 13 4 9 10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='16 15\n7 3 3\n1 13 12\n12 7 6\n5 14 2\n12 11 7\n7 5 2\n10 8 2\n12 13 13\n10 5 6\n6 7 7\n6 15 12\n4 7 1\n14 2 2\n12 16 3\n8 9 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '14 3 6 10 7 16 9 11 1 13 8 15 2 5 4 12\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
