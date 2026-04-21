import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 2 2\n1 2 3 5 2 5 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n3\n3\n4\n4\n3\n2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='20 5 4\n20 2 20 2 7 3 11 20 3 8 7 9 1 11 8 20 2 18 11 18\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '14\n14\n14\n14\n13\n13\n13\n11\n8\n9\n9\n11\n13\n14\n14\n14\n14\n14\n14\n13\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
