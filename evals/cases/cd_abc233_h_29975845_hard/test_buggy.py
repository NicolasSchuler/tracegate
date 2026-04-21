import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n3 3\n4 6\n7 4\n2 5\n6\n3 5 1\n3 5 2\n3 5 3\n3 5 4\n100 200 3\n300 200 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n2\n2\n5\n293\n489\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n100000 0\n1\n100000 0 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n100000 0\n1\n0 100000 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '200000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n0 0\n0 100000\n100000 0\n100000 100000\n16\n0 100000 1\n0 0 1\n100000 0 1\n0 100000 3\n100000 0 3\n100000 0 4\n0 100000 2\n100000 100000 2\n0 100000 4\n100000 0 2\n100000 100000 4\n0 0 4\n0 0 2\n100000 100000 3\n0 0 3\n100000 100000 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n0\n0\n100000\n100000\n200000\n100000\n100000\n200000\n100000\n200000\n200000\n100000\n100000\n100000\n0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
