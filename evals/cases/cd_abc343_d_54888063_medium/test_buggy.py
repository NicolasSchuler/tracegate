import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4\n1 10\n3 20\n2 10\n2 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n3\n2\n2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 3\n1 3\n1 4\n1 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\n1\n1\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 10\n7 2620\n9 2620\n8 3375\n1 3375\n6 1395\n5 1395\n6 2923\n10 3375\n9 5929\n5 1225\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n2\n3\n3\n4\n4\n5\n5\n6\n5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
