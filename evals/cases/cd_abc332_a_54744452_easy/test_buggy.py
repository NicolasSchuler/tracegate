import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2000 500\n1000 1\n100 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 2000 500\n1000 1\n100 6\n5000 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6600\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2000 500\n1000 1\n1000 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 10000 10000\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n10000 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '100000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='99 10000 10000\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n37 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '10989\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
