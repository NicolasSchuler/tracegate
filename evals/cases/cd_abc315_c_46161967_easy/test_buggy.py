import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n1 4\n2 10\n2 8\n3 6\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '16\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n4 10\n3 2\n2 4\n4 12\n',
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
        input='2\n1 2\n1 2\n',
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
        input='2\n1 2\n2 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n3 10\n3 10\n2 4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '15\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
