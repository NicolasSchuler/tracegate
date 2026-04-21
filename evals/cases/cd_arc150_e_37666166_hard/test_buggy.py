import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 1\r\nRRLRLLL\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '9\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 10\r\nLLRR\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='23 200\r\nRLRRRLLLLLLLLRRRLLRLRRR\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2207\r\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\r\nR\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1\r\nRL\r\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
