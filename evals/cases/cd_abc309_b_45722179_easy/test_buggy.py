import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n0101\n1101\n1111\n0000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1010\n1101\n0111\n0001\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n11\n11\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '11\n11\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n01010\n01001\n10110\n00110\n01010\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '00101\n11000\n00111\n00110\n10100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n10\n11\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '11\n10\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\n11111\n00011\n11010\n01010\n00010\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '01111\n10011\n01011\n01010\n00100\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
