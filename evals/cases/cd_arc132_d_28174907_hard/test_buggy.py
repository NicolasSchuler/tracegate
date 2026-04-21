import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 3\n10110\n01101\n',
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
        input='4 2\n000011\n110000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='12 26\n01110111101110111101001101111010110110\n10011110111011011001111011111101001110\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '22\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
