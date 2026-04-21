import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n',
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
        input='10000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '391163238\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '2\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '240\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5376\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
