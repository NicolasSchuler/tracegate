import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1\n0 80000000000\n80000000000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '80000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1 1\n0 1\n12345678912\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '24691357824\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1 2\n0 1\n98765432123\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '197530864246\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 1 3\n0 1\n56789123456\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '56789123456\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 1 1\n0 1\n99999999999\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '299999999997\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
