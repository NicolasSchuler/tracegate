import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\n2\n1\n0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n1\n2\n3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0\n',
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
        input='123\n456\n789\n987\n654\n321\n0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n321\n654\n987\n789\n456\n123\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '0\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n1000000000\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
