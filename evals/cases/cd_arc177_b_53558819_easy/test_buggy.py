import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\n1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\nA\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '1\nA\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n01\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\nAAB\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\n11\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4\nAABA\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='30\n111111111111111111111111111111\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '900\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBAAAAAAAAAAAAAAABBBBBBBBBBBBBBAAAAAAAAAAAAAABBBBBBBBBBBBBAAAAAAAAAAAAABBBBBBBBBBBBAAAAAAAAAAAABBBBBBBBBBBAAAAAAAAAAABBBBBBBBBBAAAAAAAAAABBBBBBBBBAAAAAAAAABBBBBBBBAAAAAAAABBBBBBBAAAAAAABBBBBBAAAAAABBBBBAAAAABBBBAAAABBBAAABBAABA\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
