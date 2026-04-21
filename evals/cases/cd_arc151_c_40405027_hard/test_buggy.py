import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7 2\n2 0\n4 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Takahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n1 1\n2 0\n3 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1000000000000000000 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Takahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1\n1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
