import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4 3 3 6 2 5 10\n',
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
        input='3 1 4 1 5 9 2\n',
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
        input='1 1 1 1 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Draw\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 36 14 8 48 20 90\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Draw\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 100 1 1 100 1 100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Draw\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
