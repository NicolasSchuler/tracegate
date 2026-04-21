import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 100 80 94\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Aoki\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 2 1 1\n',
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
        input='1 100 1 100\n',
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
        input='22 75 26 32\n',
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
        input='45 72 81 89\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Takahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
