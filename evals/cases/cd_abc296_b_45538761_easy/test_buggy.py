import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='.......*\n........\n........\n........\n........\n........\n........\n........\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'h8\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='........\n*.......\n........\n........\n........\n........\n........\n........\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'a7\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='........\n........\n..*.....\n........\n........\n........\n........\n........\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'c6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='........\n........\n........\n...*....\n........\n........\n........\n........\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'd5\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='........\n........\n........\n........\n......*.\n........\n........\n........\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'g4\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
