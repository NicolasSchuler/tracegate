import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 4 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '.#..\n##..\n....\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2 2 1000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '..\n..\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 10 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '##........\n##........\n..........\n..........\n..........\n..........\n..........\n..........\n..........\n#........#\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '#\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 1 1000\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '.\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
