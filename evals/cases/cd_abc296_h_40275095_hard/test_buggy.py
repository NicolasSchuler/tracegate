import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 5\n...#.\n.#...\n....#\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3 3\n###\n###\n###\n',
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
        input='10 1\n.\n#\n.\n.\n.\n.\n.\n.\n#\n.\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 7\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n#######\n',
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
        input='100 7\n#.....#\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n.......\n#.....#\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '108\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
