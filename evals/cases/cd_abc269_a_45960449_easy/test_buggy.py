import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 2 5 3\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '6\nTakahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='10 -20 30 -40\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-700\nTakahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='100 100 100 -100\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '40000\nTakahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 63 76 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '4864\nTakahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='-28 -63 57 54\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '-273\nTakahashi\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
