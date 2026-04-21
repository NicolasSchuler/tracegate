import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='2\nABB\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Alice\nBob\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='4\nAAAAA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Alice\nAlice\nAlice\nAlice\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\nBBAABABA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Bob\nBob\nAlice\nBob\nAlice\nBob\nAlice\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\nAA\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Alice\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\nAB\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Alice\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
