import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\nFor\nAgainst\nFor\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\nAgainst\nAgainst\nFor\nAgainst\nFor\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1\nFor\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='99\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\nFor\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'Yes\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='99\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\nAgainst\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'No\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
