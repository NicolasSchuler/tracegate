import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\ndpdppd\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'dddpdd\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='3\nddd\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ddd\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='11\nddpdpdppddp\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ddddpdpdddp\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\nppppp\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ddddd\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\ndpppp\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ddddd\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
