import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '13\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 0 1 0 1 0 0 0 0 1 0 0 1 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 1 1 1 0 0 0 0 1 0 1 0 1 0 1 1 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '766067858140017173\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0 0 1 0 1 1 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 1 1 0 1 0 0 1 1 1 0 0 0 1 0 0 1 0 1 1 1 0 1 1 0 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '3983686466441027124\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='1 0 1 1 0 0 0 0 1 0 1 1 0 0 1 1 1 0 0 1 1 1 0 1 0 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1 1 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0 0 1 0 0 1 0\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '5225951577507482893\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='0 1 1 1 0 1 0 0 1 0 0 1 1 1 0 0 0 0 1 0 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 0 1 1 1 1 0 1 1 0 1 1 0 0 1 0 0 0 1 0 0 1 1 1\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = '16451012835332602158\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
