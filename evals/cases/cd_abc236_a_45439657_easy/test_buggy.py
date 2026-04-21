import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='chokudai\n3 5\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'chukodai\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='aa\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'aa\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='aaaabbbb\n1 8\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'baaabbba\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='ab\n1 2\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ba\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='zzzzzzzzzz\n9 10\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'zzzzzzzzzz\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
