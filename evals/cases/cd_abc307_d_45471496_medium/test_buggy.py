import subprocess
import sys


def test_condefects_0():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='5\nc)d)e\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'c)d)e\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_1():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='6\n(c)d)e\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'd)e\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_2():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='7\n((c)d)e\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'e\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_3():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='8\n(b(c)d)e\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'e\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )


def test_condefects_4():
    proc = subprocess.run(
        [sys.executable, "buggy.py"],
        input='9\na(b(c)d)e\n',
        capture_output=True,
        text=True,
        timeout=10,
    )
    expected = 'ae\n'
    actual = proc.stdout.strip()
    assert actual == expected.strip(), (
        f"Expected {expected.strip()!r}, got {actual!r}"
    )
